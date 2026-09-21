from arcengine import GameAction
from collections import deque
from typing import Union, List
import random
import torch
import torch.nn as nn
import numpy as np


class ActionModel(nn.Module):

  def __init__(self, actionSpace: List[int], usesMouseClick: bool, gridSize=64, colorCount=16):
    super().__init__()
    self.actionSpace = actionSpace  # e.g. [1, 2, 3, 4] for ls20; the index into this list is what the head predicts
    self.usesMouseClick = usesMouseClick
    self.possibleActions = len(actionSpace)

    self.backbone = nn.Sequential(  # shared feature extractor, has a 9x9 vision due to the kernel
      nn.Conv2d(colorCount, 16, kernel_size=3, padding=1),  # # Padding 1 with kernel 3 keeps 64x64 throughout, so every pixel keeps its position
      nn.ReLU(),  # (batchSize,  16, 64, 64) # one-hot encoded over colours
      nn.Conv2d(16, 32, kernel_size=3, padding=1),
      nn.ReLU(),  # (batchSize,  32, 64, 64)
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),  # (batchSize,  64, 64, 64)
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.ReLU())  # (batchSize, 128, 64, 64) == 128 dims per cell in the grid

    self.actionHead = nn.Sequential(  # which action to choose
      nn.MaxPool2d(kernel_size=4, stride=4),  # (batchSize, 128, 16, 16)
      nn.Flatten(),
      nn.Linear(128 * (gridSize // 4) * (gridSize // 4), 256),
      nn.ReLU(),
      nn.Dropout(p=0.2),
      nn.Linear(256, self.possibleActions))

    if self.usesMouseClick:
      self.mouseClickConvolutions = nn.Sequential(  # where to click -> heatmap with score 0-1
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.ReLU(),  # (batchSize,  64, 64, 64)
        nn.Conv2d(64, 32, kernel_size=3, padding=1),
        nn.ReLU(),  # (batchSize,  32, 64, 64)
        nn.Conv2d(32, 16, kernel_size=3, padding=1),
        nn.ReLU(),  # (batchSize,  16, 64, 64)
        nn.Conv2d(16, 1, kernel_size=3, padding=1))  # (batchSize,   1, 64, 64)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    extractedFeatures = self.backbone(x)
    actionLogits = self.actionHead(extractedFeatures)  # "Which button": a global decision, so the head pools the map down before flattening. -> (B, len(actionSpace))

    if self.usesMouseClick:  # NOTE: Unnecessary logits on inference when not clicking
      mouseClickLogits = torch.flatten(self.mouseClickConvolutions(extractedFeatures), start_dim=1)
      return actionLogits, mouseClickLogits
    else:
      return actionLogits


class StochasticGoose():
  """
  Reward = 1.0 if the action changed the frame, else 0.0. It never sees the game's
  goal, so it produces broad state coverage rather than solutions. It trains online
  while recording, so episode N is played by a slightly better policy than episode 0.
  """

  def __init__(self, available_actions, seed: int = 42):
    self.actionMap = {action.value: action for action in available_actions if action is not GameAction.RESET}  # RESET is the harness's job (arc_collection.py), never the policy's
    self.actionSpace = list(self.actionMap)

    if not self.actionSpace:
      raise ValueError("Goose requires at least one policy action")

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)  # Seed model initialization and PyTorch action sampling.
    random.seed(seed)  # Keep replay sampling independent of Python's global random state.

    self.usesMouseClick = any(action.is_complex() for action in self.actionMap.values())  # "complex" = needs x/y, i.e. ACTION6
    self.experienceBuffer = deque(maxlen=10000)
    self.colorCount = 16
    self.gridSize = 64
    self.stepCount = 0
    self.episodesPlayed = 0
    self.trainingInterval = 2
    self.batchSize = 32

    self.previousFrame = None  # the reward of an action is only knowable one step later, when its result arrives
    self.previousAction = None

    self.actionModel = ActionModel(self.actionSpace, self.usesMouseClick, gridSize=self.gridSize, colorCount=self.colorCount, ).to(self.device)

    self.optimizer = torch.optim.Adam(self.actionModel.parameters(), lr=1e-3, )

  def choose_action(self, frame: np.ndarray, available_actions, episode_finished: bool = False):
    """
    Called once per step with the CURRENT frame. Does three things:
      1. scores the PREVIOUS action (did it change anything?) and stores that experience
      2. occasionally takes a gradient step
      3. samples and returns the next action
    With episode_finished=True only step 1 runs: arc_collection.py calls it that way after a
    terminal frame so the last action still gets its reward.
    """

    indexTensor = torch.from_numpy(frame).to(device=self.device, dtype=torch.long, ).unsqueeze(0)

    frameTensor = torch.zeros((self.colorCount, self.gridSize, self.gridSize), dtype=torch.float32, device=self.device, )
    frameTensor.scatter_(0, indexTensor, 1.0)

    # The current frame is the result of our previous action.
    if self.previousFrame is not None and self.previousAction is not None:
      experience = {"state": self.previousFrame.cpu().numpy().astype(bool), 
                    "action": self.previousAction, 
                    "reward": (1.0 if not torch.equal(self.previousFrame, frameTensor) else 0.0), }
      self.experienceBuffer.append(experience)

    # Process the final result without selecting another action.
    if episode_finished:
      self.previousFrame = None
      self.previousAction = None
      self.episodesPlayed += 1
      return None

    current_actions = {action.value for action in available_actions if action is not GameAction.RESET}

    if current_actions != set(self.actionSpace):
      raise RuntimeError("Available actions changed; the reference Goose assumes a fixed action space")

    self.stepCount += 1
    if self.stepCount % self.trainingInterval == 0:
      self.trainingIteration()

    self.actionModel.eval()  # actionHead has Dropout(0.2)

    with torch.no_grad():
      if self.usesMouseClick:
        actionLogits, mouseClickLogits = self.actionModel(frameTensor.unsqueeze(0))
      else:
        actionLogits = self.actionModel(frameTensor.unsqueeze(0))

    self.actionModel.train()

    actionProbabilities = torch.softmax(actionLogits.squeeze(0), dim=-1)
    actionIndex = torch.multinomial(actionProbabilities, 1).item()
    action = self.actionMap[self.actionSpace[actionIndex]]

    flatCoordinate = None
    action_data = {}

    if action.is_complex():
      coordinateProbabilities = torch.softmax(mouseClickLogits.squeeze(0), dim=-1)
      flatCoordinate = torch.multinomial(coordinateProbabilities, 1).item()

      action_data = {"x": flatCoordinate % self.gridSize, "y": flatCoordinate // self.gridSize, }

    self.previousFrame = frameTensor  # remember, so the NEXT call can score this action
    self.previousAction = (actionIndex, flatCoordinate)

    return action, action_data

  def trainingIteration(self) -> None:
    if len(self.experienceBuffer) < self.batchSize: return  # nothing happens for the first ~32 steps of episode 0, warmup   

    batch = random.sample(self.experienceBuffer, self.batchSize)
    states = torch.stack([torch.from_numpy(experience['state']) for experience in batch]).float().to(self.device)
    rewards = torch.tensor([experience['reward'] for experience in batch], dtype=torch.float32, device=self.device)
    actionIndices = torch.tensor([experience['action'][0] for experience in batch], dtype=torch.long, device=self.device)
    flatCoordinates = [experience['action'][1] for experience in batch]

    if self.usesMouseClick:
      actionLogits, mouseClickLogits = self.actionModel(states)
    else:
      actionLogits = self.actionModel(states)

    logProbabilities = torch.nn.functional.log_softmax(actionLogits, dim=-1)
    selectedLogProbabilities = logProbabilities.gather(1, actionIndices.unsqueeze(1)).squeeze(1)
    loss = -(selectedLogProbabilities * rewards).mean()

    if self.usesMouseClick:
      goodClickBatchIndices = [index for index, coordinate in enumerate(flatCoordinates) if coordinate is not None and batch[index]['reward'] == 1.0]
      if goodClickBatchIndices:
        indicesTensor = torch.tensor(goodClickBatchIndices, dtype=torch.long, device=self.device)
        goodCoordinatesWeClicked = torch.tensor([flatCoordinates[i] for i in goodClickBatchIndices], dtype=torch.long, device=self.device)

        selectedMouseClickLogits = mouseClickLogits[indicesTensor]
        mouseClickLoss = torch.nn.functional.cross_entropy(selectedMouseClickLogits, goodCoordinatesWeClicked)
        loss += mouseClickLoss * 0.1

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
