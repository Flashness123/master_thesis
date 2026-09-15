"""
These isbuild to use the PPO model that I trained on the pure ls20 images trajectories as a policy.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from arcengine import GameAction
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from master_thesis.training.lewm import ArcGridToPixels


class ArcImagePPOPolicy:
  """Use a frozen image-PPO checkpoint through the ARC policy interface."""

  def __init__(self, checkpoint: str | Path, device: str = "auto"):
    checkpoint = Path(checkpoint)

    self.settings = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))

    if self.settings["input"] != "images":
      raise ValueError("Expected an image-PPO checkpoint")

    self.model = PPO.load(str(checkpoint), device=device)
    self.model.policy.set_training_mode(False)
    self.model.policy.requires_grad_(False)

    self.history = int(self.settings["history"])
    if self.history < 1:
      raise ValueError("History must be positive")

    self.frames = deque(maxlen=self.history)

    # Reuse the same palette as image-PPO training.
    self.palette = (ArcGridToPixels(224).palette.mul(255).round().byte().numpy())

    if self.model.observation_space.shape != (3 * self.history, 64, 64):
      raise ValueError("Checkpoint observation shape does not match RGB history")

    self.actions = None

  def reset(self, *, seed: int, available_actions, game_id: str):
    if game_id != self.settings["game"]:
      raise ValueError(f"Checkpoint game {self.settings['game']} differs from {game_id}")

    # Same policy-index -> ARC-action ordering as ArcPPOEnv.
    actions = sorted((action for action in available_actions if action != GameAction.RESET), key=lambda action: action.value, )

    if (not actions or any(action.is_complex() for action in actions) or len(actions) != self.model.action_space.n):
      raise ValueError("Checkpoint requires the original simple discrete action space")

    self.actions = actions
    self.frames.clear()

    set_random_seed(seed, using_cuda=self.model.device.type == "cuda", )

  def choose_action(self, frame: np.ndarray, available_actions):
    if self.actions is None:
      raise RuntimeError("Call reset() before using the policy")

    frame = np.asarray(frame)

    if frame.shape != (64, 64) or np.any((frame < 0) | (frame > 15)):
      raise ValueError("Expected a 64x64 ARC grid with colors 0..15")

    rgb = (self.palette[frame.astype(np.uint8)].transpose(2, 0, 1).copy())

    # Match the training wrapper: repeat the first observation.
    if not self.frames:
      self.frames.extend(rgb.copy() for _ in range(self.history))
    else:
      self.frames.append(rgb)

    observation = np.concatenate(list(self.frames), axis=0)

    index, _ = self.model.predict(observation, deterministic=False, )

    action = self.actions[int(np.asarray(index).item())]

    if action not in available_actions:
      raise RuntimeError("Selected action is no longer available")

    return action, {}
