"""
JEPA-TTT (Zhang et al. 2026, arXiv:2610.00722) for a trained LeWM: keep training the predictor while the agent plays a
level, on the transitions it plays there ("test-time training"), so the model adapts to dynamics it has never seen, e.g.
the changers moving on tracks in real levels 5-7.

As in the paper (Algorithm 1, Eq. 4, Table 2), only the predictor side learns, with two adaptations to our models:
  trained      the predictor and the prediction projection (pred_proj). The paper also trains the action encoder; here
               that is off by default (train_action_encoder): its shifts change what an action means (rotated or delayed
               actions), while in ls20 a move keeps its meaning in every level and only the state's dynamics change,
               which the predictor holds. With 4 actions the action encoder is a table of 4 vectors that every
               prediction depends on, so a few windows of one level would shift all predictions at once.
  frozen       the encoder and its projector, so the latents of states and goals that the planner compares do not move;
               nothing can collapse, so SIGReg / TC-SIGReg are not needed (encoder-TTT is a later experiment). Delta-JEPA's
               action loss is left out too: it trains the action decoder on the frozen encoder's latents, unused in planning.
  loss         the run's own prediction loss (prediction_objective), as the paper keeps its pretraining loss (LeWM's
               one-step loss, Eq. 4): one-step teacher-forced for runs trained with it, the Semigroup rollout loss (from
               1 .. context real frames, num_preds steps, discount) for runs trained with that, both if both. A run would
               otherwise be pushed towards an objective it never had, e.g. one-step on a rollout-trained predictor. The
               targets are the frozen encoder's latents (no gradient).
  mode         the whole model stays in eval mode: dropout off, the BatchNorm statistics of the projections fixed
  data         its own buffer of everything played in the level. Every `frames` frames in a row of one continuous piece of
               play are one training window (the paper's dense windows: 4 for the one-step loss, 8 for the rollout loss);
               a new attempt starts a new piece, so no window spans a restart. The buffer only grows (dense replay).
  schedule     at most one AdamW step (lr 5e-5, weight decay 1e-3, batch 64, gradient clip 1.0) per update() call, which the
               caller makes once per plan; the first step comes once 64 windows exist
  persistence  predictor, optimizer state and buffer carry over every attempt in the level (resetting them per episode loses
               most of the gain in the paper); the next level gets a new TestTimeTrainer
The trained checkpoint is never changed: the trainer adapts its own copy (TestTimeTrainer.model), which the planner uses
instead. Actions are stored by their ARC id only (ls20 has no clicks).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from master_thesis.training.lewm import ArcActionEncoding, ArcGridToPixels


def prediction_objective(train_config) -> dict:
  """
  GETS:    train_config -- path of a run's train_config.yaml, as training/lewm.py saves it.
  DOES:    reads which prediction losses the run was trained with (extensions.one_step_loss / rollout_loss) and their
           settings (loss.one_step, loss.rollout, wm). Runs from before the extensions block used the one-step loss; runs
           from before 24.10.26 have no loss.rollout.context and started their rollout from 1 real frame.
  RETURNS: {"history_size": int, "one_step": its weight or None, "rollout": {"weight", "context", "num_preds", "discount"} or None}
  """
  from omegaconf import OmegaConf

  config = OmegaConf.load(train_config)
  select = lambda key, default: OmegaConf.select(config, key, default=default)
  rollout = bool(select("extensions.rollout_loss", False))
  return {"history_size": int(config.wm.history_size),
          "one_step": float(select("loss.one_step.weight", 1.0)) if bool(select("extensions.one_step_loss", True)) else None,
          "rollout": {"weight": float(config.loss.rollout.weight), "context": int(select("loss.rollout.context", 1)), "num_preds": int(config.wm.num_preds),
                      "discount": float(config.loss.rollout.discount)} if rollout else None}


class TestTimeTrainer:
  """
  GETS:    model     -- a trained JEPA; it stays untouched, the trainer adapts its own copy.
           img_size  -- the encoder's image size (train_config img_size, 224).
           objective -- the run's prediction losses, from prediction_objective(train_config).
           train_action_encoder -- also train the action encoder, as the paper does (default off, see the module docstring).
           lr, weight_decay, batch, clip -- the paper's Table 2.
           seed      -- for drawing the training windows.
  DOES:    observe() encodes every played frame once with the frozen encoder and stores it with the action taken;
           update() takes one AdamW step of the run's prediction loss on a random batch of windows from everything
           stored so far.
  RETURNS: (object) .model   -- the adapted copy, for the planner.
                    .frames  -- frames per training window.
                    .windows -- (piece, first frame) of every training window stored so far.
                    .losses  -- the loss of every update.
  """

  def __init__(self, model, img_size: int, objective: dict, train_action_encoder: bool = False, lr: float = 5e-5, weight_decay: float = 1e-3, batch: int = 64, clip: float = 1.0, seed: int = 0):
    if objective["one_step"] is None and objective["rollout"] is None:
      raise ValueError("the objective has no prediction loss")
    self.model = copy.deepcopy(model).eval()  # eval mode for good: dropout off, BatchNorm statistics fixed (the paper: trainable modules run in eval mode)
    self.model.requires_grad_(False)
    trainable = [self.model.predictor, self.model.pred_proj] + ([self.model.action_encoder] if train_action_encoder else [])  # encoder and projector always stay frozen
    for module in trainable:
      module.requires_grad_(True)
    self.parameters = [parameter for module in trainable for parameter in module.parameters()]
    self.optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=weight_decay)
    self.device = next(self.model.parameters()).device
    self.to_pixels, self.encode_action = ArcGridToPixels(img_size), ArcActionEncoding()  # the training's preprocessing of grids and actions
    self.objective, self.batch, self.clip = objective, batch, clip
    rollout = objective["rollout"]
    self.frames = max(objective["history_size"] + 1 if objective["one_step"] is not None else 0, rollout["context"] + rollout["num_preds"] if rollout else 0)  # as training/lewm.py build_dataset: 4 for the one-step loss, 8 for the rollout loss
    self.generator = torch.Generator().manual_seed(seed)
    self.pieces = []  # one entry per continuous piece of play: {"latents": the latent of every frame, "actions": the action id taken in each frame but the last}
    self.windows = []  # (piece, first frame) of every training window: `frames` frames in a row of one piece
    self.losses = []

  @torch.no_grad()
  def encode(self, grid):
    """One grid [4096] or [64, 64] -> its latent [192] from the frozen encoder. Each frame is encoded once: the encoder never changes."""
    pixels = self.to_pixels(torch.as_tensor(np.asarray(grid).reshape(1, -1), device=self.device))  # [1, 3, img, img]
    return self.model.encode({"pixels": pixels[:, None]})["emb"][0, 0]  # the encoder expects a time axis: here one frame

  def observe(self, grid, action: int, next_grid, new_attempt: bool = False):
    """
    GETS:    grid        -- the state the action was taken in.
             action      -- its ARC action id (1-7).
             next_grid   -- the state it led to.
             new_attempt -- True for the first move after a jump: a new attempt at the level, a task started from a
                            replayed state. Starts a new piece, so no window spans the jump. (A lost life is a real
                            transition of the game and stays inside the piece.)
    DOES:    stores the latent of next_grid (and of grid when a piece starts) and the action; every frame that completes
             `frames` frames of its piece adds one training window.
    """
    if new_attempt or not self.pieces:
      self.pieces.append({"latents": [self.encode(grid)], "actions": []})
    piece = self.pieces[-1]
    piece["actions"].append(int(action))
    piece["latents"].append(self.encode(next_grid))
    if len(piece["latents"]) >= self.frames:  # the newest `frames` frames form a window
      self.windows.append((len(self.pieces) - 1, len(piece["latents"]) - self.frames))

  def update(self):
    """
    DOES:    one AdamW step on `batch` windows drawn uniformly from all stored ones (no window twice in a batch), with the
             run's prediction loss, computed as training/lewm.py lewm_forward does:
               one-step  the predictor gets the real latents of the first history_size frames and the actions taken in
                         them and predicts the next latent at every position (the paper's Eq. 4)
               rollout   from the last 1, 2, ..., context real frames before the targets, the predictor imagines num_preds
                         steps feeding back its own predictions; step k is weighted by discount^(k-1), averaged over the
                         start lengths
             Nothing happens until `batch` windows exist.
    RETURNS: the loss of this step, or None when there are not enough windows yet.
    """
    if len(self.windows) < self.batch:
      return None
    chosen = [self.windows[index] for index in torch.randperm(len(self.windows), generator=self.generator)[:self.batch].tolist()]
    latents = torch.stack([torch.stack(self.pieces[piece]["latents"][first:first + self.frames]) for piece, first in chosen])  # z_0 .. z_(frames-1) of every window  [B, frames, 192]
    ids = torch.tensor([self.pieces[piece]["actions"][first:first + self.frames - 1] for piece, first in chosen], dtype=torch.float32)  # the action taken in each frame but the last  [B, frames - 1]
    actions = self.model.action_encoder(self.encode_action(torch.stack([ids, -torch.ones_like(ids), -torch.ones_like(ids)], dim=-1)).to(self.device))  # [id, x, y] -> LeWM's 9 numbers -> action latents  [B, frames - 1, 192]
    history_size, loss = self.objective["history_size"], 0.0
    if self.objective["one_step"] is not None:
      predictions = self.model.predict(latents[:, :history_size], actions[:, :history_size])  # position j predicts z_(j+1) from the real z_0 .. z_j
      loss = loss + self.objective["one_step"] * (predictions - latents[:, 1:history_size + 1]).pow(2).mean()  # squared error averaged over windows, positions and the 192 numbers
    if self.objective["rollout"] is not None:
      context, num_preds, discount = (self.objective["rollout"][key] for key in ("context", "num_preds", "discount"))
      targets = latents[:, context:context + num_preds]  # the same num_preds targets for every start length  [B, num_preds, 192]
      weights = discount ** torch.arange(num_preds, device=self.device, dtype=torch.float32)  # discount^(k-1)
      errors = [(self.model.predict_rollout(latents[:, context - real:context], actions[:, context - real:context + num_preds - 1], history_size) - targets).pow(2).mean(dim=(0, 2)) for real in range(1, context + 1)]  # per start length: error of each step  [num_preds]
      loss = loss + self.objective["rollout"]["weight"] * sum((weights / weights.sum() * error).sum() for error in errors) / context  # normalized as in training: the scale of a single-step error
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters, self.clip)
    self.optimizer.step()
    self.losses.append(loss.item())
    return self.losses[-1]
