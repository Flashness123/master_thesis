import os
import json
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_pretraining import data as dt
from functools import partial
from pathlib import Path
from PIL import Image
from tensorboard.compat.proto.summary_pb2 import Summary
from master_thesis.models.lewm import SIGReg
from master_thesis.paths import model_dir, stablewm_home, timestamp
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from master_thesis.evaluation.world_model_metrics import world_model_metrics


class PlanningEvaluation(Callback):
  """
  Every `every_steps` optimizer steps: play in the real ARC game with the current model (see evaluation/arc_policies.py).
  mode "goals": reach single goal states, logs planning/*; mode "waypoints": complete levels along human solutions, logs waypoints/*.
  The GIFs go to models/lewm/<run>/planning/step<N>/ and into TensorBoard (tab IMAGES).
  """

  def __init__(self, model, folder: Path, every_steps: int, settings: dict):
    self.model = model  # the JEPA (not the spt.Module wrapper)
    self.folder = folder  # models/lewm/<run>/planning: GIFs per evaluation
    self.every_steps = every_steps
    self.mode = settings.pop("mode", "goals")
    settings.pop("waypoint_every" if self.mode == "goals" else "goal_steps", None)  # the setting of the other mode is not an argument here
    self.settings = settings  # dataset, levels, episodes_per_level, samples, ... from the config

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    step = trainer.global_step
    if step == 0 or step % self.every_steps:
      return

    # imported here so training does not depend on the ARC game
    from master_thesis.evaluation.arc_policies import evaluate_planning, evaluate_waypoints
    from master_thesis.evaluation.world_model_metrics import planning_metrics, waypoint_metrics

    evaluate, summarize = (evaluate_waypoints, waypoint_metrics) if self.mode == "waypoints" else (evaluate_planning, planning_metrics)
    gif_dir = self.folder / f"step{step}"
    was_training = self.model.training
    self.model.eval()  # planning must be deterministic (the predictor has dropout)
    try:
      rows = evaluate(self.model, gif_dir=gif_dir, device=str(pl_module.device), **self.settings)
      metrics = {name: value for name, value in summarize(rows).items() if value == value}  # drop NaN (e.g. no distance measured)
      pl_module.log_dict(metrics, on_step=True, on_epoch=False, batch_size=1)  # next to train/* and val/* in the same TensorBoard

      writer = next(logger for logger in trainer.loggers if isinstance(logger, TensorBoardLogger)).experiment  # stable-pretraining adds its own CSV logger next to ours
      for path in sorted(gif_dir.glob("*.gif")):  # the GIFs just written, as animations in TensorBoard (tab IMAGES shows GIFs as they are)
        width, height = Image.open(path).size
        tag = path.stem.rsplit("_", 1)[0]  # without the outcome (_completed/_failed), so one episode keeps one slider over all steps
        image = Summary.Image(height=height, width=width, colorspace=3, encoded_image_string=path.read_bytes())  # the file itself; add_video would need moviepy (breaks on the pygame install)
        writer._get_file_writer().add_summary(Summary(value=[Summary.Value(tag=f"{self.mode}/{tag}", image=image)]), step)
    except Exception as error:  # a failing evaluation must not end a long training run
      print(f"Planning evaluation at step {step} failed: {error!r}")
    finally:
      self.model.train(was_training)


class SaveCheckpointEvery(Callback):
  """Every `every_steps` optimizer steps, save a full checkpoint (resumable) and the plain model weights."""

  def __init__(self, model, folder: Path, every_steps: int):
    self.model = model  # the JEPA (not the spt.Module wrapper)
    self.folder = folder  # models/lewm/<run>/checkpoints
    self.every_steps = every_steps

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    step = trainer.global_step  # optimizer steps done so far (one per batch)
    if step > 0 and step % self.every_steps == 0:
      trainer.save_checkpoint(self.folder / f"step{step}.ckpt")  # model, optimizer, lr schedule, step counter (for resuming later)
      torch.save(self.model.state_dict(), self.folder / f"weights_step{step}.pt")  # loadable like weights.pt


class ArcGridToPixels:  # ARC: flattened colour grid → normalized RGB image for the ViT
  """
  Convert an ARC-AGI-3 categorical 64x64 grid into RGB pixels for
  vanilla LeWM's existing 3-channel ViT encoder.

  Resize uses nearest-neighbor interpolation so no artificial ARC
  colors are introduced.
  """

  def __init__(self, img_size: int):
    self.img_size = img_size

    # Official ARC-AGI-3 display palette.
    self.palette = torch.tensor(
      [
        [255, 255, 255],  # 0
        [204, 204, 204],  # 1
        [153, 153, 153],  # 2
        [102, 102, 102],  # 3
        [51, 51, 51],  # 4
        [0, 0, 0],  # 5
        [229, 58, 163],  # 6
        [255, 123, 204],  # 7
        [249, 60, 49],  # 8
        [30, 147, 255],  # 9
        [136, 216, 241],  # 10
        [255, 220, 0],  # 11
        [255, 133, 27],  # 12
        [146, 18, 49],  # 13
        [79, 204, 48],  # 14
        [163, 86, 214],  # 15
      ],
      dtype=torch.float32,
    ) / 255.0

    self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, ).view(1, 3, 1, 1)  # ImageNet per-channel mean, shaped to broadcast over [T, 3, H, W]

    self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, ).view(1, 3, 1, 1)  # ImageNet per-channel std (conventional for ViTs; the ViT isn't pretrained here, so it's just a fixed scaling)

  def __call__(self, grids: torch.Tensor, ) -> torch.Tensor:  # grids: [T, 4096] float32 from Lance (T = 4 in training, 1 in ppo.py)

    if grids.shape[-1] != 64 * 64:
      raise ValueError(f"Expected flattened ARC grids with 4096 values, got shape {tuple(grids.shape)}")

    grids = grids.long().reshape(-1, 64, 64, )

    if torch.any((grids < 0) | (grids > 15)):
      raise ValueError("ARC grid contains values outside 0..15")

    palette = self.palette.to(grids.device)

    pixels = palette[grids]

    # [T, H, W, RGB]
    # ->
    # [T, RGB, H, W]
    pixels = pixels.permute(0, 3, 1, 2, )

    pixels = F.interpolate(pixels, size=(self.img_size, self.img_size, ), mode="nearest", )

    mean = self.mean.to(pixels.device)
    std = self.std.to(pixels.device)

    return (pixels - mean) / std  # normalized [T, 3, 224, 224]


class ArcActionEncoding:  # ARC: [id, x, y] → 9-D vector (the only implementation of this encoding)
  """
  Convert canonical ARC actions:

      [action_id, x, y]

  into a 9-D LeWM representation:

      7-D action one-hot + normalized x + normalized y.

  ACTION6 is the coordinate-based action.

  The boundary action [0, -1, -1] becomes an all-zero vector.
  """

  def __call__(self, actions: torch.Tensor, ) -> torch.Tensor:

    if actions.shape[-1] != 3:
      raise ValueError(f"Expected ARC actions with shape (..., 3), "
                       f"got {tuple(actions.shape)}")

    action_ids = actions[..., 0, ].long()

    x = actions[..., 1, ]

    y = actions[..., 2, ]

    boundary = (action_ids == 0)

    valid = (boundary | ((action_ids >= 1) & (action_ids <= 7)))

    if not torch.all(valid):
      raise ValueError("ARC action IDs must be 0..7")

    safe_ids = torch.where(boundary, torch.ones_like(action_ids), action_ids, )

    one_hot = F.one_hot(safe_ids - 1, num_classes=7, ).float()

    # Boundary action is not a real ARC action.
    one_hot = one_hot * (~boundary).unsqueeze(-1)

    is_click = (action_ids == 6)

    if torch.any(is_click & ((x < 0) | (x > 63) | (y < 0) | (y > 63))):
      raise ValueError("ACTION6 coordinates must be in 0..63")

    x_normalized = torch.where(is_click, x / 63.0, torch.zeros_like(x), )

    y_normalized = torch.where(is_click, y / 63.0, torch.zeros_like(y), )

    return torch.cat([one_hot, x_normalized.unsqueeze(-1), y_normalized.unsqueeze(-1), ], dim=-1, )


def build_dataset(cfg: DictConfig):  # opens the Lance data as 4-state windows and attaches the transforms
  if cfg.data.type != "arc":  # the training pipeline supports ARC data only
    raise ValueError(f"Unsupported data type: {cfg.data.type!r}; expected 'arc'")

  dataset_name = cfg.data.name

  dataset = swm.data.load_dataset(dataset_name, transform=None, frameskip=cfg.data.frameskip, num_steps=(cfg.wm.history_size + cfg.wm.num_preds), keys_to_load=list(cfg.data.keys_to_load), )

  grid_transform = (dt.transforms.WrapTorchTransform(ArcGridToPixels(cfg.img_size), source="grid", target="pixels", ))

  action_transform = (dt.transforms.WrapTorchTransform(ArcActionEncoding(), source="action", target="action", ))

  dataset.transform = (dt.transforms.Compose(grid_transform, action_transform, ))

  action_input_dim = 9  # 7 action one-hot + x + y

  with open_dict(cfg):
    cfg.model.action_encoder.input_dim = (action_input_dim)

  print(f"Dataset: {dataset_name}")
  print(f"Samples: {len(dataset)}")
  print(f"Action encoder input: "
        f"{action_input_dim}")

  return dataset


def lewm_forward(self, batch, stage, cfg, ):
  """
    LeWM training objective.

    Encode observations, predict the next latent state,
    and optimize prediction MSE + SIGReg.
    """
  history_size = cfg.wm.history_size
  num_preds = cfg.wm.num_preds

  output = self.model.encode(batch)

  embeddings = output["emb"]
  action_embeddings = output["act_emb"]

  context_embeddings = embeddings[:, :history_size, ]

  context_actions = action_embeddings[:, :history_size, ]

  targets = embeddings[:, num_preds:, ]

  predictions = self.model.predict(context_embeddings, context_actions, )

  prediction_loss = (predictions - targets).pow(2).mean()

  sigreg_loss = self.sigreg(embeddings.transpose(0, 1))

  loss = (prediction_loss + cfg.loss.sigreg.weight * sigreg_loss)

  if self.training:  # training: one value per step, no epoch averages (epochs are short when training by steps)
    self.log_dict({"train/loss": loss.detach(), "train/pred_loss": prediction_loss.detach(), "train/sigreg_loss": sigreg_loss.detach(), }, on_step=True, on_epoch=False, sync_dist=True, )

  else:  # validation: one value per epoch
    metrics = world_model_metrics(model=self.model, embeddings=embeddings, context_embeddings=context_embeddings, context_actions=context_actions, targets=targets, predictions=predictions, actions=batch["action"], )
    self.log_dict({"val/loss": loss.detach(), "val/pred_loss": prediction_loss.detach(), **{f"val/{name}": value for name, value in metrics.items()}, }, on_step=False, on_epoch=True, batch_size=embeddings.shape[0], sync_dist=True, )

  return {"loss": loss, "pred_loss": prediction_loss, "sigreg_loss": sigreg_loss, }


def build_arc_sampler(dataset, train_indices, success_probability_sampling, seed):
  """Sample levels, outcome groups, episodes, then windows."""
  p = float(success_probability_sampling)

  if not 0 < p < 1:
    raise ValueError("success_probability_sampling must be between 0 and 1")

  if not train_indices:
    raise ValueError("The training split contains no windows")

  # Read metadata directly from Lance.
  # These columns do not need to enter model-input batches.
  start_levels = torch.as_tensor(dataset.get_col_data("start_level")).reshape(-1)

  successes = torch.as_tensor(dataset.get_col_data("episode_success")).reshape(-1)

  total_rows = sum(int(length) for length in dataset.lengths)

  if start_levels.numel() != total_rows or successes.numel() != total_rows:
    raise ValueError("Expected one metadata value per stored state")

  # Validate episode labels and identify levels with eligible windows.
  eligible_episodes = sorted({episode for episode, _ in dataset.clip_indices})
  metadata = {}

  for episode in eligible_episodes:
    offset = int(dataset.offsets[episode])
    length = int(dataset.lengths[episode])

    episode_levels = start_levels[offset:offset + length]
    episode_successes = successes[offset:offset + length]

    level_value = float(episode_levels[0].item())
    success_value = float(episode_successes[0].item())

    if (not torch.isfinite(episode_levels).all().item() or level_value < 1 or not level_value.is_integer() or not (episode_levels == level_value).all().item()):
      raise ValueError(f"Episode {episode}: starting level is unknown or inconsistent")

    if (success_value not in (0.0, 1.0) or not (episode_successes == success_value).all().item()):
      raise ValueError(f"Episode {episode}: success label must be consistently 0 or 1")

    metadata[episode] = (int(level_value), bool(success_value))

  # Positions here refer to train_set, not the full dataset.
  episode_positions = {}

  for position, window_index in enumerate(train_indices):
    episode, _ = dataset.clip_indices[window_index]
    episode_positions.setdefault(episode, []).append(position)

  # Group training episodes by starting level and success.
  groups = {}

  for episode in episode_positions:
    level, successful = metadata[episode]
    groups.setdefault(level, {False: [], True: []})
    groups[level][successful].append(episode)

  dataset_levels = {level for level, _ in metadata.values()}
  missing_levels = dataset_levels - set(groups)

  if missing_levels:
    raise ValueError("Eligible levels missing from the training split: "
                     f"{sorted(missing_levels)}. Add coverage or revise the episode split.")

  weights = torch.zeros(len(train_indices), dtype=torch.double)
  level_probability = 1.0 / len(groups)

  for level, outcomes in sorted(groups.items()):
    has_success = bool(outcomes[True])
    has_failure = bool(outcomes[False])

    effective_success_probability = (p if has_success and has_failure else float(has_success))

    print(f"Sampler level {level}: "
          f"{len(outcomes[True])} successful episodes, "
          f"{len(outcomes[False])} unsuccessful episodes; "
          f"level probability={level_probability:.3f}, "
          f"success probability={effective_success_probability:.2f}")

    for successful, episodes in outcomes.items():
      if not episodes:
        continue

      outcome_probability = (effective_success_probability if successful else 1.0 - effective_success_probability)

      episode_probability = (level_probability * outcome_probability / len(episodes))

      for episode in episodes:
        positions = episode_positions[episode]
        weights[positions] = episode_probability / len(positions)

  return torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(train_indices), replacement=True, generator=torch.Generator().manual_seed(seed), )


def build_dataloaders(dataset, cfg: DictConfig):
  train_fraction = cfg.train_split
  val_fraction = cfg.val_split

  if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
    raise ValueError("train_split and val_split must be positive and sum to less than 1")

  # Include only episodes that supply at least one SWM window.
  episode_ids = sorted({episode for episode, _ in dataset.clip_indices})

  generator = torch.Generator().manual_seed(cfg.seed)
  order = torch.randperm(len(episode_ids), generator=generator).tolist()
  episode_ids = [episode_ids[index] for index in order]

  num_train = int(len(episode_ids) * train_fraction)
  num_val = int(len(episode_ids) * val_fraction)
  num_test = len(episode_ids) - num_train - num_val

  if min(num_train, num_val, num_test) < 1:
    raise ValueError("Not enough eligible episodes for these split proportions")

  episode_groups = (episode_ids[:num_train], episode_ids[num_train:num_train + num_val], episode_ids[num_train + num_val:], )

  # Map each episode to one split: 0=train, 1=val, 2=test.
  episode_to_split = {}
  for split_index, episodes in enumerate(episode_groups):
    for episode in episodes:
      episode_to_split[episode] = split_index

  # Assign every window according to its episode.
  window_groups = [[], [], []]
  for window_index, (episode, _) in enumerate(dataset.clip_indices):
    split_index = episode_to_split[episode]
    window_groups[split_index].append(window_index)

  train_set, val_set, test_set = [torch.utils.data.Subset(dataset, indices) for indices in window_groups]

  for name, episodes, indices in zip(("train", "val", "test"), episode_groups, window_groups):
    print(f"{name}: {len(episodes)} episodes, {len(indices)} windows")
  loader_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
  sampling_cfg = cfg.get("sampling")

  if sampling_cfg is not None:
    sampler = build_arc_sampler(dataset=dataset, train_indices=window_groups[0], success_probability_sampling=sampling_cfg.success_probability_sampling, seed=int(cfg.seed) + 1, )

    train_loader_cfg = dict(loader_cfg)

    # The sampler controls sample selection and order.
    train_loader_cfg.pop("shuffle", None)

    train_loader = torch.utils.data.DataLoader(train_set, **train_loader_cfg, sampler=sampler, generator=generator, )
  else:
    train_loader = torch.utils.data.DataLoader(train_set, **loader_cfg, generator=generator, )

  eval_loader_cfg = dict(loader_cfg)
  eval_loader_cfg["shuffle"] = False
  eval_loader_cfg["drop_last"] = False

  # Shuffled so each batch mixes episodes; otherwise batch statistics (SIGReg, batch_latent_std) are inflated by consecutive windows of one episode.
  val_loader = torch.utils.data.DataLoader(val_set, **{**eval_loader_cfg, "shuffle": True}, generator=generator, )
  test_loader = torch.utils.data.DataLoader(test_set, **eval_loader_cfg)

  return train_loader, val_loader, test_loader


@hydra.main(version_base=None, config_path="../../../configs", config_name=None)
def main(cfg: DictConfig) -> None:
  pl.seed_everything(cfg.seed, workers=True, )
  if "run_name" not in cfg:
    raise ValueError("No training experiment selected. Use --config-name train/arc_lewm.")

  dataset = build_dataset(cfg)

  # sample = dataset[0]
  # print("pixels:", sample["pixels"].shape, sample["pixels"].dtype, )
  # print("action:", sample["action"].shape, sample["action"].dtype, )
  # print("action[0]:", sample["action"][0], )

  train_loader, val_loader, test_loader = build_dataloaders(dataset, cfg)

  model = hydra.utils.instantiate(cfg.model, )

  init_weights = cfg.get("init_weights")

  if init_weights:
    checkpoint = Path(init_weights)

    saved_model_config = OmegaConf.load(checkpoint.parent / "model_config.json")  # architecture saved next to weights.pt

    if (OmegaConf.to_container(saved_model_config, resolve=True) != OmegaConf.to_container(cfg.model, resolve=True)):
      raise ValueError("Current model configuration differs from the source checkpoint")

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True, )

    model.load_state_dict(state_dict, strict=True)
    print(f"Initialized LeWM from: {checkpoint}")

  max_steps = cfg.trainer.get("max_steps") or -1  # if set (> 0), the run length is given in optimizer steps (use with trainer.max_epochs=-1)
  total_steps = max_steps if max_steps > 0 else cfg.trainer.max_epochs * len(train_loader)  # length of the learning-rate schedule

  if total_steps < 2:
    raise ValueError("LeWM training needs at least 2 optimizer steps "
                     "for the warmup/cosine scheduler. "
                     "Use a smaller batch size, more epochs, or more data.")

  warmup_steps = max(1, int(0.01 * total_steps), )

  optimizers = {"model_opt": {"modules": "model", "optimizer": dict(cfg.optimizer), "scheduler": {"type": "LinearWarmupCosineAnnealingLR", "warmup_steps": warmup_steps, "max_steps": total_steps, }, "interval": "step", }, }
  module = spt.Module(model=model, sigreg=SIGReg(**cfg.loss.sigreg.kwargs, ), forward=partial(lewm_forward, cfg=cfg, ), optim=optimizers, )

  data_module = spt.data.DataModule(train=train_loader, val=val_loader, )

  run_name = f"{cfg.run_name}_{timestamp()}"  # description from the config/command line + MMDD-HHMM
  run_dir = model_dir("lewm", run_name)  # models/lewm/<run_name>: weights, configs and logs of this run

  run_dir.mkdir(parents=True, exist_ok=False)  # fails if this name already exists

  with open_dict(cfg):
    cfg.run_name = run_name  # store the final folder name in the saved config

  OmegaConf.save(cfg, run_dir / "train_config.yaml", )  # full training config (interpolations unresolved)
  OmegaConf.save(OmegaConf.create(list(HydraConfig.get().overrides.task)), run_dir / "hydra_overrides.yaml")  # exactly what was typed on the command line
  (run_dir / "model_config.json").write_text(json.dumps(OmegaConf.to_container(cfg.model, resolve=True), indent=2), encoding="utf-8")  # architecture needed to rebuild the model (ppo.py, init_weights); written first so intermediate weights are usable

  logger = TensorBoardLogger(save_dir=str(run_dir), name="tensorboard", default_hp_metric=False, )

  # stable-pretraining always has a cache folder (default ~/.cache/stable-pretraining) where it writes its own last.ckpt after every epoch:
  # keep it inside the data root and switch that per-epoch checkpoint off (we save our own below)
  spt.set(cache_dir=str(stablewm_home() / "spt_cache"), requeue_checkpoint=False)

  callbacks = []
  every_steps = cfg.get("checkpoint_every_steps")  # e.g. 10000; None/0 = only the final weights
  if every_steps:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    callbacks.append(SaveCheckpointEvery(model, checkpoint_dir, every_steps))  # -> checkpoints/step10000.ckpt + weights_step10000.pt

  planning_cfg = cfg.get("planning")  # optional: plan in the real ARC game during training
  if planning_cfg and planning_cfg.get("every_steps"):
    settings = {key: value for key, value in OmegaConf.to_container(planning_cfg, resolve=True).items() if key != "every_steps"}  # dataset, goal_steps, episodes_per_level, ...
    callbacks.append(PlanningEvaluation(model, run_dir / "planning", planning_cfg.every_steps, settings))

  trainer = pl.Trainer(**cfg.trainer, default_root_dir=run_dir, logger=logger, callbacks=callbacks, )
  trainer.callbacks = [callback for callback in trainer.callbacks if type(callback).__name__ != "HardwareMonitor"]  # drop stable-pretraining's hardware/* curves (CPU, RAM, disk, network)

  manager = spt.Manager(trainer=trainer, module=module, data=data_module, )

  manager()

  torch.save(model.state_dict(), run_dir / "weights.pt")  # final weights (no optimizer state)

  # Everything is written; end the process immediately. Otherwise the persistent dataloader workers keep it
  # alive after training (on SLURM the job then runs into its time limit instead of finishing).
  logger.save()  # flush the TensorBoard events first, os._exit skips all cleanup
  print(f"Training complete: {run_dir}")
  os._exit(0)


if __name__ == "__main__":
  main()
