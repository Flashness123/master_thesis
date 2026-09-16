from __future__ import annotations

import argparse
import json
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from gymnasium import spaces
from hydra.utils import instantiate
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from master_thesis.environments.arc_ppo import ArcPPOEnv
from master_thesis.paths import model_dir, timestamp
from master_thesis.evaluation.arc_policy import LevelEvaluation
from master_thesis.training.lewm import ArcGridToPixels


class ArcPolicyInput(gym.Wrapper):
  """Convert real grids into RGB history or frozen LeWM embeddings."""

  def __init__(self, env, history: int = 3, lewm=None, img_size: int = 224, ):
    super().__init__(env)

    if history < 1:
      raise ValueError("history must be positive")

    self.history = history
    self.frames = deque(maxlen=history)
    self.lewm = lewm
    self.preprocess = ArcGridToPixels(img_size)

    if lewm is None:
      self.palette = (self.preprocess.palette.mul(255).round().byte().numpy())

      self.observation_space = spaces.Box(low=0, high=255, shape=(3 * history, 64, 64), dtype=np.uint8, )
    else:
      self.lewm.eval()
      self.lewm.requires_grad_(False)
      self.device = next(lewm.parameters()).device

      # Infer the embedding dimension from the actual loaded model.
      sample = self._convert(np.zeros((64, 64), dtype=np.uint8))

      self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(history * sample.size, ), dtype=np.float32, )

  def _convert(self, grid):
    if self.lewm is None:
      return self.palette[grid].transpose(2, 0, 1).copy()

    with torch.no_grad():
      pixels = self.preprocess(torch.as_tensor(grid, device=self.device, ).reshape(1, -1))

      # pixels: [1, C, H, W] -> [batch=1, time=1, C, H, W]
      embedding = self.lewm.encode({"pixels": pixels.unsqueeze(0), })["emb"][0, 0]

    return embedding.cpu().numpy().astype(np.float32, copy=True, )

  def reset(self, *, seed=None, options=None):
    grid, info = self.env.reset(seed=seed, options=options)
    observation = self._convert(grid)

    # At episode start there is no history, so repeat the initial observation.
    self.frames.clear()
    for _ in range(self.history):
      self.frames.append(observation.copy())

    return np.concatenate(list(self.frames), axis=0), info

  def step(self, action):
    grid, reward, terminated, truncated, info = self.env.step(action)
    self.frames.append(self._convert(grid))

    return (np.concatenate(list(self.frames), axis=0), reward, terminated, truncated, info, )


def main():
  parser = argparse.ArgumentParser(description="Train PPO on ARC images or frozen LeWM embeddings")

  parser.add_argument("--input", choices=["images", "lewm"], required=True, )
  parser.add_argument("--run-name", required=True, help="Run name without timestamp, e.g. ls20_lewm-l1_l1; saved as models/ppo/<run-name>_<MMDD-HHMM>")
  parser.add_argument("--game", default="ls20-9607627b")
  parser.add_argument("--levels", nargs="+", type=int, default=None)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--steps", type=int, default=204_800)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--history", type=int, default=3)
  parser.add_argument("--eval-every", type=int, default=20_480)
  parser.add_argument("--eval-episodes", type=int, default=10)
  parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", )
  parser.add_argument("--lewm-run", default=None, help="LeWM model folder name under models/lewm")

  args = parser.parse_args()

  # PPO collects complete batches of 2048 steps with this configuration.
  if args.steps < 2048 or args.steps % 2048:
    parser.error("--steps must be a positive multiple of 2048")

  if min(args.max_steps, args.history, args.eval_every, args.eval_episodes, ) < 1:
    parser.error("Step limits, history, and evaluation counts must be positive")

  if args.input == "lewm" and args.lewm_run is None:
    parser.error("--input lewm requires --lewm-run")

  if args.input == "images" and args.lewm_run is not None:
    parser.error("--lewm-run is only used with --input lewm")

  if args.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
  else:
    device = args.device

  if device == "cuda" and not torch.cuda.is_available():
    parser.error("CUDA was requested but is unavailable")

  set_random_seed(args.seed, using_cuda=device == "cuda", )

  lewm = None
  img_size = 224

  if args.input == "lewm":
    lewm_dir = model_dir("lewm", args.lewm_run)  # models/lewm/<lewm_run>
    training_config = OmegaConf.load(lewm_dir / "train_config.yaml")  # full LeWM training config

    if training_config.data.type != "arc":
      raise ValueError("The selected LeWM run was not configured for ARC")

    img_size = int(training_config.img_size)

    model_config = json.loads((lewm_dir / "model_config.json").read_text(encoding="utf-8"))  # architecture with _target_ entries
    lewm = instantiate(model_config)  # build JEPA with random weights (Hydra, as in lewm.py)
    lewm.load_state_dict(torch.load(lewm_dir / "weights.pt", map_location="cpu", weights_only=True), strict=True)  # then load the trained weights

    lewm.to(device).eval()
    lewm.requires_grad_(False)

  run_name = f"{args.run_name}_{timestamp()}"  # e.g. ls20_lewm-l1_l1_0916-1432
  run_dir = model_dir("ppo", run_name)  # models/ppo/<run_name>

  # Require a new run name to avoid mixing or overwriting experiments.
  run_dir.mkdir(parents=True, exist_ok=False)

  train_base = ArcPPOEnv(game_id=args.game, seed=args.seed, levels=args.levels, max_steps=args.max_steps, stop_on_success=False, )

  levels = train_base.levels

  train_env = Monitor(ArcPolicyInput(train_base, history=args.history, lewm=lewm, img_size=img_size, ), filename=str(run_dir / "train.monitor.csv"), info_keywords=("start_level", "is_success", "state"), )

  eval_env = ArcPolicyInput(ArcPPOEnv(game_id=args.game, seed=args.seed + 100_000, levels=levels, max_steps=args.max_steps, stop_on_success=True, ), history=args.history, lewm=lewm, img_size=img_size, )

  settings = vars(args).copy()
  settings.update({"run_name": run_name, "device": device, "levels": levels, "lewm_img_size": img_size if lewm is not None else None, })  # run_name with timestamp

  (run_dir / "train_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8", )

  model = PPO("CnnPolicy" if args.input == "images" else "MlpPolicy", train_env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, policy_kwargs={
    "net_arch": {
      "pi": [128, 128],
      "vf": [128, 128], }, }, seed=args.seed, device=device, tensorboard_log=str(run_dir / "tensorboard"), verbose=1,
              )

  callback = LevelEvaluation(env=eval_env, levels=levels, output_path=run_dir / "evaluation.jsonl", every_steps=args.eval_every, episodes_per_level=args.eval_episodes, seed=args.seed + 100_000, )

  try:
    model.learn(total_timesteps=args.steps, callback=callback, tb_log_name="ppo", )
    model.save(str(run_dir / "policy"))
  finally:
    train_env.close()
    eval_env.close()

  print(f"Saved PPO policy and results to {run_dir}")


if __name__ == "__main__":
  main()
