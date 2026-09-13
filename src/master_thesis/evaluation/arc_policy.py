from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback


class LevelEvaluation(BaseCallback):
  """Evaluate direct-start success separately for each enabled level."""

  def __init__(self, env, levels, output_path: Path, every_steps: int = 20_480, episodes_per_level: int = 10, seed: int = 100_000, ):
    super().__init__()

    if every_steps < 1 or episodes_per_level < 1:
      raise ValueError("Evaluation interval and episode count must be positive")

    self.env = env
    self.levels = list(levels)
    self.output_path = Path(output_path)
    self.every_steps = every_steps
    self.episodes_per_level = episodes_per_level
    self.seed = seed
    self.last_evaluation = -1

  def _evaluate(self):
    cuda_devices = (list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [])

    # Restore training's Torch RNG state after evaluation.
    with torch.random.fork_rng(devices=cuda_devices):
      with self.output_path.open("a", encoding="utf-8") as file:
        for level in self.levels:
          successes = []
          lengths = []

          for episode in range(self.episodes_per_level):
            episode_seed = self.seed + level * 10_000 + episode

            # Seed action sampling separately for each evaluation attempt.
            torch.manual_seed(episode_seed)

            observation, _ = self.env.reset(seed=episode_seed, options={"start_level": level}, )

            terminated = False
            truncated = False

            while not (terminated or truncated):
              action, _ = self.model.predict(observation, deterministic=False, )

              observation, _, terminated, truncated, info = self.env.step(int(np.asarray(action).item()))

            successes.append(int(info["is_success"]))
            lengths.append(info["steps"])

            file.write(json.dumps({"training_steps": self.num_timesteps, "level": level, "episode_seed": episode_seed, "success": bool(info["is_success"]), "actions": info["steps"], "state": info["state"], "truncated": bool(truncated), }) + "\n")

          success_rate = float(np.mean(successes))

          self.logger.record(f"eval/level_{level}/success_rate", success_rate, )
          self.logger.record(f"eval/level_{level}/mean_actions", float(np.mean(lengths)), )

          print(f"Evaluation at {self.num_timesteps} steps:"
                f" level {level},"
                f" success {sum(successes)}/{len(successes)}")

    self.last_evaluation = self.num_timesteps
    self.logger.dump(self.num_timesteps)

  def _on_training_start(self):
    self._evaluate()

  def _on_step(self):
    if self.num_timesteps - self.last_evaluation >= self.every_steps:
      self._evaluate()

    return True

  def _on_training_end(self):
    self._evaluate()
