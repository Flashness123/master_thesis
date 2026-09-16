from __future__ import annotations

import arc_agi
import gymnasium as gym
import numpy as np

from arc_agi import OperationMode
from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arcengine import GameAction, GameState
from gymnasium import spaces

from master_thesis.environments.arc_collection import reset_at_level
from master_thesis.paths import arc_environments_dir


class ArcPPOEnv(gym.Env):
  """Expose local ARC gameplay through Gymnasium."""

  def __init__(self, game_id: str, seed: int, levels: list[int] | None = None, max_steps: int = 500, stop_on_success: bool = False, ):
    super().__init__()

    if max_steps < 1:
      raise ValueError("max_steps must be positive")

    self.arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(arc_environments_dir()), )  # $STABLEWM_HOME/arc_environments

    self.arc = self.arcade.make(game_id, seed=seed, include_frame_data=True, )

    if not isinstance(self.arc, LocalEnvironmentWrapper):
      raise RuntimeError(f"Could not create local game: {game_id}")

    if self.arc._game is None:
      raise RuntimeError("Local ARC game is not loaded")

    self.game = self.arc._game
    number_of_levels = len(self.game._levels)

    self.levels = (list(range(1, number_of_levels + 1)) if levels is None else list(levels))

    if (not self.levels or len(set(self.levels)) != len(self.levels) or any(level < 1 or level > number_of_levels for level in self.levels)):
      raise ValueError(f"Choose distinct levels from 1 to {number_of_levels}")

    self.actions = sorted((action for action in self.arc.action_space if action != GameAction.RESET), key=lambda action: action.value, )

    if not self.actions or any(action.is_complex() for action in self.actions):
      raise ValueError("This pilot supports games with simple discrete actions")

    self.action_space = spaces.Discrete(len(self.actions))
    self.observation_space = spaces.Box(low=0, high=15, shape=(64, 64), dtype=np.uint8, )

    self.max_steps = max_steps
    self.stop_on_success = stop_on_success
    self.finished = True

  def _grid(self):
    grid = np.asarray(self.observation.frame[-1])

    if grid.shape != (64, 64) or np.any((grid < 0) | (grid > 15)):
      raise RuntimeError("Expected a 64x64 ARC grid with colors 0..15")

    return grid.astype(np.uint8, copy=True)

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)

    requested_level = (options or {}).get("start_level")

    self.start_level = (int(self.np_random.choice(self.levels)) if requested_level is None else int(requested_level))

    if self.start_level not in self.levels:
      raise ValueError(f"Level {self.start_level} is not enabled")

    self.observation = reset_at_level(self.arc, self.start_level)

    self.steps = 0
    self.success = False
    self.finished = False

    return self._grid(), {"start_level": self.start_level}

  def step(self, action):
    if self.finished:
      raise RuntimeError("Call reset() before stepping a finished episode")

    action = int(action)

    if not self.action_space.contains(action):
      raise ValueError(f"Invalid policy action: {action}")

    arc_action = self.actions[action]

    if arc_action not in self.arc.action_space:
      raise RuntimeError("The selected action is no longer available")

    before_completed = self.observation.levels_completed
    before_level = self.game._current_level_index

    observation = self.arc.step(arc_action)

    if observation is None:
      raise RuntimeError("ARC returned no observation")

    self.observation = observation
    self.steps += 1

    completed = bool(observation.levels_completed > before_completed or self.game._current_level_index > before_level or observation.state == GameState.WIN)

    self.success = self.success or completed

    terminated = bool(observation.state in (GameState.GAME_OVER, GameState.WIN) or (self.stop_on_success and completed))

    truncated = self.steps >= self.max_steps and not terminated
    self.finished = terminated or truncated

    info = {"start_level": self.start_level, "is_success": bool(self.success), "state": observation.state.name, "steps": self.steps, }

    return (self._grid(), float(completed), terminated, truncated, info, )
