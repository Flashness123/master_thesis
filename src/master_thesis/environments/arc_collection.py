from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import arc_agi
import numpy as np
from arc_agi import OperationMode
from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arcengine import FrameDataRaw, GameState

from master_thesis.policies.arc_random import ArcRandomPolicy
"""
ARC-AGI-3 trajectory collection.

The interaction structure is adapted from the ideas used in
InexperiencedMe/ARC-AGI-3-Playground

This version is intentionally reduced to the functionality required
for world-model data collection in this Master's thesis
"""

MODE_MAP = {"normal": OperationMode.NORMAL, "offline": OperationMode.OFFLINE, "online": OperationMode.ONLINE, }


def get_stablewm_home() -> Path:
  try:
    return Path(os.environ["STABLEWM_HOME"])
  except KeyError as exc:
    raise RuntimeError("STABLEWM_HOME is not set") from exc


def default_recording_path(game_id: str, seed: int, policy_name: str = "random", start_level: int | None = None, ) -> Path:
  timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

  recordings_dir = get_stablewm_home() / "datasets" / "arc_recordings"
  recordings_dir.mkdir(parents=True, exist_ok=True)

  short_game_id = game_id.split("-")[0]
  level_suffix = "" if start_level is None else f"-level{start_level}"

  return recordings_dir / (f"{policy_name}-{short_game_id}"
                           f"-seed{seed}{level_suffix}-{timestamp}.jsonl")


def record_frame(file: TextIO, frame_data: FrameDataRaw, start_level: int | None = None, ) -> None:
  """
  Store one ARC response using the same basic JSONL structure used
  by ARC-AGI-3-Playground.

  `frame` is added explicitly because it is not reliably included
  in FrameDataRaw.model_dump().
  """
  data = frame_data.model_dump(mode="json")
  data["frame"] = [np.asarray(frame).tolist() for frame in frame_data.frame]

  event = {"timestamp": datetime.now(timezone.utc).isoformat(), "data": data, }

  if start_level is not None:
    event["start_level"] = start_level

  file.write(json.dumps(event) + "\n")
  file.flush()


def reset_at_level(env: LocalEnvironmentWrapper, start_level: int, ) -> FrameDataRaw:
  """Reset a local game to a one-based starting level."""
  if not isinstance(env, LocalEnvironmentWrapper):
    raise ValueError("Starting at a selected level requires a local environment")

  game = env._game

  if game is None:
    raise RuntimeError("Local ARC game is not loaded")

  if not 1 <= start_level <= len(game._levels):
    raise ValueError(f"start_level must be between 1 and {len(game._levels)}, "
                     f"got {start_level}")

  def reset_selected_level() -> None:
    game.full_reset()
    game.set_level(start_level - 1)

  original_handle_reset = game.handle_reset

  try:
    game.handle_reset = reset_selected_level
    observation = env.reset()
  finally:
    game.handle_reset = original_handle_reset

  if observation is None:
    raise RuntimeError("ARC reset returned no observation")

  if game._current_level_index != start_level - 1:
    raise RuntimeError("ARC reset did not select the requested level")

  if observation.state != GameState.NOT_FINISHED:
    raise RuntimeError(f"Unexpected initial game state: {observation.state}")

  return observation


def collect_game(game_id: str, max_steps: int, seed: int, mode: str, output_path: Path, policy_name: str = "random", start_level: int | None = None) -> Path:
  """
  Collect one ARC trajectory using a random policy.

  `max_steps` counts actual policy actions only.

  RESET actions caused by GAME_OVER are not counted as policy steps.
  Every initial/reset state and every state resulting from an executed
  action is recorded.
  """
  if max_steps < 1:
    raise ValueError("max_steps must be at least 1")

  if mode not in MODE_MAP:
    raise ValueError(f"Unknown ARC mode: {mode}")

  if start_level is not None:
    if start_level < 1:
      raise ValueError("start_level must be at least 1")
    if mode == "online":
      raise ValueError("--start-level requires normal or offline mode")

  stablewm_home = (get_stablewm_home())

  environments_dir = (stablewm_home / "arc_environments")

  environments_dir.mkdir(parents=True, exist_ok=True, )

  arcade = arc_agi.Arcade(operation_mode=MODE_MAP[mode], environments_dir=str(environments_dir), )

  env = arcade.make(game_id, seed=seed, include_frame_data=True, )

  if env is None:
    raise RuntimeError(f"Could not create ARC game: {game_id}")

  observation = (env.observation_space)

  if start_level is not None:
    observation = reset_at_level(env, start_level)
  else:
    observation = env.observation_space

    if observation is None:
      observation = env.reset()

  if observation is None:
    raise RuntimeError("ARC environment returned no initial observation")

  if observation is None:
    raise RuntimeError("ARC environment returned no initial observation")

  if policy_name == "random":
    policy = ArcRandomPolicy(seed=seed)

  elif policy_name == "goose":
    from master_thesis.policies.arc_goose import StochasticGoose

    policy = StochasticGoose(available_actions=env.action_space, seed=seed, )

  else:
    raise ValueError(f"Unknown policy: {policy_name}")

  if output_path is None:
    output_path = (default_recording_path(game_id=game_id, seed=seed, policy_name=policy_name, start_level=start_level))

  output_path = Path(output_path)

  output_path.parent.mkdir(parents=True, exist_ok=True, )

  # Exclusive creation prevents accidentally overwriting
  # an existing trajectory.
  with output_path.open("x", encoding="utf-8", ) as file:

    # s0
    record_frame(file, observation, start_level=start_level)

    actions_executed = 0
    episodes = 1

    while (actions_executed < max_steps):
      if (observation.state is GameState.WIN):
        print("Game won after "
              f"{actions_executed} actions")
        break

      if (observation.state is GameState.GAME_OVER):
        observation = (reset_at_level(env, start_level) if start_level is not None else env.reset())

        if observation is None:
          raise RuntimeError("ARC reset returned no observation")

        episodes += 1

        # New episode initial state.
        record_frame(file, observation, start_level=start_level)

        continue

      frame = np.asarray(observation.frame[-1])

      action, action_data = (policy.choose_action(frame=frame, available_actions=(env.action_space), ))

      next_observation = env.step(action, data=action_data, )

      if next_observation is None:
        raise RuntimeError("ARC environment returned None "
                           "after executing an action")

      actions_executed += 1

      # Immediately record the state produced by this action.
      #
      # This is intentionally different from the original
      # Playground loop, which records at the start of the
      # next iteration and can therefore miss the final
      # post-action state when an agent reaches its step limit.
      record_frame(file, next_observation, start_level=start_level)
      observation = next_observation

      if (policy_name == "goose" and observation.state in (GameState.GAME_OVER, GameState.WIN)):
        policy.choose_action(frame=np.asarray(observation.frame[-1]), available_actions=env.action_space, episode_finished=True, )

        print(f"Goose episode ended: {observation.state.name}")
        break

  print(f"Game: {game_id}")

  print(f"Policy actions: "
        f"{actions_executed}")

  print(f"Episodes: {episodes}")

  print(f"Recording: {output_path}")

  return output_path


def collect_runs(game_id: str, max_steps: int, seed: int, mode: str, runs: int = 1, policy_name: str = "random", output_path: Path | None = None, start_level: int | None = None) -> list[Path]:
  """Collect one or more recordings with consecutive seeds.

  An explicit output path is supported only for a single recording.
  Otherwise each run receives its own automatically generated path.
  """
  if runs < 1:
    raise ValueError("runs must be at least 1")

  if max_steps < 1:
    raise ValueError("max_steps must be at least 1")

  if mode not in MODE_MAP:
    raise ValueError(f"Unknown ARC mode: {mode}")

  if output_path is not None and runs != 1:
    raise ValueError("output_path can only be used with runs=1")

  recordings = []

  for run_index in range(runs):
    run_seed = seed + run_index
    print(f"\n--- Run {run_index + 1}/{runs} (seed={run_seed}) ---")

    recording = collect_game(game_id=game_id, max_steps=max_steps, seed=run_seed, mode=mode, output_path=output_path, policy_name=policy_name, start_level=start_level)
    recordings.append(recording)

  print("\nCollection complete")
  print(f"Runs: {len(recordings)}")
  print(f"Total requested actions (upper bound): {runs * max_steps}")

  return recordings


def main() -> None:
  parser = argparse.ArgumentParser(description=("Collect ARC-AGI-3 trajectories for world-model training"))
  parser.add_argument("game", type=str, help=("ARC game ID, e.g. ls20"), )
  parser.add_argument("--max-steps", type=int, default=50, help=("Maximum number of actual policy actions per run"))
  parser.add_argument("--seed", type=int, default=42, )
  parser.add_argument("--mode", choices=["normal", "offline", "online", ], default="normal", )
  parser.add_argument("--output", type=Path, default=None, help=("Optional explicit JSONL recording path; requires --runs 1"))
  parser.add_argument("--runs", type=int, default=1, help=("Number of recordings; seeds start at --seed and increase by one"))
  parser.add_argument("--policy", choices=["random", "goose"], default="random")
  parser.add_argument("--start-level", type=int, default=None, help="Start each recording at this level, numbered from 1; local modes only", )
  args = parser.parse_args()

  if args.runs < 1:
    parser.error("--runs must be at least 1")
  if args.max_steps < 1:
    parser.error("--max-steps must be at least 1")
  if args.output is not None and args.runs != 1:
    parser.error("--output can only be used with --runs 1")
  if args.start_level is not None:
    if args.start_level < 1:
      parser.error("--start-level must be at least 1")
    if args.mode == "online":
      parser.error("--start-level requires normal or offline mode")

  collect_runs(game_id=args.game, max_steps=args.max_steps, seed=args.seed, mode=args.mode, runs=args.runs, policy_name=args.policy, output_path=args.output, start_level=args.start_level)


if __name__ == "__main__":
  main()
