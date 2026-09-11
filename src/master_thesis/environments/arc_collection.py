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


def default_recording_path(game_id: str, seed: int, ) -> Path:
  timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

  recordings_dir = (get_stablewm_home() / "datasets" / "arc_recordings")

  recordings_dir.mkdir(parents=True, exist_ok=True, )

  short_game_id = game_id.split("-")[0]

  return (recordings_dir / (f"random-{short_game_id}"
                            f"-seed{seed}"
                            f"-{timestamp}.jsonl"))


def record_frame(file: TextIO, frame_data: FrameDataRaw, ) -> None:
  """
  Store one ARC response using the same basic JSONL structure used
  by ARC-AGI-3-Playground.

  `frame` is added explicitly because it is not reliably included
  in FrameDataRaw.model_dump().
  """
  data = frame_data.model_dump(mode="json")

  data["frame"] = [np.asarray(frame).tolist() for frame in frame_data.frame]

  event = {"timestamp": datetime.now(timezone.utc).isoformat(), "data": data, }

  file.write(json.dumps(event) + "\n")

  file.flush()


def collect_game(game_id: str, max_steps: int, seed: int, mode: str, output_path: Path | None = None, ) -> Path:
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

  stablewm_home = (get_stablewm_home())

  environments_dir = (stablewm_home / "arc_environments")

  environments_dir.mkdir(parents=True, exist_ok=True, )

  arcade = arc_agi.Arcade(operation_mode=MODE_MAP[mode], environments_dir=str(environments_dir), )

  env = arcade.make(game_id, seed=seed, include_frame_data=True, )

  if env is None:
    raise RuntimeError(f"Could not create ARC game: {game_id}")

  observation = (env.observation_space)

  if observation is None:
    observation = env.reset()

  if observation is None:
    raise RuntimeError("ARC environment returned no initial observation")

  policy = ArcRandomPolicy(seed=seed)

  if output_path is None:
    output_path = (default_recording_path(game_id=game_id, seed=seed, ))

  output_path = Path(output_path)

  output_path.parent.mkdir(parents=True, exist_ok=True, )

  # Exclusive creation prevents accidentally overwriting
  # an existing trajectory.
  with output_path.open("x", encoding="utf-8", ) as file:

    # s0
    record_frame(file, observation, )

    actions_executed = 0
    episodes = 1

    while (actions_executed < max_steps):
      if (observation.state is GameState.WIN):
        print("Game won after "
              f"{actions_executed} actions")
        break

      if (observation.state is GameState.GAME_OVER):
        observation = (env.reset())

        if observation is None:
          raise RuntimeError("ARC reset returned no observation")

        episodes += 1

        # New episode initial state.
        record_frame(file, observation, )

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
      record_frame(file, next_observation, )

      observation = (next_observation)

  print(f"Game: {game_id}")

  print(f"Policy actions: "
        f"{actions_executed}")

  print(f"Episodes: {episodes}")

  print(f"Recording: {output_path}")

  return output_path


def collect_runs(game_id: str, max_steps: int, seed: int, mode: str, runs: int = 1, output_path: Path | None = None, ) -> list[Path]:
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

    recording = collect_game(game_id=game_id, max_steps=max_steps, seed=run_seed, mode=mode, output_path=output_path, )
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

  args = parser.parse_args()

  if args.runs < 1:
    parser.error("--runs must be at least 1")
  if args.max_steps < 1:
    parser.error("--max-steps must be at least 1")
  if args.output is not None and args.runs != 1:
    parser.error("--output can only be used with --runs 1")

  collect_runs(game_id=args.game, max_steps=args.max_steps, seed=args.seed, mode=args.mode, runs=args.runs, output_path=args.output, )


if __name__ == "__main__":
  main()
