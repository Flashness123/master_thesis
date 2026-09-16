"""
ARC-AGI-3 trajectory collection.

The interaction structure is adapted from the ideas used in
InexperiencedMe/ARC-AGI-3-Playground

This version is intentionally reduced to the functionality required
for world-model data collection in this Master's thesis

One call creates one collection:
$STABLEWM_HOME/recordings/<game>/
└── <name>_<MMDD-HHMM>/                 ← one collection (name defaults to <policy>_<levels>, e.g. goose_l1-7)
    ├── collection.json                 ← settings of this call (policy, seed, runs, start levels, PPO run …)
    ├── level_1/                        ← one folder per --start-levels entry
    │   ├── goose_seed10042.jsonl       ← one recording = one episode (Goose/PPO); seed = seed + level * 10000 + run
    │   └── …
    └── level_7/
Without --start-levels the recordings lie directly in the collection folder (normal ARC reset, level unknown).
"""
# (this text must stay ABOVE the imports; only then Python treats it as the module docstring)

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import arc_agi
import numpy as np
from arc_agi import OperationMode
from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arcengine import FrameDataRaw, GameState

from master_thesis.paths import arc_environments_dir, recordings_dir, timestamp  # storage layout lives in paths.py
from master_thesis.policies.arc_random import ArcRandomPolicy

MODE_MAP = {"normal": OperationMode.NORMAL, "offline": OperationMode.OFFLINE, "online": OperationMode.ONLINE, }
SEED_LEVEL_OFFSET = 10_000  # run seed = seed + start_level * SEED_LEVEL_OFFSET + run_index (the rule both existing collections followed)


def levels_tag(start_levels: list[int] | None) -> str:  # short text for collection names
    if not start_levels:  # no direct starts: normal ARC reset
        return "reset"
    if start_levels == list(range(start_levels[0], start_levels[-1] + 1)) and len(start_levels) > 1:  # consecutive, e.g. [1..7]
        return f"l{start_levels[0]}-{start_levels[-1]}"  # -> "l1-7"
    return "l" + ".".join(str(level) for level in start_levels)  # single or scattered levels -> "l1" or "l2.5"


def record_frame(file: TextIO, frame_data: FrameDataRaw, start_level: int | None = None, ) -> None:
    """
  Store one ARC response using the same basic JSONL structure used by ARC-AGI-3-Playground.

  `frame` is added explicitly because it is not reliably included in FrameDataRaw.model_dump().
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
        raise ValueError("Starting at a selected level requires a local environment") # direct starts need a local game

    game = env._game  # loading the game object

    if game is None:  # failed to load game's python file
        raise RuntimeError("Local ARC game is not loaded")

    if not 1 <= start_level <= len(game._levels):
        raise ValueError(f"start_level must be between 1 and {len(game._levels)}, "
                         f"got {start_level}")

    def reset_selected_level() -> None:
        game.full_reset()  # engine: fresh copies of all levels, completion counter (score) = 0, action_count = 0, full_reset flag = True, level 0, state NOT_FINISHED
        game.set_level(start_level - 1)

    original_handle_reset = game.handle_reset

    try:
        game.handle_reset = reset_selected_level  # override the method
        observation = env.reset()  # call our own reset_selected_level()
    finally:
        game.handle_reset = original_handle_reset

    if observation is None:
        raise RuntimeError("ARC reset returned no observation")

    if game._current_level_index != start_level - 1:
        raise RuntimeError("ARC reset did not select the requested level")

    if observation.state != GameState.NOT_FINISHED:
        raise RuntimeError(f"Unexpected initial game state: {observation.state}")

    return observation  # first frame of level k. NOTE: levels_completed is 0 here even for k>1, because full_reset set score=0


def collect_game(game_id: str, max_steps: int, seed: int, mode: str, output_path: Path, policy_name: str = "random", start_level: int | None = None, ppo_policy=None, ) -> Path:  # output_path is always chosen by collect_runs
    """
  Collect one ARC trajectory using a policy.

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

    environments_dir = arc_environments_dir()  # $STABLEWM_HOME/arc_environments

    environments_dir.mkdir(parents=True, exist_ok=True, )

    arcade = arc_agi.Arcade(operation_mode=MODE_MAP[mode], environments_dir=str(environments_dir), ) # Arcade: scans environments_dir for metadata.json files and, in NORMAL/ONLINE mode, can use the API

    env = arcade.make(game_id, seed=seed, include_frame_data=True, )  # splits "ls20-9607627b" into game + version; OFFLINE: finds the local copy; NORMAL: downloads it if needed. Returns a LocalEnvironmentWrapper, which runs the game file, creates the game with this seed, and ALREADY RESETS it. Also creates a local scorecard.

    if env is None:
        raise RuntimeError(f"Could not create ARC game: {game_id}")

    # observation = (env.observation_space)  # arc_agi's "observation_space" is NOT a Gym space: it is the last FrameDataRaw returned

    if start_level is not None:
        observation = reset_at_level(env, start_level)  # second reset, this time to level k
    else:
        observation = env.observation_space   # use the first fram from make()'s reset

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

        policy = StochasticGoose(available_actions=env.action_space, seed=seed)

    elif policy_name == "ppo_images":  # !!! Inspect this !!! - I dont hink that giving a policy as a function argument is the cleanest way here!
        if ppo_policy is None:
            raise ValueError("Use collect_runs(..., ppo_run=...) for PPO collection")

        policy = ppo_policy
        policy.reset(seed=seed, available_actions=env.action_space, game_id=observation.game_id, )

    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    output_path = Path(output_path)  # also accept a string
    output_path.parent.mkdir(parents=True, exist_ok=True, )  # level_<k> folder

    # Exclusive creation prevents accidentally overwriting
    # an existing trajectory.
    with output_path.open("x", encoding="utf-8", ) as file:  # mode "x" = create a new file, error if it exists (never overwrites)

        # s0
        record_frame(file, observation, start_level=start_level)

        actions_executed = 0
        episodes = 1

        while (actions_executed < max_steps):
            if (observation.state is GameState.WIN):  # only happens if all 7 levels completed
                print(f"Game won after {actions_executed} actions")
                break

            if (observation.state is GameState.GAME_OVER): #random policy can reach this becasue it continues then we have multiple episodes in one recording
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
                raise RuntimeError("ARC environment returned None after executing an action")

            actions_executed += 1

            # Immediately record the state produced by this action.
            #
            # This is intentionally different from the original
            # Playground loop, which records at the start of the
            # next iteration and can therefore miss the final
            # post-action state when an agent reaches its step limit.
            record_frame(file, next_observation, start_level=start_level)
            observation = next_observation

            if policy_name in ("goose", "ppo_images") and observation.state in (GameState.GAME_OVER, GameState.WIN):
                if policy_name == "goose":
                    policy.choose_action(frame=np.asarray(observation.frame[-1]), available_actions=env.action_space, episode_finished=True, )

                print(f"{policy_name} episode ended: {observation.state.name}")
                break

    print(f"Game: {game_id}")

    print(f"Policy actions: "
          f"{actions_executed}")

    print(f"Episodes: {episodes}")

    print(f"Recording: {output_path}")

    return output_path


def collect_runs(game_id: str, max_steps: int, seed: int, mode: str, runs: int = 1, policy_name: str = "random", start_levels: list[int] | None = None, ppo_run: str | None = None, device: str = "auto", name: str | None = None) -> Path:
    """
  Record `runs` episodes per start level into one new collection folder.
  A frozen PPO policy is loaded once for the whole collection.
  Returns the collection folder.
  """

    if runs < 1 or max_steps < 1:
        raise ValueError("runs and max_steps must be positive")

    if mode not in MODE_MAP:
        raise ValueError(f"Unknown ARC mode: {mode}")

    if start_levels and mode == "online":  # direct starts patch the local game
        raise ValueError("start_levels require normal or offline mode")

    if start_levels and (len(set(start_levels)) != len(start_levels) or min(start_levels) < 1):  # each level once, counted from 1
        raise ValueError("start_levels must be distinct and at least 1")

    ppo_policy = None

    if policy_name == "ppo_images":
        if ppo_run is None:
            raise ValueError("ppo_images requires ppo_run")

        from master_thesis.policies.arc_ppo import ArcImagePPOPolicy

        ppo_policy = ArcImagePPOPolicy(ppo_run, device=device, )  # loads models/ppo/<ppo_run>/policy.zip once

    elif ppo_run is not None:
        raise ValueError("ppo_run is only used with ppo_images")

    collection_name = f"{name or f'{policy_name}_{levels_tag(start_levels)}'}_{timestamp()}"  # e.g. goose_l1-7_0916-1432
    collection_dir = recordings_dir(game_id) / collection_name  # recordings/<game>/<collection_name>
    collection_dir.mkdir(parents=True, exist_ok=False)  # never mix two collections

    settings = {"game_id": game_id, "policy": policy_name, "runs_per_level": runs, "seed": seed, "seed_rule": f"seed + start_level * {SEED_LEVEL_OFFSET} + run_index", "start_levels": start_levels, "max_steps": max_steps, "mode": mode, "device": device, "ppo_run": ppo_run, }  # everything needed to repeat the collection

    (collection_dir / "collection.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    recordings = []

    for start_level in (start_levels or [None]):  # [None] = one pass with normal ARC reset
        level_dir = collection_dir if start_level is None else collection_dir / f"level_{start_level}"  # recordings of one start level

        for run_index in range(runs):
            run_seed = seed + (start_level or 0) * SEED_LEVEL_OFFSET + run_index  # distinct seeds across levels, e.g. 42 -> 10042, 10043, …

            path = level_dir / f"{policy_name}_seed{run_seed}.jsonl"

            print(f"\n--- Level {start_level}, run {run_index + 1}/{runs} (seed={run_seed}) ---")

            recordings.append(collect_game(game_id=game_id, max_steps=max_steps, seed=run_seed, mode=mode, output_path=path, policy_name=policy_name, start_level=start_level, ppo_policy=ppo_policy, ))

    print(f"\nCollection complete: {len(recordings)} recordings in {collection_dir}")
    return collection_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=("Collect ARC-AGI-3 trajectories for world-model training"))
    parser.add_argument("game", type=str, help=("ARC game ID, e.g. ls20"), )
    parser.add_argument("--max-steps", type=int, default=50, help=("Maximum number of actual policy actions per run"))
    parser.add_argument("--seed", type=int, default=42, )
    parser.add_argument("--mode", choices=["normal", "offline", "online", ], default="normal", )
    parser.add_argument("--runs", type=int, default=1, help=("Recordings per start level"))
    parser.add_argument("--policy", choices=["random", "goose", "ppo_images"], default="random")
    parser.add_argument("--start-levels", type=int, nargs="+", default=None, help="Start levels, numbered from 1, e.g. --start-levels 1 2 3; local modes only", )
    parser.add_argument("--ppo-run", default=None, help="Image-PPO model folder name under models/ppo (for --policy ppo_images)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", )
    parser.add_argument("--name", default=None, help="Collection name without timestamp; default <policy>_<levels>")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.start_levels is not None:
        if min(args.start_levels) < 1:
            parser.error("--start-levels must be at least 1")
        if args.mode == "online":
            parser.error("--start-levels requires normal or offline mode")

    collect_runs(game_id=args.game, max_steps=args.max_steps, seed=args.seed, mode=args.mode, runs=args.runs, policy_name=args.policy, start_levels=args.start_levels, ppo_run=args.ppo_run, device=args.device, name=args.name, )


if __name__ == "__main__":
    main()
