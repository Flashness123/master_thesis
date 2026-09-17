from __future__ import annotations
import numpy as np
import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from stable_worldmodel.data.format import get_format

from master_thesis.paths import dataset_path

"""
┌──────────┬────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
│          │          JSONL (after collection)          │                              Lance (after arc_recording.py)                               │
├──────────┼────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Unit     │ 1 line = 1 state + the action that         │ 1 row = 1 state + the action taken in it                                                  │
│          │ produced it                                │                                                                                           │
├──────────┼────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Episodes │ 1 file = 1 episode (Goose/PPO)             │ episode_idx column, numbered by the writer                                                │
├──────────┼────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Grid     │ nested list, possibly several animation    │ only the last frame, float32 with 4096 values                                             │
│          │ frames                                     │                                                                                           │
├──────────┼────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Metadata │ levels_completed, state, full_reset, guid, │ terminal, levels_completed, available_actions, game_id, start_level (0 = unknown),        │
│          │  timestamp, …                              │ episode_success (the same on every row)                                                   │
├──────────┼────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Access   │ read the file line by line                 │ random access to any 4-state window (what the LeWM dataloader uses)                       │
└──────────┴────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────┘
"""

GRID_SIZE = 64
COLOR_COUNT = 16

NUM_ACTIONS = 7
CLICK_ACTION = 6


@dataclass
class ArcTransition:
    game_id: str
    episode: int
    start_level: int

    state: np.ndarray  # 64×64 uint8 grid BEFORE the action
    action: np.ndarray  # [action_id, x, y] as int16 (x=y=-1 for non-click actions)
    next_state: np.ndarray  # 64×64 uint8 grid AFTER the action

    terminal: bool  # True if next_state is WIN or GAME_OVER
    game_state: str  # next_state's state as text: "NOT_FINISHED" / "WIN" / "GAME_OVER"

    levels_completed_before: int  # completion counter in state
    levels_completed_after: int  # completion counter in next_state (higher = a level was just completed)

    available_actions: tuple[int, ...]  # action IDs allowed in state, e.g. (1, 2, 3, 4)
    next_available_actions: tuple[int, ...]  # action IDs allowed in next_state


def load_recording(path: str | Path) -> list[dict[str, Any]]: # read a JSONL file into a list of dicts (one per line - one line is one move in the game)
    path = Path(path)

    events = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc

    if not events:
        raise ValueError(f"Recording is empty: {path}")

    return events  # list of {"timestamp", "data", ["start_level"]}


def get_final_frame(event: dict[str, Any]) -> np.ndarray:
    """
  ARC can return multiple frames for one action because of animations.

  For the first LeWM baseline we use only the final settled frame.
  """
    frames = event["data"].get("frame")

    if not frames:
        raise ValueError("Recording event contains no frame data")

    frame = np.asarray(frames[-1], dtype=np.uint8, )

    if frame.shape != (GRID_SIZE, GRID_SIZE):  # must be 64×64
        raise ValueError(f"Expected ARC frame shape (64, 64), got {frame.shape}")

    if frame.min() < 0 or frame.max() >= COLOR_COUNT:
        raise ValueError("ARC frame values must be between "
                         f"0 and {COLOR_COUNT - 1}")

    return frame  # 64×64 uint8 grid


def get_raw_action(event: dict[str, Any]) -> np.ndarray | None:
    """
  Return the action that produced this event's state.

  Canonical representation:

      [action_id, x, y]

  Simple action:
      ACTION2 -> [2, -1, -1]

  Complex click:
      ACTION6(17, 42) -> [6, 17, 42]

  RESET:
      [0, -1, -1]

  -1 means that the coordinate does not exist.
  """
    action_input = event["data"].get("action_input")

    if action_input is None:
        return None

    action_id = int(action_input["id"])
    action_data = action_input.get("data") or {}

    if action_id == 0:
        return np.asarray([0, -1, -1], dtype=np.int16, )

    if not 1 <= action_id <= NUM_ACTIONS:
        raise ValueError(f"Unknown ARC action id: {action_id}")

    if action_id == CLICK_ACTION:
        if "x" not in action_data or "y" not in action_data:
            raise ValueError("ACTION6 requires x and y coordinates")

        x = int(action_data["x"])
        y = int(action_data["y"])

        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            raise ValueError(f"ACTION6 coordinates out of range: ({x}, {y})")

        return np.asarray([action_id, x, y], dtype=np.int16, )  # click in canonical form

    return np.asarray([action_id, -1, -1], dtype=np.int16, )  # simple action (ls20: always this case)


def encode_available_actions(actions: tuple[int, ...], ) -> np.ndarray:  # allowed action IDs → 7-value 0/1 mask
    """
  Encode the currently available ARC action types as a 7-D mask.

  Example:
      [1, 2, 4]

  becomes:
      [1, 1, 0, 1, 0, 0, 0]
  """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32, )

    for action_id in actions:
        if not 1 <= action_id <= NUM_ACTIONS:
            raise ValueError(f"Unknown available ARC action: {action_id}")

        mask[action_id - 1] = 1.0

    return mask  # e.g. ls20: [1,1,1,1,0,0,0]; stored in Lance but not used by training yet


def build_lance_episodes(transitions: list[ArcTransition], ) -> list[dict[str, list]]:
    """
  Convert transitions into episode-contiguous data suitable for SWM.

  For:

      s0 + a0 -> s1
      s1 + a1 -> s2

  store:

      grid:   [s0, s1, s2]
      action: [a0, a1, boundary]

  The final boundary action is ACTION0 and is never treated as a
  learnable ARC action.
  """
    if not transitions:
        return []

    grouped: dict[int, list[ArcTransition]] = {}

    for transition in transitions:
        grouped.setdefault(transition.episode, [], ).append(transition)

    episodes = []

    boundary_action = np.asarray([0, -1, -1], dtype=np.int16, )

    for episode_transitions in grouped.values():
        first = episode_transitions[0]
        if any(transition.start_level != first.start_level for transition in episode_transitions):
            raise ValueError(f"Episode {first.episode} contains inconsistent starting levels")

        counters_before = {transition.levels_completed_before for transition in episode_transitions}  # completion counter before each action
        completed_early = any(t.levels_completed_after > t.levels_completed_before for t in episode_transitions[:-1])  # a completion before the last action?
        if len(counters_before) != 1 or completed_early:  # only the last action may complete the level, otherwise the episode spans more than one level
            raise ValueError(f"Episode {first.episode} spans more than one level")

        # Valid levels are one-based; 0 explicitly means unknown.
        stored_start_level = (first.start_level if first.start_level is not None else 0)

        grids = [first.state, *[transition.next_state for transition in episode_transitions], ]

        actions = [*[transition.action for transition in episode_transitions], boundary_action, ]

        terminal = [False, *[transition.terminal for transition in episode_transitions], ]

        levels_completed = [first.levels_completed_before, *[transition.levels_completed_after for transition in episode_transitions], ]

        available_actions = [encode_available_actions(first.available_actions), *[encode_available_actions(transition.next_available_actions) for transition in episode_transitions], ]

        game_ids = [first.game_id] * len(grids)

        episode_success = any(transition.levels_completed_after > transition.levels_completed_before or transition.game_state == "WIN" for transition in episode_transitions)

        episode_data = {"grid": grids, "action": actions, "terminal": terminal, "levels_completed": levels_completed, "available_actions": available_actions, "game_id": game_ids, "start_level": [stored_start_level] * len(grids), "episode_success": [episode_success] * len(grids)}

        lengths = {key: len(value) for key, value in episode_data.items()}

        if len(set(lengths.values())) != 1:
            raise ValueError(f"Episode columns have different lengths: {lengths}")

        episodes.append(episode_data)

    return episodes  # list of episode dicts (the episode number itself is not stored; the writer assigns its own episode_idx)


def get_start_level(event: dict[str, Any]) -> int | None:
    """Read an explicit one-based starting level; None means unknown."""
    level = event.get("start_level")

    if level is None:
        return None

    # bool is a subclass of int, so reject it explicitly.
    if isinstance(level, bool) or not isinstance(level, int) or level < 1:
        raise ValueError(f"start_level must be a positive integer, got {level!r}")

    return level # f.e. 3


def build_transitions(events: list[dict[str, Any]], ) -> list[ArcTransition]:
    """
  Convert Playground recording semantics into world-model transitions.

  Playground rows contain:

      current_state
      +
      action_input = action that produced current_state

  We convert:

      row 0: s0, RESET
      row 1: s1, a0
      row 2: s2, a1

  into:

      s0 + a0 -> s1
      s1 + a1 -> s2
  """
    transitions = []

    previous_data = None
    previous_frame = None

    episode = -1
    episode_start_level: int | None = None

    for row_index, event in enumerate(events):
        data = event["data"]
        if "frame" not in data:  # not a game step, e.g. the scorecard summary at the end of ARC Prize human recordings
            continue
        recorded_start_level = get_start_level(event)
        frame = get_final_frame(event)
        action = get_raw_action(event)

        is_reset = (bool(data.get("full_reset", False)) or (action is not None and int(action[0]) == 0))

        if previous_data is None or previous_frame is None or is_reset:  # new episode: first line, after WIN/GAME_OVER, or after a RESET (the reset jump is never a learnable transition)
            episode += 1  # each episode gets its own number, so build_lance_episodes never merges two of them
            episode_start_level = (recorded_start_level or 1) + int(data.get("levels_completed", 0))  # level being played = recording's start level (1 if not recorded, e.g. human runs) + levels completed so far

            previous_data = data
            previous_frame = frame

            continue

        if action is None:
            raise ValueError(f"Event {row_index} has no action_input")

        action_id = int(action[0])

        available_actions = tuple(int(value) for value in previous_data.get("available_actions", [], ))

        # The action belongs to the transition FROM the
        # previous state TO the current state.
        if (available_actions and action_id not in available_actions):
            raise ValueError(f"Event {row_index}: action {action_id} "
                             f"was not available in source state "
                             f"{available_actions}. "
                             "This likely indicates action/state misalignment.")

        game_state = str(data.get("state", "NOT_FINISHED", ))

        terminal = game_state in {"WIN", "GAME_OVER", }

        transitions.append(
          ArcTransition(game_id=str(data["game_id"]), episode=max(episode, 0), start_level=episode_start_level, state=previous_frame, action=action, next_state=frame, terminal=terminal, game_state=game_state, levels_completed_before=int(previous_data.get("levels_completed", 0,
                                                                                                                                                                                                                                                               )),
                        levels_completed_after=int(data.get("levels_completed", 0,
                                                            )), available_actions=available_actions, next_available_actions=tuple(int(value) for value in data.get("available_actions", [],
                                                                                                                                                                   )),
                        ))

        # Never create a transition out of a terminal
        # state. The next RESET starts a new episode.
        if terminal:
            previous_data = None
            previous_frame = None

        else:
            previous_data = data
            previous_frame = frame

            if transitions[-1].levels_completed_after > transitions[-1].levels_completed_before:  # this action completed a level
                episode += 1  # the completion frame (already showing the next level) ends this episode and starts the next one
                episode_start_level = (recorded_start_level or 1) + transitions[-1].levels_completed_after  # the new episode plays the next level

    return transitions  # all transitions of the file, in order


def recording_files(source: Path) -> list[Path]:  # one .jsonl file, or every .jsonl inside a collection folder (incl. level_<k>/)
    files = sorted(source.rglob("*.jsonl")) if source.is_dir() else [source]  # sorted -> reproducible episode order

    if not files:
        raise ValueError(f"No .jsonl recordings found in {source}")

    return files


def iter_episodes(files: list[Path], levels: list[int] | None = None) -> Iterable[dict[str, list]]:  # yields Lance episodes file by file (only one file in memory)
    for file in files:
        for episode in build_lance_episodes(build_transitions(load_recording(file))):  # converted per file, so episodes of different files never merge
            if levels is None or episode["start_level"][0] in levels:  # keep only episodes that play one of the requested levels
                yield episode


def write_lance_dataset(episodes: Iterable[dict[str, list]], output_path: str | Path, mode: str = "error", sources: list[Path] = (), levels: list[int] | None = None, ) -> None:
    """
  Write ARC episodes using stable-worldmodel's native Lance writer
  and record where they came from in <dataset>.json next to the table.
  """
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True, )  # datasets/<game>/

    counts = {"episodes": 0, "states": 0}  # filled while the writer consumes the episodes

    def counted():  # passes episodes through unchanged and counts them
        for episode in episodes:
            counts["episodes"] += 1
            counts["states"] += len(episode["grid"])
            yield episode

    with get_format("lance").open_writer(output_path, mode=mode, ) as writer:  # mode: error = refuse existing, append = add, overwrite = replace
        writer.write_episodes(counted())  # streams all episodes into one new Lance version

    if counts["episodes"] == 0:
        raise ValueError("No ARC episodes available to write")

    record_path = output_path.with_suffix(".json")  # datasets/<game>/<dataset>.json
    record = json.loads(record_path.read_text(encoding="utf-8")) if mode == "append" and record_path.exists() else {"writes": []}  # appending keeps the earlier history
    record["writes"].append({"time": datetime.now().isoformat(timespec="seconds"), "mode": mode, "sources": [str(Path(source).resolve()) for source in sources], "levels": levels, **counts})  # one entry per write; levels = level filter (None = all)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"\nWrote Lance dataset: {output_path}")
    print(f"Episodes: {counts['episodes']}")
    print(f"States: {counts['states']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=("Inspect ARC-AGI-3 recordings and optionally convert them into a Lance dataset"))
    parser.add_argument("recording", type=Path, help="one .jsonl file or a collection folder (all .jsonl files inside are used)")
    parser.add_argument("--show", type=int, default=5, help="number of transitions of the first file to print")
    parser.add_argument("--dataset", default=None, help="write datasets/<game>/<DATASET>.lance, e.g. goose_l1-7")
    parser.add_argument("--mode", choices=["error", "append", "overwrite", ], default="error", help="what to do if the dataset already exists")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="keep only episodes of these levels, e.g. --levels 1 (default: all)")
    args = parser.parse_args()

    files = recording_files(args.recording)
    events = load_recording(files[0])  # the first file is inspected below
    transitions = build_transitions(events)

    print(f"Recording files: {len(files)}")
    print(f"Recording: {files[0]}")
    print(f"Events: {len(events)}")
    print(f"Valid transitions: {len(transitions)}")

    if transitions:
        print(f"Game: {transitions[0].game_id}")
        print(f"Grid shape: {transitions[0].state.shape}")
        print(f"Actions seen: {sorted({ int(t.action[0]) for t in transitions})}")

    for index, transition in enumerate(transitions[:args.show]):
        changed_cells = int(np.count_nonzero(transition.state != transition.next_state))

        print(f"\nTransition {index}")
        print(f"episode: {transition.episode}")
        print(f"raw action [id, x, y]: {transition.action.tolist()}")
        print(f"changed cells: {changed_cells}")
        print(f"next state: {transition.game_state}")
        print("  levels: "
              f"{transition.levels_completed_before}"
              " -> "
              f"{transition.levels_completed_after}")
        print(f"terminal: {transition.terminal}")

    if args.dataset is not None:
        game_id = str(events[0]["data"]["game_id"])  # e.g. "ls20-9607627b" -> stored under datasets/ls20/
        write_lance_dataset(iter_episodes(files, args.levels), dataset_path(game_id, args.dataset), mode=args.mode, sources=[args.recording], levels=args.levels, )


if __name__ == "__main__":
    main()
