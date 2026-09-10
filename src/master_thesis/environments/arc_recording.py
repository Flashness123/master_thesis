from __future__ import annotations
import numpy as np
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from stable_worldmodel.data.format import get_format

GRID_SIZE = 64
COLOR_COUNT = 16

NUM_ACTIONS = 7
CLICK_ACTION = 6


@dataclass
class ArcTransition:
  game_id: str
  episode: int

  state: np.ndarray
  action: np.ndarray
  next_state: np.ndarray

  terminal: bool
  game_state: str

  levels_completed_before: int
  levels_completed_after: int

  available_actions: tuple[int, ...]
  next_available_actions: tuple[int, ...]


def load_recording(path: str | Path) -> list[dict[str, Any]]:
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

  return events


def get_final_frame(event: dict[str, Any]) -> np.ndarray:
  """
  ARC can return multiple frames for one action because of animations.

  For the first LeWM baseline we use only the final settled frame.
  """
  frames = event["data"].get("frame")

  if not frames:
    raise ValueError("Recording event contains no frame data")

  frame = np.asarray(frames[-1], dtype=np.uint8, )

  if frame.shape != (GRID_SIZE, GRID_SIZE):
    raise ValueError(f"Expected ARC frame shape (64, 64), got {frame.shape}")

  if frame.min() < 0 or frame.max() >= COLOR_COUNT:
    raise ValueError("ARC frame values must be between "
                     f"0 and {COLOR_COUNT - 1}")

  return frame


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

    return np.asarray([action_id, x, y], dtype=np.int16, )

  return np.asarray([action_id, -1, -1], dtype=np.int16, )


def encode_arc_action(action: np.ndarray, ) -> np.ndarray:
  """
  Convert our raw ARC action into the representation given to LeWM.

  Output:

      7 action-type dimensions
      + normalized x
      + normalized y

  Total: 9 dimensions.
  """
  action = np.asarray(action)

  if action.shape != (3, ):
    raise ValueError(f"Expected raw ARC action shape (3,), got {action.shape}")

  action_id, x, y = (int(value) for value in action)

  if not 1 <= action_id <= NUM_ACTIONS:
    raise ValueError("LeWM actions must be ACTION1..ACTION7, "
                     f"got {action_id}")

  encoded = np.zeros(NUM_ACTIONS + 2, dtype=np.float32, )

  # ACTION1 -> index 0
  # ...
  # ACTION7 -> index 6
  encoded[action_id - 1] = 1.0

  if action_id == CLICK_ACTION:
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
      raise ValueError(f"ACTION6 coordinates out of range: ({x}, {y})")

    encoded[-2] = x / (GRID_SIZE - 1)
    encoded[-1] = y / (GRID_SIZE - 1)

  return encoded


def encode_available_actions(actions: tuple[int, ...], ) -> np.ndarray:
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

  return mask


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

    grids = [first.state, *[transition.next_state for transition in episode_transitions], ]

    actions = [*[transition.action for transition in episode_transitions], boundary_action, ]

    terminal = [False, *[transition.terminal for transition in episode_transitions], ]

    levels_completed = [first.levels_completed_before, *[transition.levels_completed_after for transition in episode_transitions], ]

    available_actions = [encode_available_actions(first.available_actions), *[encode_available_actions(transition.next_available_actions) for transition in episode_transitions], ]

    game_ids = [first.game_id] * len(grids)

    episode_data = {"grid": grids, "action": actions, "terminal": terminal, "levels_completed": levels_completed, "available_actions": available_actions, "game_id": game_ids, }

    lengths = {key: len(value) for key, value in episode_data.items()}

    if len(set(lengths.values())) != 1:
      raise ValueError(f"Episode columns have different lengths: {lengths}")

    episodes.append(episode_data)

  return episodes


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

  for row_index, event in enumerate(events):
    data = event["data"]

    frame = get_final_frame(event)
    action = get_raw_action(event)

    is_reset = (bool(data.get("full_reset", False)) or (action is not None and int(action[0]) == 0))

    if is_reset:
      episode += 1

      previous_data = data
      previous_frame = frame

      continue

    # A recording could theoretically begin without
    # containing its original RESET event.
    if (previous_data is None or previous_frame is None):
      episode = max(episode, 0)

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
      ArcTransition(game_id=str(data["game_id"]), episode=max(episode, 0), state=previous_frame, action=action, next_state=frame, terminal=terminal, game_state=game_state, levels_completed_before=int(previous_data.get("levels_completed", 0,
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

  return transitions


def write_lance_dataset(transitions: list[ArcTransition], output_path: str | Path, mode: str = "overwrite", ) -> None:
  """
  Write parsed ARC trajectories using stable-worldmodel's
  native Lance writer.
  """
  output_path = Path(output_path)

  output_path.parent.mkdir(parents=True, exist_ok=True, )

  episodes = build_lance_episodes(transitions)

  if not episodes:
    raise ValueError("No ARC episodes available to write")

  with get_format("lance").open_writer(output_path, mode=mode, ) as writer:
    writer.write_episodes(episodes)

  num_states = sum(len(episode["grid"]) for episode in episodes)

  print(f"\nWrote Lance dataset: {output_path}")
  print(f"Episodes: {len(episodes)}")
  print(f"States: {num_states}")


def main() -> None:
  parser = argparse.ArgumentParser(description=("Inspect an ARC-AGI-3 Playground recording"))
  parser.add_argument("recording", type=Path, )
  parser.add_argument("--show", type=int, default=5, )
  parser.add_argument("--output", type=Path, default=None, help="Optional output .lance dataset", )
  parser.add_argument("--mode", choices=["overwrite", "append", "error", ], default="overwrite", )
  args = parser.parse_args()

  events = load_recording(args.recording)
  transitions = build_transitions(events)

  print(f"Recording: {args.recording}")
  print(f"Events: {len(events)}")
  print(f"Valid transitions: {len(transitions)}")

  if transitions:
    print(f"Game: {transitions[0].game_id}")
    print(f"Grid shape: {transitions[0].state.shape}")
    print(f"Actions seen: {sorted({ int(t.action[0]) for t in transitions})}")

  for index, transition in enumerate(transitions[:args.show]):
    encoded = encode_arc_action(transition.action)
    changed_cells = int(np.count_nonzero(transition.state != transition.next_state))

    print(f"\nTransition {index}")
    print(f"episode: {transition.episode}")
    print(f"raw action [id, x, y]: {transition.action.tolist()}")
    print(f"LeWM action (9D): {encoded.tolist()}")
    print(f"changed cells: {changed_cells}")
    print(f"next state: {transition.game_state}")
    print("  levels: "
          f"{transition.levels_completed_before}"
          " -> "
          f"{transition.levels_completed_after}")
    print(f"terminal: {transition.terminal}")

  if args.output is not None:
    write_lance_dataset(transitions, args.output, mode=args.mode, )


if __name__ == "__main__":
  main()
