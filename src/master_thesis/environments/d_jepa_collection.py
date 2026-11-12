"""
Training data for D-JEPA's relational ranker (models/lewm/d_jepa.py, fitted by training/d_jepa.py): situations from
recorded episodes, and for each the real outcome of every plan the exhaustive planner can choose. It needs no model:
whether a plan reaches a goal is a fact about the game, so one collection serves the ranker of every LeWM checkpoint.

A situation is a recorded state and, as its goal, the recorded state goal_ahead (5) moves later: the short goal the
planner works on every time it plans (its horizon is 5 moves; far goals and level completion are split into such goals,
e.g. the waypoints of a solution). From the situation's real state the game plays every plan of `horizon` moves,
4^5 = 1,024 of them, and notes after which move each plan first shows the goal (0 = never).

Which situations (as in the paper, which takes its starts and goals from recorded segments of the environment the world
models were trained on and does not restrict them to held-out data; checked here too: LeWM's errors on training and
validation episodes are practically equal, only new levels differ):
  - every episode of the dataset's levels, from any split
  - the levels are split once (by --seed) into fit, calibration and test levels: the ranker is fitted on the fit levels,
    its gate calibrated on the calibration levels, and the test levels stay for testing it as the paper does (same kind
    of levels, situations never used for fitting); new levels (syn1000_unseen, real levels 5-7) are a test beyond that
  - per level the same number of situations, drawn uniformly over the states that have a goal goal_ahead moves later
    (never the completion frame of a successful episode nor the last state of an episode that lost a life); situations
    that already show their goal are skipped (nothing to plan)
  - a situation whose plans all reach the goal or all fail teaches nothing and is dropped (as in their code)

Output: datasets/ls20/d_jepa_<name>/part<i>.npz, one per array task:
  start, goal   [N, 4096] grids
  reached       [N, 1024] int8: the move after which the plan first shows the goal (1-5), 0 = never; the plans in
                itertools.product order of the moves 0-3 (= ACTION1-4), as lewm_analysis.ExhaustivePlanner lists them
  level, episode, step  [N]  where the situation comes from
  part          [N] "fit", "calibration" or "test"
and part<i>.json with the settings and counts.

Usage (one array task per process; a Barnard array job runs --task-index 0..n-1 with --tasks n):
  python -m master_thesis.environments.d_jepa_collection --name syn1000 --dataset syn1000_train --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from master_thesis.paths import d_jepa_data_dir, dataset_path

GAME = "ls20"
PARTS = ("fit", "calibration", "test")


_WORKER_GAMES = {}  # in each worker process: game_id -> game (making a game takes seconds for syn1000, so each worker makes it once)
def label_situations_of_level(task):
  """
  GETS:    task -- (game_id, level, horizon, situations); a situation is a dict with its key, the start path ("prefix",
           for walk runs that begin inside the level), the recorded grids from the episode's first state up to the
           situation's state, the recorded actions between them, and the goal grid.
  DOES:    runs in a worker process. Replays every situation from the level start (start path, then the recorded
           actions, checking every grid against the recording; a mismatch drops the situation). From the reached state
           it plays every plan of `horizon` moves as a tree: each partial plan once, on a copy of its parent
           (copy_game), so the 4^horizon plans cost 4 + 16 + ... + 4^horizon moves instead of horizon x 4^horizon.
           A plan reaches the goal when the game shows the goal situation after one of its moves (only the energy bar
           may differ: classify_change), unless a move before has lost a life, ended the game or left the level. Once a
           partial plan has reached the goal or failed, every plan that starts with it shares that outcome and is not
           played further.
  RETURNS: per situation {"key", "status" ("ok" or why it was dropped), "reached" [4^horizon] int8: the move after which
           the plan first shows the goal (1 .. horizon), 0 = never; plans in itertools.product order}.
  """
  from arcengine import ActionInput, GameAction, GameState

  from master_thesis.environments.arc_collection import reset_at_level
  from master_thesis.environments.arc_levelgen import make_game
  from master_thesis.evaluation.lewm_analysis import ACTIONS, classify_change, copy_game

  game_id, level, horizon, situations = task
  if game_id not in _WORKER_GAMES:
    _WORKER_GAMES[game_id] = make_game(game_id)
  start_grid = np.asarray(reset_at_level(_WORKER_GAMES[game_id], level).frame[-1], dtype=np.uint8).reshape(-1)
  level_start = copy_game(_WORKER_GAMES[game_id]._game)
  play = lambda game, action: game.perform_action(ActionInput(id=GameAction.from_id(int(action))), raw=True)  # its last frame is what the player sees
  grid_of = lambda frame: np.asarray(frame.frame[-1], dtype=np.uint8).reshape(-1)

  def expand(node, depth, first, goal, reached):  # node: the game after `depth` moves, shared by the plans first .. first + 4^(horizon - depth) - 1
    block = len(ACTIONS) ** (horizon - depth - 1)  # how many plans start with the same depth + 1 moves
    for position, action in enumerate(ACTIONS):
      child, plans = copy_game(node), slice(first + position * block, first + (position + 1) * block)
      frame = play(child, action)
      if frame.state in (GameState.GAME_OVER, GameState.WIN) or child._current_level_index != node._current_level_index or child.aqygnziho < node.aqygnziho:  # aqygnziho: lives left
        continue  # lost a life, ended the game or left the level: every plan starting with these moves fails (reached stays 0)
      if classify_change(grid_of(frame), goal) != "mechanics":
        reached[plans] = depth + 1  # the goal situation is shown: every plan starting with these moves reaches it here
      elif depth + 1 < horizon:
        expand(child, depth + 1, plans.start, goal, reached)

  results = []
  for situation in situations:
    game, grid = copy_game(level_start), start_grid
    for action in situation["prefix"]:
      grid = grid_of(play(game, action))
    status = "ok" if np.array_equal(grid, situation["grids"][0]) else "start differs from the recording"
    for step, action in enumerate(situation["actions"]):
      if status != "ok":
        break
      grid = grid_of(play(game, action))
      if not np.array_equal(grid, situation["grids"][step + 1]):
        status = f"differs from the recording at step {step + 1}"
    reached = np.zeros(len(ACTIONS) ** horizon, dtype=np.int8)
    if status == "ok":
      expand(game, 0, 0, situation["goal"], reached)
    results.append({"key": situation["key"], "status": status, "reached": reached})
  return results


def collect_situations(name: str, dataset: str, counts: dict, shares: dict, goal_ahead: int, horizon: int, seed: int, task_index: int, tasks: int, workers: int):
  """
  GETS:    name       -- the collection's name: datasets/ls20/d_jepa_<name>/.
           dataset    -- the Lance dataset the situations come from (its latest version), e.g. syn1000_train.
           counts     -- {"fit", "calibration", "test"}: situations wanted in each part, spread evenly over its levels.
           shares     -- {"calibration", "test"}: share of the levels in these parts; the rest are fit levels.
           goal_ahead -- the goal is the recorded state this many moves later.
           horizon    -- moves per plan (the planner's horizon).
           seed       -- for the level split and the draws.
           task_index, tasks -- this array task takes every tasks-th level, starting at task_index.
           workers    -- processes that play the plans.
  DOES:    splits the levels into the three parts (the same split in every task), draws this task's situations (module
           docstring), replays and labels them in parallel over levels (label_situations_of_level) and drops the
           situations with only one outcome.
  RETURNS: the written part<task_index>.npz; part<task_index>.json next to it holds the settings and counts.
  """
  from master_thesis.evaluation.lewm_analysis import classify_change, list_episodes, read_recorded_rows

  episodes, version = list_episodes(dataset)
  levels = sorted(set(episodes.level.tolist()))
  shuffled = np.random.default_rng([seed, 0]).permutation(levels).tolist()
  calibration_count, test_count = round(len(levels) * shares["calibration"]), round(len(levels) * shares["test"])
  part_of = {level: "calibration" if index < calibration_count else "test" if index < calibration_count + test_count else "fit" for index, level in enumerate(shuffled)}
  levels_in = {part: sum(value == part for value in part_of.values()) for part in PARTS}
  per_level = {part: math.ceil(counts[part] / max(levels_in[part], 1)) for part in PARTS}
  task_levels = levels[task_index::tasks]

  # candidates per level: 3x the needed count, drawn uniformly over the states with a goal goal_ahead moves later
  candidates = []
  for level in task_levels:
    rng = np.random.default_rng([seed, 1, level])
    level_episodes = episodes[episodes.level == level]
    usable = (level_episodes.length - (level_episodes.success | level_episodes.terminal_end)).to_numpy()  # states that may be start or goal
    starts = np.maximum(usable - goal_ahead, 0)  # states whose goal goal_ahead moves later is still usable
    owner = np.repeat(np.arange(len(level_episodes)), starts)  # every candidate state of the level: its episode ...
    step = np.arange(len(owner)) - np.repeat(np.cumsum(starts) - starts, starts)  # ... and its step in that episode
    for index in rng.permutation(len(owner))[:3 * per_level[part_of[level]]]:
      candidates.append((level_episodes.iloc[owner[index]], int(step[index])))
  if not candidates:
    raise ValueError(f"task {task_index}: no usable states in levels {task_levels}")

  # keep the first per_level candidates of each level that do not already show their goal
  pairs, _, _ = read_recorded_rows(dataset, version, np.array([[row.offset + step, row.offset + step + goal_ahead] for row, step in candidates]).reshape(-1))  # start, goal, start, goal, ...
  kept, taken = [], {}
  for (row, step), start, goal in zip(candidates, pairs[0::2], pairs[1::2]):
    if classify_change(start, goal) != "mechanics" or taken.get(row.level, 0) >= per_level[part_of[row.level]]:
      continue
    taken[row.level] = taken.get(row.level, 0) + 1
    kept.append((row, step))

  # their recorded grids and actions from the episode start (for the replay), and the start paths of walk runs
  grids, actions, game_ids = read_recorded_rows(dataset, version, np.concatenate([np.arange(row.offset, row.offset + step + goal_ahead + 1) for row, step in kept]))
  provenance = dataset_path(GAME, dataset).with_suffix(".episodes.jsonl")  # one line per episode, written with the Lance table; recordings from the level start have no or an empty start path
  wanted = {int(row.episode_idx) for row, _ in kept}
  prefixes = {}
  if provenance.exists():
    with provenance.open(encoding="utf-8") as file:
      prefixes = {record["episode_idx"]: record.get("prefix") or [] for record in map(json.loads, file) if record["episode_idx"] in wanted}
  situations, jobs, position = [], {}, 0
  for key, (row, step) in enumerate(kept):
    length = step + goal_ahead + 1
    episode_grids, episode_actions, game_id = grids[position:position + length], actions[position:position + length], game_ids[position]
    position += length
    situations.append({"level": int(row.level), "episode": int(row.episode), "step": step, "part": part_of[row.level], "start": episode_grids[step], "goal": episode_grids[-1]})
    jobs.setdefault((game_id, int(row.level)), []).append({"key": key, "prefix": prefixes.get(int(row.episode_idx), []), "grids": episode_grids[:step + 1], "actions": episode_actions[:step], "goal": episode_grids[-1]})

  # label every situation in the game, one job per (game, level)
  jobs = [(game_id, level, horizon, level_situations) for (game_id, level), level_situations in sorted(jobs.items())]
  if workers > 1:
    from master_thesis.environments.d_jepa_collection import label_situations_of_level as worker  # imported by its package name, so the worker processes can find it
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
      labelled = [result for results in pool.map(worker, jobs) for result in results]
  else:
    labelled = [result for job in jobs for result in label_situations_of_level(job)]
  dropped = {}
  for result in labelled:
    reached = result["reached"]
    status = result["status"] if result["status"] != "ok" else "only one outcome" if (reached > 0).all() or (reached == 0).all() else "ok"
    situations[result["key"]].update(reached=reached, status=status)
    if status != "ok":
      dropped[status] = dropped.get(status, 0) + 1
  good = [situation for situation in situations if situation["status"] == "ok"]

  folder = d_jepa_data_dir(GAME, name)
  folder.mkdir(parents=True, exist_ok=True)
  path = folder / f"part{task_index}.npz"
  np.savez_compressed(path, **{key: np.stack([situation[key] for situation in good]) for key in ("start", "goal", "reached")},
                      **{key: np.array([situation[key] for situation in good]) for key in ("level", "episode", "step", "part")})
  summary = {"levels": len(task_levels), "candidates": len(candidates), "situations": len(situations), "kept": {part: sum(situation["part"] == part for situation in good) for part in PARTS}, "dropped": dropped,
             "success_share": float(np.mean([(situation["reached"] > 0).mean() for situation in good])) if good else None}  # how many of the 1,024 plans reach the goal, on average
  settings = {"dataset": dataset, "version": version, "goal_ahead": goal_ahead, "horizon": horizon, "seed": seed, "counts": counts, "shares": shares, "levels_per_part": levels_in,
              "task_index": task_index, "tasks": tasks, "summary": summary}
  path.with_suffix(".json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
  print(json.dumps(summary))
  return path


def main():
  parser = argparse.ArgumentParser(description="Label every plan of situations from recorded episodes, for D-JEPA's ranker (no model needed)")
  parser.add_argument("--name", required=True, help="collection name: datasets/ls20/d_jepa_<name>/")
  parser.add_argument("--dataset", default="syn1000_train", help="Lance dataset the situations come from")
  parser.add_argument("--fit-situations", type=int, default=2000, help="situations to fit the ranker on, spread evenly over the fit levels")
  parser.add_argument("--calibration-situations", type=int, default=500, help="situations to calibrate its gate on, spread evenly over the calibration levels")
  parser.add_argument("--test-situations", type=int, default=500, help="situations to test it on, spread evenly over the test levels")
  parser.add_argument("--calibration-share", type=float, default=0.15, help="share of the levels used for calibration")
  parser.add_argument("--test-share", type=float, default=0.15, help="share of the levels used for testing")
  parser.add_argument("--goal-ahead", type=int, default=5, help="the goal is the recorded state this many moves later")
  parser.add_argument("--horizon", type=int, default=5, help="moves per plan (the planner's horizon)")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--task-index", type=int, default=0, help="this array task: levels task_index, task_index + tasks, ...")
  parser.add_argument("--tasks", type=int, default=1, help="number of array tasks")
  parser.add_argument("--workers", type=int, default=1, help="processes that play the plans")
  args = parser.parse_args()
  collect_situations(args.name, args.dataset, {"fit": args.fit_situations, "calibration": args.calibration_situations, "test": args.test_situations},
                     {"calibration": args.calibration_share, "test": args.test_share}, args.goal_ahead, args.horizon, args.seed, args.task_index, args.tasks, args.workers)


if __name__ == "__main__":
  main()
