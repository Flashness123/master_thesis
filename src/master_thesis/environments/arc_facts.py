"""
The game's facts for every row of a recorded dataset, for training LeWM with fact heads or a shaped latent
(training/lewm.py: extensions.fact_heads, extensions.latent_shaping). The Lance tables store only grids and actions; the
facts (the player's cell, its energy, the refills left, the key's shape and rotation, ...) are read from the game itself
by replaying every episode, as lewm_analysis does for its probes. It needs no model, so one collection serves every run
on the dataset, and the dataset itself stays unchanged.

Every episode is replayed from its level start: the start path of walk runs (<dataset>.episodes.jsonl), then the
recorded actions, and every replayed grid is checked against the recorded one. At every row the game state is read
(lewm_analysis.read_game_state) before the row's action is played, so row t gets the facts of the grid it shows. Where a
grid differs from the recording, that row and the rest of its episode keep -1, which training ignores.

Output: datasets/ls20/<dataset>_facts/part<i>.npz, one per array task:
  rows   [N] int64: the row in the Lance table, numbered as stable-worldmodel and lewm_analysis.list_episodes number them
  facts  [N, len(FACTS)] int16: the facts in FACTS order, -1 everywhere where the replay differed from the recording
         (tell such rows by their level, which is >= 1 otherwise: the energy itself goes below 0 on the frame it runs out)
and part<i>.json with the dataset version, FACTS and the counts. Task 0 also writes key_patches.npz: the status bar's
key box (KEY_BOX) for every key shape, colour and rotation, rendered by the game (render_key_patches). Training pastes
them into recorded frames to show the same state with another key shape (training/lewm.py extensions.counterfactual_keys):
the training levels have no shape changers, so no recorded pair of frames differs only in the key's shape.

Usage (one array task per process; a Barnard array job runs --task-index 0..n-1 with --tasks n):
  python -m master_thesis.environments.arc_facts --dataset syn1000_train --workers 8
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from master_thesis.paths import dataset_path, facts_data_dir

GAME = "ls20"
FACTS = ("level", "player_x", "player_y", "energy", "lives", "refills_left", "key_shape", "key_color", "key_rotation")  # names as lewm_analysis.read_game_state returns them; level = the level the grid shows (the completion frame of a solved level already shows the next one)
KEY_BOX = (slice(55, 61), slice(3, 9))  # rows and columns of the status bar's key box on the 64 x 64 grid (ls20's render_interface draws the 6 x 6 key sprite at row 55, column 3)


def render_key_patches(game_id: str):
  """
  GETS:    game_id -- the game the dataset was recorded in.
  DOES:    in a copy of the game at the start of level 1, gives the key every shape, colour and rotation the way the game's
           own changers do (the shape's pixels, recoloured from the base colour to the key's colour, turned by the
           sprite's rotation) and renders the frame as perform_action does (camera.render); the key box is all that
           changes between them.
  RETURNS: patches [shapes, colours, rotations, 6, 6] uint8 (6 x 4 x 4 for ls20): the key box of every combination.
  """
  from master_thesis.environments.arc_collection import reset_at_level
  from master_thesis.environments.arc_levelgen import make_game
  from master_thesis.evaluation.lewm_analysis import copy_game

  env = make_game(game_id)
  reset_at_level(env, 1)
  game = env._game
  base_colour = type(game).qetwzqzzik.__globals__["epeqflmtfc"]  # the colour the shapes are stored in before recolouring: a constant of the game's module
  patches = np.zeros((len(game.ijessuuig), len(game.tnkekoeuk), len(game.dhksvilbb), 6, 6), dtype=np.uint8)  # ijessuuig: shapes, tnkekoeuk: colours, dhksvilbb: rotations
  for shape, colour, rotation in itertools.product(*(range(size) for size in patches.shape[:3])):
    copy = copy_game(game)
    copy.fwckfzsyc, copy.hiaauhahz, copy.cklxociuu = shape, colour, rotation  # the key's shape, colour and rotation index
    copy.htkmubhry.pixels = copy.ijessuuig[shape].pixels.copy()  # htkmubhry: the key sprite of the status bar
    copy.htkmubhry.color_remap(base_colour, copy.tnkekoeuk[colour])
    copy.htkmubhry.set_rotation(copy.dhksvilbb[rotation])
    patches[shape, colour, rotation] = np.asarray(copy.camera.render(copy.current_level.get_sprites()), dtype=np.uint8).reshape(64, 64)[KEY_BOX]
  return patches


_WORKER_GAMES = {}  # in each worker process: game_id -> game (making a game takes seconds for syn1000, so each worker makes it once)
def replay_facts_of_level(task):
  """
  GETS:    task -- (dataset, version, level, episodes); an episode is a dict with its first row in the table ("offset"),
           its number of rows ("length") and the start path of walk runs ("prefix", empty for recordings from the level start).
  DOES:    runs in a worker process. Reads the level's recorded grids and actions, replays every episode from the level
           start (start path, then the recorded actions, checking every grid against the recording) and reads the
           game's facts at every row before its action is played.
  RETURNS: (rows [N] int64, facts [N, len(FACTS)] int16 with -1 from the first differing grid of an episode on,
           {status: number of episodes}).
  """
  from arcengine import ActionInput, GameAction

  from master_thesis.environments.arc_collection import reset_at_level
  from master_thesis.environments.arc_levelgen import make_game
  from master_thesis.evaluation.lewm_analysis import copy_game, read_game_state, read_recorded_rows

  dataset, version, level, episodes = task
  rows = np.concatenate([np.arange(episode["offset"], episode["offset"] + episode["length"]) for episode in episodes])
  grids, actions, game_ids = read_recorded_rows(dataset, version, rows)
  if len(set(game_ids)) != 1:
    raise ValueError(f"level {level} of {dataset} was recorded in several games: {sorted(set(game_ids))}")
  if game_ids[0] not in _WORKER_GAMES:
    _WORKER_GAMES[game_ids[0]] = make_game(game_ids[0])
  grid_of = lambda frame: np.asarray(frame.frame[-1], dtype=np.uint8).reshape(-1)  # its last frame is what the player sees
  start_grid = grid_of(reset_at_level(_WORKER_GAMES[game_ids[0]], level))
  level_start = copy_game(_WORKER_GAMES[game_ids[0]]._game)
  play = lambda game, action: game.perform_action(ActionInput(id=GameAction.from_id(int(action))), raw=True)

  facts = np.full((len(rows), len(FACTS)), -1, dtype=np.int16)
  statuses, first = {}, 0  # first: the episode's first position in rows
  for episode in episodes:
    length, game, grid, status = episode["length"], copy_game(level_start), start_grid, "ok"
    for action in episode["prefix"]:
      grid = grid_of(play(game, action))
    if not np.array_equal(grid, grids[first]):
      status = "start differs from the recording"
    for step in range(length if status == "ok" else 0):
      state = read_game_state(game)
      facts[first + step] = [state[name] for name in FACTS]
      if step + 1 < length:  # the last row's action ends the episode and is not played
        grid = grid_of(play(game, actions[first + step]))
        if not np.array_equal(grid, grids[first + step + 1]):
          status = "differs from the recording"  # the rows from here on keep -1
          break
    statuses[status] = statuses.get(status, 0) + 1
    first += length
  return rows, facts, statuses


def collect_facts(dataset: str, task_index: int, tasks: int, workers: int):
  """
  GETS:    dataset    -- the Lance dataset (its latest version), e.g. syn1000_train.
           task_index, tasks -- this array task takes every tasks-th level, starting at task_index.
           workers    -- processes that replay the levels.
  DOES:    lists the episodes of this task's levels with their start paths and replays them level by level
           (replay_facts_of_level), in parallel over levels.
  RETURNS: the written part<task_index>.npz; part<task_index>.json next to it holds the settings and counts.
  """
  from master_thesis.evaluation.lewm_analysis import list_episodes

  episodes, version = list_episodes(dataset)
  levels = sorted(set(episodes.level.tolist()))[task_index::tasks]
  episodes = episodes[episodes.level.isin(levels)]
  provenance = dataset_path(GAME, dataset).with_suffix(".episodes.jsonl")  # one line per episode, written with the Lance table; recordings from the level start have no or an empty start path
  wanted = set(episodes.episode_idx.astype(int).tolist())
  prefixes = {}
  if provenance.exists():
    with provenance.open(encoding="utf-8") as file:
      prefixes = {record["episode_idx"]: record.get("prefix") or [] for record in map(json.loads, file) if record["episode_idx"] in wanted}
  jobs = [(dataset, version, level, [{"offset": int(row.offset), "length": int(row.length), "prefix": prefixes.get(int(row.episode_idx), [])} for row in episodes[episodes.level == level].itertuples()])
          for level in levels]

  if workers > 1:
    from master_thesis.environments.arc_facts import replay_facts_of_level as worker  # imported by its package name, so the worker processes can find it
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
      results = list(pool.map(worker, jobs))
  else:
    results = [replay_facts_of_level(job) for job in jobs]
  rows, facts = np.concatenate([result[0] for result in results]), np.concatenate([result[1] for result in results])
  statuses = {}
  for _, _, level_statuses in results:
    for status, count in level_statuses.items():
      statuses[status] = statuses.get(status, 0) + count

  folder = facts_data_dir(GAME, dataset)
  folder.mkdir(parents=True, exist_ok=True)
  path = folder / f"part{task_index}.npz"
  np.savez_compressed(path, rows=rows, facts=facts)
  valid = facts[:, 0] >= 0
  summary = {"levels": len(levels), "episodes": len(episodes), "rows": len(rows), "rows_with_facts": int(valid.sum()), "episodes_by_status": statuses,
             "ranges": {name: [int(facts[valid, index].min()), int(facts[valid, index].max())] if valid.any() else None for index, name in enumerate(FACTS)}}
  settings = {"dataset": dataset, "version": version, "facts": list(FACTS), "task_index": task_index, "tasks": tasks, "summary": summary}
  path.with_suffix(".json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
  if task_index == 0:  # one task writes the key patches, the same for every part
    from master_thesis.evaluation.lewm_analysis import read_recorded_rows

    game_id = read_recorded_rows(dataset, version, [0])[2][0]
    np.savez_compressed(folder / "key_patches.npz", patches=render_key_patches(game_id), rows=np.arange(64)[KEY_BOX[0]], columns=np.arange(64)[KEY_BOX[1]])
  print(json.dumps(summary))
  return path


def main():
  parser = argparse.ArgumentParser(description="Read the game's facts at every row of a recorded dataset by replaying its episodes (no model needed)")
  parser.add_argument("--dataset", default="syn1000_train", help="Lance dataset whose rows get facts: datasets/ls20/<dataset>_facts/")
  parser.add_argument("--task-index", type=int, default=0, help="this array task: levels task_index, task_index + tasks, ...")
  parser.add_argument("--tasks", type=int, default=1, help="number of array tasks")
  parser.add_argument("--workers", type=int, default=1, help="processes that replay the levels")
  args = parser.parse_args()
  collect_facts(args.dataset, args.task_index, args.tasks, args.workers)


if __name__ == "__main__":
  main()
