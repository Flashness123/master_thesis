"""
┌─────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┬────────────────────┐
│                     Experiment                      │                        Needs                         │        Uses        │
├─────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼────────────────────┤
│ latent_variance, latent_maps, checkpoint_similarity │ the state's latent                                   │ states             │
├─────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼────────────────────┤
│ state_probes                                        │ latent + game state (+ move success)                 │ states             │
├─────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼────────────────────┤
│ action_decoding                                     │ latent change of every move that changed the game    │ states             │
├─────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼────────────────────┤
│ state_aliasing                                      │ latent + what the 4 moves did                        │ states             │
├─────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼────────────────────┤
│ one_step_prediction                                 │ state + its recorded next state                      │ states (next_grid) │
├─────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼────────────────────┤
│ counterfactual_actions                              │ state + the 2 states before it + the 4 move outcomes │ states (history)   │
├─────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼────────────────────┤
│ action_ranking                                      │ state + 4 outcomes + the state 5 steps later         │ states (goal_grid) │
├─────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼────────────────────┤
│ multi_step_rollout                                  │ state + 2 before it + 10 actions + 10 states after   │ windows            │
├─────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼────────────────────┤
│ cem_planning                                        │ whole solutions                                      │ its own episodes   │
└─────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┴────────────────────┘

Every experiment runs on three groups of episodes ("splits"), so we can see whether the model only remembers what it
was trained on:
  seen_train  episodes the model was trained on
  seen_val    other episodes of the same levels, which the model never saw during training
  unseen      levels that were not in the training data at all (--unseen). For syn1000 these are every 50th synthetic
              level (57, 107, ..., 1007: the same tier mix and solution lengths as all levels), converted to their own
              dataset syn1000_unseen; training uses syn1000_train with the other 980 levels.
For seen_train and seen_val we recreate the exact split the training used (same dataset version, seed and fractions).
The training's test episodes are never touched. The unseen episodes are split in two halves per level: "adapt" is only
used to fine-tune the probes on the new levels, "eval" is where the probes are then tested.

How the script works:
  1. Load the checkpoint and read from its training config how it was trained.
  2. Find the episodes of the three splits.
  3. Pick the states to look at: per level 200 states and 50 rollout windows (13 recorded states in a row). The picks
     depend only on the data and the seed, so every checkpoint is tested on exactly the same states.
  4. Replay the picked episodes in the real game and check that every grid matches the recording. At each picked state,
     read what the game knows (where the player is, the symbol, energy, lives, the nearest goal, ...) and try all 4
     actions on copies of the game to see what each one really does. Together this is a split's "evaluation set".
  5. Encode every grid with the checkpoint's encoder.
  6. Run all experiments and save their results.

The experiments, in the groups the dashboard will show:
  encoder: does the representation keep the information needed to tell possible futures apart?
    latent_variance         is the representation collapsed, and does it mostly store which level we are in?
    state_probes            which facts about the game can be read from the latent?
    action_decoding         can the move be read from how the latent changes? (what Delta-JEPA trains for)
    state_aliasing          do states that look close to the model really behave the same in the game?
    latent_maps             PCA and UMAP maps of the latents, only for looking at them
    checkpoint_similarity   how similar are the representations of two checkpoints? (only with --compare-to)
  predictor: given a useful representation, does the predictor know how actions change the future?
    counterfactual_actions  does it know what each of the 4 actions would do from the same state?
    multi_step_rollout      does the error grow when it keeps predicting from its own predictions? does real history help?
  world_model: how well do encoder and predictor together predict the next state?
    one_step_prediction     the next state from one frame and the action, compared with "nothing changes"
  planning: can the model steer the game?
    cem_planning            reach goal states and follow a recorded solution with CEM in the real game
    action_ranking          does the predictor rank the actions towards a goal like the real game does?
  generalization: what changes between known and new data?
    split_gaps              for every metric: seen_val minus seen_train, and unseen minus seen_val
    state_probes            its results on the unseen levels (zero-shot and fine-tuned) belong to this group
    action_decoding         its results on the unseen levels (zero-shot and fine-tuned) belong to this group
Every metric row carries its group, so the dashboard can sort them without knowing the experiments.

How a step changed the grid is sorted into three kinds (classify_change):
  mechanics    something in the game changed: the player moved, the symbol changed, a refill was taken, a life was lost, ...
  energy_only  only the energy bar changed: a blocked move, which still costs energy
  none         nothing changed, e.g. walking into a goal with the wrong symbol costs nothing

All settings (how many states, CEM, probes, ...) are in EXPERIMENT_PARAMETERS, chosen with --experiment-parameters:
"full" (about 45k states, about 20 minutes of encoding on a Mac and a few minutes on an H100) or "smoke" (tiny, only to
check that everything runs).

Files, below $STABLEWM_HOME/evaluation/ls20/lewm_analysis/<checkpoint>/ (overwritten by every run):
  results.json        the checkpoint and how it was trained, the parameters, the evaluated levels, how many states each
                      split had (and how many episodes were dropped), and every metric as one row: group, experiment,
                      metric, split, level, subset, value, n
  latent_maps.json    PCA and UMAP coordinates of every state with the labels to colour them by

Usage:
  python -m master_thesis.evaluation.lewm_analysis --checkpoint <run>/weights.pt --unseen syn1000_unseen
  python -m master_thesis.evaluation.lewm_analysis --checkpoint <run>/checkpoints/weights_step100000.pt --compare-to <other run>/weights.pt --unseen syn1000_unseen
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from master_thesis.paths import dataset_path, evaluation_dir, model_dir

GAME = "ls20"
ACTIONS = (1, 2, 3, 4)  # the 4 moves of ls20: ACTION1 .. ACTION4
ACTION_NAMES = {1: "up", 2: "down", 3: "left", 4: "right"}
DIRECTIONS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # how each move changes (column, row) on the playing field
ACTION_DIM = 9  # LeWM's action vector: 7 action types (one-hot) + the x and y of a click (training/lewm.py ArcActionEncoding)
SPLITS = ("seen_train", "seen_val", "unseen")
ROLLOUT_HORIZONS = (1, 3, 5, 10)  # how many steps multi_step_rollout predicts ahead, all on the same windows
ENERGY_BAR = (slice(61, 63), slice(13, 55))  # rows and columns of the energy bar on the 64 x 64 grid
CHANGE_KINDS = ("none", "energy_only", "mechanics")  # how a step changed the grid (classify_change)
D = 192  # size of a latent (embed_dim of every LeWM run so far)
BATCH_SIZE = 256  # grids or latents per pass through the model; changes only speed and memory, no result

# Every setting of the experiments, chosen with --experiment-parameters.
#   full:  about 45k states and 11k rollout windows for 100 seen and ~20 unseen levels; encoding takes about 20 minutes
#          on a Mac (MPS, ~290 grids per second) and a few minutes on an H100.
#   smoke: tiny, only to check that everything runs.
EXPERIMENT_PARAMETERS = {"full": {
  "seed": 0,                    # for every random pick: levels, states, windows, planning episodes, the unseen adapt/eval halves
  # which states the experiments look at
  "seen_levels": 100,           # seen levels evaluated, the same for seen_train and seen_val: all real levels plus a random choice of the others (0 = all)
  "states_per_level": 200,      # picked at random from all states of the level's episodes
  "windows_per_level": 50,      # rollout windows: 13 recorded states in a row (history - 1 before the rollout start, the start, 10 after it)
  "goal_ahead": 5,              # action_ranking: the goal is the recorded state this many steps later
  "history": 3,                 # counterfactual_actions and multi_step_rollout: most real frames they predict from (the state and the ones before it; never more than the model's history_size)
  # state_probes and state_aliasing
  "probe_steps": 2000,          # training steps (batches of 256) on seen_train
  "probe_adapt_steps": 300,     # fine-tuning steps on the unseen adapt half
  "knn": 10,                    # neighbours per state
  "projection_levels": [],      # latent_maps: levels that also get a map of their own
  # cem_planning
  "planning_levels": 5,         # levels per split
  "goal_steps": [5, 10, 25],    # goals this many steps after the start of a recorded solution
  "waypoint_every": 5,          # every how many recorded states a waypoint is placed
  "samples": 300,               # CEM: action sequences tried per round
  "iterations": 10,             # CEM: rounds per plan
  "horizon": 5,                 # CEM: actions per plan
  "topk": 30,                   # CEM: how many of the best sequences the next round learns from
  "replan_every": 5,            # actions of a plan played before planning again
  "cost_on": "path",            # score a plan by its closest imagined point to the goal ("path") or its last imagined state ("end")
}}
EXPERIMENT_PARAMETERS["smoke"] = {**EXPERIMENT_PARAMETERS["full"], "seen_levels": 10, "states_per_level": 30, "windows_per_level": 20, "probe_steps": 50, "probe_adapt_steps": 10,
                                  "planning_levels": 1, "goal_steps": [5], "samples": 20, "iterations": 2, "topk": 5}

# The dashboard's groups and the experiments in them. state_probes appears twice: its seen-level results belong to
# encoder, its unseen-level results to generalization.
DASHBOARD_GROUPS = {"encoder": ("latent_variance", "state_probes", "action_decoding", "state_aliasing", "latent_maps", "checkpoint_similarity"),
                    "predictor": ("counterfactual_actions", "multi_step_rollout"), "world_model": ("one_step_prediction",),
                    "planning": ("cem_planning", "action_ranking"), "generalization": ("state_probes", "action_decoding", "split_gaps")}


##########################
## The game and labels ##
##########################


def copy_game(game):
  """
  Make a copy of the game that we can try one action on, without touching the original.
  Copying all 1000 levels takes 1.2 seconds so we only copy the current level and keep others as is. Deepcopy accepts arguments what not to copy
  """
  current = game._levels[game._current_level_index]
  shared = {id(level): level for level in game._levels if level is not current}  # deepcopy uses these objects as they are
  shared.update({id(level): level for level in game._clean_levels})  # the untouched originals the game resets levels from
  return copy.deepcopy(game, shared)


def read_game_state(game):
  """
  No grid parsing, we ask the game directly about its current state and use that as labels for the probes. x,y coordinates correspond to the top left corner of the object
  Positions are (column, row) on the 12 x 12 playing field. "Nearest" means the smallest manhatten distance away from the player.
  Returns a dict of numbers; -1 means the object does not exist (e.g. no refill left)
  """
  from master_thesis.environments.arc_levelgen import REFILL, ROTATOR

  cell = lambda sprite, dx=0, dy=0: ((sprite.x + dx - 4) // 5, (sprite.y + dy) // 5)  # pixel position -> cell, as arc_levelgen.spec_from_level
  player = cell(game.gudziatsk)
  distance = lambda target: abs(target[0] - player[0]) + abs(target[1] - player[1])  # moves needed if nothing is in the way
  nearest = lambda cells: min(cells, key=distance, default=(-1, -1))
  key = (game.fwckfzsyc, game.hiaauhahz, game.cklxociuu)

  unreached = [index for index in range(len(game.plrpelhym)) if not game.lvrnuajbl[index]]
  goal = min(unreached, key=lambda index: distance(cell(game.plrpelhym[index])), default=None)
  goal_cell = cell(game.plrpelhym[goal]) if goal is not None else (-1, -1)
  goal_key = (game.ldxlnycps[goal], game.yjdexjsoa[goal], game.ehwheiwsk[goal]) if goal is not None else (-1, -1, -1)
  refills = [cell(sprite, -1, -1) for sprite in game.current_level.get_sprites_by_tag(REFILL)]  # a refill sits 1 pixel inside its cell; taken ones are removed from the level
  rotators = [cell(sprite) for sprite in game.current_level.get_sprites_by_tag(ROTATOR)]

  return {"level": game._current_level_index + 1, "player_x": player[0], "player_y": player[1], "energy": int(game._step_counter_ui.current_steps), "lives": int(game.aqygnziho),
          "key_shape": key[0], "key_color": key[1], "key_rotation": key[2],
          "goal_x": goal_cell[0], "goal_y": goal_cell[1], "goal_shape": goal_key[0], "goal_color": goal_key[1], "goal_rotation": goal_key[2],
          "goal_distance": distance(goal_cell) if goal is not None else -1, "key_matches_goal": int(goal is not None and key == goal_key),
          "refills_left": len(refills), "refill_x": nearest(refills)[0], "refill_y": nearest(refills)[1],
          "rotator_present": int(bool(rotators)), "rotator_x": nearest(rotators)[0], "rotator_y": nearest(rotators)[1]}


def classify_change(before, after):
  """
  Say how the grid changed from `before` to `after`: "mechanics", "energy_only" or "none" (see the module docstring).
  Works on two single grids or on whole arrays of grids (the last axis is the 4096 pixels). A refill counts as mechanics.
  """
  differs = np.asarray(before).reshape(*np.shape(before)[:-1], 64, 64) != np.asarray(after).reshape(*np.shape(after)[:-1], 64, 64)
  in_bar = np.zeros((64, 64), bool)
  in_bar[ENERGY_BAR] = True
  return np.where((differs & ~in_bar).any(axis=(-2, -1)), "mechanics", np.where((differs & in_bar).any(axis=(-2, -1)), "energy_only", "none"))


_WORKER_GAMES = {}  # in each worker process: game_id -> game. Making a game takes seconds for syn1000, so each worker makes it once.
def replay_episodes_of_level(task):
  """
  An episode is: one continuous piece of play within one level. It has no fixed length
  We have three kinds of episodes so far: Human Recordings (Start->End), BFS Solutions (Start->End), Synthetic walks (middle-first lost life)
  For probes we sometimes need goal, refills, rotator etc - lance only saves grids and actions -> we need to replay what we are interested in and extrect goal, refills, rotator etc from the game state
  We also need this function for replaying the game up to a popint where we copy it and then can choose and compare multiple actions and outcomes for counterfactual_actions.

  This runs in a separate worker process. `task` is (game_id, level, episode(s)), it gets build by build_evaluation_set and the games are buffered in _WORKER_GAMES so that they dont have to be created each time

  For every episode:
    - copy the level start, play the start path and check that we arrive at the recorded first grid,
    - play the recorded actions one by one and check every grid against the recording, up to the last grid that any
      picked state, goal or rollout window needs. If a grid differs, the episode is dropped.
    - at every picked step, read the game state (read_game_state) and play each of the 4 actions on its own copy.

  What each tried action did is stored so that states of different levels can be compared: how far the player moved
  (dx, dy), how far the symbol's shape / colour / rotation changed etc - we need this for state aliasing, and reaplying and querying the engine is faster than grid parsing

  Returns one dict per episode: its key, "ok" or why it was dropped, the game state at each picked step, the grids after
  each of the 4 actions, and what each action did.
  """
  from arcengine import ActionInput, GameAction, GameState

  from master_thesis.environments.arc_collection import reset_at_level
  from master_thesis.environments.arc_levelgen import make_game

  game_id, level, episodes = task
  if game_id not in _WORKER_GAMES:
    _WORKER_GAMES[game_id] = make_game(game_id)
  start_grid = np.asarray(reset_at_level(_WORKER_GAMES[game_id], level).frame[-1], dtype=np.uint8).reshape(-1)
  level_start = copy_game(_WORKER_GAMES[game_id]._game)
  play = lambda game, action: game.perform_action(ActionInput(id=GameAction.from_id(int(action))), raw=True)  # its last frame is what the player sees

  results = []
  for episode in episodes:
    game, grids, positions = copy_game(level_start), episode["grids"], set(episode["positions"])
    grid = start_grid
    for action in episode["prefix"]:
      grid = np.asarray(play(game, action).frame[-1], dtype=np.uint8).reshape(-1)
    result = {"key": episode["key"], "status": "ok", "labels": [], "outcome_grids": [], "outcomes": []}
    if not np.array_equal(grid, grids[0]):
      result["status"] = "start differs from the recording" if episode["prefix"] else "start differs from the recording (no start path)"
      results.append(result)
      continue

    for step in range(len(grids)):
      if step in positions:
        before = read_game_state(game)
        row = []
        for action in ACTIONS:
          trial = copy_game(game)
          frame = play(trial, action)
          row.append(np.asarray(frame.frame[-1], dtype=np.uint8).reshape(-1))
          after = read_game_state(trial)
          completed = trial._current_level_index != game._current_level_index or frame.state == GameState.WIN
          life_lost = not completed and after["lives"] < before["lives"]
          reset = completed or life_lost  # the game now shows the next level or the level start again, so position and symbol say nothing about this action
          dx, dy = DIRECTIONS[action] if completed else (0, 0) if life_lost else (after["player_x"] - before["player_x"], after["player_y"] - before["player_y"])
          result["outcomes"].append({"action": action, "dx": dx, "dy": dy, "d_shape": 0 if reset else (after["key_shape"] - before["key_shape"]) % 6,
                                     "d_color": 0 if reset else (after["key_color"] - before["key_color"]) % 4, "d_rotation": 0 if reset else (after["key_rotation"] - before["key_rotation"]) % 4,
                                     "refill_taken": int(not reset and after["refills_left"] < before["refills_left"]), "life_lost": int(life_lost), "completed": int(completed),
                                     "success": int(completed or (not life_lost and (dx, dy) != (0, 0))), "change": str(classify_change(grid, row[-1]))})
        result["labels"].append(before)
        result["outcome_grids"].append(np.stack(row))
      if step + 1 < len(grids):  # follow the recording up to the last grid we need
        grid = np.asarray(play(game, episode["actions"][step]).frame[-1], dtype=np.uint8).reshape(-1)
        if not np.array_equal(grid, grids[step + 1]):
          result.update(status=f"differs from the recording at step {step + 1}", labels=[], outcome_grids=[], outcomes=[])
          break
    results.append(result)
  return results


#################
## Checkpoints ##
#################


def load_checkpoint(checkpoint: str, device: str):
  """
  Rebuild a trained LeWM from a weights file, and read how it was trained, recreate its data splits.
  From train_config.yaml we need the image size the encoder expects, how many past latents the predictor looks at,
  the training dataset, and everything that decided the train/val - we keep these information in memory and discard the later
  """
  from hydra.utils import instantiate
  from omegaconf import OmegaConf

  weights_file = Path(checkpoint) if Path(checkpoint).is_absolute() else model_dir("lewm", ".") / checkpoint
  folder = next(parent for parent in weights_file.resolve().parents if (parent / "model_config.json").exists())  # the run folder
  run, weights = folder.name, weights_file.resolve().relative_to(folder).as_posix()
  model = instantiate(json.loads((folder / "model_config.json").read_text(encoding="utf-8")))
  model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True), strict=True)
  model.to(device).eval()

  config = OmegaConf.load(folder / "train_config.yaml")
  history_size = int(config.wm.history_size)
  rollout_loss = bool(OmegaConf.select(config, "extensions.rollout_loss", default=False))  # older runs have no extensions block: they used the one-step loss
  info = {"id": run if weights == "weights.pt" else f"{run}__{Path(weights).with_suffix('').as_posix().replace('/', '_')}", "run": run, "weights": weights,
          "img_size": int(config.img_size), "history_size": history_size,
          "dataset": str(config.data.dataset), "frameskip": int(config.data.frameskip), "window": int(OmegaConf.select(config, "loss.rollout.context", default=1)) + int(config.wm.num_preds) if rollout_loss else history_size + 1,  # as training/lewm.py build_dataset; runs before 24.10.26 started their rollout from 1 real frame
          "seed": int(config.seed), "train_split": float(config.train_split), "val_split": float(config.val_split),
          "started": datetime.fromtimestamp((folder / "train_config.yaml").stat().st_mtime).isoformat(timespec="seconds")}
  return model, info


@torch.no_grad()
def encode_grids(model, img_size: int, grids, device: str):
  """
  exactly as in trainingTurn grids into latents with the checkpoint's encoder, 
  """
  from master_thesis.training.lewm import ArcGridToPixels

  to_pixels = ArcGridToPixels(img_size)
  unique, position = np.unique(np.asarray(grids, dtype=np.uint8).reshape(-1, 4096), axis=0, return_inverse=True)
  if not len(unique):  # nothing to encode, e.g. a set without rollout windows
    return np.zeros((0, D), np.float32)
  latents = []
  for start in range(0, len(unique), BATCH_SIZE):
    pixels = to_pixels(torch.as_tensor(unique[start:start + BATCH_SIZE], device=device))
    latents.append(model.encode({"pixels": pixels[:, None]})["emb"][:, 0].float().cpu())  # the encoder expects a time axis: here one frame
  return torch.cat(latents).numpy()[position.reshape(-1)]


##########################
## Splits and sampling ##
##########################


def list_episodes(dataset: str, version: int | None = None):
  """
  List all episodes of a Lance dataset, numbered exactly as the training code numbers them, only reads: episode_idx, start_level, episode_success, terminal and uses these in sample_states_and_windows 
  """
  import lance

  table = lance.dataset(str(dataset_path(GAME, dataset)), version=version)
  columns = table.to_table(columns=["episode_idx", "start_level", "episode_success", "terminal"])
  episode_idx = columns.column("episode_idx").to_numpy()
  offsets = np.flatnonzero(np.r_[True, np.diff(episode_idx) != 0])  # an episode = the rows with the same episode_idx
  lengths = np.diff(np.r_[offsets, len(episode_idx)])
  value = lambda name, rows: columns.column(name).combine_chunks().flatten().to_numpy()[rows]
  episodes = pd.DataFrame({"dataset": dataset, "episode": np.arange(len(offsets)), "episode_idx": episode_idx[offsets], "offset": offsets, "length": lengths,
                           "level": value("start_level", offsets).astype(int), "success": value("episode_success", offsets).astype(bool),  # the same on every row of an episode
                           "terminal_end": value("terminal", offsets + lengths - 1).astype(bool)})  # only the last state of an episode can be terminal
  return episodes, table.version


def rebuild_training_split(info: dict):
  """
  Find the episodes the checkpoint was trained on ("train") and the ones held out for validation ("val"), exactly as
  training/lewm.py split them.
  Datasets can be extended after a run, and new episodes would change the shuffle, so we open the dataset in the version it had when the run started (Lance keeps old versions). 
  Returns the train and val episodes, the dataset version, and all levels of the training dataset (the model has seen all of them beacuse vald dataset contains episodes not entire hold out levels).
  """
  import lance

  started = datetime.fromisoformat(info["started"])
  versions = [entry["version"] for entry in lance.dataset(str(dataset_path(GAME, info["dataset"]))).versions() if entry["timestamp"] <= started]
  if not versions:
    raise ValueError(f"{info['dataset']} has no version from before the run started ({info['started']}): the training split cannot be rebuilt")
  episodes, version = list_episodes(info["dataset"], versions[-1])

  long_enough = episodes[episodes.length >= info["window"] * info["frameskip"]].episode.tolist()  # what training could cut windows from
  order = torch.randperm(len(long_enough), generator=torch.Generator().manual_seed(info["seed"])).tolist()  # the same shuffle as training
  shuffled = [long_enough[index] for index in order]
  num_train, num_val = int(len(shuffled) * info["train_split"]), int(len(shuffled) * info["val_split"])
  part = {**{episode: "train" for episode in shuffled[:num_train]}, **{episode: "val" for episode in shuffled[num_train:num_train + num_val]}}
  training_levels = set(episodes.level.tolist())
  episodes = episodes.assign(part=episodes.episode.map(part)).dropna(subset=["part"])  # test episodes and too short ones drop out
  return episodes, version, training_levels


def unseen_level_episodes(specs: list[str], training_levels: set, seed: int):
  """
  Collect the episodes of the levels given with --unseen ("DATASET" or "DATASET:LEVEL,LEVEL").
  Stops with an error if one of these levels is in the checkpoint's training data, since then it is not unseen. Each
  level's episodes are split in two halves: "adapt" for fine-tuning the probes on
  the new levels, "eval" is where they are tested.
  """
  frames, versions = [], {}
  for spec in specs:
    dataset, _, levels = spec.partition(":")
    episodes, versions[dataset] = list_episodes(dataset)
    if levels:
      episodes = episodes[episodes.level.isin([int(level) for level in levels.split(",")])]
    seen = sorted(set(episodes.level) & training_levels)
    if seen:
      raise ValueError(f"--unseen {spec}: levels {seen} are in the checkpoint's training data")
    frames.append(episodes)
  episodes = pd.concat(frames, ignore_index=True)
  part = pd.Series("eval", index=episodes.index)
  for level, rows in episodes.groupby("level").groups.items():
    order = np.random.default_rng([seed, 1, level]).permutation(rows.to_numpy())
    part[order[:len(order) // 2]] = "adapt"
  return episodes.assign(part=part), versions


def read_recorded_rows(dataset: str, version: int, rows):
  """Read selected rows from the lance dataset and provide grids, action ids, game ids"""
  import lance

  table = lance.dataset(str(dataset_path(GAME, dataset)), version=version).take(np.asarray(rows), columns=["grid", "action", "game_id"])
  grids = table.column("grid").combine_chunks().flatten().to_numpy().reshape(-1, 4096).astype(np.uint8)  # older datasets store the grid as float32
  actions = table.column("action").combine_chunks().flatten().to_numpy().reshape(-1, 3)[:, 0].astype(int)  # [action id, x, y] -> action id
  return grids, actions, table.column("game_id").to_pylist()


def sample_states_and_windows(split: str, episodes, parameters):
  """
  Pick the states and rollout windows of one split that the experiments look at, picks:
    - 200 states at random, for almost all experiments
    - 50 windows at random, only for multi_step_rollout
  Per level (each level has its own random generator, so adding or removing a level never changes the picks of the others):
    - states: pick states_per_level at random from all states of the level's episodes. Two kinds of states are never picked: the last state of an episode that completed the level (it already shows the next level), and a game over (no action can be played there).
    one-step and rollout results come from slightly different mixes of episodes, because rollout discard episodes of 12 or shorter: a window holds the 2 real frames before its rollout start and rolls out up to 10 times
  Every usable state of every episode has the same chance, so long episodes give more states than short ones.
  """
  span = ROLLOUT_HORIZONS[-1] + parameters.history - 1  # steps a window covers after its first state: the real frames before the rollout start, then the horizon
  states, windows = [], []
  for level in sorted(set(episodes.level)):
    rng = np.random.default_rng([parameters.seed, SPLITS.index(split), level])
    level_episodes = episodes[episodes.level == level]
    usable = (level_episodes.length - (level_episodes.success | level_episodes.terminal_end)).to_numpy()  # states we may pick in each episode
    starts = np.maximum(level_episodes.length.to_numpy() - span, 0)  # possible window starts in each episode
    for picked, available, count in ((states, usable, parameters.states_per_level), (windows, starts, parameters.windows_per_level)):
      owner = np.repeat(np.arange(len(level_episodes)), available)  # every candidate of the level: its episode ...
      step = np.arange(len(owner)) - np.repeat(np.cumsum(available) - available, available)  # ... and its step in that episode
      keep = np.sort(rng.choice(len(owner), min(count, len(owner)), replace=False))
      picked.append(level_episodes.iloc[owner[keep]].assign(step=step[keep]))
  table = lambda parts: pd.concat(parts, ignore_index=True) if parts else episodes.iloc[:0].assign(step=[])
  return table(states), table(windows)


######################
## Evaluation sets ##
######################


@dataclass
class EvaluationSet:
  """
  Model independent!
  We build three of these: seen_train, seen_val and unseen. with usually 200 states and 50 windows of len 11. Every metric is computed on each one separately, for comparing memorization and generalization.
    counts    how many episodes were candidates, replayed and dropped, and how many states and windows remain
    dropped   why episodes were dropped: reason -> number of episodes
    states    one row per picked state: where it comes from (dataset, episode, step, level, part), the recorded action, has_next / has_goal, next_change, history_frames (real frames up to the state, at most history), and the game state as label_player_x, label_energy
    outcomes  one row per (state, move): what the move really did (dx, dy, life_lost, success, change)
    windows   one row per rollout window: where it starts
    arrays    the grids: grid, next_grid, goal_grid (the state goal_ahead steps later), outcome_grid (after each of
              the 4 actions), history_grid (the history - 1 recorded grids before the state, oldest first; zeros before the
              episode start) and history_action (the actions taken in them), window_grid (13 grids per window: the
              history - 1 real frames before the rollout start, the start, 10 after it) and window_action (the 12
              actions between them)
  """
  split: str
  counts: dict
  dropped: dict
  states: pd.DataFrame
  outcomes: pd.DataFrame
  windows: pd.DataFrame
  arrays: dict


def build_evaluation_set(split: str, episodes, versions: dict, parameters) -> EvaluationSet:
  """
  Build the evaluation set of one split.

  Steps: pick the states and windows (sample_states_and_windows), read the rows they need from Lance (read_recorded_rows), read the start
  paths of synthetic walk runs from the dataset's provenance file, replay every episode a state or window comes from in
  the real game (in parallel, one task per level), and drop the episodes whose replay does not match the recording (with their states and windows). 
  Everything here depends only on the data and the seed, so every checkpoint gets exactly the same set.
  """
  states, windows = sample_states_and_windows(split, episodes, parameters)
  span = ROLLOUT_HORIZONS[-1] + parameters.history - 1  # steps a window covers after its first state, as in sample_states_and_windows

  # for every episode we need: its row in `episodes`, and the last of its rows we need (a picked state's next state and goal, or a window's end)
  needed = {}
  for table, extra in ((states, max(1, parameters.goal_ahead)), (windows, span)):
    for row in table.itertuples(index=False):
      key = (row.dataset, row.episode)
      needed[key] = (row, max(min(row.step + extra, row.length - 1), needed[key][1] if key in needed else 0))

  recorded = {}  # (dataset, episode) -> its recorded grids, actions and game id, up to the last row we need
  for dataset in sorted({key[0] for key in needed}):
    keys = [key for key in needed if key[0] == dataset]
    grids, actions, game_ids = read_recorded_rows(dataset, versions[dataset], np.concatenate([np.arange(needed[key][0].offset, needed[key][0].offset + needed[key][1] + 1) for key in keys]))
    start = 0
    for key in keys:
      length = needed[key][1] + 1
      recorded[key] = {"grid": grids[start:start + length], "action": actions[start:start + length], "game_id": game_ids[start]}
      start += length

  # start paths of recordings that do not begin at the level start (synthetic walk runs), from datasets/ls20/<dataset>.episodes.jsonl
  # (written together with the Lance table, one line per episode); recordings that begin at the level start have none or an empty one
  prefixes = {}
  for dataset in sorted({key[0] for key in needed}):
    provenance = dataset_path(GAME, dataset).with_suffix(".episodes.jsonl")
    wanted = {int(row.episode_idx) for key, (row, _) in needed.items() if key[0] == dataset}
    if provenance.exists():
      with provenance.open(encoding="utf-8") as file:
        records = (json.loads(line) for line in file)
        prefixes[dataset] = {record["episode_idx"]: record.get("prefix") or [] for record in records if record["episode_idx"] in wanted}

  # replay every episode a state or window comes from; one task per (game, level), so a worker needs one level start per task
  positions = {key: sorted(int(step) for step in steps) for key, steps in states.groupby(["dataset", "episode"]).step} if len(states) else {}
  tasks = {}
  for (dataset, episode), (row, _) in needed.items():
    data = recorded[(dataset, episode)]
    tasks.setdefault((data["game_id"], int(row.level)), []).append({"key": (dataset, int(episode)), "prefix": prefixes.get(dataset, {}).get(int(row.episode_idx), []),
                                                                     "grids": data["grid"], "actions": data["action"][:-1], "positions": positions.get((dataset, episode), [])})
  if parameters.workers > 1:
    from master_thesis.evaluation.lewm_analysis import replay_episodes_of_level as worker  # imported by its package name, so the worker processes can find it
    with ProcessPoolExecutor(max_workers=parameters.workers, mp_context=get_context("spawn")) as pool:
      replayed = [episode for result in pool.map(worker, [(game_id, level, jobs) for (game_id, level), jobs in sorted(tasks.items())]) for episode in result]
  else:
    replayed = [episode for (game_id, level), jobs in sorted(tasks.items()) for episode in replay_episodes_of_level((game_id, level, jobs))]
  replayed = {episode["key"]: episode for episode in replayed}
  ok = {key for key, episode in replayed.items() if episode["status"] == "ok"}
  dropped = pd.Series([episode["status"] for episode in replayed.values() if episode["status"] != "ok"], dtype=str).value_counts().to_dict()  # reason -> number of episodes

  # put the states of the episodes that replayed correctly together with their game state, action outcomes and grids
  state_rows, outcome_rows, grid, next_grid, goal_grid, outcome_grid, history_grid, history_action = [], [], [], [], [], [], [], []
  for (dataset, episode), steps in states.groupby(["dataset", "episode"], sort=True).step:
    if (dataset, episode) not in ok:
      continue
    result, data = replayed[(dataset, episode)], recorded[(dataset, episode)]
    row = needed[(dataset, episode)][0]
    for position, step in enumerate(sorted(steps)):
      has_next, has_goal = step + 1 < row.length, step + parameters.goal_ahead < row.length
      state = len(state_rows)
      state_rows.append({"state": state, "dataset": dataset, "episode": int(episode), "episode_idx": int(row.episode_idx), "step": int(step), "level": int(row.level), "part": row.part,
                         "game_id": data["game_id"], "action": int(data["action"][step]), "has_next": has_next, "has_goal": has_goal, "history_frames": min(int(step), parameters.history - 1) + 1,
                         "next_change": str(classify_change(data["grid"][step], data["grid"][step + 1])) if has_next else "", **{f"label_{key}": value for key, value in result["labels"][position].items()}})
      outcome_rows += [{"state": state, **outcome} for outcome in result["outcomes"][position * len(ACTIONS):(position + 1) * len(ACTIONS)]]
      grid.append(data["grid"][step])
      next_grid.append(data["grid"][step + 1] if has_next else np.zeros(4096, dtype=np.uint8))
      goal_grid.append(data["grid"][step + parameters.goal_ahead] if has_goal else np.zeros(4096, dtype=np.uint8))
      outcome_grid.append(result["outcome_grids"][position])
      before = range(step - parameters.history + 1, step)  # the recorded steps before the state, oldest first (negative: before the episode start)
      history_grid.append([data["grid"][earlier] if earlier >= 0 else np.zeros(4096, dtype=np.uint8) for earlier in before])
      history_action.append([int(data["action"][earlier]) if earlier >= 0 else 0 for earlier in before])
  states = pd.DataFrame(state_rows)
  mismatched = states.label_level != states.level if len(states) else pd.Series(dtype=bool)  # the game plays another level than the recording says: should never happen
  if mismatched.any():
    raise RuntimeError(f"{split}: {int(mismatched.sum())} states are in another level than their episode's start_level")

  windows = windows[[(row.dataset, row.episode) in ok for row in windows.itertuples(index=False)]].reset_index(drop=True)
  window_grid = np.stack([recorded[(row.dataset, row.episode)]["grid"][row.step:row.step + span + 1] for row in windows.itertuples(index=False)]) if len(windows) else np.zeros((0, span + 1, 4096), np.uint8)
  window_action = np.stack([recorded[(row.dataset, row.episode)]["action"][row.step:row.step + span] for row in windows.itertuples(index=False)]) if len(windows) else np.zeros((0, span), int)

  if dropped:
    print(f"WARNING {split}: {len(replayed) - len(ok)} of {len(replayed)} episodes dropped ({dropped}); their states and windows are not evaluated")
  counts = {"candidate_episodes": len(episodes), "replayed_episodes": len(replayed), "dropped_episodes": len(replayed) - len(ok), "levels": int(states.level.nunique()) if len(states) else 0,
            "states": len(states), "windows": len(windows)}
  arrays = {"grid": np.asarray(grid, np.uint8).reshape(-1, 4096), "next_grid": np.asarray(next_grid, np.uint8).reshape(-1, 4096), "goal_grid": np.asarray(goal_grid, np.uint8).reshape(-1, 4096),
            "outcome_grid": np.asarray(outcome_grid, np.uint8).reshape(-1, len(ACTIONS), 4096), "window_grid": window_grid.astype(np.uint8), "window_action": window_action.astype(np.int16),
            "history_grid": np.asarray(history_grid, np.uint8).reshape(len(states), parameters.history - 1, 4096), "history_action": np.asarray(history_action, np.int16).reshape(len(states), parameters.history - 1)}
  return EvaluationSet(split=split, counts=counts, dropped=dropped, states=states, outcomes=pd.DataFrame(outcome_rows), windows=windows, arrays=arrays)


def encode_evaluation_set(model, info: dict, evaluation_set: EvaluationSet, parameters) -> dict:
  """
  Encode all grids of an evaluation set with one checkpoint.
  Stores in memory: 
    - state: the picked states;
    - next: their recorded next states;
    - goal: their states 5 steps later (for action ranking);
    - outcome: the 4 grids after each move, per state;
    - history: the recorded states before each state (for counterfactual_actions), NaN before the episode start;
    - window: the 13 frames of each rollout window.
  """
  names = ("state", "next", "goal", "outcome", "history", "window")
  arrays = evaluation_set.arrays
  parts = dict(zip(names, (arrays["grid"], arrays["next_grid"], arrays["goal_grid"], arrays["outcome_grid"], arrays["history_grid"], arrays["window_grid"])))
  latents = encode_grids(model, info["img_size"], np.concatenate([part.reshape(-1, 4096) for part in parts.values()]), parameters.device)  # all at once, so a grid in several parts is encoded once
  encoded, start = {}, 0
  for name, part in parts.items():
    count = int(np.prod(part.shape[:-1]))
    encoded[name] = latents[start:start + count].reshape(*part.shape[:-1], latents.shape[1])
    start += count
  if len(evaluation_set.states):
    encoded["next"][~evaluation_set.states.has_next.to_numpy()] = np.nan
    encoded["goal"][~evaluation_set.states.has_goal.to_numpy()] = np.nan
    encoded["history"][np.arange(parameters.history - 1)[None] < (parameters.history - evaluation_set.states.history_frames.to_numpy())[:, None]] = np.nan  # the zero grids before the episode start
  return encoded


#####################
## Shared helpers ##
#####################


@dataclass
class AnalysisContext:
  """Everything the experiments need: the parameters, the checkpoint and its training info, and per split the episodes, the evaluation set and its latents."""
  parameters: argparse.Namespace
  model: torch.nn.Module
  info: dict
  episodes: dict  # (seen_train, seen_val, unseen) for cem_planning - whole solution from paths from beginning to end
  sets: dict  # three EvaluationSets (seen_train, seen_val, unseen), different from episodes as they dont include whole runs
  latents: dict  # EvaluationSets encoded
  compare_to: tuple | None  # the --compare-to checkpoint

def metric_row(experiment: str, name: str, split: str, value, n: int, level="all", subset: str = "all", group: str | None = None) -> dict:
  """One row of the metrics table. The group is the experiment's dashboard group unless given; level and subset are "all" or a specific value."""
  group = group or next(group for group, experiments in DASHBOARD_GROUPS.items() if experiment in experiments)
  return {"group": group, "experiment": experiment, "metric": name, "split": split, "level": str(level), "subset": str(subset),
          "value": float(value) if value == value else None, "n": int(n)}  # an undefined value (NaN) becomes null: JSON has no NaN


def metric_rows_per_level(experiment: str, name: str, split: str, values, levels, subset: str = "all", denominator=None, group: str | None = None) -> list:
  """
  We get the State, Level, Error values and compute averages from them: one over everything, and one per level. Then we write them with the help of metric_row
  """
  values, levels = np.asarray(values, dtype=float), np.asarray(levels)
  denominator = np.ones_like(values) if denominator is None else np.asarray(denominator, dtype=float)
  defined = ~np.isnan(values) & ~np.isnan(denominator)
  average = lambda selected: values[selected].mean() / denominator[selected].mean() if selected.any() and denominator[selected].mean() > 0 else np.nan
  rows = [metric_row(experiment, name, split, average(defined), defined.sum(), subset=subset, group=group)]
  for level in np.unique(levels[defined]):
    selected = defined & (levels == level)
    rows.append(metric_row(experiment, name, split, average(selected), selected.sum(), level=int(level), subset=subset, group=group))
  return rows


@torch.no_grad()
def encode_actions(context: AnalysisContext):
  """
  Encode the 4 moves (up, down, left, right) with the model's action encoder [1 0 0 0 0 0 0 | 0 0] -> 192 dims , then we save the actions because they dont change
  """
  return context.model.action_encoder(torch.eye(len(ACTIONS), ACTION_DIM, device=context.parameters.device)[:, None])[:, 0]


@torch.no_grad()
def predict_next_latent(context: AnalysisContext, latents, actions):
  """
  Predict the latent after the last action from real latents [N, frames, 192] and the action ids (1-4) taken in them [N, frames].
  With one frame this is the planner's first step; with more, the predictor also sees the real history before it.
  """
  device, encoded_actions = context.parameters.device, encode_actions(context)
  predictions = [context.model.predict(torch.as_tensor(latents[start:start + BATCH_SIZE], device=device), encoded_actions[torch.as_tensor(np.asarray(actions[start:start + BATCH_SIZE], dtype=np.int64) - 1, device=device)])[:, -1].float().cpu()
                 for start in range(0, len(latents), BATCH_SIZE)]
  return torch.cat(predictions).numpy() if predictions else np.zeros((0, latents.shape[-1]), np.float32)


def train_probe(inputs, targets, outputs: int, steps: int, seed: int, device: str, probe=None, class_weight=None):
  """
  Train a small MLP probe on frozen latents, or keep training an existing one (to fine-tune it on unseen levels).
  Every checkpoint gets the same probe (192 -> 256 -> outputs), the same optimizer and the same number of steps. For classes (outputs > 1), rare classes get a higher loss weight
  """
  torch.manual_seed(seed)
  probe = probe if probe is not None else torch.nn.Sequential(torch.nn.Linear(inputs.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, outputs))
  probe.to(device).train()
  optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
  loss = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(class_weight, dtype=torch.float32, device=device)) if outputs > 1 else torch.nn.MSELoss()
  inputs, targets = torch.as_tensor(inputs, dtype=torch.float32, device=device), torch.as_tensor(targets, dtype=torch.long if outputs > 1 else torch.float32, device=device)  # float32: MPS has no float64
  generator = torch.Generator().manual_seed(seed)
  for _ in range(steps):
    batch = torch.randint(len(inputs), (256,), generator=generator).to(device)
    output = probe(inputs[batch])
    value = loss(output, targets[batch]) if outputs > 1 else loss(output[:, 0], targets[batch])
    optimizer.zero_grad()
    value.backward()
    optimizer.step()
  return probe.eval()


class CemPlanner:
  """
  Plan with the world model, as LewmRolloutPlanner does: tries many action sequences, the model imagines where each one leads, and we play the first replan_every actions
  """

  def __init__(self, model, img_size: int, goal_grid, parameters):
    import gymnasium as gym
    from types import SimpleNamespace

    from stable_worldmodel.planning.solver.categorical_cem import CategoricalCEMSolver

    from master_thesis.training.lewm import ArcGridToPixels

    self.device, self.replan_every = parameters.device, min(parameters.replan_every, parameters.horizon)
    self.to_pixels = ArcGridToPixels(img_size)
    self.goal = self.pixels(goal_grid)
    self.dummy_action = torch.zeros(1, 1, ACTION_DIM, device=parameters.device)  # JEPA.get_cost needs an action entry; the imagined plan replaces it
    self.queue = []  # the actions of the current plan still to play
    pad = lambda candidates: torch.nn.functional.pad(candidates, (0, ACTION_DIM - candidates.shape[-1]))  # 4 moves (one-hot) -> LeWM's 9 numbers

    def cost(info, candidates):
      costs = model.get_cost(info, pad(candidates))  # imagines every candidate plan; fills info["predicted_emb"] and info["goal_emb"]
      if parameters.cost_on == "end":
        return costs
      distance = (info["predicted_emb"][:, :, 1:] - info["goal_emb"][:, None, -1:]).square().sum(dim=-1)  # distance to the goal after move 1, 2, ...
      return distance.min(dim=-1).values

    self.solver = CategoricalCEMSolver(cost=SimpleNamespace(get_cost=cost), num_samples=parameters.samples, n_steps=parameters.iterations, topk=parameters.topk, device=parameters.device, seed=parameters.seed)
    self.solver.configure(action_space=gym.spaces.Discrete(len(ACTIONS)), n_envs=1, config=SimpleNamespace(horizon=parameters.horizon, action_block=1))

  def pixels(self, grid):
    return self.to_pixels(torch.as_tensor(np.asarray(grid).reshape(1, -1), device=self.device))[None]  # with a batch and a time axis of 1

  @torch.no_grad()
  def act(self, grid) -> int:
    """Return the next action (1-4) for the current grid; make a new plan when the old one is used up."""
    if not self.queue:
      plan = self.solver.solve({"pixels": self.pixels(grid), "goal": self.goal, "action": self.dummy_action})
      self.queue = [int(action) + 1 for action in plan["actions"][0, :self.replan_every, 0]]  # the solver counts actions from 0
    return self.queue.pop(0)


#####################################
## Encoder: what does it store? ##
#####################################


def latent_variance(context: AnalysisContext, split: str):
  """
  This doesnt test gaussianity at all! Instead we take the 192 dims and calculate its total variance and then do that for every single level by encoding different states of that level and looking how much latent vectors vary across different states
  If it barely changes but the total variance is high, it suggests we occupy a lot of the space with level identification and not so much by dynamics
  """
  latents, levels = context.latents[split]["state"], context.sets[split].states.level.to_numpy()
  if len(latents) < 2:
    return []
  overall = latents.var(axis=0).mean()
  rows, within = [metric_row("latent_variance", "latent_variance", split, overall, len(latents))], []
  for level in np.unique(levels):
    selected = levels == level
    if selected.sum() >= 2:  # a single state has no variance
      within.append(latents[selected].var(axis=0).mean())
      rows.append(metric_row("latent_variance", "latent_variance", split, within[-1], selected.sum(), level=int(level)))
  rows.append(metric_row("latent_variance", "within_level_share", split, np.mean(within) / overall if within and overall > 0 else np.nan, len(within)))
  return rows


def state_probes(context: AnalysisContext):
  """
  Which facts about the game can be read from the latent, and does that still work in levels the model never saw?
  Implemented as a small MLP per fact that gets the frozen latent of a state and has to predict one fact about it. 
    level                                       which level it is (only on seen levels: there is no answer for a new level)
    player_x, player_y                          where the player is
    energy, lives                               current step and lifes left, ground truth taken from the game engine
    key_shape, key_color, key_rotation          the symbol the player carries
    goal_shape, goal_color, goal_rotation       the symbol the nearest goal needs
    key_matches_goal                            does the player's symbol fit that goal?
    goal_distance                               grid distance to the nearest goal (Manhattan, walls ignored)
    refill_left, refill_x, refill_y             is a refill left, and where is the nearest one?
    rotator_present, rotator_x, rotator_y       is there a rotator, and where is the nearest one?
    success_up, _down, _left, _right            would this move work? (e.g. right fails when a wall is to the right)
  Facts with a few possible answers are scored with balanced accuracy, numbers with the mean absolute error.

  How the splits are used:
    - train on seen_train, test on seen_train and on seen_val. seen_val holds other episodes of the same levels, so a
      big difference means the probe learned the training states by heart instead of reading real information.
    - test the same probes on the unseen levels ("zero_shot"): is the information stored the same way in new levels?
    - fine-tune a copy on the unseen "adapt" half and test it again on the same unseen "eval" states ("adapted"): is
      the information there at all, even if it is stored differently?
  Every result comes with two baselines on the same states: always guessing the most common answer (global_baseline),
  and guessing the most common answer of the state's level (level_baseline). If a probe is not better than
  level_baseline, the latent only tells us the level, not the fact itself. varying_levels says in what share of the
  levels the fact changes at all: a fact that never changes inside a level can be solved by knowing the level.
  A fact with fewer than 10 known training states or always the same answer is skipped (and printed).
  """
  from sklearn.metrics import balanced_accuracy_score

  parameters = context.parameters
  if "seen_train" not in context.sets:
    print("state_probes: skipped (needs the seen_train split)")
    return []
  splits = [name for name in SPLITS if name in context.sets]

  def facts(name):  # fact -> (kind, answer per state, which states have an answer) for one split
    states = context.sets[name].states
    label = lambda key: states[f"label_{key}"].to_numpy()
    success = context.sets[name].outcomes.pivot(index="state", columns="action", values="success").reindex(states.state).to_numpy()  # one column per action
    every, has_goal, has_refill, has_rotator = np.ones(len(states), bool), label("goal_x") >= 0, label("refills_left") > 0, label("rotator_present") > 0
    return {"level": ("class", states.level.to_numpy(), every & (name != "unseen")), "player_x": ("number", label("player_x"), every), "player_y": ("number", label("player_y"), every),
            "energy": ("number", label("energy"), every), "lives": ("class", label("lives"), every),
            **{f"key_{part}": ("class", label(f"key_{part}"), every) for part in ("shape", "color", "rotation")},
            **{f"goal_{part}": ("class", label(f"goal_{part}"), has_goal) for part in ("shape", "color", "rotation")},
            "key_matches_goal": ("class", label("key_matches_goal"), has_goal), "goal_distance": ("number", label("goal_distance"), has_goal),
            "refill_left": ("class", has_refill.astype(int), every), "refill_x": ("number", label("refill_x"), has_refill), "refill_y": ("number", label("refill_y"), has_refill),
            "rotator_present": ("class", has_rotator.astype(int), every), "rotator_x": ("number", label("rotator_x"), has_rotator), "rotator_y": ("number", label("rotator_y"), has_rotator),
            **{f"success_{ACTION_NAMES[action]}": ("class", success[:, position], every) for position, action in enumerate(ACTIONS)}}

  answers = {name: facts(name) for name in splits}
  mean, std = context.latents["seen_train"]["state"].mean(axis=0), context.latents["seen_train"]["state"].std(axis=0) + 1e-6
  inputs = {name: (context.latents[name]["state"] - mean) / std for name in splits}  # every split standardized like the training states
  levels = {name: context.sets[name].states.level.to_numpy() for name in splits}
  parts = {name: context.sets[name].states.part.to_numpy() for name in splits}
  score = lambda kind, truth, guess: balanced_accuracy_score(truth, guess) if kind == "class" else float(np.abs(truth - guess).mean())
  # what knowing only the level gives: per level the most common answer (or the average number) of the known states; a level without known states gets the overall one
  level_guess = lambda kind, known_levels, known, asked_levels: pd.Series(asked_levels).map(pd.Series(known).groupby(known_levels).agg(lambda values: values.mode()[0] if kind == "class" else values.mean())).fillna(
    pd.Series(known).mode()[0] if kind == "class" else known.mean()).to_numpy()

  rows = []
  for fact, (kind, train_answers, train_known) in answers["seen_train"].items():
    known = train_answers[train_known]
    if train_known.sum() < 10 or len(np.unique(known)) < 2:
      print(f"state_probes: {fact} skipped ({int(train_known.sum())} known training states, {len(np.unique(known))} different answers)")
      continue
    if kind == "class":
      classes, counts = np.unique(known, return_counts=True)
      to_target = lambda values: np.searchsorted(classes, values)  # answer -> class index
      from_output = lambda output: classes[output.argmax(axis=1)]
      outputs, class_weight, most_common = len(classes), len(known) / (len(classes) * counts), classes[counts.argmax()]
    else:
      center, scale = known.mean(), known.std() + 1e-6
      to_target = lambda values: (values - center) / scale
      from_output = lambda output: output[:, 0] * scale + center
      outputs, class_weight, most_common = 1, None, center
    name = f"{fact}/{'balanced_accuracy' if kind == 'class' else 'mae'}"
    probe = train_probe(inputs["seen_train"][train_known], to_target(known), outputs, parameters.probe_steps, parameters.seed, parameters.device, class_weight=class_weight)
    guess = lambda model, selected, split: from_output(model(torch.as_tensor(inputs[split][selected], dtype=torch.float32, device=parameters.device)).detach().cpu().numpy())

    for split in splits:
      _, split_answers, split_known = answers[split][fact]
      group = "generalization" if split == "unseen" else "encoder"
      tested = split_known & (parts[split] == "eval") if split == "unseen" else split_known  # on unseen levels the probes are compared on the eval half only
      if not tested.any():
        continue
      truth = split_answers[tested]
      result = {"zero_shot" if split == "unseen" else "probe": score(kind, truth, guess(probe, tested, split)), "global_baseline": score(kind, truth, np.full(len(truth), most_common))}
      if split != "unseen":
        result["level_baseline"] = score(kind, truth, level_guess(kind, levels["seen_train"][train_known], known, levels[split][tested]))
      else:
        adapt = split_known & (parts[split] == "adapt") & (np.isin(split_answers, classes) if kind == "class" else True)  # a class the probe does not know cannot be fine-tuned on
        if adapt.any():
          adapted = train_probe(inputs[split][adapt], to_target(split_answers[adapt]), outputs, parameters.probe_adapt_steps, parameters.seed, parameters.device, probe=copy.deepcopy(probe), class_weight=class_weight)
          result["adapted"] = score(kind, truth, guess(adapted, tested, split))
          result["adapted_minus_zero_shot"] = result["adapted"] - result["zero_shot"]
          result["level_baseline"] = score(kind, truth, level_guess(kind, levels[split][adapt], split_answers[adapt], levels[split][tested]))
      rows += [metric_row("state_probes", name, split, value, len(truth), subset=subset, group=group) for subset, value in result.items()]
      varying = pd.DataFrame({"level": levels[split][tested], "answer": truth}).groupby("level").answer.nunique().gt(1)
      rows.append(metric_row("state_probes", f"{fact}/varying_levels", split, varying.mean(), len(varying), group=group))
  return rows


def action_decoding(context: AnalysisContext):
  """
  Can the move be read from how the latent changes? This is what Delta-JEPA trains its action decoder for
  (val/action_accuracy), measured here the same way for every model, with or without a decoder.

  Data: every picked state and the real outcomes of all 4 moves from the replay, so all moves are equally common.
  Only moves that changed the game count (change kind mechanics): a blocked move only costs energy and looks the same
  whichever direction was pressed, so no model can tell those apart (Delta-JEPA leaves them out as well).
  How: a small MLP, as in state_probes, gets the latent change z(after the move) - z(before) and has to name the move.
  balanced_accuracy: 1 = every move recognized, 0.25 = chance. If the latent change does not depend on the direction,
  the planner cannot tell where its moves lead. The splits are used as in state_probes:
    - probe: trained on seen_train, tested on seen_train and seen_val.
    - zero_shot: the same probe on the eval half of the unseen levels: does the same readout work in new levels?
    - adapted: a copy fine-tuned on the unseen adapt half (probe_adapt_steps), tested on the same eval states. If
      zero_shot is low, this tells why: high adapted = the move is in the latent change, but stored differently in new
      levels; low adapted = it is missing. adapted_minus_zero_shot is the difference.
  """
  from sklearn.metrics import balanced_accuracy_score

  parameters = context.parameters
  if "seen_train" not in context.sets:
    print("action_decoding: skipped (needs the seen_train split)")
    return []
  splits = [name for name in SPLITS if name in context.sets]

  def moves(name, part=None):  # latent change, move (0-3) and level of every move that changed the game (of one half of the unseen levels if part is given)
    evaluation_set, latents = context.sets[name], context.latents[name]
    keep = evaluation_set.outcomes.sort_values(["state", "action"]).change.to_numpy().reshape(-1, len(ACTIONS)) == "mechanics"  # per state and move
    if part is not None:
      keep &= (evaluation_set.states.part.to_numpy() == part)[:, None]
    change = latents["outcome"] - latents["state"][:, None]  # [states, 4, 192]
    return change[keep], np.nonzero(keep)[1], np.repeat(evaluation_set.states.level.to_numpy()[:, None], len(ACTIONS), axis=1)[keep]

  change, move, _ = moves("seen_train")
  counts = np.bincount(move, minlength=len(ACTIONS))
  if counts.min() < 10:
    print(f"action_decoding: skipped (moves per direction in seen_train: {counts.tolist()})")
    return []
  mean, std = change.mean(axis=0), change.std(axis=0) + 1e-6  # every split standardized like the training moves
  class_weight = len(move) / (len(ACTIONS) * counts)
  probe = train_probe((change - mean) / std, move, len(ACTIONS), parameters.probe_steps, parameters.seed, parameters.device, class_weight=class_weight)
  guess = lambda model, change: model(torch.as_tensor((change - mean) / std, dtype=torch.float32, device=parameters.device)).argmax(dim=1).detach().cpu().numpy()

  def results(subset, split, truth, guessed, levels):  # balanced accuracy over all moves and per level
    group = "generalization" if split == "unseen" else "encoder"
    rows = [metric_row("action_decoding", "balanced_accuracy", split, balanced_accuracy_score(truth, guessed), len(truth), subset=subset, group=group)]
    for level in np.unique(levels):
      selected = levels == level
      rows.append(metric_row("action_decoding", "balanced_accuracy", split, balanced_accuracy_score(truth[selected], guessed[selected]), selected.sum(), level=int(level), subset=subset, group=group))
    return rows

  rows = []
  for split in splits:
    change, truth, levels = moves(split, "eval" if split == "unseen" else None)
    if not len(truth):
      continue
    probe_guess = guess(probe, change)
    rows += results("zero_shot" if split == "unseen" else "probe", split, truth, probe_guess, levels)
    if split != "unseen":
      continue
    adapt_change, adapt_move, _ = moves(split, "adapt")
    if not len(adapt_move):
      continue
    adapted = guess(train_probe((adapt_change - mean) / std, adapt_move, len(ACTIONS), parameters.probe_adapt_steps, parameters.seed, parameters.device, probe=copy.deepcopy(probe), class_weight=class_weight), change)
    rows += results("adapted", split, truth, adapted, levels)
    rows.append(metric_row("action_decoding", "balanced_accuracy", split, balanced_accuracy_score(truth, adapted) - balanced_accuracy_score(truth, probe_guess), len(truth), subset="adapted_minus_zero_shot", group="generalization"))
  return rows


def state_aliasing(context: AnalysisContext, split: str):
  """
  Do states that are close together in latent space really behave the same in the game? Are they too close together?

  If two different situations get almost the same latent, the planner cannot tell them apart. That is a problem when the
  same action does different things in them, e.g. "right" moves the player in one state but hits a wall in the other.
  This is called aliasing.

  Data: the split's picked states, and what each of the 4 actions really does in each of them (from the replay).
  How: for every state we look for its 10 (knn) nearest states in latent space: once among the states of the same
  level, once among the states of other levels. A neighbour with exactly the same grid is the same situation (e.g.
  visited by two episodes), not an alias, so it is skipped. For each pair and each action we check whether the action
  has a different effect (a different move, symbol change, refill, life lost or level completed). As a baseline we do
  the same with 10 random states of the same group: if the nearest neighbours do not agree more often than random
  states, the distance between latents says little about how states behave. Within a level, the neighbours can only
  come from that level's picked states.
  Result: effect_mismatch_up / _down / _left / _right and effect_mismatch (the average over the 4 actions), for the
  groups same_level, other_levels, same_level_random and other_levels_random.
  """
  evaluation_set, parameters = context.sets[split], context.parameters
  count, k = len(evaluation_set.states), parameters.knn
  if count < 2:
    return []
  effect_columns = ["dx", "dy", "d_shape", "d_color", "d_rotation", "refill_taken", "life_lost", "completed"]
  effects = evaluation_set.outcomes.sort_values(["state", "action"])[effect_columns].to_numpy().reshape(count, len(ACTIONS), len(effect_columns))  # per state and action: what it did
  levels = evaluation_set.states.level.to_numpy()
  grid_ids = np.unique(evaluation_set.arrays["grid"], axis=0, return_inverse=True)[1].reshape(-1)  # equal grids get the same id
  latents = torch.as_tensor(context.latents[split]["state"], device=parameters.device)
  level_ids, grid_ids_device = torch.as_tensor(levels, device=parameters.device), torch.as_tensor(grid_ids, device=parameters.device)
  rng = np.random.default_rng([parameters.seed, SPLITS.index(split), 2])

  neighbours = {}  # group -> the neighbours of every state; -1 where a group has fewer than k states
  for group in ("same_level", "other_levels"):
    indices = []
    for start in range(0, count, 1024):  # 1024 states at a time: all distances at once can take gigabytes
      distance = torch.cdist(latents[start:start + 1024], latents)
      same_level = level_ids[start:start + 1024, None] == level_ids[None, :]
      excluded = (grid_ids_device[start:start + 1024, None] == grid_ids_device[None, :]) | (~same_level if group == "same_level" else same_level)  # the state itself, its copies, and the other group
      values, index = distance.masked_fill(excluded, float("inf")).topk(min(k, count), dim=1, largest=False)
      indices.append(index.masked_fill(torch.isinf(values), -1).cpu())
    neighbours[group] = torch.cat(indices).numpy()
    random_index = np.full((count, min(k, count)), -1)
    for state in range(count):  # the baseline: random states of the same group
      members = np.flatnonzero(((levels == levels[state]) if group == "same_level" else (levels != levels[state])) & (grid_ids != grid_ids[state]))
      chosen = rng.choice(members, min(k, len(members)), replace=False) if len(members) else []
      random_index[state, :len(chosen)] = chosen
    neighbours[f"{group}_random"] = random_index

  rows = []
  for group, index in neighbours.items():
    query, neighbour = np.nonzero(index >= 0)[0], index[index >= 0]  # one entry per (state, neighbour) pair
    mismatch = (effects[query] != effects[neighbour]).any(axis=-1)  # per pair and action: does the action do something different?
    for position, action in enumerate(ACTIONS):
      rows += metric_rows_per_level("state_aliasing", f"effect_mismatch_{ACTION_NAMES[action]}", split, mismatch[:, position], levels[query], group)
    rows += metric_rows_per_level("state_aliasing", "effect_mismatch", split, mismatch.mean(axis=1), levels[query], group)
  return rows


def latent_maps(context: AnalysisContext):
  """
  Maps of the latent space for the dashboard (PCA and UMAP). They are only for looking at; they do not measure anything.

  Data: the picked states of all splits together, and again only the states of each level in projection_levels.
  How: PCA finds the 3 directions in which the latents vary most, and how much of the variance each of the first 10
  directions explains. UMAP draws a 2-d and a 3-d map that keeps nearby states nearby; it distorts distances and the
  size of clusters, so read it as a picture only. It starts from the PCA layout (its default start fails when many
  states are identical) and is seeded, so the map comes out the same every time.
  Result: the metric explained_variance per PCA direction (subset pc1, pc2, ...), and the maps for latent_maps.json:
  per map ("all", or "level_<n>") one row per state with its coordinates and what the dashboard needs to colour and
  identify it: split, level, player position, goal distance, whether the symbol fits the goal, and where the state
  comes from.
  """
  import umap
  from sklearn.decomposition import PCA

  columns = ["state", "dataset", "episode", "episode_idx", "step", "level", "label_player_x", "label_player_y", "label_goal_distance", "label_key_matches_goal"]
  table = pd.concat([context.sets[name].states[columns].assign(split=name) for name in context.sets], ignore_index=True)
  latents = np.concatenate([context.latents[name]["state"] for name in context.sets])
  rows, maps = [], {}
  for level in [None, *context.parameters.projection_levels]:  # None = all states
    selected = np.ones(len(table), bool) if level is None else (table.level == level).to_numpy()
    if selected.sum() < 20:  # UMAP needs more states than its 15 neighbours
      print(f"latent_maps: level {level} has {selected.sum()} states, skipped")
      continue
    pca = PCA(n_components=min(10, *latents[selected].shape)).fit(latents[selected])
    coordinates = table[selected].reset_index(drop=True)
    coordinates[["pca_1", "pca_2", "pca_3"]] = pca.transform(latents[selected])[:, :3]
    for dimensions in (2, 3):
      reducer = umap.UMAP(n_components=dimensions, n_neighbors=15, min_dist=0.1, init="pca", random_state=context.parameters.seed)
      coordinates[[f"umap{dimensions}_{axis + 1}" for axis in range(dimensions)]] = reducer.fit_transform(latents[selected])
    rows += [metric_row("latent_maps", "explained_variance", "all", share, selected.sum(), level="all" if level is None else level, subset=f"pc{component + 1}") for component, share in enumerate(pca.explained_variance_ratio_)]
    maps["all" if level is None else f"level_{level}"] = coordinates
  return rows, maps


def checkpoint_similarity(context: AnalysisContext, split: str):
  """
  How similar are the representations of two checkpoints? (only with --compare-to)

  Both checkpoints encode exactly the same states, and linear CKA compares the two sets of latents. 1 means the same
  representation (up to rotating and scaling it), 0 means unrelated. This shows how much, e.g., a new loss changed the
  representation; it says nothing about which one is better.
  Result: cka over all states and per level (levels with at least 3 states).
  """
  x, y = (latents[split]["state"].astype(np.float64) for latents in (context.latents, context.compare_to[2]))
  levels = context.sets[split].states.level.to_numpy()

  def linear_cka(a, b):
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    scale = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    return np.linalg.norm(b.T @ a) ** 2 / scale if scale > 0 else np.nan

  if len(x) < 3:
    return []
  rows = [metric_row("checkpoint_similarity", "cka", split, linear_cka(x, y), len(x))]
  rows += [metric_row("checkpoint_similarity", "cka", split, linear_cka(x[levels == level], y[levels == level]), (levels == level).sum(), level=int(level)) for level in np.unique(levels) if (levels == level).sum() >= 3]
  return rows


################################################
## World model: encoder and predictor together ##
################################################


def one_step_prediction(context: AnalysisContext, split: str):
  """
  How well do encoder and predictor together predict the next state?

  Data: the split's picked states that have a recorded next state, with the action that was taken there.
  How: encode the state, let the predictor predict the next latent from this one frame and the action (as the
  planner's first step does: only one real frame, no history), and compare it with the encoded real next state.
  counterfactual_actions predicts the same with up to 3 real frames, so for the recorded action the two together show
  what real history adds. prediction_error is the squared difference, averaged over
  the 192 latent numbers (prediction_error). As a baseline we pretend nothing changes and compare the current latent
  with the real next one (copy_error); this also shows how much one step changes the latent at all.
  error_ratio = prediction_error / copy_error can be compared between models, whose latents have different scales:
  below 1 the model is better than "nothing changes".
  We report this for all steps, and separately for each kind of change: mechanics (can it predict the real dynamics?),
  energy_only (does it see that a blocked move only costs energy?) and none (does it see that nothing happens?).
  Without this split, a model that always predicts "nothing happens" could look good just because many moves are blocked.
  """
  states, latents = context.sets[split].states, context.latents[split]
  selected = states.has_next.to_numpy() if len(states) else np.zeros(0, bool)
  if not selected.any():
    return []
  current, following = latents["state"][selected], latents["next"][selected]
  predicted = predict_next_latent(context, current[:, None], states.action.to_numpy()[selected][:, None])
  prediction_error = ((predicted - following) ** 2).mean(axis=1)  # per step: squared difference, averaged over the 192 numbers
  copy_error = ((current - following) ** 2).mean(axis=1)
  levels, change = states.level.to_numpy()[selected], states.next_change.to_numpy()[selected]
  rows = []
  for subset in ("all", *CHANGE_KINDS):
    mask = np.ones(len(change), bool) if subset == "all" else change == subset
    rows += metric_rows_per_level("one_step_prediction", "prediction_error", split, prediction_error[mask], levels[mask], subset)
    rows += metric_rows_per_level("one_step_prediction", "copy_error", split, copy_error[mask], levels[mask], subset)
    rows += metric_rows_per_level("one_step_prediction", "error_ratio", split, prediction_error[mask], levels[mask], subset, denominator=copy_error[mask])
  return rows


###########################################
## Predictor: does it know the dynamics? ##
###########################################


def counterfactual_actions(context: AnalysisContext, split: str):
  """
  Does the predictor know what every action would do, not just the one that was recorded?

  The recordings only show one action per state, so they cannot test this. During the replay all 4 actions were played
  from the same real state, so here we know the real result of each one.

  Data: the split's picked states, the recorded states and actions before them, and the real outcomes of all 4 actions.
  History: the predictor gets the state and the up to 2 recorded states before it (the history parameter; fewer at an
  episode start, never more than the model's history_size), with the actions really taken there; only the last action
  is varied. This is the most real context the predictor can use, as in the one-step training loss, so it shows what the
  predictor knows. (one_step_prediction uses one frame; for the recorded action the difference is what history adds.)
  How, for every state:
    - each action: predict its outcome and compare it with the encoded real outcome
      (prediction_error), and with "nothing changes" (giving error_ratio, as in one_step_prediction). Reported per
      action (up, down, left, right) and per kind of change the action really caused (mechanics, energy_only, none).
    - each pair of actions (6 per state): the predicted difference between their two outcomes should match the real
      difference (effect_error). The pairs are sorted by what the real game did:
        identical: both actions lead to the same grid (e.g. two moves into walls). The model should predict the same
                   outcome for both; effect_error then says how far apart it predicts them.
        different: e.g. left and up move the player to different cells. The model should predict different outcomes
                   that differ in the right way. effect_error_ratio compares effect_error with the real difference
                   (1 = no better than predicting no difference at all); predicted_difference and real_difference are
                   the sizes of the two differences.
  """
  evaluation_set, latents = context.sets[split], context.latents[split]
  count = len(evaluation_set.states)
  if not count:
    return []
  current, outcomes = latents["state"], latents["outcome"]  # outcomes: per state the latents after each of the 4 actions
  sequence = np.repeat(np.concatenate([latents["history"], current[:, None]], axis=1), len(ACTIONS), axis=0)  # per (state, action) its real latents, oldest first  [N * 4, history, 192]
  taken = np.concatenate([np.repeat(evaluation_set.arrays["history_action"], len(ACTIONS), axis=0), np.tile(ACTIONS, count)[:, None]], axis=1)  # the recorded actions, then the tried one  [N * 4, history]
  frames = np.repeat(np.minimum(evaluation_set.states.history_frames.to_numpy(), context.info["history_size"]), len(ACTIONS))  # real frames per (state, action)
  predicted = np.zeros((len(sequence), current.shape[1]), np.float32)
  for length in np.unique(frames):  # predicted together in groups with the same number of real frames
    rows = frames == length
    predicted[rows] = predict_next_latent(context, sequence[rows][:, -length:], taken[rows][:, -length:])
  predicted = predicted.reshape(outcomes.shape)
  prediction_error = ((predicted - outcomes) ** 2).mean(axis=-1)  # per state and action
  copy_error = ((current[:, None] - outcomes) ** 2).mean(axis=-1)
  levels = evaluation_set.states.level.to_numpy()

  rows = []
  for position, action in enumerate(ACTIONS):
    rows += metric_rows_per_level("counterfactual_actions", "prediction_error", split, prediction_error[:, position], levels, ACTION_NAMES[action])
    rows += metric_rows_per_level("counterfactual_actions", "error_ratio", split, prediction_error[:, position], levels, ACTION_NAMES[action], denominator=copy_error[:, position])
  action_levels = np.repeat(levels[:, None], len(ACTIONS), axis=1)  # the level of every (state, action)
  rows += metric_rows_per_level("counterfactual_actions", "prediction_error", split, prediction_error.reshape(-1), action_levels.reshape(-1))
  rows += metric_rows_per_level("counterfactual_actions", "error_ratio", split, prediction_error.reshape(-1), action_levels.reshape(-1), denominator=copy_error.reshape(-1))
  change = evaluation_set.outcomes.sort_values(["state", "action"]).change.to_numpy().reshape(count, len(ACTIONS))  # what each action really did
  for kind in CHANGE_KINDS:
    rows += metric_rows_per_level("counterfactual_actions", "prediction_error_by_change", split, prediction_error[change == kind], action_levels[change == kind], kind)
    rows += metric_rows_per_level("counterfactual_actions", "error_ratio_by_change", split, prediction_error[change == kind], action_levels[change == kind], kind, denominator=copy_error[change == kind])

  pairs = [(a, b) for a in range(len(ACTIONS)) for b in range(a + 1, len(ACTIONS))]  # the 6 pairs of actions (positions in ACTIONS)
  first, second = np.array(pairs).T
  identical = (evaluation_set.arrays["outcome_grid"][:, first] == evaluation_set.arrays["outcome_grid"][:, second]).all(axis=-1)  # per state and pair: the real game shows the same grid
  predicted_difference, real_difference = predicted[:, first] - predicted[:, second], outcomes[:, first] - outcomes[:, second]
  effect_error = ((predicted_difference - real_difference) ** 2).mean(axis=-1)
  predicted_size, real_size = (predicted_difference ** 2).mean(axis=-1), (real_difference ** 2).mean(axis=-1)
  pair_levels = np.repeat(levels[:, None], len(pairs), axis=1)
  rows += metric_rows_per_level("counterfactual_actions", "effect_error", split, effect_error[identical], pair_levels[identical], "identical")
  rows += metric_rows_per_level("counterfactual_actions", "effect_error", split, effect_error[~identical], pair_levels[~identical], "different")
  rows += metric_rows_per_level("counterfactual_actions", "effect_error_ratio", split, effect_error[~identical], pair_levels[~identical], "different", denominator=real_size[~identical])
  rows += metric_rows_per_level("counterfactual_actions", "predicted_difference", split, predicted_size[~identical], pair_levels[~identical], "different")
  rows += metric_rows_per_level("counterfactual_actions", "real_difference", split, real_size[~identical], pair_levels[~identical], "different")
  return rows


def multi_step_rollout(context: AnalysisContext, split: str):
  """
  Does the error grow when the model keeps predicting from its own predictions, and does real history help?

  When the planner thinks ahead, it predicts the next state, then predicts from that prediction, and so on. Small
  mistakes can add up over the steps. The rollout loss is meant to prevent that, so this shows whether it works.

  Data: the split's rollout windows: 13 recorded states in a row and the 12 actions between them. The rollout starts at
  the 3rd state; the 2 before it are its real history.
  How: give the model the start state and the actions. It predicts one step at a time and always uses its own
  predictions as the history for the next step (it looks at its last history_size latents, like the planner): its
  history is gathered by itself. This is done 3 times on the same windows: from the start state alone (subsets k1 ..
  k10, the planner's situation: it only gets the current frame) and with the 1 or 2 real states before it as well
  (k1_real2 .. k10_real3, like val/rollout_k*_real2/3 in training). The difference is what real history adds.
  After 1, 3, 5 and 10 steps we compare the prediction with the real state at that point (prediction_error). The
  baseline is "the start state never changed" (copy_error), and error_ratio = prediction_error / copy_error. All step
  counts and start lengths use the same windows and targets, so they can be compared directly: a ratio that grows with
  the steps means the errors pile up.
  """
  evaluation_set, frames = context.sets[split], context.latents[split]["window"]  # per window the latents of its 13 states
  if not len(frames):
    return []
  device, encoded_actions = context.parameters.device, encode_actions(context)
  actions = evaluation_set.arrays["window_action"].astype(int)  # per window the 12 recorded actions
  start, horizon = context.parameters.history - 1, ROLLOUT_HORIZONS[-1]  # the rollout starts at frame `start`, the frames before it are its real history
  levels, rows = evaluation_set.windows.level.to_numpy(), []
  for real in range(1, min(context.parameters.history, context.info["history_size"]) + 1):  # real frames the rollout starts from
    predicted = []
    with torch.no_grad():
      for first in range(0, len(frames), BATCH_SIZE):
        real_frames = torch.as_tensor(frames[first:first + BATCH_SIZE, start - real + 1:start + 1], device=device)  # the start state and the real ones before it
        taken = encoded_actions[torch.as_tensor(actions[first:first + BATCH_SIZE, start - real + 1:start + horizon] - 1, device=device)]  # the actions taken in them, then the next ones
        predicted.append(context.model.predict_rollout(real_frames, taken, context.info["history_size"]).float().cpu())
    predicted = torch.cat(predicted).numpy()  # per window the predictions for steps 1 .. 10
    for k in ROLLOUT_HORIZONS:
      subset = f"k{k}" + (f"_real{real}" if real > 1 else "")
      prediction_error = ((predicted[:, k - 1] - frames[:, start + k]) ** 2).mean(axis=1)
      copy_error = ((frames[:, start] - frames[:, start + k]) ** 2).mean(axis=1)
      rows += metric_rows_per_level("multi_step_rollout", "prediction_error", split, prediction_error, levels, subset)
      if real == 1:  # the same for every start length
        rows += metric_rows_per_level("multi_step_rollout", "copy_error", split, copy_error, levels, subset)
      rows += metric_rows_per_level("multi_step_rollout", "error_ratio", split, prediction_error, levels, subset, denominator=copy_error)
  return rows


#####################################
## Planning: can it steer the game? ##
#####################################


def cem_planning(context: AnalysisContext, split: str):
  """
  Can the model actually steer the game to a goal?

  Data: recorded solutions of the split's levels, i.e. episodes that completed their level and start at the level
  start (so planning can start at the same place). For planning_levels random levels we take one such episode each;
  a recording that starts in the middle of a level (e.g. a walk run) cannot be used, at most 10 are tried per level.
  The episodes are read from the dataset directly; the training's test episodes are never among them.
  How: CEM (CemPlanner) plans with the model, and its actions are played in the real game.
    goal reaching: the recorded state 5, 10 or 25 (goal_steps) steps after the start is the goal, and the planner
                   gets 3 times that many actions. Reached means the game shows the same situation as the goal; only
                   the energy bar may differ, because a path of another length to the same place also counts.
    waypoints:     every 5th (waypoint_every) recorded state, and the final state, become waypoints the planner has
                   to reach one after the other, with 3 x 5 actions for each. It stops at the first waypoint it misses.
  A game over ends an attempt.
  Result: success_rate and steps per goal distance, waypoints_reached_fraction and level_completion_rate, also per level.
  """
  from arcengine import GameAction, GameState

  from master_thesis.environments.arc_collection import reset_at_level
  from master_thesis.environments.arc_levelgen import make_game

  parameters = context.parameters
  episodes, versions = context.episodes[split]
  successful = episodes[episodes.success]
  levels = np.random.default_rng([parameters.seed, SPLITS.index(split), 3]).permutation(sorted(set(successful.level)))[:parameters.planning_levels]
  games, attempts = {}, []
  for level in sorted(int(level) for level in levels):
    candidates = successful[successful.level == level]
    episode, env = None, None
    for row in candidates.iloc[np.random.default_rng([parameters.seed, SPLITS.index(split), level, 3]).permutation(len(candidates))[:10]].itertuples(index=False):
      solution, _, game_ids = read_recorded_rows(row.dataset, versions[row.dataset], np.arange(row.offset, row.offset + row.length))  # its last grid completes the level
      if game_ids[0] not in games:
        games[game_ids[0]] = make_game(game_ids[0])
      env = games[game_ids[0]]
      if np.array_equal(solution[0], np.asarray(reset_at_level(env, level).frame[-1], dtype=np.uint8).reshape(-1)):
        episode = row
        break
    if episode is None:
      print(f"cem_planning {split}: level {level} skipped (no successful episode starts at the level start)")
      continue

    def play_towards(targets, budget):  # from the level start, plan towards each target in turn, with at most `budget` actions per target
      observation = reset_at_level(env, level)
      grid = np.asarray(observation.frame[-1], dtype=np.uint8).reshape(-1)
      played, reached = 0, 0
      for target in targets:
        planner, steps = CemPlanner(context.model, context.info["img_size"], target, parameters), 0
        while classify_change(grid, target) == "mechanics" and steps < budget and observation.state not in (GameState.GAME_OVER, GameState.WIN):
          observation = env.step(GameAction.from_id(planner.act(grid)))
          grid = np.asarray(observation.frame[-1], dtype=np.uint8).reshape(-1)
          steps += 1
        played += steps
        if classify_change(grid, target) == "mechanics":  # missed this target: stop
          break
        reached += 1
      return {"reached": reached, "steps": played, "completed": bool(observation.levels_completed >= 1 or observation.state == GameState.WIN)}

    for goal_steps in (g for g in parameters.goal_steps if g < episode.length):
      attempts.append({"level": level, "kind": "goal", "goal_steps": goal_steps, "targets": 1, **play_towards([solution[goal_steps]], 3 * goal_steps)})
    waypoints = list(solution[parameters.waypoint_every::parameters.waypoint_every])
    if not waypoints or not np.array_equal(waypoints[-1], solution[-1]):
      waypoints.append(solution[-1])  # the final state (level completed) is always the last waypoint
    attempts.append({"level": level, "kind": "waypoints", "goal_steps": parameters.waypoint_every, "targets": len(waypoints), **play_towards(waypoints, 3 * parameters.waypoint_every)})

  if not attempts:
    return []
  attempts = pd.DataFrame(attempts)
  goals, followed = attempts[attempts.kind == "goal"], attempts[attempts.kind == "waypoints"]
  rows = metric_rows_per_level("cem_planning", "success_rate", split, goals.reached, goals.level) + metric_rows_per_level("cem_planning", "steps", split, goals.steps, goals.level)
  for goal_steps, attempt in goals.groupby("goal_steps"):
    rows += metric_rows_per_level("cem_planning", "success_rate", split, attempt.reached, attempt.level, f"goal{goal_steps}")
    rows += metric_rows_per_level("cem_planning", "steps", split, attempt.steps, attempt.level, f"goal{goal_steps}")
  rows += metric_rows_per_level("cem_planning", "waypoints_reached_fraction", split, followed.reached / followed.targets, followed.level, "waypoints")
  rows += metric_rows_per_level("cem_planning", "level_completion_rate", split, followed.completed, followed.level, "waypoints")
  return rows


def action_ranking(context: AnalysisContext, split: str):
  """
  Does the predictor rank the 4 actions by "which brings me closer to the goal" like the real game does?

  This is what the planner relies on, but it is much cheaper to measure than planning itself. So it predicts like the
  planner: from one real frame, without history (the planner only gets the current frame).
  Data: the split's picked states that have a recorded state 5 (goal_ahead) steps later in their episode; that later
  state is the goal.
  How: for every action, the distance to the goal's latent, once from the predicted outcome and once from the real
  outcome (from the replay). spearman says how well the two orders agree (1 = same order, 0 = unrelated, -1 = reversed);
  states where all real outcomes are equally far away have no order and are left out. action_spread says how
  different the 4 actions look at all (from the real outcomes): near 0 means every move looks the same to the planner,
  so it has no direction.
  """
  evaluation_set, latents = context.sets[split], context.latents[split]
  selected = evaluation_set.states.has_goal.to_numpy() if len(evaluation_set.states) else np.zeros(0, bool)
  if not selected.any():
    return []
  count = int(selected.sum())
  goals = latents["goal"][selected][:, None]
  predicted = predict_next_latent(context, np.repeat(latents["state"][selected], len(ACTIONS), axis=0)[:, None], np.tile(ACTIONS, count)[:, None]).reshape(count, len(ACTIONS), -1)
  predicted_distance, real_distance = ((predicted - goals) ** 2).sum(axis=-1), ((latents["outcome"][selected] - goals) ** 2).sum(axis=-1)  # per state and action: squared distance to the goal

  rank = lambda distance: (distance[:, None, :] < distance[:, :, None]).sum(axis=-1) + ((distance[:, None, :] == distance[:, :, None]).sum(axis=-1) + 1) / 2  # rank 1-4 per state; equal distances share their rank (two blocked moves: 1, 2.5, 2.5, 4)
  predicted_rank, real_rank = (r - r.mean(axis=1, keepdims=True) for r in (rank(predicted_distance), rank(real_distance)))
  ordered = (real_rank ** 2).sum(axis=1) > 0  # states whose real distances are not all equal
  spearman = np.where(ordered, (predicted_rank * real_rank).sum(axis=1) / np.sqrt(np.maximum((predicted_rank ** 2).sum(axis=1) * (real_rank ** 2).sum(axis=1), 1e-12)), np.nan)  # all predictions equal -> 0
  spread = real_distance.std(axis=1)  # how much the real distance to the goal depends on the action

  levels = evaluation_set.states.level.to_numpy()[selected]
  return metric_rows_per_level("action_ranking", "spearman", split, spearman, levels) + metric_rows_per_level("action_ranking", "action_spread", split, spread, levels)


#######################################################
## Generalization: what changes between known and new? ##
#######################################################


def split_gaps(metrics: list):
  """
  What changes between known and new data? The differences between the splits, for every metric of this checkpoint.

  seen_val - seen_train shows memorization: does the model do worse on episodes it was not trained on?
  unseen - seen_val shows how well it transfers to levels it never saw.
  For the probes, the zero-shot result on unseen levels is compared with the probe's result on the seen splits.
  We only compare splits of the same checkpoint (the same encoder, so even raw latent errors can be compared);
  comparing different checkpoints is left to the dashboard. Works on the metric rows of all other experiments.
  """
  if not metrics:
    return []
  table = pd.DataFrame(metrics)
  table = table[(table.level == "all") & table.split.isin(SPLITS)]
  table.loc[table.experiment.isin(["state_probes", "action_decoding"]) & (table.subset == "zero_shot"), "subset"] = "probe"  # on unseen levels, the trained probe's result is its zero-shot result
  pivot = table.pivot_table(index=["experiment", "metric", "subset"], columns="split", values=["value", "n"], aggfunc="first")
  rows = []
  for (experiment, name, subset), entry in pivot.iterrows():
    for later, earlier, label in (("seen_val", "seen_train", "val_minus_train"), ("unseen", "seen_val", "unseen_minus_val")):
      if not (pd.isna(entry.get(("value", later))) or pd.isna(entry.get(("value", earlier)))):
        rows.append(metric_row("split_gaps", f"{experiment}/{name}/{subset}", "all", entry[("value", later)] - entry[("value", earlier)], min(entry[("n", later)], entry[("n", earlier)]), subset=label))
  return rows


##########
## Main ##
##########


def main() -> None:
  parser = argparse.ArgumentParser(description="Measure what a trained LeWM learned about ls20: representation, dynamics, planning, generalization")
  parser.add_argument("--checkpoint", required=True, help="the weights file to evaluate, below models/lewm/ (e.g. <run>/weights.pt, <run>/checkpoints/weights_step100000.pt) or absolute")
  parser.add_argument("--compare-to", default=None, help="a second weights file; only used by checkpoint_similarity")
  parser.add_argument("--unseen", nargs="+", default=[], help="unseen levels: DATASET or DATASET:LEVEL,LEVEL, e.g. syn1000_unseen, or human_l1-7:3 for real level 3")
  parser.add_argument("--experiment-parameters", choices=list(EXPERIMENT_PARAMETERS), default="full", help="which settings of EXPERIMENT_PARAMETERS to use (smoke: tiny, only to check that everything runs)")
  parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
  parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)), help="processes that replay episodes in the game")
  cli = parser.parse_args()
  device = cli.device if cli.device != "auto" else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  parameters = argparse.Namespace(**EXPERIMENT_PARAMETERS[cli.experiment_parameters], device=device, workers=cli.workers, output=evaluation_dir(GAME) / "lewm_analysis")
  splits = ["seen_train", "seen_val", *(["unseen"] if cli.unseen else [])]

  model, info = load_checkpoint(cli.checkpoint, device)
  print(f"checkpoint {info['id']} on {device}: trained on {info['dataset']} (window {info['window']}, seed {info['seed']})")

  # the seen levels we evaluate, the same for seen_train and seen_val so that their difference compares the same levels:
  # every real level (levels 1 .. NUM_REAL_LEVELS in every game version) that has training episodes, plus a random choice of the others up to seen_levels
  from master_thesis.environments.arc_levelgen import NUM_REAL_LEVELS

  seen, version, training_levels = rebuild_training_split(info)
  seen_levels = sorted(set(seen.level[seen.part == "train"]))
  if parameters.seen_levels and len(seen_levels) > parameters.seen_levels:
    real = [level for level in seen_levels if level <= NUM_REAL_LEVELS]
    others = [level for level in seen_levels if level > NUM_REAL_LEVELS]
    seen_levels = real + sorted(np.random.default_rng([parameters.seed, 4]).choice(others, max(parameters.seen_levels - len(real), 0), replace=False).tolist())
  seen = seen[seen.level.isin(seen_levels)]

  # the evaluation sets: the picked states and windows of every split, replayed in the game
  episodes, sets = {}, {}
  for split in splits:
    episodes[split] = unseen_level_episodes(cli.unseen, training_levels, parameters.seed) if split == "unseen" else (seen[seen.part == split.removeprefix("seen_")], {info["dataset"]: version})
    sets[split] = build_evaluation_set(split, *episodes[split], parameters)
    print(f"{split}: {sets[split].counts}")

  latents = {split: encode_evaluation_set(model, info, evaluation_set, parameters) for split, evaluation_set in sets.items()}
  compare_to = None
  if cli.compare_to:
    other_model, other_info = load_checkpoint(cli.compare_to, device)
    compare_to = (other_model, other_info, {split: encode_evaluation_set(other_model, other_info, evaluation_set, parameters) for split, evaluation_set in sets.items()})
  context = AnalysisContext(parameters=parameters, model=model, info=info, episodes=episodes, sets=sets, latents=latents, compare_to=compare_to)

  # every experiment, every time
  metrics = []
  for split in splits:
    for experiment in (latent_variance, state_aliasing, one_step_prediction, counterfactual_actions, multi_step_rollout, action_ranking, cem_planning):
      metrics += experiment(context, split)
      print(f"{experiment.__name__} / {split}: done")
    if compare_to is not None:
      metrics += checkpoint_similarity(context, split)
  metrics += state_probes(context)  # trained once on seen_train, tested on every split
  metrics += action_decoding(context)  # the same
  map_metrics, maps = latent_maps(context)
  metrics += map_metrics
  metrics += split_gaps(metrics)  # the differences between the splits of all metrics above

  folder = parameters.output / info["id"]
  folder.mkdir(parents=True, exist_ok=True)
  results = {"checkpoint": info, "compare_to": compare_to[1] if compare_to else None, "evaluated": datetime.now().isoformat(timespec="seconds"), "unseen": cli.unseen,
             "experiment_parameters": {"name": cli.experiment_parameters, **{key: value for key, value in vars(parameters).items() if key != "output"}},
             "training_levels": sorted(training_levels), "seen_levels_evaluated": seen_levels, "dataset_version": version,
             "evaluation_sets": {split: {**evaluation_set.counts, "dropped": evaluation_set.dropped} for split, evaluation_set in sets.items()},
             "metrics": metrics}
  (folder / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
  (folder / "latent_maps.json").write_text(json.dumps({name: json.loads(table.to_json(orient="records")) for name, table in maps.items()}), encoding="utf-8")
  print(f"{len(metrics)} metrics -> {folder / 'results.json'}, maps -> {folder / 'latent_maps.json'}")


if __name__ == "__main__":
  main()
