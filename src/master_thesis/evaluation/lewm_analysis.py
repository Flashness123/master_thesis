"""
LeWM analysis: evaluate trained LeWorldModel checkpoints on ls20 and keep every detail for a later dashboard.

Stages (cached separately, so a new or changed metric never redoes the expensive parts)
  1. evaluation sets  CPU, independent of the weights. Per split: which episodes, states and rollout windows are
                      evaluated. Every sampled episode is replayed in the real engine from its start (level
                      start, plus the start path of synthetic walk runs from the dataset's provenance file) and checked
                      grid by grid against the recording. At each sampled state the labels are read from the engine
                      (read_state), and each of the 4 actions is played on its own copy of the game (counterfactual
                      outcomes). Parallel over levels (--workers), every worker with its own game.
  2. embeddings       once per checkpoint and set: every distinct grid of the set encoded once (batched, one process).
  3. experiments      the metrics, from 1 + 2. Each is saved on its own and skipped when it exists with the same inputs
                      (--force recomputes). planning plays the real game with CEM and is the only slow one.

Splits
  seen_train  the episodes the checkpoint was trained on     } rebuilt exactly as training/lewm.py split them, on the
  seen_val    held-out episodes of the same levels           } dataset as it was when the run started
  unseen      levels missing from the training data (--unseen), split by episode into adapt (only to fine-tune the
              probes) and eval (where zero-shot and adapted probes are compared); every other metric uses both parts
  The training's test split is never read.

Samples (they depend only on the split and --seed: checkpoints with the same split are evaluated on the same states)
  episodes  per level at most --episodes-per-level, so levels with many episodes do not dominate
  states    per episode at most --states-per-episode at random positions, per level at most --states-per-level; never
            the completion frame that ends a successful episode (it already shows the next level) nor a game over
            (no action can be played there)
  windows   rollout starts with ROLLOUT_HORIZONS[-1] recorded actions after them, from the same episodes, same caps
  planning  not part of the set (so planning settings never rebuild it): the planning experiment draws successful
            episodes of at most --planning-levels levels per split itself, the start checked against the level start
  An episode whose replay does not reproduce its recording is dropped with all its states and windows; set.json counts
  them per reason. Dropping a whole kind of episode (e.g. walk runs without a start path) biases every metric of the split.

Experiment      group(s)                 sample                            measures
  variance      encoder                  states                            latent variance, global and per level
  probes        encoder, generalization  states                            frozen-encoder MLP probes: trained on seen_train, measured on
                                                                           seen_val; zero-shot and fine-tuned on unseen (generalization)
  aliasing      encoder                  states + their action outcomes    do latent nearest neighbours react to actions alike?
  projections   encoder                  states of all splits              PCA and UMAP coordinates for plots
  cka           encoder                  states                            linear CKA with the --compare checkpoint (only then)
  one_step      world_model              states with a recorded next one   one-step error vs copy baseline: all / changed / unchanged
  counterfactual predictor               states x 4 actions                every action from the same state: errors and action effects
  rollout       predictor                windows                           autoregressive from one real frame, k in ROLLOUT_HORIZONS
  planning      planning                 planning episodes                 CEM goal reaching and waypoint following in the real game
  ranking       planning                 states with a recorded goal       spearman of predicted vs real-latent action ranking, spread
  generalization generalization          saved results of the above        differences between the splits, per metric
Every metric row carries its group, so the dashboard groups without knowing the experiments. The groups are only labels.

Output, below $STABLEWM_HOME/evaluation/ls20/lewm_analysis/
  sets/<split>_<hash>/          set.json, states.parquet, outcomes.parquet, windows.parquet, arrays.npz
  embeddings/<checkpoint>/<set>.npz
  results/<checkpoint>/checkpoint.json
  results/<checkpoint>/<experiment>/<split>.parquet    metric rows: group, experiment, metric, split, level, subset, value, n
  results/<checkpoint>/<experiment>/<split>.json       the inputs; a result with the same inputs is not computed again
  results/<checkpoint>/<experiment>/<split>_<name>.*   details (per state, per pair, coordinates, probe weights, ...)
  <split> is "all" for experiments that span the splits (probes, projections, generalization). Every (checkpoint,
  experiment, split) writes only its own files, so they can run as separate Slurm array tasks.

Usage
  python -m master_thesis.evaluation.lewm_analysis --run ls20_multi-no-l3_220k_succ0.1_0918-1951 --unseen human_l1-7:3
  python -m master_thesis.evaluation.lewm_analysis --run <run> --weights checkpoints/weights_step100000.pt --groups encoder
  python -m master_thesis.evaluation.lewm_analysis --run <run> --experiments planning --splits seen_val unseen
  python -m master_thesis.evaluation.lewm_analysis --run <run> --compare <other run> --experiments cka --unseen syn1000_dev
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
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
ACTIONS = (1, 2, 3, 4)  # the movement actions ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right
DIRECTIONS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # action -> (dcol, drow) on the playfield (arc_solver.ACTIONS)
ACTION_DIM = 9  # LeWM's action vector: 7-way one-hot + x + y (training/lewm.py ArcActionEncoding)
SPLITS = ("seen_train", "seen_val", "unseen")
ROLLOUT_HORIZONS = (3, 5, 10)  # rollout steps; k = 1 is one_step (same one-frame context)
ENERGY_BAR = (slice(61, 63), slice(13, 55))  # rows, cols of the energy bar: the only display that changes without the game state changing (every move costs energy, also a blocked one)
SET_VERSION = 1  # part of every set's hash: raise it when what a set contains changes, so old sets are rebuilt
D = 192  # latent size (embed_dim of every LeWM run so far): the shape letter D in docstrings, and the width of an empty encoding

# Dashboard groups -> experiments. probes reports the seen splits under encoder and unseen (zero-shot, adapted) under generalization.
GROUPS = {"encoder": ("variance", "probes", "aliasing", "projections", "cka"), "predictor": ("counterfactual", "rollout"), "world_model": ("one_step",),
          "planning": ("planning", "ranking"), "generalization": ("probes", "generalization")}


#####################
## Game and labels ##
#####################


def light_copy(game):
  """
  A copy of a game to try one action on.

  GETS:    game -- an arcengine game (ls20).
  DOES:    a deepcopy that shares every level object except the one being played: actions only change that one, and the
           clean levels are only ever cloned, never changed. A full deepcopy also copies every other level: 1.6 s with
           syn1000's 1007 levels, against 1.2 ms here (checked: the same outcome as the real game for every move of 4
           solutions, level completions included, and the original game unchanged).
  RETURNS: the copy.
  """
  current = game._levels[game._current_level_index]
  memo = {id(level): level for level in game._levels if level is not current}
  memo.update({id(level): level for level in game._clean_levels})
  return copy.deepcopy(game, memo)


def read_state(game):
  """
  The labels of the game's current state, read from the engine.

  GETS:    game -- an arcengine Ls20 game. Its names are obfuscated (see Ls20.on_set_level and Ls20.step in the game file):
           gudziatsk = player sprite, fwckfzsyc / hiaauhahz / cklxociuu = the current symbol's shape / colour index /
           rotation index, plrpelhym = goal sprites, lvrnuajbl = goal reached flags, ldxlnycps / yjdexjsoa / ehwheiwsk =
           shape / colour index / rotation index each goal needs, aqygnziho = lives, _step_counter_ui.current_steps = energy.
  DOES:    positions as (col, row) cells of the 12 x 12 playfield (pixel origin -> cell as arc_levelgen.spec_from_level;
           a refill sits 1 px inside its cell). "Nearest" = smallest Manhattan distance in cells from the player, the first
           in the level on ties. Moving changers (levels 5-7) are read where they are now.
  RETURNS: dict of ints -- level, player_x, player_y, energy, lives, key_shape, key_color, key_rotation (the current
           symbol); goal_x, goal_y, goal_shape, goal_color, goal_rotation, goal_distance, key_matches_goal (nearest
           unreached goal); refills_left, refill_x, refill_y (nearest refill); rotator_present, rotator_x, rotator_y
           (nearest rotator). -1 where the object does not exist.
  """
  from master_thesis.environments.arc_levelgen import REFILL, ROTATOR

  cell = lambda sprite, dx=0, dy=0: ((sprite.x + dx - 4) // 5, (sprite.y + dy) // 5)
  player = cell(game.gudziatsk)
  distance = lambda target: abs(target[0] - player[0]) + abs(target[1] - player[1])
  nearest = lambda cells: min(cells, key=distance, default=(-1, -1))
  key = (game.fwckfzsyc, game.hiaauhahz, game.cklxociuu)

  unreached = [index for index in range(len(game.plrpelhym)) if not game.lvrnuajbl[index]]
  goal = min(unreached, key=lambda index: distance(cell(game.plrpelhym[index])), default=None)
  goal_cell = cell(game.plrpelhym[goal]) if goal is not None else (-1, -1)
  goal_key = (game.ldxlnycps[goal], game.yjdexjsoa[goal], game.ehwheiwsk[goal]) if goal is not None else (-1, -1, -1)
  refills = [cell(sprite, -1, -1) for sprite in game.current_level.get_sprites_by_tag(REFILL)]  # taken refills are removed from the level
  rotators = [cell(sprite) for sprite in game.current_level.get_sprites_by_tag(ROTATOR)]

  return {"level": game._current_level_index + 1, "player_x": player[0], "player_y": player[1], "energy": int(game._step_counter_ui.current_steps), "lives": int(game.aqygnziho),
          "key_shape": key[0], "key_color": key[1], "key_rotation": key[2],
          "goal_x": goal_cell[0], "goal_y": goal_cell[1], "goal_shape": goal_key[0], "goal_color": goal_key[1], "goal_rotation": goal_key[2],
          "goal_distance": distance(goal_cell) if goal is not None else -1, "key_matches_goal": int(goal is not None and key == goal_key),
          "refills_left": len(refills), "refill_x": nearest(refills)[0], "refill_y": nearest(refills)[1],
          "rotator_present": int(bool(rotators)), "rotator_x": nearest(rotators)[0], "rotator_y": nearest(rotators)[1]}


def meaningful_change(before, after):
  """
  Did the game state change between two grids, apart from the energy?

  GETS:    before, after -- grids [..., 4096] (colour indices).
  DOES:    compares them outside ENERGY_BAR. Every move costs energy, so a blocked move changes only the bar; anything
           else (player, symbol, refills, goals, lives, moving changers, a new level) changes the playfield or the HUD.
  RETURNS: bool array [...]: True = meaningful change.
  """
  before, after = (np.asarray(grid).reshape(*np.shape(grid)[:-1], 64, 64).copy() for grid in (before, after))
  for grid in (before, after):
    grid[(..., *ENERGY_BAR)] = 0
  return (before != after).any(axis=(-2, -1))


_GAMES = {}  # per worker process: game_id -> game (arc_levelgen.make_game takes seconds for syn1000, so each worker makes it once)


def replay_level(task):
  """
  Worker: replay the sampled episodes of one level in the real engine and record what every action does at the sampled states.

  GETS:    task -- (game_id, level, episodes); each episode a dict with "key" (dataset, episode), "prefix" (action ids
           played from the level start before the recording begins; [] for recordings that start there), "grids"
           [L, 4096] uint8 (the recorded grids up to the last one needed), "actions" [L - 1] (the recorded action ids
           between them) and "positions" (sorted steps to evaluate).
  DOES:    per episode: copy the level-start game, play the prefix, check the first recorded grid, then follow the
           recorded actions, checking every grid. At each position: read_state, then each of the 4 actions on its own
           light_copy, with the outcome's grid and labels.
  RETURNS: one dict per episode: "key", "status" ("ok" or why it was dropped), "labels" (one read_state dict per
           position), "outcome_grids" [P, 4, 4096] uint8, "outcomes" (P * 4 dicts: the action's effect, see below).
           Effect of action a in state s, relative so that states of different levels compare: dx, dy (player
           displacement in cells; the action's direction when it completes the level, 0 when it costs a life),
           d_shape, d_color, d_rotation (how far the symbol cycled), refill_taken, life_lost, completed,
           success (completed, or moved without losing a life), changed (meaningful_change).
  """
  from arcengine import ActionInput, GameAction, GameState

  from master_thesis.environments.arc_collection import reset_at_level
  from master_thesis.environments.arc_levelgen import make_game

  game_id, level, episodes = task
  if game_id not in _GAMES:
    _GAMES[game_id] = make_game(game_id)
  start_grid = np.asarray(reset_at_level(_GAMES[game_id], level).frame[-1], dtype=np.uint8).reshape(-1)
  level_start = light_copy(_GAMES[game_id]._game)
  play = lambda game, action: game.perform_action(ActionInput(id=GameAction.from_id(int(action))), raw=True)  # -> FrameDataRaw; its last frame is what the player sees

  results = []
  for episode in episodes:
    game, grids, positions = light_copy(level_start), episode["grids"], episode["positions"]
    grid = start_grid
    for action in episode["prefix"]:
      grid = np.asarray(play(game, action).frame[-1], dtype=np.uint8).reshape(-1)
    result = {"key": episode["key"], "status": "ok", "labels": [], "outcome_grids": [], "outcomes": []}
    if not np.array_equal(grid, grids[0]):
      result["status"] = "start differs from the recording" if episode["prefix"] else "start differs from the recording (no start path)"
      results.append(result)
      continue

    for step in range(positions[-1] + 1):
      if step in positions:
        before = read_state(game)
        row = []
        for action in ACTIONS:
          trial = light_copy(game)
          frame = play(trial, action)
          row.append(np.asarray(frame.frame[-1], dtype=np.uint8).reshape(-1))
          after = read_state(trial)
          completed = trial._current_level_index != game._current_level_index or frame.state == GameState.WIN
          life_lost = not completed and after["lives"] < before["lives"]
          still = completed or life_lost  # the symbol and refills of the next level / after the reset say nothing about this action
          dx, dy = DIRECTIONS[action] if completed else (0, 0) if life_lost else (after["player_x"] - before["player_x"], after["player_y"] - before["player_y"])
          result["outcomes"].append({"action": action, "dx": dx, "dy": dy, "d_shape": 0 if still else (after["key_shape"] - before["key_shape"]) % 6,
                                     "d_color": 0 if still else (after["key_color"] - before["key_color"]) % 4, "d_rotation": 0 if still else (after["key_rotation"] - before["key_rotation"]) % 4,
                                     "refill_taken": int(not still and after["refills_left"] < before["refills_left"]), "life_lost": int(life_lost), "completed": int(completed),
                                     "success": int(completed or (not life_lost and (dx, dy) != (0, 0))), "changed": int(meaningful_change(grid, row[-1]))})
        result["labels"].append(before)
        result["outcome_grids"].append(np.stack(row))
      if step < positions[-1]:
        grid = np.asarray(play(game, episode["actions"][step]).frame[-1], dtype=np.uint8).reshape(-1)
        if not np.array_equal(grid, grids[step + 1]):
          result.update(status=f"differs from the recording at step {step + 1}", labels=[], outcome_grids=[], outcomes=[])
          break
    results.append(result)
  return results


###################
## Checkpoints ##
###################


def load_checkpoint(spec: str, device: str):
  """
  Rebuild a trained LeWM and what the evaluation needs to know about its training.

  GETS:    spec -- "<run>" or "<run>:<weights file inside the run>" (default weights.pt; e.g. checkpoints/weights_step10000.pt);
           device -- "cpu", "cuda" or "mps".
  DOES:    the architecture comes from model_config.json, which every run saves (plain LeWM, TC-SIGReg, Delta-JEPA, rollout
           loss and later variants all rebuild from it); the weights are loaded strictly; eval mode, since the predictor
           has dropout. From train_config.yaml: the preprocessing (img_size), the predictor's history, the training data
           and how it was split (window length as in training/lewm.py build_dataset: 1 + num_preds with the rollout
           loss, else history_size + 1).
  RETURNS: (model, info) -- info is a JSON-able dict: id (folder name for the results), run, weights, sha256 of the
           weights, img_size, history_size, dataset, frameskip, window, seed, train_split, val_split, started (when the
           run began: train_config.yaml's modification time).
  """
  from hydra.utils import instantiate
  from omegaconf import OmegaConf

  run, _, weights = spec.partition(":")
  weights = weights or "weights.pt"
  folder = model_dir("lewm", run)
  model = instantiate(json.loads((folder / "model_config.json").read_text(encoding="utf-8")))
  model.load_state_dict(torch.load(folder / weights, map_location="cpu", weights_only=True), strict=True)
  model.to(device).eval()

  config = OmegaConf.load(folder / "train_config.yaml")
  history_size = int(config.wm.history_size)
  rollout_loss = bool(OmegaConf.select(config, "extensions.rollout_loss", default=False))  # runs before the extensions block trained with the one-step loss only
  info = {"id": run if weights == "weights.pt" else f"{run}__{Path(weights).with_suffix('').as_posix().replace('/', '_')}", "run": run, "weights": weights,
          "sha256": hashlib.sha256((folder / weights).read_bytes()).hexdigest(), "img_size": int(config.img_size), "history_size": history_size,
          "dataset": str(config.data.dataset), "frameskip": int(config.data.frameskip), "window": 1 + int(config.wm.num_preds) if rollout_loss else history_size + 1,
          "seed": int(config.seed), "train_split": float(config.train_split), "val_split": float(config.val_split),
          "started": datetime.fromtimestamp((folder / "train_config.yaml").stat().st_mtime).isoformat(timespec="seconds")}
  return model, info


@torch.no_grad()
def encode_grids(model, img_size: int, grids, device: str, batch_size: int):
  """
  Encode grids with a checkpoint's encoder (ViT + projector), as training does.

  GETS:    model -- the JEPA in eval mode; img_size -- from its train_config; grids -- [M, 4096] uint8 colour indices;
           device; batch_size -- grids per forward pass.
  DOES:    encodes every distinct grid once (blocked moves and revisited states repeat grids a lot), with training's own
           preprocessing (training/lewm.py ArcGridToPixels: palette, nearest-neighbour resize, ImageNet normalization).
  RETURNS: [M, D] float32 numpy latents, in the order of `grids`.
  """
  from master_thesis.training.lewm import ArcGridToPixels

  to_pixels = ArcGridToPixels(img_size)
  unique, inverse = np.unique(np.asarray(grids, dtype=np.uint8).reshape(-1, 4096), axis=0, return_inverse=True)
  if not len(unique):  # nothing to encode (e.g. a set without windows)
    return np.zeros((0, D), np.float32)
  latents = []
  for start in range(0, len(unique), batch_size):
    pixels = to_pixels(torch.as_tensor(unique[start:start + batch_size], device=device))  # [B, 3, img, img]
    latents.append(model.encode({"pixels": pixels[:, None]})["emb"][:, 0].float().cpu())  # encode expects [B, T, ...]; T = 1
  return torch.cat(latents).numpy()[inverse.reshape(-1)]


##########################
## Splits and samples ##
##########################


def read_episodes(dataset: str, version: int | None = None):
  """
  Every episode of a Lance dataset, numbered as stable-worldmodel numbers them.

  GETS:    dataset -- name under datasets/ls20/; version -- Lance version to read (None = the latest).
  DOES:    episodes are runs of equal episode_idx in row order (stable-worldmodel's _compute_episode_structure), so
           episode e here is episode e of the training's clip_indices.
  RETURNS: (episodes, version) -- DataFrame with one row per episode: dataset, episode, episode_idx, offset, length,
           level, success, terminal_end (its last state is a game over or the win of the last level: no action can
           be played there); and the Lance version read.
  """
  import lance

  table = lance.dataset(str(dataset_path(GAME, dataset)), version=version)
  columns = table.to_table(columns=["episode_idx", "start_level", "episode_success", "terminal"])
  episode_idx = columns.column("episode_idx").to_numpy()
  offsets = np.flatnonzero(np.r_[True, np.diff(episode_idx) != 0])
  lengths = np.diff(np.r_[offsets, len(episode_idx)])
  value = lambda name, rows: columns.column(name).combine_chunks().flatten().to_numpy()[rows]
  episodes = pd.DataFrame({"dataset": dataset, "episode": np.arange(len(offsets)), "episode_idx": episode_idx[offsets], "offset": offsets, "length": lengths,
                           "level": value("start_level", offsets).astype(int), "success": value("episode_success", offsets).astype(bool),  # constant within an episode: read on its first row
                           "terminal_end": value("terminal", offsets + lengths - 1).astype(bool)})  # only an episode's last state can be terminal (arc_recording ends episodes there)
  return episodes, table.version


def training_split(info: dict):
  """
  The checkpoint's own train and val episodes, rebuilt exactly as training/lewm.py build_dataloaders drew them.

  GETS:    info -- from load_checkpoint.
  DOES:    opens the training dataset at the version that was current when the run started (Lance keeps every version),
           so episodes appended later cannot shift the split; keeps the episodes long enough for one training window
           (length >= window * frameskip, as stable-worldmodel's clip_indices); draws the same permutation (torch, the
           run's seed) and cuts it with the same fractions. The test part is dropped here and never read.
  RETURNS: (episodes, version, training_levels) -- episodes as read_episodes with part "train" / "val"; the Lance
           version used; every level in the training dataset (all three parts: the model saw these levels).
  """
  import lance

  started = datetime.fromisoformat(info["started"])
  versions = [entry["version"] for entry in lance.dataset(str(dataset_path(GAME, info["dataset"]))).versions() if entry["timestamp"] <= started]
  if not versions:
    raise ValueError(f"{info['dataset']} has no version from before the run started ({info['started']}): the training split cannot be rebuilt")
  episodes, version = read_episodes(info["dataset"], versions[-1])

  eligible = episodes[episodes.length >= info["window"] * info["frameskip"]].episode.tolist()  # training's sorted({episode for episode, _ in clip_indices})
  order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(info["seed"])).tolist()
  shuffled = [eligible[index] for index in order]
  num_train, num_val = int(len(shuffled) * info["train_split"]), int(len(shuffled) * info["val_split"])
  part = {**{episode: "train" for episode in shuffled[:num_train]}, **{episode: "val" for episode in shuffled[num_train:num_train + num_val]}}
  training_levels = set(episodes.level.tolist())
  episodes = episodes.assign(part=episodes.episode.map(part)).dropna(subset=["part"])  # test episodes and too-short ones drop out
  return episodes, version, training_levels


def unseen_episodes(specs: list[str], training_levels: set, seed: int):
  """
  The unseen split: episodes of levels the checkpoint never trained on.

  GETS:    specs -- --unseen values, "DATASET" or "DATASET:LEVEL,LEVEL"; training_levels -- from training_split; seed.
  DOES:    reads each dataset (latest version) and keeps the given levels (all if none). Refuses a level of the
           training data. Per level, a seeded permutation of its episodes puts the first half in "adapt" (only used to
           fine-tune the probes) and the rest in "eval"; a level with one episode has only "eval".
  RETURNS: (episodes, versions) -- episodes as read_episodes with part "adapt" / "eval"; {dataset: Lance version}.
  """
  frames, versions = [], {}
  for spec in specs:
    dataset, _, levels = spec.partition(":")
    episodes, versions[dataset] = read_episodes(dataset)
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


def read_rows(dataset: str, version: int, rows):
  """
  Recorded rows of a Lance dataset.

  GETS:    dataset -- name under datasets/ls20/; version -- the Lance version its episodes were read from; rows [R] row indices.
  RETURNS: (grids [R, 4096] uint8, actions [R] recorded action ids (0 on an episode's last state), game_ids [R]).
  """
  import lance

  table = lance.dataset(str(dataset_path(GAME, dataset)), version=version).take(np.asarray(rows), columns=["grid", "action", "game_id"])
  grids = table.column("grid").combine_chunks().flatten().to_numpy().reshape(-1, 4096).astype(np.uint8)  # float32 in datasets written before 20.10.26
  actions = table.column("action").combine_chunks().flatten().to_numpy().reshape(-1, 3)[:, 0].astype(int)  # [id, x, y] -> id
  return grids, actions, table.column("game_id").to_pylist()


def sample_split(split: str, episodes, args):
  """
  Which states and rollout windows of a split are evaluated. (Planning draws its own episodes: experiment_planning.)

  GETS:    split -- one of SPLITS; episodes -- its candidate episodes (training_split / unseen_episodes); args -- the CLI.
  DOES:    per level its own seeded generator (--seed, the split, the level), so adding or removing a level never
           changes what is drawn in the others: at most --episodes-per-level episodes; in each, at most
           --states-per-episode state positions (never the completion frame of a successful episode or a terminal
           last state) and at most --windows-per-episode rollout starts (with ROLLOUT_HORIZONS[-1] recorded steps after
           them), uniformly at random; then at most --states-per-level and --windows-per-level per level.
           --max-levels caps the levels first.
  RETURNS: (states, windows) -- DataFrames with one row per sampled state / window start: the episode's columns plus "step".
  """
  horizon = ROLLOUT_HORIZONS[-1]
  levels = sorted(set(episodes.level))
  if args.max_levels is not None and len(levels) > args.max_levels:
    levels = sorted(np.random.default_rng([args.seed, SPLITS.index(split)]).choice(levels, args.max_levels, replace=False).tolist())

  states, windows = [], []
  for level in levels:
    rng = np.random.default_rng([args.seed, SPLITS.index(split), level])
    candidates = episodes[episodes.level == level]
    chosen = candidates.iloc[rng.permutation(len(candidates))[:args.episodes_per_level]]
    level_states, level_windows = [], []
    for episode in chosen.itertuples(index=False):
      usable = episode.length - 1 if episode.success or episode.terminal_end else episode.length  # not the last state of a successful episode (the completion frame shows the next level) or a terminal one (game over: no action can be played)
      for step in np.sort(rng.choice(usable, min(args.states_per_episode, usable), replace=False)):
        level_states.append({**episode._asdict(), "step": int(step)})
      starts = episode.length - horizon  # window starts with `horizon` recorded actions after them
      for step in np.sort(rng.choice(starts, min(args.windows_per_episode, starts), replace=False)) if starts > 0 else []:
        level_windows.append({**episode._asdict(), "step": int(step)})
    states += [level_states[index] for index in np.sort(rng.permutation(len(level_states))[:args.states_per_level])]
    windows += [level_windows[index] for index in np.sort(rng.permutation(len(level_windows))[:args.windows_per_level])]
  return pd.DataFrame(states), pd.DataFrame(windows)


######################
## Evaluation sets ##
######################


@dataclass
class EvalSet:
  """
  One split's evaluation set, loaded from sets/<split>_<hash>/ (written by build_set). N states, W windows.

  meta      set.json: definition, counts, dropped episodes
  states    one row per state: dataset, episode, episode_idx, step, level, part, game_id, action (recorded there; 0 on
            an episode's last state), has_next, has_goal (a recorded state --goal-ahead steps later exists),
            next_changed (meaningful_change to the recorded next state), and the read_state labels as label_<name>
  outcomes  one row per (state, action): state, action and the effect columns of replay_level
  windows   one row per rollout start: dataset, episode, episode_idx, step, level, part
  arrays    grid [N, 4096], next_grid [N, 4096] (zeros without has_next), goal_grid [N, 4096] (zeros without has_goal),
            outcome_grid [N, 4, 4096], window_grid [W, H + 1, 4096] uint8; window_action [W, H] (H = ROLLOUT_HORIZONS[-1])
  """
  split: str
  path: Path
  meta: dict
  states: pd.DataFrame
  outcomes: pd.DataFrame
  windows: pd.DataFrame
  arrays: dict


def build_set(split: str, episodes, versions: dict, args) -> Path:
  """
  Build (or find) the evaluation set of one split.

  GETS:    split; episodes -- its candidate episodes; versions -- {dataset: Lance version} they were read from; args.
  DOES:    the set's folder name hashes everything it depends on (SET_VERSION, the candidate episodes, the dataset
           versions, the sampling settings), so a set is built once and shared by every checkpoint with the same split.
           Samples (sample_split), reads the needed rows of the sampled episodes from Lance, replays the sampled
           episodes in the engine (replay_level, parallel over levels), drops episodes that do not replay with their
           states and windows, and writes the folder atomically (a concurrent job building the same set loses the race
           and uses the winner's).
  RETURNS: the set folder.
  """
  sampling = {key: getattr(args, key) for key in ("seed", "episodes_per_level", "states_per_episode", "states_per_level", "windows_per_episode", "windows_per_level", "max_levels", "goal_ahead")}
  definition = {"set_version": SET_VERSION, "split": split, "versions": versions, "sampling": sampling, "horizons": ROLLOUT_HORIZONS,
                "candidates": hashlib.sha256(pd.util.hash_pandas_object(episodes[["dataset", "episode", "offset", "length", "part"]], index=False).values.tobytes()).hexdigest()}
  folder = args.output / "sets" / f"{split}_{hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:12]}"
  if (folder / "set.json").exists():
    return folder

  states, windows = sample_split(split, episodes, args)
  horizon = ROLLOUT_HORIZONS[-1]

  # (dataset, episode) -> (its episode row, last row needed): a sampled state's next state and goal, or a window's end
  needed = {}
  for frame, extra in ((states, max(1, args.goal_ahead)), (windows, horizon)):
    for row in frame.itertuples(index=False):
      key = (row.dataset, row.episode)
      needed[key] = (row, max(min(row.step + extra, row.length - 1), needed[key][1] if key in needed else 0))

  recorded = {}  # (dataset, episode) -> {"grid": [L, 4096] uint8, "action": [L], "game_id"} up to the last needed row
  for dataset in sorted({key[0] for key in needed}):
    keys = [key for key in needed if key[0] == dataset]
    grids, actions, game_ids = read_rows(dataset, versions[dataset], np.concatenate([np.arange(needed[key][0].offset, needed[key][0].offset + needed[key][1] + 1) for key in keys]))
    start = 0
    for key in keys:
      length = needed[key][1] + 1
      recorded[key] = {"grid": grids[start:start + length], "action": actions[start:start + length], "game_id": game_ids[start]}
      start += length

  # start paths of recordings that do not begin at the level start (synthetic walk runs): datasets/ls20/<dataset>.episodes.jsonl, one line per
  # episode written with the Lance table: {"episode_idx", "source", "start_level", "prefix"}; recordings that start at the level start have no or an empty prefix
  prefixes = {}
  for dataset in sorted({key[0] for key in needed}):
    provenance = dataset_path(GAME, dataset).with_suffix(".episodes.jsonl")
    wanted = {int(row.episode_idx) for key, (row, _) in needed.items() if key[0] == dataset}
    if provenance.exists():
      with provenance.open(encoding="utf-8") as file:
        records = (json.loads(line) for line in file)
        prefixes[dataset] = {record["episode_idx"]: record.get("prefix") or [] for record in records if record["episode_idx"] in wanted}

  # replay every episode with sampled states, grouped by (game, level) so each worker task needs one level start
  tasks = {}
  for (dataset, episode), steps in states.groupby(["dataset", "episode"]).step:
    row = needed[(dataset, episode)][0]
    data = recorded[(dataset, episode)]
    tasks.setdefault((data["game_id"], int(row.level)), []).append({"key": (dataset, int(episode)), "prefix": prefixes.get(dataset, {}).get(int(row.episode_idx), []),
                                                                     "grids": data["grid"], "actions": data["action"][:-1], "positions": sorted(int(step) for step in steps)})
  if args.workers > 1:
    from master_thesis.evaluation.lewm_analysis import replay_level as worker  # imported by its package name, so spawned workers can import it too
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
      replayed = [episode for result in pool.map(worker, [(game_id, level, jobs) for (game_id, level), jobs in sorted(tasks.items())]) for episode in result]
  else:
    replayed = [episode for (game_id, level), jobs in sorted(tasks.items()) for episode in replay_level((game_id, level, jobs))]
  replayed = {episode["key"]: episode for episode in replayed}
  ok = {key for key, episode in replayed.items() if episode["status"] == "ok"}
  dropped = pd.Series([episode["status"] for episode in replayed.values() if episode["status"] != "ok"], dtype=str).value_counts().to_dict()  # reason -> episodes

  # assemble the states (only those of episodes that replayed), their outcomes and arrays
  state_rows, outcome_rows, grid, next_grid, goal_grid, outcome_grid = [], [], [], [], [], []
  for (dataset, episode), steps in states.groupby(["dataset", "episode"], sort=True).step:
    if (dataset, episode) not in ok:
      continue
    result, data = replayed[(dataset, episode)], recorded[(dataset, episode)]
    row = needed[(dataset, episode)][0]
    for position, step in enumerate(sorted(steps)):
      has_next, has_goal = step + 1 < row.length, step + args.goal_ahead < row.length
      state = len(state_rows)
      state_rows.append({"state": state, "dataset": dataset, "episode": int(episode), "episode_idx": int(row.episode_idx), "step": int(step), "level": int(row.level), "part": row.part,
                         "game_id": data["game_id"], "action": int(data["action"][step]), "has_next": has_next, "has_goal": has_goal,
                         "next_changed": bool(has_next and meaningful_change(data["grid"][step], data["grid"][step + 1])), **{f"label_{key}": value for key, value in result["labels"][position].items()}})
      outcome_rows += [{"state": state, **outcome} for outcome in result["outcomes"][position * len(ACTIONS):(position + 1) * len(ACTIONS)]]
      grid.append(data["grid"][step])
      next_grid.append(data["grid"][step + 1] if has_next else np.zeros(4096, dtype=np.uint8))
      goal_grid.append(data["grid"][step + args.goal_ahead] if has_goal else np.zeros(4096, dtype=np.uint8))
      outcome_grid.append(result["outcome_grids"][position])
  states = pd.DataFrame(state_rows)
  mismatched = states.label_level != states.level if len(states) else pd.Series(dtype=bool)  # the engine plays another level than the recording says: should never happen
  if mismatched.any():
    raise RuntimeError(f"{split}: {int(mismatched.sum())} states are in another level than their episode's start_level")

  windows = windows[[(row.dataset, row.episode) in ok for row in windows.itertuples(index=False)]].reset_index(drop=True)
  window_grid = np.stack([recorded[(row.dataset, row.episode)]["grid"][row.step:row.step + horizon + 1] for row in windows.itertuples(index=False)]) if len(windows) else np.zeros((0, horizon + 1, 4096), np.uint8)
  window_action = np.stack([recorded[(row.dataset, row.episode)]["action"][row.step:row.step + horizon] for row in windows.itertuples(index=False)]) if len(windows) else np.zeros((0, horizon), int)

  meta = {**definition, "built": datetime.now().isoformat(timespec="seconds"), "levels": int(states.level.nunique()) if len(states) else 0,
          "counts": {"candidate_episodes": len(episodes), "replayed_episodes": len(replayed), "dropped_episodes": len(replayed) - len(ok), "states": len(states), "windows": len(windows)},
          "dropped": dropped}
  temporary = folder.with_name(f"{folder.name}.tmp{os.getpid()}")
  temporary.mkdir(parents=True)
  states.to_parquet(temporary / "states.parquet")
  pd.DataFrame(outcome_rows).to_parquet(temporary / "outcomes.parquet")
  windows.to_parquet(temporary / "windows.parquet")
  np.savez_compressed(temporary / "arrays.npz", grid=np.asarray(grid, np.uint8).reshape(-1, 4096), next_grid=np.asarray(next_grid, np.uint8).reshape(-1, 4096), goal_grid=np.asarray(goal_grid, np.uint8).reshape(-1, 4096),
                      outcome_grid=np.asarray(outcome_grid, np.uint8).reshape(-1, len(ACTIONS), 4096), window_grid=window_grid.astype(np.uint8), window_action=window_action.astype(np.int16))
  (temporary / "set.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")  # written last: its presence marks a complete set
  try:
    temporary.rename(folder)
  except OSError:  # another job finished the same set first
    shutil.rmtree(temporary)
  if dropped:
    print(f"WARNING {split}: {len(replayed) - len(ok)} of {len(replayed)} sampled episodes dropped ({dropped}); their states and windows are not evaluated")
  return folder


def load_set(split: str, folder: Path) -> EvalSet:
  """Read an evaluation set written by build_set."""
  with np.load(folder / "arrays.npz") as arrays:
    return EvalSet(split=split, path=folder, meta=json.loads((folder / "set.json").read_text(encoding="utf-8")), states=pd.read_parquet(folder / "states.parquet"),
                   outcomes=pd.read_parquet(folder / "outcomes.parquet"), windows=pd.read_parquet(folder / "windows.parquet"), arrays={name: arrays[name] for name in arrays.files})


def embed_set(model, info: dict, evaluation_set: EvalSet, args) -> dict:
  """
  A set encoded by one checkpoint, cached in embeddings/<checkpoint>/<set>.npz.

  GETS:    model, info -- from load_checkpoint; evaluation_set; args (device, batch size).
  DOES:    encodes the set's states, recorded next states, goals, action outcomes and window frames in one pass, so a
           grid that occurs in several of them is encoded once.
  RETURNS: {"state": [N, D], "next": [N, D], "goal": [N, D] (NaN where the state has none), "outcome": [N, 4, D], "window": [W, H + 1, D]}.
  """
  path = args.output / "embeddings" / info["id"] / f"{evaluation_set.path.name}.npz"
  names = ("state", "next", "goal", "outcome", "window")
  if path.exists():
    with np.load(path) as stored:
      if str(stored["sha256"]) == info["sha256"]:  # the same weights: retrained runs with the same name are encoded again
        return {name: stored[name] for name in names}

  arrays = evaluation_set.arrays
  parts = dict(zip(names, (arrays["grid"], arrays["next_grid"], arrays["goal_grid"], arrays["outcome_grid"], arrays["window_grid"])))
  latents = encode_grids(model, info["img_size"], np.concatenate([part.reshape(-1, 4096) for part in parts.values()]), args.device, args.batch_size)
  embeddings, start = {}, 0
  for name, part in parts.items():
    count = int(np.prod(part.shape[:-1]))
    embeddings[name] = latents[start:start + count].reshape(*part.shape[:-1], latents.shape[1])
    start += count
  if len(evaluation_set.states):
    embeddings["next"][~evaluation_set.states.has_next.to_numpy()] = np.nan
    embeddings["goal"][~evaluation_set.states.has_goal.to_numpy()] = np.nan
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path.with_suffix(".tmp.npz"), sha256=np.array(info["sha256"]), **embeddings)
  path.with_suffix(".tmp.npz").replace(path)
  return embeddings


#################
## Experiments ##
#################


@dataclass
class Evaluation:
  """Everything an experiment needs: the checkpoint, the sets and their embeddings per split, and the CLI arguments."""
  args: argparse.Namespace
  model: torch.nn.Module
  info: dict
  episodes: dict  # split -> (its candidate episodes, {dataset: Lance version}): planning draws its episodes from them
  sets: dict  # split -> EvalSet
  embeddings: dict  # split -> embed_set's dict
  compare: tuple | None  # (model, info, {split: embeddings}) of --compare, or None


def metric(experiment: str, name: str, split: str, value, n: int, level="all", subset: str = "all", group: str | None = None) -> dict:
  """One row of the metrics table. group defaults to the experiment's first group in GROUPS; level and subset are "all" or a value."""
  group = group or next(group for group, experiments in GROUPS.items() if experiment in experiments)
  return {"group": group, "experiment": experiment, "metric": name, "split": split, "level": str(level), "subset": str(subset), "value": float(value), "n": int(n)}


def summarize(experiment: str, name: str, split: str, values, levels, subset: str = "all", denominator=None, group: str | None = None) -> list:
  """
  Metric rows for a per-item quantity, over all items and within every level.

  GETS:    values [M] per item (NaN = not defined there, left out); levels [M]; denominator [M] -- optional: then the
           value is mean(values) / mean(denominator), e.g. prediction error / copy error (NaN where that mean is 0);
           the other fields as in metric().
  RETURNS: list of metric rows: level "all", then one per level.
  """
  values, levels = np.asarray(values, dtype=float), np.asarray(levels)
  denominator = np.ones_like(values) if denominator is None else np.asarray(denominator, dtype=float)
  defined = ~np.isnan(values) & ~np.isnan(denominator)
  value = lambda selected: values[selected].mean() / denominator[selected].mean() if selected.any() and denominator[selected].mean() > 0 else np.nan
  rows = [metric(experiment, name, split, value(defined), defined.sum(), subset=subset, group=group)]
  for level in np.unique(levels[defined]):
    selected = defined & (levels == level)
    rows.append(metric(experiment, name, split, value(selected), selected.sum(), level=int(level), subset=subset, group=group))
  return rows


@torch.no_grad()
def action_latents(evaluation: Evaluation):
  """
  The action latents of ACTION1..4 as the planner encodes them: one-hot over the 7 action types, x = y = 0 [4, D].
  The action encoder works per time step (a kernel-1 convolution), so they also serve for action sequences.
  """
  return evaluation.model.action_encoder(torch.eye(len(ACTIONS), ACTION_DIM, device=evaluation.args.device)[:, None])[:, 0]


@torch.no_grad()
def predict_next(evaluation: Evaluation, latents, actions):
  """
  The predictor's next latent from one real frame, as the planner's first step.

  GETS:    latents [M, D] numpy; actions [M] action ids 1..4.
  DOES:    z_hat = JEPA.predict([z], [e(a)]) at its only position, in batches of --batch-size (eval mode: no dropout).
  RETURNS: [M, D] numpy.
  """
  device, size, encoded = evaluation.args.device, evaluation.args.batch_size, action_latents(evaluation)
  predictions = [evaluation.model.predict(torch.as_tensor(latents[start:start + size], device=device)[:, None], encoded[torch.as_tensor(np.asarray(actions[start:start + size]) - 1, device=device)][:, None])[:, -1].float().cpu()
                 for start in range(0, len(latents), size)]
  return torch.cat(predictions).numpy() if predictions else np.zeros((0, latents.shape[1]), np.float32)


def train_probe(inputs, targets, outputs: int, steps: int, seed: int, device: str, probe=None, class_weight=None):
  """
  Train, or fine-tune, one probe on frozen latents; the same procedure for every checkpoint.

  GETS:    inputs [M, D] standardized latents; targets [M] class indices (outputs > 1) or standardized values (outputs = 1);
           steps -- optimizer steps on batches of 256 drawn with replacement; seed; device; probe -- one to fine-tune
           (None = a new MLP D -> 256 -> outputs); class_weight [outputs] -- loss weight per class (inverse frequency
           in the training data, so a rare class counts as much as a common one: probes are scored by balanced accuracy).
  DOES:    AdamW (learning rate 1e-3, weight decay 1e-4) on cross-entropy (classes) or MSE (values).
  RETURNS: the probe in eval mode.
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
  Model-predictive control with LeWM, adapted from evaluation/arc_policies.py LewmRolloutPlanner: stable-worldmodel's
  CategoricalCEMSolver searches action sequences, JEPA.rollout imagines where they lead, and the first --replan-every
  actions of the best one are played before planning again.

  GETS:    model -- the JEPA in eval mode; img_size; goal_grid [4096]; args -- samples, iterations, horizon, topk,
           replan_every, cost_on, seed, device.
  DOES:    a plan's cost is the squared latent distance to the goal at its closest imagined state ("path") or at its
           last one ("end", LeWM's own criterion). Actions are one-hot over ACTION1..4 padded to LeWM's 9-d action.
  """

  def __init__(self, model, img_size: int, goal_grid, args):
    import gymnasium as gym
    from types import SimpleNamespace

    from stable_worldmodel.planning.solver.categorical_cem import CategoricalCEMSolver

    from master_thesis.training.lewm import ArcGridToPixels

    self.device, self.replan_every = args.device, min(args.replan_every, args.horizon)
    self.to_pixels = ArcGridToPixels(img_size)
    self.goal = self.pixels(goal_grid)
    self.dummy_action = torch.zeros(1, 1, ACTION_DIM, device=args.device)  # JEPA.get_cost expects an action entry; the rollout replaces it
    self.queue, self.costs = [], []  # actions of the current plan still to play; the solver's cost of every plan
    pad = lambda candidates: torch.nn.functional.pad(candidates, (0, ACTION_DIM - candidates.shape[-1]))  # one-hot over ACTION1..4 -> 9-d (x, y stay 0)

    def cost(info, candidates):
      costs = model.get_cost(info, pad(candidates))  # runs the rollout: fills info["predicted_emb"] [1, samples, horizon + 1, D] and info["goal_emb"]
      if args.cost_on == "end":
        return costs
      distance = (info["predicted_emb"][:, :, 1:] - info["goal_emb"][:, None, -1:]).square().sum(dim=-1)  # [1, samples, horizon]: after move 1, 2, ...
      return distance.min(dim=-1).values

    self.solver = CategoricalCEMSolver(cost=SimpleNamespace(get_cost=cost), num_samples=args.samples, n_steps=args.iterations, topk=args.topk, device=args.device, seed=args.seed)
    self.solver.configure(action_space=gym.spaces.Discrete(len(ACTIONS)), n_envs=1, config=SimpleNamespace(horizon=args.horizon, action_block=1))

  def pixels(self, grid):
    return self.to_pixels(torch.as_tensor(np.asarray(grid).reshape(1, -1), device=self.device))[None]  # [1, 1, 3, img, img] = (batch, time, ...)

  @torch.no_grad()
  def act(self, grid) -> int:
    """The next action id (1..4) for the current grid; plans again when the current plan is used up."""
    if not self.queue:
      plan = self.solver.solve({"pixels": self.pixels(grid), "goal": self.goal, "action": self.dummy_action})
      self.costs.append(float(plan["costs"][0]))
      self.queue = [int(action) + 1 for action in plan["actions"][0, :self.replan_every, 0]]  # solver index 0..3 -> action id 1..4
    return self.queue.pop(0)


def experiment_variance(evaluation: Evaluation, split: str):
  """
  Latent variance: is the representation collapsed, and does level identity dominate it? (group encoder)

  GETS:    evaluation; split.
  DOES:    on the split's states z [N, D]: V = 1/D * sum_j Var_i(z_j) (population variance per dimension, averaged), over
           all states and within every level; within/global = mean over levels of the within-level V divided by the
           global V (near 0: the latents mostly tell levels apart, little varies inside a level).
  RETURNS: (rows, details) -- metrics variance (subset all / per level), within_over_global.
  """
  latents, levels = evaluation.embeddings[split]["state"], evaluation.sets[split].states.level.to_numpy()
  if len(latents) < 2:
    return [], {}
  overall = latents.var(axis=0).mean()  # population variance per dimension, averaged over the D dimensions
  rows, within = [metric("variance", "variance", split, overall, len(latents))], []
  for level in np.unique(levels):
    selected = levels == level
    if selected.sum() >= 2:  # one state has no variance
      within.append(latents[selected].var(axis=0).mean())
      rows.append(metric("variance", "variance", split, within[-1], selected.sum(), level=int(level)))
  rows.append(metric("variance", "within_over_global", split, np.mean(within) / overall if within and overall > 0 else np.nan, len(within)))
  return rows, {}


def experiment_probes(evaluation: Evaluation, split: str):
  """
  Frozen-encoder MLP probes: which control-relevant facts can be read off the latent? (encoder; unseen rows: generalization)

  GETS:    evaluation; split -- "all" (the probes are trained once and measured on every split).
  DOES:    targets from the read_state labels and the action outcomes: level (seen splits only: a level classifier
           has no class for an unseen level), player_x, player_y, energy, lives, key_shape/_color/_rotation (current
           symbol), goal_shape/_color/_rotation, key_matches_goal, goal_distance (nearest unreached goal),
           refill_left (refills_left > 0), refill_x/_y (states with a refill left), rotator_present, rotator_x/_y
           (levels with a rotator), success_up/_down/_left/_right (does the action move the player or complete the
           level). Classification -> balanced accuracy, regression -> MAE in the label's unit. One architecture and
           budget for every checkpoint (train_probe, --probe-steps; latents and regression targets standardized with
           seen_train's statistics). Trained on seen_train, measured on seen_train (overfitting) and seen_val. On
           unseen: zero-shot on the eval part, then a copy fine-tuned on the adapt part (--probe-adapt-steps; states
           with a class the seen probe does not know are left out of the fine-tuning) and measured on the same eval
           states. Controls on the same states: global_baseline (seen_train's majority class / mean), level_baseline
           (per level the majority / mean of the training labels -- seen_train, or the adapt part on unseen -- i.e.
           what knowing the level alone gives), and varying_levels (share of levels where the label takes more than
           one value: a target that is constant within levels can be solved by level identity). A target with fewer
           than 10 labelled seen_train states or a constant label is skipped and listed.
  RETURNS: (rows, details) -- metrics <target>/balanced_accuracy or <target>/mae with subset probe / global_baseline /
           level_baseline (seen splits, group encoder) or zero_shot / adapted / adapted_minus_zero_shot /
           global_baseline / level_baseline (unseen, group generalization), and <target>/varying_levels;
           details: predictions (per split, target and state), weights (every probe), skipped.
  """
  from sklearn.metrics import balanced_accuracy_score

  args = evaluation.args
  if "seen_train" not in evaluation.sets:
    print("probes: skipped (needs the seen_train split)")
    return [], {}
  splits = [name for name in SPLITS if name in evaluation.sets]

  def targets(name):  # target -> (kind, labels [N], usable [N]) for one split
    states = evaluation.sets[name].states
    label = lambda key: states[f"label_{key}"].to_numpy()
    success = evaluation.sets[name].outcomes.pivot(index="state", columns="action", values="success").reindex(states.state).to_numpy()  # [N, 4]
    every, has_goal, has_refill, has_rotator = np.ones(len(states), bool), label("goal_x") >= 0, label("refills_left") > 0, label("rotator_present") > 0
    return {"level": ("class", states.level.to_numpy(), every & (name != "unseen")), "player_x": ("regression", label("player_x"), every), "player_y": ("regression", label("player_y"), every),
            "energy": ("regression", label("energy"), every), "lives": ("class", label("lives"), every),
            **{f"key_{part}": ("class", label(f"key_{part}"), every) for part in ("shape", "color", "rotation")},
            **{f"goal_{part}": ("class", label(f"goal_{part}"), has_goal) for part in ("shape", "color", "rotation")},
            "key_matches_goal": ("class", label("key_matches_goal"), has_goal), "goal_distance": ("regression", label("goal_distance"), has_goal),
            "refill_left": ("class", has_refill.astype(int), every), "refill_x": ("regression", label("refill_x"), has_refill), "refill_y": ("regression", label("refill_y"), has_refill),
            "rotator_present": ("class", has_rotator.astype(int), every), "rotator_x": ("regression", label("rotator_x"), has_rotator), "rotator_y": ("regression", label("rotator_y"), has_rotator),
            **{f"success_{direction}": ("class", success[:, position], every) for position, direction in enumerate(("up", "down", "left", "right"))}}

  labels = {name: targets(name) for name in splits}
  mean, std = evaluation.embeddings["seen_train"]["state"].mean(axis=0), evaluation.embeddings["seen_train"]["state"].std(axis=0) + 1e-6
  inputs = {name: (evaluation.embeddings[name]["state"] - mean) / std for name in splits}
  levels = {name: evaluation.sets[name].states.level.to_numpy() for name in splits}
  parts = {name: evaluation.sets[name].states.part.to_numpy() for name in splits}
  states = {name: evaluation.sets[name].states.state.to_numpy() for name in splits}
  score = lambda kind, truth, guess: balanced_accuracy_score(truth, guess) if kind == "class" else float(np.abs(truth - guess).mean())
  level_guess = lambda kind, known_levels, known, asked_levels: pd.Series(asked_levels).map(pd.Series(known).groupby(known_levels).agg(lambda values: values.mode()[0] if kind == "class" else values.mean())).fillna(
    pd.Series(known).mode()[0] if kind == "class" else known.mean()).to_numpy()  # per level the majority / mean of the known labels; a level without known labels gets the overall one

  rows, predictions, weights, skipped = [], [], {}, []
  for target, (kind, train_labels, train_usable) in labels["seen_train"].items():
    known = train_labels[train_usable]
    if train_usable.sum() < 10 or len(np.unique(known)) < 2:
      skipped.append({"target": target, "reason": f"{int(train_usable.sum())} labelled seen_train states, {len(np.unique(known))} distinct labels"})
      continue
    if kind == "class":
      classes, counts = np.unique(known, return_counts=True)
      encode = lambda values: np.searchsorted(classes, values)  # label -> class index (only for labels in classes)
      decode = lambda output: classes[output.argmax(axis=1)]
      outputs, class_weight, overall = len(classes), len(known) / (len(classes) * counts), classes[counts.argmax()]
    else:
      location, scale = known.mean(), known.std() + 1e-6
      encode = lambda values: (values - location) / scale
      decode = lambda output: output[:, 0] * scale + location
      outputs, class_weight, overall = 1, None, location
    name = f"{target}/{'balanced_accuracy' if kind == 'class' else 'mae'}"
    probe = train_probe(inputs["seen_train"][train_usable], encode(known), outputs, args.probe_steps, args.seed, args.device, class_weight=class_weight)
    guess = lambda model, selected, split: decode(model(torch.as_tensor(inputs[split][selected], dtype=torch.float32, device=args.device)).detach().cpu().numpy())
    weights.update({f"{target}/{key}": value.cpu().numpy() for key, value in probe.state_dict().items()})

    for split in splits:
      _, split_labels, usable = labels[split][target]
      group = "generalization" if split == "unseen" else "encoder"
      evaluated = usable & (parts[split] == "eval") if split == "unseen" else usable  # unseen: the probes are compared on the eval part only
      if not evaluated.any():
        continue
      truth = split_labels[evaluated]
      result = {"zero_shot" if split == "unseen" else "probe": score(kind, truth, guess(probe, evaluated, split)), "global_baseline": score(kind, truth, np.full(len(truth), overall))}
      if split != "unseen":
        result["level_baseline"] = score(kind, truth, level_guess(kind, levels["seen_train"][train_usable], known, levels[split][evaluated]))
      else:
        adapt = usable & (parts[split] == "adapt") & (np.isin(split_labels, classes) if kind == "class" else True)
        if adapt.any():
          adapted = train_probe(inputs[split][adapt], encode(split_labels[adapt]), outputs, args.probe_adapt_steps, args.seed, args.device, probe=copy.deepcopy(probe), class_weight=class_weight)
          weights.update({f"{target}/adapted/{key}": value.cpu().numpy() for key, value in adapted.state_dict().items()})
          result["adapted"] = score(kind, truth, guess(adapted, evaluated, split))
          result["adapted_minus_zero_shot"] = result["adapted"] - result["zero_shot"]
          result["level_baseline"] = score(kind, truth, level_guess(kind, levels[split][adapt], split_labels[adapt], levels[split][evaluated]))
          predictions.append(pd.DataFrame({"split": split, "target": target, "kind": "adapted", "state": states[split][evaluated], "label": truth, "prediction": guess(adapted, evaluated, split)}))
      rows += [metric("probes", name, split, value, len(truth), subset=subset, group=group) for subset, value in result.items()]
      varying = pd.DataFrame({"level": levels[split][evaluated], "label": truth}).groupby("level").label.nunique().gt(1)
      rows.append(metric("probes", f"{target}/varying_levels", split, varying.mean(), len(varying), group=group))
      predictions.append(pd.DataFrame({"split": split, "target": target, "kind": "zero_shot" if split == "unseen" else "probe", "state": states[split][evaluated], "label": truth, "prediction": guess(probe, evaluated, split)}))

  for entry in skipped:
    print(f"probes: {entry['target']} skipped ({entry['reason']})")
  details = {"predictions": pd.concat(predictions, ignore_index=True).astype({"label": float, "prediction": float}) if predictions else pd.DataFrame(), "weights": weights, "skipped": pd.DataFrame(skipped, columns=["target", "reason"])}
  return rows, details


def experiment_aliasing(evaluation: Evaluation, split: str):
  """
  State aliasing: do states that are close in latent space react to actions the same way? (group encoder)

  GETS:    evaluation; split.
  DOES:    for every state its k = --knn nearest neighbours in the latent space (Euclidean), once within its level and
           once among the other levels of the split. A neighbour with the very same grid is the same situation (e.g. a
           state visited by two episodes), not an alias, and is left out. A_k(a) = share of (state, neighbour) pairs
           whose effect of action a differs (effect = replay_level's relative columns dx, dy, d_shape, d_color,
           d_rotation, refill_taken, life_lost, completed, so states of different levels compare). Baseline: k random
           members of the same pool. Within a level the pool is that level's sampled states (--states-per-level).
  RETURNS: (rows, details) -- metrics mismatch_a<id> and mismatch (mean over the actions) with subset = pool
           (within_level, cross_level, within_level_random, cross_level_random); details: neighbour indices (-1 = none)
           and distances per pool.
  """
  evaluation_set, args = evaluation.sets[split], evaluation.args
  count, k = len(evaluation_set.states), args.knn
  if count < 2:
    return [], {}
  effect_columns = ["dx", "dy", "d_shape", "d_color", "d_rotation", "refill_taken", "life_lost", "completed"]
  effects = evaluation_set.outcomes.sort_values(["state", "action"])[effect_columns].to_numpy().reshape(count, len(ACTIONS), len(effect_columns))  # y(s, a)  [N, 4, 8]
  levels = evaluation_set.states.level.to_numpy()
  grid_ids = np.unique(evaluation_set.arrays["grid"], axis=0, return_inverse=True)[1].reshape(-1)  # equal grids share an id
  latents = torch.as_tensor(evaluation.embeddings[split]["state"], device=args.device)
  level_ids, grid_ids_device = torch.as_tensor(levels, device=args.device), torch.as_tensor(grid_ids, device=args.device)
  rng = np.random.default_rng([args.seed, SPLITS.index(split), 2])

  neighbours = {}  # pool -> (index [N, k], distance [N, k]); -1 / inf where a pool has fewer than k members
  for pool in ("within_level", "cross_level"):
    indices, distances = [], []
    for start in range(0, count, 1024):  # 1024 queries at a time: the full [N, N] distance matrix can be gigabytes
      distance = torch.cdist(latents[start:start + 1024], latents)  # [b, N]
      same_level = level_ids[start:start + 1024, None] == level_ids[None, :]
      excluded = (grid_ids_device[start:start + 1024, None] == grid_ids_device[None, :]) | (~same_level if pool == "within_level" else same_level)  # the state itself, its copies, the other pool
      values, index = distance.masked_fill(excluded, float("inf")).topk(min(k, count), dim=1, largest=False)
      indices.append(index.masked_fill(torch.isinf(values), -1).cpu())
      distances.append(values.cpu())
    neighbours[pool] = (torch.cat(indices).numpy(), torch.cat(distances).numpy())
    random_index = np.full((count, min(k, count)), -1)
    for state in range(count):  # baseline: random members of the same pool
      members = np.flatnonzero(((levels == levels[state]) if pool == "within_level" else (levels != levels[state])) & (grid_ids != grid_ids[state]))
      chosen = rng.choice(members, min(k, len(members)), replace=False) if len(members) else []
      random_index[state, :len(chosen)] = chosen
    neighbours[f"{pool}_random"] = (random_index, np.full(random_index.shape, np.nan))

  rows = []
  for pool, (index, _) in neighbours.items():
    query, neighbour = np.nonzero(index >= 0)[0], index[index >= 0]  # one entry per (state, neighbour) pair
    mismatch = (effects[query] != effects[neighbour]).any(axis=-1)  # [pairs, 4]: the action has a different effect
    for position, action in enumerate(ACTIONS):
      rows += summarize("aliasing", f"mismatch_a{action}", split, mismatch[:, position], levels[query], pool)
    rows += summarize("aliasing", "mismatch", split, mismatch.mean(axis=1), levels[query], pool)
  return rows, {"neighbours": {f"{pool}_{kind}": array for pool, (index, distance) in neighbours.items() for kind, array in (("index", index), ("distance", distance))}}


def experiment_projections(evaluation: Evaluation, split: str):
  """
  PCA and UMAP of the states of all splits, for plots. (group encoder)

  GETS:    evaluation; split -- "all".
  DOES:    over the states of every split together, and again within each of --projection-levels: PCA (the first 3
           components as coordinates; explained variance of the first 10) and UMAP in 2-d and 3-d (15 neighbours,
           PCA start: the default spectral start fails on many identical states; seeded). Every coordinate row keeps
           the state's identifiers (split, state, dataset, episode, episode_idx, step) and the colouring labels
           (level, player_x, player_y, goal_distance, key_matches_goal). PCA and UMAP are for looking, not measuring:
           UMAP distorts distances and cluster sizes.
  RETURNS: (rows, details) -- metric explained_variance with subset pc<i> (level "all" or the projection level);
           details: coordinates (all states) and coordinates_level<L>.
  """
  import umap
  from sklearn.decomposition import PCA

  columns = ["state", "dataset", "episode", "episode_idx", "step", "level", "label_player_x", "label_player_y", "label_goal_distance", "label_key_matches_goal"]
  table = pd.concat([evaluation.sets[name].states[columns].assign(split=name) for name in evaluation.sets], ignore_index=True)
  latents = np.concatenate([evaluation.embeddings[name]["state"] for name in evaluation.sets])
  rows, details = [], {}
  for level in [None, *evaluation.args.projection_levels]:  # None = all states
    selected = np.ones(len(table), bool) if level is None else (table.level == level).to_numpy()
    if selected.sum() < 20:  # too few points for UMAP's 15 neighbours
      print(f"projections: level {level} has {selected.sum()} states, skipped")
      continue
    pca = PCA(n_components=min(10, *latents[selected].shape)).fit(latents[selected])
    coordinates = table[selected].reset_index(drop=True)
    coordinates[["pca_1", "pca_2", "pca_3"]] = pca.transform(latents[selected])[:, :3]
    for dimensions in (2, 3):
      reducer = umap.UMAP(n_components=dimensions, n_neighbors=15, min_dist=0.1, init="pca", random_state=evaluation.args.seed)
      coordinates[[f"umap{dimensions}_{axis + 1}" for axis in range(dimensions)]] = reducer.fit_transform(latents[selected])
    rows += [metric("projections", "explained_variance", "all", ratio, selected.sum(), level="all" if level is None else level, subset=f"pc{component + 1}") for component, ratio in enumerate(pca.explained_variance_ratio_)]
    details["coordinates" if level is None else f"coordinates_level{level}"] = coordinates
  return rows, details


def experiment_cka(evaluation: Evaluation, split: str):
  """
  Linear CKA between this checkpoint and --compare on identical states. (group encoder; only with --compare)

  GETS:    evaluation; split.
  DOES:    linear CKA of the two [N, D] latent matrices of the same ordered states (centred): ||Y^T X||_F^2 /
           (||X^T X||_F ||Y^T Y||_F), globally and per level (levels with at least 3 states). 1 = the same
           representation up to rotation and scale; it measures similarity, not quality.
  RETURNS: (rows, details) -- metric cka.
  """
  x, y = (embeddings[split]["state"].astype(np.float64) for embeddings in (evaluation.embeddings, evaluation.compare[2]))
  levels = evaluation.sets[split].states.level.to_numpy()

  def linear_cka(a, b):
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    norm = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)  # Frobenius norms
    return np.linalg.norm(b.T @ a) ** 2 / norm if norm > 0 else np.nan

  if len(x) < 3:
    return [], {}
  rows = [metric("cka", "cka", split, linear_cka(x, y), len(x))]
  rows += [metric("cka", "cka", split, linear_cka(x[levels == level], y[levels == level]), (levels == level).sum(), level=int(level)) for level in np.unique(levels) if (levels == level).sum() >= 3]
  return rows, {}


def experiment_one_step(evaluation: Evaluation, split: str):
  """
  One-step prediction from one real frame, as the planner's first step. (group world_model)

  GETS:    evaluation; split.
  DOES:    for every state with a recorded next state: z_hat = predictor(z_t, a_t); L_1step = mean ||z_hat - z_t+1||^2 / D,
           L_copy = mean ||z_t - z_t+1||^2 / D, R = L_1step / L_copy (NaN where L_copy = 0); for all transitions, the
           meaningfully changed ones (next_changed) and the unchanged ones; globally and per level. Raw errors depend on
           each model's latent scale: compare them between splits of one checkpoint, R between checkpoints.
  RETURNS: (rows, details) -- metrics l1step, lcopy, ratio per subset; details: per-state errors.
  """
  states, latents = evaluation.sets[split].states, evaluation.embeddings[split]
  selected = states.has_next.to_numpy() if len(states) else np.zeros(0, bool)
  if not selected.any():
    return [], {}
  current, following = latents["state"][selected], latents["next"][selected]  # z_t, z_t+1  [M, D]
  predicted = predict_next(evaluation, current, states.action.to_numpy()[selected])  # z_hat_t+1  [M, D]
  error = ((predicted - following) ** 2).mean(axis=1)  # ||z_hat - z_t+1||^2 / D per transition
  copy_error = ((current - following) ** 2).mean(axis=1)  # ||z_t - z_t+1||^2 / D: the baseline "nothing changes"
  levels, changed = states.level.to_numpy()[selected], states.next_changed.to_numpy()[selected]
  rows = []
  for subset, mask in (("all", np.ones_like(changed)), ("changed", changed), ("unchanged", ~changed)):
    rows += summarize("one_step", "l1step", split, error[mask], levels[mask], subset)
    rows += summarize("one_step", "lcopy", split, copy_error[mask], levels[mask], subset)
    rows += summarize("one_step", "ratio", split, error[mask], levels[mask], subset, denominator=copy_error[mask])
  return rows, {"transitions": pd.DataFrame({"state": states.state.to_numpy()[selected], "level": levels, "changed": changed, "error": error, "copy_error": copy_error})}


def experiment_counterfactual(evaluation: Evaluation, split: str):
  """
  Counterfactual actions: every action from the same real state. (group predictor)

  GETS:    evaluation; split.
  DOES:    per state and action a: e(a) = ||z_hat^a - z^a||^2 / D (z^a = encoded real outcome of a, z_hat^a predicted
           from the one real frame z_s), and ratio = mean e(a) / mean ||z_s - z^a||^2 / D (relative to "nothing
           changes"). Per action pair (a, b): E_effect(a, b) = ||(z_hat^a - z_hat^b) - (z^a - z^b)||^2 / D. Pairs are
           split by the real outcome grids: identical (both actions lead to the same grid, so z^a = z^b and E_effect is
           how far apart the model predicts them) or different; for different pairs also effect_ratio = mean E_effect /
           mean ||z^a - z^b||^2 / D (1 = no better than predicting no difference) and the predicted and real distances.
  RETURNS: (rows, details) -- metrics error, ratio (subset a<id> / all), effect_error (identical / different),
           effect_ratio, pred_distance, real_distance (different); details: per (state, action) and per (state, pair).
  """
  evaluation_set, latents = evaluation.sets[split], evaluation.embeddings[split]
  count = len(evaluation_set.states)
  if not count:
    return [], {}
  current, outcomes = latents["state"], latents["outcome"]  # z_s [N, D], z^a [N, 4, D]
  predicted = predict_next(evaluation, np.repeat(current, len(ACTIONS), axis=0), np.tile(ACTIONS, count)).reshape(outcomes.shape)  # z_hat^a  [N, 4, D]
  error = ((predicted - outcomes) ** 2).mean(axis=-1)  # e(a)  [N, 4]
  copy_error = ((current[:, None] - outcomes) ** 2).mean(axis=-1)  # [N, 4]
  levels = evaluation_set.states.level.to_numpy()

  rows = []
  for position, action in enumerate(ACTIONS):
    rows += summarize("counterfactual", "error", split, error[:, position], levels, f"a{action}")
    rows += summarize("counterfactual", "ratio", split, error[:, position], levels, f"a{action}", denominator=copy_error[:, position])
  rows += summarize("counterfactual", "error", split, error.reshape(-1), np.repeat(levels, len(ACTIONS)))
  rows += summarize("counterfactual", "ratio", split, error.reshape(-1), np.repeat(levels, len(ACTIONS)), denominator=copy_error.reshape(-1))

  pairs = [(a, b) for a in range(len(ACTIONS)) for b in range(a + 1, len(ACTIONS))]  # 6 action pairs (positions in ACTIONS)
  first, second = np.array(pairs).T
  identical = (evaluation_set.arrays["outcome_grid"][:, first] == evaluation_set.arrays["outcome_grid"][:, second]).all(axis=-1)  # [N, 6]: the real game shows the same grid
  predicted_difference, real_difference = predicted[:, first] - predicted[:, second], outcomes[:, first] - outcomes[:, second]  # [N, 6, D]
  effect_error = ((predicted_difference - real_difference) ** 2).mean(axis=-1)  # E_effect  [N, 6]
  predicted_distance, real_distance = (predicted_difference ** 2).mean(axis=-1), (real_difference ** 2).mean(axis=-1)  # [N, 6]
  pair_levels = np.repeat(levels[:, None], len(pairs), axis=1)
  rows += summarize("counterfactual", "effect_error", split, effect_error[identical], pair_levels[identical], "identical")
  rows += summarize("counterfactual", "effect_error", split, effect_error[~identical], pair_levels[~identical], "different")
  rows += summarize("counterfactual", "effect_ratio", split, effect_error[~identical], pair_levels[~identical], "different", denominator=real_distance[~identical])
  rows += summarize("counterfactual", "pred_distance", split, predicted_distance[~identical], pair_levels[~identical], "different")
  rows += summarize("counterfactual", "real_distance", split, real_distance[~identical], pair_levels[~identical], "different")

  states = evaluation_set.states.state.to_numpy()
  per_action = pd.DataFrame({"state": np.repeat(states, len(ACTIONS)), "level": np.repeat(levels, len(ACTIONS)), "action": np.tile(ACTIONS, count), "error": error.reshape(-1), "copy_error": copy_error.reshape(-1),
                             "changed": evaluation_set.outcomes.sort_values(["state", "action"]).changed.to_numpy()})
  per_pair = pd.DataFrame({"state": np.repeat(states, len(pairs)), "level": pair_levels.reshape(-1), "action_a": np.tile(np.array(ACTIONS)[first], count), "action_b": np.tile(np.array(ACTIONS)[second], count),
                           "identical": identical.reshape(-1), "effect_error": effect_error.reshape(-1), "pred_distance": predicted_distance.reshape(-1), "real_distance": real_distance.reshape(-1)})
  return rows, {"actions": per_action, "pairs": per_pair}


def experiment_rollout(evaluation: Evaluation, split: str):
  """
  Autoregressive rollout from one real frame with the recorded actions. (group predictor)

  GETS:    evaluation; split.
  DOES:    per window: start at the real z_t, feed the predictions back (the predictor sees its last history_size latents,
           as JEPA.rollout); for k in ROLLOUT_HORIZONS: E_AR(k) = mean ||z_hat_t+k - z_t+k||^2 / D, the copy baseline
           mean ||z_t - z_t+k||^2 / D and their ratio. The same windows for every k, so the horizons compare.
  RETURNS: (rows, details) -- metrics error, copy_error, ratio with subset k<k>; details: per-window errors.
  """
  evaluation_set, frames = evaluation.sets[split], evaluation.embeddings[split]["window"]  # z_t .. z_t+H  [W, H + 1, D]
  if not len(frames):
    return [], {}
  device, size, encoded = evaluation.args.device, evaluation.args.batch_size, action_latents(evaluation)
  actions = evaluation_set.arrays["window_action"].astype(int)  # a_t .. a_t+H-1 recorded  [W, H]
  with torch.no_grad():
    predicted = torch.cat([evaluation.model.predict_rollout(torch.as_tensor(frames[start:start + size, :1], device=device), encoded[torch.as_tensor(actions[start:start + size] - 1, device=device)],
                                                            evaluation.info["history_size"]).float().cpu() for start in range(0, len(frames), size)]).numpy()  # z_hat_t+1 .. z_hat_t+H  [W, H, D]
  levels, rows, details = evaluation_set.windows.level.to_numpy(), [], []
  for k in ROLLOUT_HORIZONS:
    error = ((predicted[:, k - 1] - frames[:, k]) ** 2).mean(axis=1)  # ||z_hat_t+k - z_t+k||^2 / D per window
    copy_error = ((frames[:, 0] - frames[:, k]) ** 2).mean(axis=1)  # ||z_t - z_t+k||^2 / D: "nothing changes in k steps"
    rows += summarize("rollout", "error", split, error, levels, f"k{k}")
    rows += summarize("rollout", "copy_error", split, copy_error, levels, f"k{k}")
    rows += summarize("rollout", "ratio", split, error, levels, f"k{k}", denominator=copy_error)
    details.append(pd.DataFrame({"window": np.arange(len(frames)), "level": levels, "k": k, "error": error, "copy_error": copy_error}))
  return rows, {"windows": pd.concat(details, ignore_index=True)}


def experiment_planning(evaluation: Evaluation, split: str):
  """
  Planning with CEM in the real game. (group planning)

  GETS:    evaluation; split.
  DOES:    draws its own episodes, not from the evaluation set (planning settings never rebuild a set): at most
           --planning-levels of the split's levels with successful episodes (seeded), per level the first of its
           successful episodes, in a seeded order, whose first grid is the level start (a recording that starts
           elsewhere, e.g. a walk run, cannot be planned from the level start; at most 10 are tried), read from Lance at
           the split's version. The test split is never among the candidates. Goal reaching: for each g in --goal-steps shorter than
           the episode, the recorded state g steps after the start is the goal; CemPlanner (--samples, --iterations,
           --horizon, --topk, --replan-every, --cost-on) plays from the level start for at most 3 g actions; reached =
           no meaningful_change between the played grid and the goal (the energy bar is ignored). Waypoints: every
           --waypoint-every-th recorded state and the completion frame, in turn, at most 3 x waypoint-every actions
           each, giving up at the first one missed; completed = the level was completed. A game over ends an attempt.
  RETURNS: (rows, details) -- metrics success_rate and steps (subset goal<g> / all), reached_fraction and
           completion_rate (subset waypoints), per level too; details: attempts (one row per attempt, its actions and
           the solver's cost of every plan) and skipped levels.
  """
  from arcengine import GameAction, GameState

  from master_thesis.environments.arc_collection import reset_at_level
  from master_thesis.environments.arc_levelgen import make_game

  args = evaluation.args
  episodes, versions = evaluation.episodes[split]
  successful = episodes[episodes.success]
  levels = np.random.default_rng([args.seed, SPLITS.index(split), 3]).permutation(sorted(set(successful.level)))[:args.planning_levels]
  games, attempts, skipped = {}, [], []
  for level in sorted(int(level) for level in levels):
    candidates = successful[successful.level == level]
    episode, env = None, None
    for row in candidates.iloc[np.random.default_rng([args.seed, SPLITS.index(split), level, 3]).permutation(len(candidates))[:10]].itertuples(index=False):
      solution, _, game_ids = read_rows(row.dataset, versions[row.dataset], np.arange(row.offset, row.offset + row.length))  # the recorded run; its last grid completes the level
      if game_ids[0] not in games:
        games[game_ids[0]] = make_game(game_ids[0])
      env = games[game_ids[0]]
      if np.array_equal(solution[0], np.asarray(reset_at_level(env, level).frame[-1], dtype=np.uint8).reshape(-1)):
        episode = row
        break
    if episode is None:
      skipped.append({"level": level, "reason": "no successful episode of the split starts at the level start"})
      continue

    def follow(targets, budget):  # play from the level start towards each target in turn, at most `budget` actions per target
      observation = reset_at_level(env, level)
      grid = np.asarray(observation.frame[-1], dtype=np.uint8).reshape(-1)
      actions, costs, reached = [], [], 0
      for target in targets:
        planner, steps = CemPlanner(evaluation.model, evaluation.info["img_size"], target, args), 0
        while meaningful_change(grid, target) and steps < budget and observation.state not in (GameState.GAME_OVER, GameState.WIN):
          action = planner.act(grid)
          observation = env.step(GameAction.from_id(action))
          grid = np.asarray(observation.frame[-1], dtype=np.uint8).reshape(-1)
          actions.append(action)
          steps += 1
        costs += planner.costs
        if meaningful_change(grid, target):  # missed: give up
          break
        reached += 1
      return {"reached": reached, "steps": len(actions), "actions": actions, "costs": costs, "completed": bool(observation.levels_completed >= 1 or observation.state == GameState.WIN)}

    identity = {"level": int(level), "dataset": episode.dataset, "episode": int(episode.episode), "recorded_steps": int(episode.length - 1)}
    for goal_steps in (g for g in args.goal_steps if g < episode.length):
      played = follow([solution[goal_steps]], 3 * goal_steps)
      attempts.append({**identity, "kind": "goal", "goal_steps": goal_steps, "targets": 1, **played})
    waypoints = list(solution[args.waypoint_every::args.waypoint_every])
    if not waypoints or not np.array_equal(waypoints[-1], solution[-1]):
      waypoints.append(solution[-1])  # the completion frame is always the last waypoint
    attempts.append({**identity, "kind": "waypoints", "goal_steps": args.waypoint_every, "targets": len(waypoints), **follow(waypoints, 3 * args.waypoint_every)})

  if not attempts:
    return [], {"skipped": pd.DataFrame(skipped, columns=["level", "reason"])}
  attempts = pd.DataFrame(attempts)
  goals, followed = attempts[attempts.kind == "goal"], attempts[attempts.kind == "waypoints"]
  rows = summarize("planning", "success_rate", split, goals.reached, goals.level) + summarize("planning", "steps", split, goals.steps, goals.level)
  for goal_steps, attempt in goals.groupby("goal_steps"):
    rows += summarize("planning", "success_rate", split, attempt.reached, attempt.level, f"goal{goal_steps}") + summarize("planning", "steps", split, attempt.steps, attempt.level, f"goal{goal_steps}")
  rows += summarize("planning", "reached_fraction", split, followed.reached / followed.targets, followed.level, "waypoints")
  rows += summarize("planning", "completion_rate", split, followed.completed, followed.level, "waypoints")
  return rows, {"attempts": attempts, "skipped": pd.DataFrame(skipped, columns=["level", "reason"])}


def experiment_ranking(evaluation: Evaluation, split: str):
  """
  Action ranking towards a goal: does the predictor order the actions like the real outcomes do? (group planning)

  GETS:    evaluation; split.
  DOES:    for states with a recorded state --goal-ahead steps later in their episode (the goal g): C_pred(a) =
           ||z_hat^a - z_g||^2, C_real(a) = ||z^a - z_g||^2; spearman over the 4 actions per state (ties share their
           mean rank; states whose real costs are all equal have no order and are left out), averaged; action spread
           = std over actions of C_real, averaged (encoder only: ~0 means every move looks alike to the planner).
  RETURNS: (rows, details) -- metrics spearman, action_spread; details: per-state costs.
  """
  evaluation_set, latents = evaluation.sets[split], evaluation.embeddings[split]
  selected = evaluation_set.states.has_goal.to_numpy() if len(evaluation_set.states) else np.zeros(0, bool)
  if not selected.any():
    return [], {}
  count = int(selected.sum())
  goals = latents["goal"][selected][:, None]  # z_g  [M, 1, D]
  predicted = predict_next(evaluation, np.repeat(latents["state"][selected], len(ACTIONS), axis=0), np.tile(ACTIONS, count)).reshape(count, len(ACTIONS), -1)  # z_hat^a  [M, 4, D]
  predicted_cost, real_cost = ((predicted - goals) ** 2).sum(axis=-1), ((latents["outcome"][selected] - goals) ** 2).sum(axis=-1)  # C_pred, C_real  [M, 4]

  rank = lambda cost: (cost[:, None, :] < cost[:, :, None]).sum(axis=-1) + ((cost[:, None, :] == cost[:, :, None]).sum(axis=-1) + 1) / 2  # rank 1..4 per state; ties share their mean rank (two blocked moves: 1, 2.5, 2.5, 4)
  predicted_rank, real_rank = (r - r.mean(axis=1, keepdims=True) for r in (rank(predicted_cost), rank(real_cost)))  # centred ranks  [M, 4]
  ordered = (real_rank ** 2).sum(axis=1) > 0  # states whose real costs are not all equal; the others have no order to predict
  spearman = np.where(ordered, (predicted_rank * real_rank).sum(axis=1) / np.sqrt(np.maximum((predicted_rank ** 2).sum(axis=1) * (real_rank ** 2).sum(axis=1), 1e-12)), np.nan)  # all predictions equal -> 0
  spread = real_cost.std(axis=1)  # population std over the 4 actions: how much one move changes the real distance to the goal

  levels = evaluation_set.states.level.to_numpy()[selected]
  rows = summarize("ranking", "spearman", split, spearman, levels) + summarize("ranking", "action_spread", split, spread, levels)
  costs = pd.DataFrame({"state": evaluation_set.states.state.to_numpy()[selected], "level": levels, "spearman": spearman, "action_spread": spread,
                        **{f"pred_cost_a{action}": predicted_cost[:, position] for position, action in enumerate(ACTIONS)}, **{f"real_cost_a{action}": real_cost[:, position] for position, action in enumerate(ACTIONS)}})
  return rows, {"states": costs}


def experiment_generalization(evaluation: Evaluation, split: str):
  """
  Seen -> unseen comparison of this checkpoint's saved results. (group generalization)

  GETS:    evaluation; split -- "all".
  DOES:    reads every saved metrics table of the checkpoint and, per (experiment, metric, subset) over all levels,
           reports seen_val - seen_train (overfitting) and unseen - seen_val (transfer). A probe's zero-shot score on
           unseen is compared with its score on the seen splits. Only differences within one checkpoint (one encoder,
           so even raw latent errors compare); comparing checkpoints is left to the dashboard. Recomputed on every
           run, since it depends on the other results.
  RETURNS: (rows, details) -- metrics <experiment>/<metric>/<subset> with subset val_minus_train / unseen_minus_val;
           n = the smaller n of the two.
  """
  folder = evaluation.args.output / "results" / evaluation.info["id"]
  tables = [pd.read_parquet(path) for path in sorted(folder.glob("*/*.parquet")) if path.stem in (*SPLITS, "all") and path.parent.name != "generalization"]  # the metric tables (details are <split>_<name>.parquet)
  if not tables:
    return [], {}
  table = pd.concat(tables, ignore_index=True)
  table = table[(table.level == "all") & table.split.isin(SPLITS)]
  table.loc[(table.experiment == "probes") & (table.subset == "zero_shot"), "subset"] = "probe"  # a seen-level probe on unseen states is its zero-shot score
  pivot = table.pivot_table(index=["experiment", "metric", "subset"], columns="split", values=["value", "n"], aggfunc="first")
  rows = []
  for (experiment, name, subset), entry in pivot.iterrows():
    for later, earlier, label in (("seen_val", "seen_train", "val_minus_train"), ("unseen", "seen_val", "unseen_minus_val")):
      if not (pd.isna(entry.get(("value", later))) or pd.isna(entry.get(("value", earlier)))):
        rows.append(metric("generalization", f"{experiment}/{name}/{subset}", "all", entry[("value", later)] - entry[("value", earlier)], min(entry[("n", later)], entry[("n", earlier)]), subset=label))
  return rows, {}


# name -> (function, scope): "each" = run once per split, "all" = once over all splits (results saved under split "all")
EXPERIMENTS = {"variance": (experiment_variance, "each"), "probes": (experiment_probes, "all"), "aliasing": (experiment_aliasing, "each"),
               "projections": (experiment_projections, "all"), "cka": (experiment_cka, "each"), "one_step": (experiment_one_step, "each"),
               "counterfactual": (experiment_counterfactual, "each"), "rollout": (experiment_rollout, "each"), "planning": (experiment_planning, "each"),
               "ranking": (experiment_ranking, "each"), "generalization": (experiment_generalization, "all")}


##########
## Main ##
##########


def main() -> None:
  parser = argparse.ArgumentParser(description="Evaluate a trained LeWM on ls20: representation, dynamics, planning, generalization")
  parser.add_argument("--run", required=True, help="model folder under models/lewm, optionally :<weights file>, e.g. <run>:checkpoints/weights_step100000.pt")
  parser.add_argument("--weights", default=None, help="weights file inside the run (instead of the :<file> suffix of --run)")
  parser.add_argument("--compare", default=None, help="a second checkpoint (<run>[:<weights>]), only for cka")
  parser.add_argument("--unseen", nargs="+", default=[], help="unseen levels: DATASET or DATASET:LEVEL,LEVEL, e.g. human_l1-7:3 syn1000_dev")
  parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
  parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=None, help="default: all (or those of --groups)")
  parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=None, help="run the experiments of these dashboard groups")
  parser.add_argument("--seed", type=int, default=0, help="sampling seed (states, windows, planning episodes, unseen adapt/eval)")
  parser.add_argument("--episodes-per-level", type=int, default=10)
  parser.add_argument("--states-per-episode", type=int, default=3)
  parser.add_argument("--states-per-level", type=int, default=30)
  parser.add_argument("--windows-per-episode", type=int, default=2)
  parser.add_argument("--windows-per-level", type=int, default=20)
  parser.add_argument("--max-levels", type=int, default=None, help="evaluate at most this many levels per split (seeded choice)")
  parser.add_argument("--planning-levels", type=int, default=5, help="levels per split with a planning episode")
  parser.add_argument("--goal-steps", type=int, nargs="+", default=[5, 10, 25], help="planning: goals this many steps after the episode start")
  parser.add_argument("--waypoint-every", type=int, default=5, help="planning: every how many recorded states a waypoint is placed")
  parser.add_argument("--goal-ahead", type=int, default=5, help="ranking: the goal is the recorded state this many steps later")
  parser.add_argument("--samples", type=int, default=300, help="CEM: action sequences per round")
  parser.add_argument("--iterations", type=int, default=10, help="CEM: rounds per plan")
  parser.add_argument("--horizon", type=int, default=5, help="CEM: planning horizon")
  parser.add_argument("--topk", type=int, default=30, help="CEM: elite samples")
  parser.add_argument("--replan-every", type=int, default=5, help="CEM: actions played per plan")
  parser.add_argument("--cost-on", choices=["path", "end"], default="path", help="CEM: a plan counts by its closest imagined approach to the goal (path) or its last imagined state (end)")
  parser.add_argument("--probe-steps", type=int, default=2000, help="probes: optimizer steps (batches of 256) on seen_train")
  parser.add_argument("--probe-adapt-steps", type=int, default=300, help="probes: fine-tuning steps on the unseen adapt part")
  parser.add_argument("--knn", type=int, default=10, help="aliasing: neighbours per state")
  parser.add_argument("--projection-levels", type=int, nargs="*", default=[], help="projections: levels that also get their own PCA/UMAP")
  parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
  parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)), help="processes replaying episodes in the engine")
  parser.add_argument("--batch-size", type=int, default=256, help="grids per encoder pass")
  parser.add_argument("--output", type=Path, default=None, help="default: $STABLEWM_HOME/evaluation/ls20/lewm_analysis")
  parser.add_argument("--force", action="store_true", help="recompute results that exist with the same inputs")
  args = parser.parse_args()
  if args.topk > args.samples:
    parser.error("--topk (CEM elites) cannot exceed --samples")

  if args.weights:
    args.run = f"{args.run.partition(':')[0]}:{args.weights}"
  if args.device == "auto":
    args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  args.output = args.output or evaluation_dir(GAME) / "lewm_analysis"
  experiments = args.experiments or ([name for group in args.groups for name in GROUPS[group]] if args.groups else list(EXPERIMENTS))
  experiments = list(dict.fromkeys(experiments))  # unique, in order
  splits = [split for split in args.splits if split != "unseen" or args.unseen]

  model, info = load_checkpoint(args.run, args.device)
  print(f"checkpoint {info['id']} on {args.device}: trained on {info['dataset']} (window {info['window']}, seed {info['seed']})")

  # evaluation sets: built once per split definition, shared by every checkpoint with the same split
  seen, version, training_levels = training_split(info)
  episodes, sets = {}, {}
  for split in splits:
    episodes[split] = unseen_episodes(args.unseen, training_levels, args.seed) if split == "unseen" else (seen[seen.part == split.removeprefix("seen_")], {info["dataset"]: version})
    sets[split] = load_set(split, build_set(split, *episodes[split], args))
    print(f"{split}: {sets[split].meta['counts']} -> {sets[split].path.name}")

  embeddings = {split: embed_set(model, info, evaluation_set, args) for split, evaluation_set in sets.items()}
  compare = None
  if args.compare:
    other_model, other_info = load_checkpoint(args.compare, args.device)
    compare = (other_model, other_info, {split: embed_set(other_model, other_info, evaluation_set, args) for split, evaluation_set in sets.items()})
  evaluation = Evaluation(args=args, model=model, info=info, episodes=episodes, sets=sets, embeddings=embeddings, compare=compare)

  results = args.output / "results" / info["id"]
  results.mkdir(parents=True, exist_ok=True)
  (results / "checkpoint.json").write_text(json.dumps({**info, "training_levels": sorted(training_levels), "dataset_version": version}, indent=2), encoding="utf-8")
  settings = {key: value for key, value in vars(args).items() if key not in ("run", "weights", "compare", "splits", "experiments", "groups", "device", "workers", "batch_size", "output", "force")}

  for name in experiments:
    function, scope = EXPERIMENTS[name]
    if name == "cka" and compare is None:
      print("cka: skipped (needs --compare)")
      continue
    for split in (splits if scope == "each" else ["all"]):
      inputs = {"checkpoint": info["sha256"], "compare": compare[1]["sha256"] if compare and name == "cka" else None, "sets": {s: sets[s].path.name for s in (splits if split == "all" else [split])}, "settings": settings}
      target = results / name / split
      if name != "generalization" and not args.force and target.with_suffix(".json").exists() and json.loads(target.with_suffix(".json").read_text(encoding="utf-8")) == inputs:  # generalization is cheap and depends on the other results: always redone
        print(f"{name} / {split}: up to date")
        continue
      rows, details = function(evaluation, split)
      target.parent.mkdir(parents=True, exist_ok=True)
      pd.DataFrame(rows).to_parquet(target.with_suffix(".parquet"))
      for detail, value in details.items():  # per-state tables as parquet, arrays as npz
        value.to_parquet(target.parent / f"{split}_{detail}.parquet") if isinstance(value, pd.DataFrame) else np.savez(target.parent / f"{split}_{detail}.npz", **value)
      target.with_suffix(".json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")  # last: marks the result as complete
      print(f"{name} / {split}: {len(rows)} metrics")


if __name__ == "__main__":
  main()
