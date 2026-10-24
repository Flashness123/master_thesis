"""
Training data for the synthetic ls20 levels (made by arc_levelgen): each level's optimal solution, and one-life runs
that start anywhere in the level. The real levels 1-4 work the same way: their spec is read from the game file and
solved here (5-7 have changers moving on tracks, which arc_solver cannot model).

One call creates one collection, structured like the real recordings so arc_recording converts it unchanged:
$STABLEWM_HOME/synthetic_data/<game>/
└── <name>_<MMDD-HHMM>/                  one collection (name defaults to <version>_walk-reachable_<levels>)
    ├── collection.json                  settings of this call
    ├── level_8/
    │   ├── solution.jsonl               the level's optimal solution (BFS, certified), recorded from the start
    │   ├── walk_seed80042.jsonl         one life from a situation no earlier run of the level visited; seed = seed + level * 10000 + run
    │   └── ...
    └── level_37/

Where runs start: the solver (arc_solver.reachable) lists every situation one life can reach in the level (a cell, a
symbol and the refills taken: the frame apart from energy and lives), each with the path that gets there with the most
energy left. Runs can start in those with at least MIN_MOVES_LEFT moves of energy left (the start situations). Each run
draws one uniformly among the start situations that no earlier run of the level has visited, replays (not records) its
path from the level start, and the walk plays from there. So every run begins where the data has nothing yet, and the
runs cover the level instead of revisiting its well-known parts (most situations lie deep in the level, where runs that
branch off the solution rarely get).
How many runs: a level is done when every start situation has been visited (or after max_runs). The number of runs
therefore grows with the level's size by itself: simulated on syn30's 30 levels, 30-722 runs (median 111, about one
run per 4-5 situations), covering a median 94 % of the reachable situations (min 73 %: the rest can only be reached with
almost no energy left, so no run starts there and walks rarely arrive). A fixed number per level would waste runs on
small levels and leave large ones thin (60 runs covered 16-45 % of the largest).
Each run is one life on its level: it ends at the first life loss (the level would restart at its start, and with all
3 lives most of a run were such replays of the start) or when the level is completed.
Policy: a SolverWalk per run (arc_solver_walk), which moves uniformly among the moves that change something and makes
any move with probability 0.1. It replaced Goose here: in ls20 Goose wastes a third of its moves on walls, and once
trained not to, its choices become one-sided and it explores less (see arc_solver_walk). The walk also reports the
situations it was in, which decides where the next runs start.
The paths are played by the real engine, so every recorded state is real. A run raises if its replay does not end with
the energy the solver predicts and all 3 lives, i.e. if the solver and the engine disagree.

Usage:
  python -m master_thesis.environments.arc_synthetic_collection ls20-syn30 --max-runs 1000 --max-steps 500 --seed 42
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from master_thesis.environments.arc_collection import SEED_LEVEL_OFFSET, collect_game, levels_tag, record_frame, reset_at_level
from master_thesis.environments.arc_levelgen import NUM_REAL_LEVELS, load_game_module, make_game, source_game_file, spec_from_level
from master_thesis.environments.arc_solver import Goal, LevelSpec, reachable, solve
from master_thesis.paths import synthetic_dir, timestamp
from master_thesis.policies.arc_solver_walk import SolverWalk

MIN_MOVES_LEFT = 5  # a run starts only where at least 5 moves of energy are left: training windows had 6 states (1 + wm.num_preds) when this was set (now 8 with the rollout loss), so a run that ends sooner yields none


def lives(frame):
  """Lives left (3 at the start) in an ls20 frame [64, 64]: the HUD shows one pair of red pixels per life, right of the energy bar."""
  return int((np.asarray(frame)[61, 56:64] == 8).sum()) // 2


def load_levels(game_id):
  """
  The spec and optimal solution of every synthetic level of a game version.

  GETS:    game_id -- a version made by arc_levelgen, e.g. "ls20-syn30".
  DOES:    reads the version's levels.json and turns each spec back into a LevelSpec (arc_levelgen.main stores cells
           as JSON lists, sets as sorted lists and platforms as [cell, direction] pairs; the order of the refills is
           kept, since the solver's refill bitmask indexes them).
  RETURNS: {level number: (LevelSpec, solution as a list of action ids 1..4)}.
  """
  records = json.loads((source_game_file(game_id).parent / "levels.json").read_text(encoding="utf-8"))["levels"]
  cells = lambda values: {tuple(value) for value in values}
  levels = {}
  for level, record in records.items():
    spec = record["spec"]
    levels[int(level)] = (LevelSpec(**{**spec, "walls": cells(spec["walls"]), "start": tuple(spec["start"]), "refills": [tuple(cell) for cell in spec["refills"]],
                                       "goals": [Goal(tuple(goal["cell"]), goal["shape"], goal["color"], goal["rotation"]) for goal in spec["goals"]],
                                       "rotators": cells(spec["rotators"]), "colorers": cells(spec["colorers"]), "shapers": cells(spec["shapers"]),
                                       "platforms": {tuple(cell): direction for cell, direction in spec["platforms"]}}), record["solution"])
  return levels


def record_solution(env, level, actions, output_path):
  """
  Play a level's solution in the real engine and record it as one JSONL episode, like the human recordings.

  GETS:    env -- the game (arc_levelgen.make_game); level -- 1-based game level; actions -- the solution;
           output_path -- the .jsonl file to create.
  DOES:    resets to the level, records the start frame, then every frame an action produces. The last recorded frame
           is the completion frame, which already shows the next level (arc_recording keeps it in this episode).
  RETURNS: output_path. Raises if the solution does not complete the level.
  """
  from arcengine import GameAction

  observation = reset_at_level(env, level)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("x", encoding="utf-8") as file:  # "x": never overwrite a recording
    record_frame(file, observation, start_level=level, extra={"source": "solution"})
    for action in actions:
      observation = env.step(GameAction.from_id(action))
      record_frame(file, observation, start_level=level)
  if observation.levels_completed < 1:
    raise RuntimeError(f"level {level}: the recorded solution does not complete the level")
  return output_path


def collect_synthetic(game_id, max_runs, max_steps, seed, levels=None, name=None):
  """
  Record the solution and one-life walk runs for every synthetic level until each level is covered, into one new collection folder.

  GETS:    game_id -- e.g. "ls20-syn30"; max_runs -- most runs per level; max_steps -- walk actions per run;
           seed -- base seed (run seed = seed + level * SEED_LEVEL_OFFSET + run index, as in arc_collection);
           levels -- level numbers to collect (default: all synthetic levels; the real levels 1-4 can be added);
           name -- collection name without timestamp.
  DOES:    per level: solution.jsonl, then runs that each start in a start situation no earlier run of the level has
           visited (see the module docstring) and end at their first life loss or level completion, until every start
           situation has been visited or max_runs is reached. All recordings share one game (arc_levelgen.make_game).
  RETURNS: the collection folder.
  """
  synthetic_levels = load_levels(game_id)
  levels = sorted(synthetic_levels) if levels is None else levels
  real_levels = load_game_module(source_game_file(game_id)).levels[:NUM_REAL_LEVELS] if min(levels) <= NUM_REAL_LEVELS else []  # levels.json has only the synthetic levels: the real ones come from the game file
  version = game_id.split("-", 1)[1]
  collection_dir = synthetic_dir(game_id) / f"{name or f'{version}_walk-reachable_{levels_tag(levels)}'}_{timestamp()}"
  collection_dir.mkdir(parents=True, exist_ok=False)  # never mix two collections
  settings = {"game_id": game_id, "levels": levels, "policy": "SolverWalk per run, seeded with the run seed: uniform over the moves that change something, any move with probability 0.1", "max_runs_per_level": max_runs, "max_steps": max_steps, "seed": seed,
              "seed_rule": f"seed + level * {SEED_LEVEL_OFFSET} + run_index", "start": "uniform over the start situations (reachable, >= min_moves_left moves of energy left) no earlier run of the level has visited, via the path with the most energy left",
              "min_moves_left": MIN_MOVES_LEFT, "level_end": "every start situation visited, or max_runs_per_level", "run_end": "first life loss, level completion, game over or max_steps"}
  (collection_dir / "collection.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

  env = make_game(game_id)  # one game for all recordings: making one per run would copy all levels of the game file every time
  for level in levels:
    if level <= NUM_REAL_LEVELS:
      spec = spec_from_level(real_levels[level - 1], level)  # raises for levels 5-7 (moving changers)
      solution = solve(spec)  # record_solution below checks it in the real engine, as certify does for the synthetic levels
    else:
      spec, solution = synthetic_levels[level]
    level_dir = collection_dir / f"level_{level}"
    record_solution(env, level, solution, level_dir / "solution.jsonl")
    situations = reachable(spec)
    starts = {situation: (energy, path) for situation, (energy, path) in situations.items() if energy // spec.steps_decrement >= MIN_MOVES_LEFT}  # where runs can start
    visited = set()  # every situation a run of this level has been in (SolverWalk.visited)
    runs = 0
    for run_index in range(max_runs):
      unvisited = [situation for situation in starts if situation not in visited]  # in BFS order, so the draw below is reproducible
      if not unvisited:  # the level is covered: runs from here on would only start where the data already is
        break
      run_seed = seed + level * SEED_LEVEL_OFFSET + run_index
      situation = unvisited[int(np.random.default_rng(run_seed).integers(len(unvisited)))]  # uniform over the start situations nobody has visited yet
      energy, path = starts[situation]
      branch = {"source": "reachable", "cell": list(situation[0]), "path_length": len(path), "energy": energy, "prefix": path}  # stored on the first recorded line, like arc_collection's branches; prefix = the replayed path, so the run's start can be rebuilt in the engine (evaluation/lewm_analysis)
      print(f"\n--- level {level}, run {run_index + 1} (seed={run_seed}, start at cell {situation[0]}: {len(path)} moves from the level start, energy {energy}/{spec.step_counter}; {len(unvisited)} of {len(starts)} start situations unvisited) ---")
      walk = SolverWalk(spec, (*situation[:4], energy, *situation[4:]), seed=run_seed)  # starts at the situation the path leads to
      output_path = collect_game(game_id=game_id, max_steps=max_steps, seed=run_seed, mode="offline", output_path=level_dir / f"walk_seed{run_seed}.jsonl", policy_name="walk", start_level=level,
                                 prefix=path, branch=branch, env=env, stop=lambda observation: lives(observation.frame[-1]) < 3 or observation.levels_completed >= 1, policy=walk)  # one life on this level
      start = np.asarray(json.loads(output_path.open(encoding="utf-8").readline())["data"]["frame"][-1])  # the recorded state the path led to
      if int((start[61, 13:55] == 11).sum()) != energy or lives(start) != 3:  # the energy bar has one yellow pixel per unit of energy (42 at the start)
        raise RuntimeError(f"level {level}, run {run_index + 1}: the replayed path did not end with energy {energy} and 3 lives, so the solver and the engine disagree")
      visited |= walk.visited  # the start situation and everything the walk reached from it
      runs += 1
    print(f"level {level}: {runs} runs, {len(visited)} of {len(situations)} reachable situations visited")

  print(f"\nCollection complete: {len(levels)} levels x (1 solution + up to {max_runs} one-life walk runs) in {collection_dir}")
  return collection_dir


def main():
  parser = argparse.ArgumentParser(description="Record solutions and one-life walk runs for synthetic ls20 levels, until each level is covered")
  parser.add_argument("game", help="a version made by arc_levelgen, e.g. ls20-syn30")
  parser.add_argument("--max-runs", type=int, default=1000, help="most runs per level; a level ends earlier once every start situation has been visited (syn30, simulated: 30-722 runs)")
  parser.add_argument("--max-steps", type=int, default=500, help="maximum walk actions per run")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--levels", type=int, nargs="+", default=None, help="levels to collect (default: all synthetic levels); the real levels 1-4 work too, e.g. --levels 1 2 4")
  parser.add_argument("--name", default=None, help="collection name without timestamp; default <version>_walk-reachable_<levels>")
  args = parser.parse_args()
  if args.max_runs < 1 or args.max_steps < 1:
    parser.error("--max-runs and --max-steps must be positive")
  collect_synthetic(args.game, max_runs=args.max_runs, max_steps=args.max_steps, seed=args.seed, levels=args.levels, name=args.name)


if __name__ == "__main__":
  main()
