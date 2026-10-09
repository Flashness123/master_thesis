"""
Training data for the synthetic ls20 levels (made by arc_levelgen): the optimal solutions and Goose runs that
branch off them.

One call creates one collection, structured like the real recordings so arc_recording converts it unchanged:
$STABLEWM_HOME/synthetic_data/<game>/
└── <name>_<MMDD-HHMM>/                  one collection (name defaults to <version>_goose-branch_<levels>)
    ├── collection.json                  settings of this call
    ├── level_8/
    │   ├── solution.jsonl               the level's optimal solution (BFS, certified), recorded from the start
    │   ├── goose_seed80042.jsonl        Goose from a random point along the solution; seed = seed + level * 10000 + run
    │   └── ...
    └── level_37/

Branching: the solution is replayed (not recorded) up to a step drawn uniformly from 0 .. len - 1, then Goose
plays from there. Step 0 is a plain Goose run from the level start; the completing move is never replayed.
This is arc_collection's --branch-from, with the levels' solutions instead of human episodes as the source.

Usage:
  python -m master_thesis.environments.arc_synthetic_collection ls20-syn30 --runs 20 --max-steps 500 --seed 42
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from master_thesis.environments.arc_collection import SEED_LEVEL_OFFSET, collect_game, levels_tag, record_frame, reset_at_level
from master_thesis.environments.arc_levelgen import source_game_file
from master_thesis.paths import arc_environments_dir, synthetic_dir, timestamp


def load_solutions(game_id):
  """
  The optimal solution of every synthetic level of a game version.

  GETS:    game_id -- a version made by arc_levelgen, e.g. "ls20-syn30".
  RETURNS: {level number: list of action ids 1..4}, read from the version's levels.json.
  """
  records = json.loads((source_game_file(game_id).parent / "levels.json").read_text(encoding="utf-8"))["levels"]
  return {int(level): record["solution"] for level, record in records.items()}


def record_solution(game_id, level, actions, output_path):
  """
  Play a level's solution in the real engine and record it as one JSONL episode, like the human recordings.

  GETS:    game_id; level -- 1-based game level; actions -- the solution; output_path -- the .jsonl file to create.
  DOES:    resets to the level, records the start frame, then every frame an action produces. The last recorded frame
           is the completion frame, which already shows the next level (arc_recording keeps it in this episode).
  RETURNS: output_path. Raises if the solution does not complete the level.
  """
  import arc_agi
  from arc_agi import OperationMode
  from arcengine import GameAction

  arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(arc_environments_dir()))
  env = arcade.make(game_id, seed=0, include_frame_data=True)
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


def collect_synthetic(game_id, runs, max_steps, seed, levels=None, name=None):
  """
  Record the solution and `runs` branched Goose runs for every synthetic level, into one new collection folder.

  GETS:    game_id -- e.g. "ls20-syn30"; runs -- Goose runs per level; max_steps -- Goose actions per run;
           seed -- base seed (run seed = seed + level * SEED_LEVEL_OFFSET + run index, as in arc_collection);
           levels -- level numbers to collect (default: all synthetic levels); name -- collection name without timestamp.
  DOES:    per level: solution.jsonl, then Goose runs that each replay the solution up to a random step first.
  RETURNS: the collection folder.
  """
  solutions = load_solutions(game_id)
  levels = sorted(solutions) if levels is None else levels
  version = game_id.split("-", 1)[1]
  collection_dir = synthetic_dir(game_id) / f"{name or f'{version}_goose-branch_{levels_tag(levels)}'}_{timestamp()}"
  collection_dir.mkdir(parents=True, exist_ok=False)  # never mix two collections
  settings = {"game_id": game_id, "levels": levels, "policy": "goose", "runs_per_level": runs, "max_steps": max_steps, "seed": seed,
              "seed_rule": f"seed + level * {SEED_LEVEL_OFFSET} + run_index", "branch_from": "the levels' optimal solutions (levels.json)", "branch_step": "uniform over 0 .. len(solution) - 1"}
  (collection_dir / "collection.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

  for level in levels:
    solution = solutions[level]
    level_dir = collection_dir / f"level_{level}"
    record_solution(game_id, level, solution, level_dir / "solution.jsonl")
    for run_index in range(runs):
      run_seed = seed + level * SEED_LEVEL_OFFSET + run_index
      step = int(np.random.default_rng(run_seed).integers(len(solution)))  # reproducible branch point per run
      branch = {"source": "solution", "step": step}  # stored on the first recorded line, like arc_collection's branches
      print(f"\n--- level {level}, run {run_index + 1}/{runs} (seed={run_seed}, branch at step {step}/{len(solution)}) ---")
      collect_game(game_id=game_id, max_steps=max_steps, seed=run_seed, mode="offline", output_path=level_dir / f"goose_seed{run_seed}.jsonl",
                   policy_name="goose", start_level=level, prefix=solution[:step], branch=branch)

  print(f"\nCollection complete: {len(levels)} levels x (1 solution + {runs} Goose runs) in {collection_dir}")
  return collection_dir


def main():
  parser = argparse.ArgumentParser(description="Record solutions and branched Goose runs for synthetic ls20 levels")
  parser.add_argument("game", help="a version made by arc_levelgen, e.g. ls20-syn30")
  parser.add_argument("--runs", type=int, default=20, help="Goose runs per level")
  parser.add_argument("--max-steps", type=int, default=500, help="maximum Goose actions per run")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--levels", type=int, nargs="+", default=None, help="levels to collect (default: all synthetic levels)")
  parser.add_argument("--name", default=None, help="collection name without timestamp; default <version>_goose-branch_<levels>")
  args = parser.parse_args()
  if args.runs < 1 or args.max_steps < 1:
    parser.error("--runs and --max-steps must be positive")
  collect_synthetic(args.game, runs=args.runs, max_steps=args.max_steps, seed=args.seed, levels=args.levels, name=args.name)


if __name__ == "__main__":
  main()
