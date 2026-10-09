"""
Synthetic ls20 levels: generate LevelSpecs, solve them, emit them into a new ls20 game version, certify them.

A level is plain data (sprites on a 12 x 12 cell grid + a settings dict), so we never reimplement the game:
we write new Level(...) entries next to the 7 real ones and the original Ls20 code runs them. The dynamics are
therefore identical to the real game by construction.

Pipeline (main):
  random_spec(tier)  -> arc_solver.solve()  -> keep if solvable, in-tier and of the right length
  write_game_version()  -> $STABLEWM_HOME/arc_environments/ls20/<version>/ls20.py: real levels 1-7 + synthetic 8..
  certify()  -> replay every solution in the REAL engine; only levels whose solution completes are kept

Tiers (along the real level progression):
  1 -- like levels 1-2: walls, rotation cross, refills. The goal needs a different rotation than the start.
  2 -- like level 3: + colour changer and push platforms. The optimal solution must use both.

Usage:
  python -m master_thesis.environments.arc_levelgen --check-real                 # validate the solver on real levels 1-4
  python -m master_thesis.environments.arc_levelgen --version syn30 --tier1 15 --tier2 15 --seed 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from dataclasses import asdict
from pathlib import Path

from master_thesis.environments.arc_solver import COLORS, GRID, NUM_SHAPES, PUSH_DIRECTIONS, ROTATIONS, Goal, LevelSpec, mechanics_used, solve
from master_thesis.paths import arc_environments_dir

SOURCE_GAME = "ls20-9607627b"  # the real game; its file is copied and its 7 levels stay levels 1-7
NUM_REAL_LEVELS = 7

# sprite templates of the real game (names are obfuscated in the original)
WALL = "ihdgageizm"
PLAYER = "sfqyzhzkij"
GOAL = "rjlbuycveu"
REFILL = "npxgalaybz"
ROTATOR = "rhsxkxzdjz"
COLORER = "soyhouuebz"
PLATFORM = {"t": "lujfinsby_t", "b": "kapcaakvb_b", "l": "tihiodtoj_l", "r": "yjgargdic_r"}
PLATFORM_MOUNT = {"t": "krdypjjivz", "b": "mxfhnkdzvf", "l": "ubyunwkbpx", "r": "fesygzfqui"}  # the wall piece behind each platform direction (as in every real level)
# the ARC-AGI-3 palette, same as ArcGridToPixels in training/lewm.py
PALETTE = [(255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102), (51, 51, 51), (0, 0, 0), (229, 58, 163), (255, 123, 204), (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0), (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214)]

TIER_LENGTHS = {1: (12, 40), 2: (25, 50)}  # optimal solution lengths the levels are spread over; real optima: 13 (L1), 45 (L2), 39 (L3)
LENGTH_TOLERANCE = 3  # a candidate is accepted within this many moves of its level's target length
TIER_MECHANICS = {1: {"rotation"}, 2: {"color", "platform"}}  # what the optimal solution must use


def pixel(cell):
  """Cell (col, row) -> pixel origin (x, y) of the 5x5 cell on the 64x64 frame."""
  return 4 + 5 * cell[0], 5 * cell[1]


################
## Real levels ##
################


def load_game_module(path):
  """Import a game file (e.g. the real ls20.py) as a module, to read its levels and sprite templates."""
  spec = importlib.util.spec_from_file_location(f"ls20_{Path(path).parent.name}", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def source_game_file(game_id=SOURCE_GAME):
  """Path of a downloaded game's source file, e.g. arc_environments/ls20/9607627b/ls20.py."""
  name, version = game_id.split("-", 1)
  return arc_environments_dir() / name / version / f"{name}.py"


def spec_from_level(level):
  """
  Read a real Level into a LevelSpec (used to validate the solver on the real levels).

  GETS:    level -- an arcengine Level from the game file.
  DOES:    maps every sprite to its cell. Walls use the cell their origin falls into, exactly as the engine's
           movement check does (one real wall, in level 2, is off the grid). Refills sit 1 px inside their cell,
           platforms 1 px behind it (in their push direction).
  RETURNS: the LevelSpec. Raises ValueError for moving changers on tracks (levels 5-7), which the solver lacks.
  """
  tagged = lambda tag: [s for s in level._sprites if s.tags and tag in s.tags]
  if tagged("xfmluydglp"):
    raise ValueError("moving changers on tracks are not supported")
  as_list = lambda value: value if isinstance(value, list) else [value]
  cell = lambda x, y: ((x - 4) // 5, y // 5)
  shapes, colors, rotations = (as_list(level.get_data(key)) for key in ("kvynsvxbpi", "GoalColor", "GoalRotation"))
  platforms = {}
  for sprite in tagged("gbvqrjtaqo"):
    dcol, drow = PUSH_DIRECTIONS[sprite.name[-1]]
    platforms[cell(sprite.x + dcol, sprite.y + drow)] = sprite.name[-1]
  decrement = level.get_data("StepsDecrement")
  player = tagged(PLAYER)[0]
  return LevelSpec(
    walls={cell(s.x, s.y) for s in tagged(WALL)},
    start=cell(player.x, player.y),
    goals=[Goal(cell(g.x, g.y), shapes[i], COLORS.index(colors[i]), ROTATIONS.index(rotations[i])) for i, g in enumerate(tagged(GOAL))],
    refills=[cell(s.x - 1, s.y - 1) for s in tagged(REFILL)],
    rotators={cell(s.x, s.y) for s in tagged(ROTATOR)},
    colorers={cell(s.x, s.y) for s in tagged(COLORER)},
    shapers={cell(s.x, s.y) for s in tagged("ttfwljgohq")},
    platforms=platforms,
    start_shape=level.get_data("StartShape"),
    start_color=COLORS.index(level.get_data("StartColor")),
    start_rotation=ROTATIONS.index(level.get_data("StartRotation")),
    step_counter=level.get_data("StepCounter"),
    steps_decrement=2 if decrement is None else decrement,
  )


###############
## Generation ##
###############


def random_walls(rng):
  """
  A layout in the style of the real levels: complete border, open rooms and corridors inside.

  GETS:    rng -- a random.Random.
  DOES:    scatters horizontal / vertical wall segments over the 10 x 10 interior until 30-60 interior cells are
           walls (the real levels have 35-63), then walls in every floor cell that is not connected to the largest
           floor region, so every floor cell is reachable.
  RETURNS: the set of wall cells.
  """
  walls = {(c, r) for c in range(GRID) for r in range(GRID) if c in (0, GRID - 1) or r in (0, GRID - 1)}
  walls.add((1, 10))  # covered by the bottom-left symbol box in every real level
  target = rng.randint(30, 60)
  interior = lambda cell: 0 < cell[0] < GRID - 1 and 0 < cell[1] < GRID - 1
  while sum(1 for w in walls if interior(w)) < target:
    col, row = rng.randint(1, GRID - 2), rng.randint(1, GRID - 2)
    dcol, drow = rng.choice([(1, 0), (0, 1)])
    for k in range(rng.randint(1, 5)):
      if interior((col + dcol * k, row + drow * k)):
        walls.add((col + dcol * k, row + drow * k))

  floor = {(c, r) for c in range(GRID) for r in range(GRID)} - walls
  regions, seen = [], set()
  for cell in floor:
    if cell in seen:
      continue
    region, stack = set(), [cell]
    while stack:
      current = stack.pop()
      if current in seen or current not in floor:
        continue
      seen.add(current)
      region.add(current)
      stack.extend((current[0] + dc, current[1] + dr) for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    regions.append(region)
  largest = max(regions, key=len)
  return walls | (floor - largest)


def random_spec(rng, tier):
  """
  One random candidate level of a tier (not yet checked for solvability).

  GETS:    rng; tier -- 1 or 2.
  DOES:    layout, then start, goal and objects on distinct floor cells. The goal's symbol differs from the start
           in rotation (tier 1) or in colour, and sometimes rotation (tier 2), so the changers are needed. Platforms
           sit on floor cells with a wall behind them (that wall becomes the platform's mount).
           The cell diagonally up-left of the goal stays free of objects and mounts: the goal's frame sprite has no
           tag, and the engine's per-cell check stops at it, which would hide anything listed after it.
  RETURNS: a LevelSpec.
  """
  while True:  # the real levels have 37-65 floor cells; redraw layouts that are too closed
    walls = random_walls(rng)
    floor = sorted({(c, r) for c in range(GRID) for r in range(GRID)} - walls)
    if len(floor) >= 30:
      break
  rng.shuffle(floor)
  goal_cell, start = floor.pop(), floor.pop()
  blocked = {(goal_cell[0] - 1, goal_cell[1] - 1)}
  free = [cell for cell in floor if cell not in blocked]

  shape, color, rotation = rng.randrange(NUM_SHAPES), rng.randrange(len(COLORS)), rng.randrange(len(ROTATIONS))
  if tier == 1:
    goal = Goal(goal_cell, shape, color, (rotation + rng.randint(1, 3)) % len(ROTATIONS))
  else:
    goal = Goal(goal_cell, shape, (color + rng.randint(1, 3)) % len(COLORS), (rotation + rng.choice([0, 0, 1, 2, 3])) % len(ROTATIONS))

  spec = LevelSpec(walls=walls, start=start, goals=[goal], start_shape=shape, start_color=color, start_rotation=rotation, steps_decrement=rng.choice([1, 2]))
  take = lambda: free.pop()
  if goal.rotation != rotation:
    spec.rotators = {take() for _ in range(rng.randint(1, 2))}
  if tier == 2:
    spec.colorers = {take()}
    mountable = walls - blocked  # each wall can be the mount of one platform only
    for _ in range(rng.randint(1, 3)):
      candidates = [cell for cell in free if any((cell[0] - dc, cell[1] - dr) in mountable for dc, dr in PUSH_DIRECTIONS.values())]
      if not candidates:
        break
      cell = rng.choice(candidates)
      free.remove(cell)
      direction = rng.choice([d for d, (dc, dr) in PUSH_DIRECTIONS.items() if (cell[0] - dc, cell[1] - dr) in mountable])
      spec.platforms[cell] = direction
      dcol, drow = PUSH_DIRECTIONS[direction]
      mountable.discard((cell[0] - dcol, cell[1] - drow))
  spec.refills = [take() for _ in range(rng.randint(0, 2 if tier == 1 else 3))]
  return spec


def generate_level(rng, tier, target_length, max_tries=20000):
  """
  Rejection sampling until a candidate is solvable, uses its tier's mechanics and has about the target length.

  GETS:    rng; tier; target_length -- wanted optimal solution length; max_tries -- give up after this many candidates.
  DOES:    most random candidates are short, so accepting the first one in a range would bunch the levels at the
           range's lower end; a per-level target spreads the difficulty instead (main() spaces the targets evenly).
  RETURNS: (spec, solution) with the optimal solution as a list of action ids 1..4.
  """
  for _ in range(max_tries):
    spec = random_spec(rng, tier)
    solution = solve(spec, max_states=2_000_000)
    if solution is None or abs(len(solution) - target_length) > LENGTH_TOLERANCE:
      continue
    if not TIER_MECHANICS[tier] <= mechanics_used(spec, solution):
      continue
    if any(spec.platforms) and _useless_platform(spec):
      continue
    return spec, solution
  raise RuntimeError(f"no tier-{tier} level of about {target_length} moves found in {max_tries} tries")


def _useless_platform(spec):
  """True if some platform cannot push at least 2 cells (the engine stops before the next wall or goal)."""
  stops = spec.walls | {goal.cell for goal in spec.goals}
  for cell, direction in spec.platforms.items():
    dcol, drow = PUSH_DIRECTIONS[direction]
    if (cell[0] + dcol, cell[1] + drow) in stops or (cell[0] + 2 * dcol, cell[1] + 2 * drow) in stops:
      return True
  return False


#############
## Emitting ##
#############


def emit_level_source(spec):
  """
  The Python source of one Level(...) entry, in the exact style of the real game file.

  GETS:    spec -- a LevelSpec with one goal.
  DOES:    the fixed interface sprites (symbol box, frame), the goal with its decoration sprites at the same offsets
           as in the real levels, walls (mount pieces behind platforms), player and objects, listed alphabetically
           by template name like the real levels (the engine processes a cell's sprites in this order). Settings
           as the real data dict; StepsDecrement is omitted for 2, which is what the game then uses.
  RETURNS: the source text, ready to be placed inside `levels = [ ... ]`.
  """
  (goal,) = spec.goals
  gx, gy = pixel(goal.cell)
  entries = [
    ("eqatonpohu", (1, 53), ""), ("ghizzeqtoh", (1, 53), ""), ("wgmbtyhvbc", (3, 55), ".set_scale(2)"), ("xvrpzkggig", None, ""),
    (GOAL, (gx, gy), ""), ("kvynsvxbpi", (gx + 1, gy + 1), ""), ("nszegiawib", (gx - 2, gy - 2), ""), ("hoswmpiqkw", (gx - 1, gy - 1), ""), ("vjotnebuqo", (gx - 1, gy - 1), ""),
    (PLAYER, pixel(spec.start), ""),
  ]
  mounts = {}
  for cell, direction in spec.platforms.items():
    dcol, drow = PUSH_DIRECTIONS[direction]
    mounts[(cell[0] - dcol, cell[1] - drow)] = PLATFORM_MOUNT[direction]
    x, y = pixel(cell)
    entries.append((PLATFORM[direction], (x - dcol, y - drow), ""))
  entries += [(mounts.get(cell, WALL), pixel(cell), "") for cell in spec.walls]
  entries += [(REFILL, (pixel(cell)[0] + 1, pixel(cell)[1] + 1), "") for cell in spec.refills]
  entries += [(ROTATOR, pixel(cell), "") for cell in spec.rotators]
  entries += [(COLORER, pixel(cell), "") for cell in spec.colorers]

  lines = []
  for name, position, extra in sorted(entries, key=lambda e: (e[0], e[1] or (0, 0))):
    placed = "" if position is None else f".set_position({position[0]}, {position[1]})"
    lines.append(f'            sprites["{name}"].clone(){placed}{extra},')
  data = {"StepCounter": spec.step_counter, "kvynsvxbpi": goal.shape, "GoalColor": COLORS[goal.color], "GoalRotation": ROTATIONS[goal.rotation],
          "StartShape": spec.start_shape, "StartColor": COLORS[spec.start_color], "StartRotation": ROTATIONS[spec.start_rotation], "Fog": False}
  if spec.steps_decrement != 2:
    data["StepsDecrement"] = spec.steps_decrement
  data_lines = "\n".join(f'            "{key}": {value},' for key, value in data.items())
  return "    Level(\n        sprites=[\n" + "\n".join(lines) + f"\n        ],\n        grid_size=(64, 64),\n        data={{\n{data_lines}\n        }},\n    ),\n"


def write_game_version(version, specs, source_game=SOURCE_GAME):
  """
  A new local ls20 version: the real game file with the synthetic levels appended after the 7 real ones.

  GETS:    version -- folder / version name, e.g. "syn30"; specs -- the synthetic LevelSpecs.
  DOES:    copies the real game file and inserts the emitted levels at the end of its `levels = [ ... ]` list;
           writes metadata.json so arc_agi finds the game as "ls20-<version>" (class Ls20, file ls20.py).
  RETURNS: the version folder. Synthetic level i (0-based) is game level NUM_REAL_LEVELS + 1 + i.
  """
  source = source_game_file(source_game).read_text(encoding="utf-8")
  end = source.rfind("]", 0, source.index("\nBACKGROUND_COLOR"))  # the closing bracket of the levels list
  game_text = source[:end] + "    # ---- synthetic levels (master_thesis.environments.arc_levelgen) ----\n" + "".join(emit_level_source(s) for s in specs) + source[end:]

  folder = arc_environments_dir() / "ls20" / version
  folder.mkdir(parents=True, exist_ok=True)
  (folder / "ls20.py").write_text(game_text, encoding="utf-8")
  metadata = json.loads((source_game_file(source_game).parent / "metadata.json").read_text(encoding="utf-8"))
  metadata.update({"game_id": f"ls20-{version}", "title": f"LS20 + {len(specs)} synthetic levels", "baseline_actions": None})
  metadata.pop("local_dir", None)
  (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  return folder


###################
## Certification ##
###################


def certify(game_id, level_number, actions):
  """
  The ground truth: replay a solution in the real engine.

  GETS:    game_id -- e.g. "ls20-syn30"; level_number -- 1-based game level; actions -- action ids 1..4.
  DOES:    resets the game to that level (as arc_collection does) and plays the actions.
  RETURNS: (completed, first_frame) -- whether the level is completed, and the level's start frame (64 x 64)
           for the preview.
  """
  import arc_agi
  import numpy as np
  from arc_agi import OperationMode
  from arcengine import GameAction, GameState

  from master_thesis.environments.arc_collection import reset_at_level

  arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(arc_environments_dir()))
  env = arcade.make(game_id, seed=0, include_frame_data=True)
  observation = reset_at_level(env, level_number)
  first_frame = np.asarray(observation.frame[-1])
  for action in actions:
    observation = env.step(GameAction.from_id(action))
    if observation.state is GameState.GAME_OVER:
      return False, first_frame
  return observation.levels_completed >= 1 or observation.state is GameState.WIN, first_frame


def save_preview(frames, labels, path, columns=6, scale=4):
  """A contact sheet of the levels' start frames, to look at the generated levels."""
  from PIL import Image, ImageDraw

  rows = (len(frames) + columns - 1) // columns
  size = 64 * scale
  sheet = Image.new("RGB", (columns * (size + 8), rows * (size + 22)), (40, 40, 40))
  draw = ImageDraw.Draw(sheet)
  for index, (frame, label) in enumerate(zip(frames, labels)):
    image = Image.new("RGB", (64, 64))
    image.putdata([PALETTE[int(v)] for v in frame.reshape(-1)])
    x, y = (index % columns) * (size + 8), (index // columns) * (size + 22)
    sheet.paste(image.resize((size, size), Image.NEAREST), (x, y + 18))
    draw.text((x + 2, y + 3), label, fill=(230, 230, 230))
  sheet.save(path)


def check_real_levels(levels=(1, 2, 3, 4)):
  """
  Validate the solver against the real game: solve real levels and replay each solution in the real engine.

  GETS:    levels -- real level numbers without moving changers (levels 5-7 have tracks).
  RETURNS: True if every solution completes its level. Rerun this after any change to arc_solver.
  """
  module = load_game_module(source_game_file())
  all_ok = True
  for number in levels:
    spec = spec_from_level(module.levels[number - 1])
    solution = solve(spec)
    completed = solution is not None and certify(SOURCE_GAME, number, solution)[0]
    all_ok &= completed
    print(f"real level {number}: optimal {len(solution) if solution else '-':>3} moves, mechanics {sorted(mechanics_used(spec, solution)) if solution else []} -> {'completes in the real engine' if completed else 'FAILS'}")
  return all_ok


def spaced_targets(count, low, high):
  """count target lengths evenly spaced over [low, high]."""
  return [round(low + (high - low) * i / max(1, count - 1)) for i in range(count)]


def main():
  parser = argparse.ArgumentParser(description="Generate, solve and certify synthetic ls20 levels")
  parser.add_argument("--version", default="syn30", help="game version folder; the game id becomes ls20-<version>")
  parser.add_argument("--tier1", type=int, default=15, help="number of tier-1 levels (rotation)")
  parser.add_argument("--tier2", type=int, default=15, help="number of tier-2 levels (+ colour, push platforms)")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--check-real", action="store_true", help="only validate the solver on the real levels 1-4, generate nothing")
  args = parser.parse_args()

  if args.check_real:
    raise SystemExit(0 if check_real_levels() else 1)

  rng = random.Random(args.seed)
  plan = [(1, target) for target in spaced_targets(args.tier1, *TIER_LENGTHS[1])] + [(2, target) for target in spaced_targets(args.tier2, *TIER_LENGTHS[2])]
  generated = [(tier, target, *generate_level(rng, tier, target)) for tier, target in plan]
  folder = write_game_version(args.version, [spec for _, _, spec, _ in generated])

  game_id, records, frames, labels = f"ls20-{args.version}", {}, [], []
  for index, (tier, target, spec, solution) in enumerate(generated):
    level = NUM_REAL_LEVELS + 1 + index
    completed, frame = certify(game_id, level, solution)
    if not completed:
      raise RuntimeError(f"level {level}: the solver's solution does not complete it in the real engine")
    spec_dict = asdict(spec)
    spec_dict.update(walls=sorted(spec.walls), rotators=sorted(spec.rotators), colorers=sorted(spec.colorers), shapers=sorted(spec.shapers),
                     platforms=[[list(cell), direction] for cell, direction in spec.platforms.items()])
    records[level] = {"tier": tier, "target_length": target, "solution": solution, "length": len(solution), "mechanics": sorted(mechanics_used(spec, solution)), "spec": spec_dict}
    frames.append(frame)
    labels.append(f"L{level} tier {tier} | {len(solution)} moves")
    print(f"level {level:3d}  tier {tier}  optimal {len(solution):2d} moves  mechanics {records[level]['mechanics']}  -> certified")

  (folder / "levels.json").write_text(json.dumps({"seed": args.seed, "source_game": SOURCE_GAME, "levels": records}, indent=1), encoding="utf-8")
  save_preview(frames, labels, folder / "preview.png")
  print(f"\n{len(records)} levels certified -> {folder} (game id {game_id}, levels {NUM_REAL_LEVELS + 1}-{NUM_REAL_LEVELS + len(records)})")


if __name__ == "__main__":
  main()
