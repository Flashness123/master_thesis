"""
Search-based solver for ls20 levels (tiers 1-3: no moving changers on tracks).

A pure reimplementation of the ls20 rules for breadth-first search, with no arc_agi dependency, so it is fast
and testable. It is deliberately NOT trusted on its own: arc_levelgen.certify() replays every solution in the
real engine and keeps a level only if that replay completes it. A rule mistake here can therefore only make us
discard a solvable level, never ship a level with a wrong solution.

Rules follow Ls20.step() / txnfzvzetn() / twkzhcfelv (push platforms) in the original game file.
Validated by `python -m master_thesis.environments.arc_levelgen --check-real` (real levels 1-4).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

GRID = 12  # the playfield is a 12 x 12 cell grid
COLORS = [12, 9, 14, 8]  # the colour changer cycles orange -> blue -> green -> red (Ls20.tnkekoeuk)
ROTATIONS = [0, 90, 180, 270]  # the rotation cross adds 90 degrees (Ls20.dhksvilbb)
NUM_SHAPES = 6  # the shape changer cycles through 6 symbols (Ls20.ijessuuig)

# ARC action id -> cell step (dcol, drow). ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right.
ACTIONS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
# push-platform name suffix -> push direction (twkzhcfelv.__init__)
PUSH_DIRECTIONS = {"t": (0, -1), "b": (0, 1), "l": (-1, 0), "r": (1, 0)}


@dataclass
class Goal:
  """
  One goal cell and the symbol needed to enter it.

  GETS:    cell -- (col, row); shape -- 0..5; color -- index into COLORS; rotation -- index into ROTATIONS.
  """
  cell: tuple[int, int]
  shape: int
  color: int
  rotation: int


@dataclass
class LevelSpec:
  """
  A level as the solver and the generator see it: cells, not pixels.

  GETS:    walls       -- blocked cells.
           start       -- the player's start cell.
           goals       -- one or more Goal; all must be reached to finish the level.
           refills     -- cells of the yellow energy refills (consumed when taken).
           rotators / colorers / shapers -- cells of the rotation cross, colour changer, shape changer.
           platforms   -- {cell: direction} with direction in "t", "b", "l", "r".
           start_shape / start_color / start_rotation -- the symbol at the start (indices).
           step_counter     -- energy at the start and after a refill (StepCounter, 42 in all real levels).
           steps_decrement  -- energy per move (StepsDecrement; the game uses 2 when a level does not set it).
  """
  walls: set[tuple[int, int]]
  start: tuple[int, int]
  goals: list[Goal]
  refills: list[tuple[int, int]] = field(default_factory=list)
  rotators: set[tuple[int, int]] = field(default_factory=set)
  colorers: set[tuple[int, int]] = field(default_factory=set)
  shapers: set[tuple[int, int]] = field(default_factory=set)
  platforms: dict[tuple[int, int], str] = field(default_factory=dict)
  start_shape: int = 5
  start_color: int = 1
  start_rotation: int = 0
  step_counter: int = 42
  steps_decrement: int = 2


def _in_grid(cell):
  return 0 <= cell[0] < GRID and 0 <= cell[1] < GRID


def _apply_objects(spec, cell, shape, color, rotation, energy, refills_taken):
  """
  What stepping onto (or being pushed onto) a free cell does, as in txnfzvzetn().

  GETS:    the cell and the current symbol / energy / taken refills.
  DOES:    a not-yet-taken refill restores energy to step_counter; each changer cycles its attribute by one.
  RETURNS: (shape, color, rotation, energy, refills_taken, refilled) with refilled = a refill was taken here.
  """
  refilled = False
  if cell in spec.refills:
    index = spec.refills.index(cell)
    if not refills_taken & (1 << index):
      refills_taken |= 1 << index
      energy = spec.step_counter
      refilled = True
  if cell in spec.shapers:
    shape = (shape + 1) % NUM_SHAPES
  if cell in spec.colorers:
    color = (color + 1) % len(COLORS)
  if cell in spec.rotators:
    rotation = (rotation + 1) % len(ROTATIONS)
  return shape, color, rotation, energy, refills_taken, refilled


def transition(spec, state, action):
  """
  One action in the abstract game.

  GETS:    spec -- the LevelSpec; state -- (cell, shape, color, rotation, energy, refills_taken, goals_reached)
           with the two bitmasks over spec.refills / spec.goals; action -- 1..4.
  DOES:    applies the ls20 rules in the engine's order.
  RETURNS: the next state, "SOLVED" if this action finishes the level, or None if the action is useless or fatal
           (wall bump, goal with the wrong symbol, running out of energy) -- BFS never needs those.
  """
  cell, shape, color, rotation, energy, refills_taken, goals_reached = state
  dcol, drow = ACTIONS[action]
  target = (cell[0] + dcol, cell[1] + drow)

  if not _in_grid(target) or target in spec.walls:
    return None  # a wall bump costs energy and changes nothing: never useful

  goal_cells = [goal.cell for goal in spec.goals]
  if target in goal_cells and not goals_reached & (1 << goal_cells.index(target)):
    index = goal_cells.index(target)
    goal = spec.goals[index]
    if (shape, color, rotation) != (goal.shape, goal.color, goal.rotation):
      return None  # blocked, no energy cost: a no-op
    goals_reached |= 1 << index
    if goals_reached == (1 << len(spec.goals)) - 1:
      return "SOLVED"  # the engine checks the goal before running out of energy, so the last step may hit 0
    energy -= spec.steps_decrement
    if energy < 0:
      return None
    return (target, shape, color, rotation, energy, refills_taken, goals_reached)

  shape, color, rotation, energy, refills_taken, refilled = _apply_objects(spec, target, shape, color, rotation, energy, refills_taken)
  if not refilled:
    energy -= spec.steps_decrement
    if energy < 0:
      return None  # out of energy: the engine takes a life and resets; a clean solution never does that

  if target in spec.platforms:  # standing on a platform cell triggers it (it is not consumed)
    pcol, prow = PUSH_DIRECTIONS[spec.platforms[target]]
    stops = spec.walls | set(goal_cells)  # the engine stops before any wall or goal origin (ullzqnksoj)
    travel = 0
    for k in range(1, 12):
      if (target[0] + pcol * k, target[1] + prow * k) in stops:
        travel = k - 1
        break
    if travel > 0:  # no stop within 11 cells -> no push, as in the engine
      target = (target[0] + pcol * travel, target[1] + prow * travel)
      shape, color, rotation, energy, refills_taken, _ = _apply_objects(spec, target, shape, color, rotation, energy, refills_taken)

  return (target, shape, color, rotation, energy, refills_taken, goals_reached)


def solve(spec, max_states=20_000_000):
  """
  Shortest solution by breadth-first search.

  GETS:    spec -- the LevelSpec; max_states -- give up after expanding this many states.
  DOES:    BFS over (cell, symbol, energy, taken refills, reached goals); every state is visited once.
  RETURNS: the shortest list of action ids 1..4 that finishes the level, or None if there is none (or the
           search hit max_states).
  """
  start = (spec.start, spec.start_shape, spec.start_color, spec.start_rotation, spec.step_counter, 0, 0)
  parent = {start: None}
  queue = deque([start])
  while queue:
    state = queue.popleft()
    for action in ACTIONS:
      nxt = transition(spec, state, action)
      if nxt is None:
        continue
      if nxt == "SOLVED":
        path = [action]
        while parent[state] is not None:
          state, step_action = parent[state]
          path.append(step_action)
        return path[::-1]
      if nxt not in parent:
        parent[nxt] = (state, action)
        if len(parent) > max_states:
          return None
        queue.append(nxt)
  return None


def mechanics_used(spec, actions):
  """
  Which mechanics a solution actually needs, for tiering and for rejecting trivial levels.

  GETS:    spec and a solution (list of action ids).
  DOES:    replays the abstract game and records the objects the path touches.
  RETURNS: a set out of {"rotation", "color", "shape", "refill", "platform"}.
  """
  used = set()
  state = (spec.start, spec.start_shape, spec.start_color, spec.start_rotation, spec.step_counter, 0, 0)
  for action in actions:
    cell = state[0]
    dcol, drow = ACTIONS[action]
    target = (cell[0] + dcol, cell[1] + drow)
    used |= {name for name, cells in (("rotation", spec.rotators), ("color", spec.colorers), ("shape", spec.shapers), ("refill", spec.refills)) if target in cells}
    if target in spec.platforms:
      used.add("platform")
    state = transition(spec, state, action)
    if state in ("SOLVED", None):
      break
  return used
