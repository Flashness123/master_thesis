"""
A random walk over the moves that change something, for synthetic ls20 levels (arc_levelgen), whose LevelSpec is known.

Goose cannot give this: in ls20 every move uses energy, so with Goose's reward (did the frame change?) walking into a
wall counts as a change; with that masked out, Goose learns to avoid walls but its choices become one-sided (its loss
only raises rewarded actions), and a level's runs repeat the same paths. In the synthetic pilot (6 levels, 60 one-life
runs each) a fresh Goose wasted 34 % of its moves, a trained one 5 %, but found a third fewer situations.
"""

from __future__ import annotations

import numpy as np
from arcengine import GameAction

from master_thesis.environments.arc_solver import ACTIONS, transition


class SolverWalk:
  """
  Picks uniformly among the moves that change the player's situation, and sometimes any move.

  GETS:    spec -- the level's LevelSpec; state -- the solver state at the run's start, (cell, shape, color, rotation,
           energy, taken refills, reached goals), e.g. from arc_solver.reachable; seed; explore -- chance of a uniformly
           random move instead (so the data keeps some wall bumps, whose outcome the world model must learn too).
  DOES:    follows the run with arc_solver.transition (the same rules the engine plays: replayed solver paths always
           match the engine's frames). Energy is held at full in this copy: it only decides when a life ends, which
           the collector's stop condition handles, while the moves that change something do not depend on it.
           A move changes something unless it hits a wall or a goal that the current symbol cannot enter.
           visited collects the situations (states without energy) the run has really been in, for the collector to
           steer later runs to situations nobody has visited yet.
  """

  def __init__(self, spec, state, seed: int, explore: float = 0.1):
    self.spec = spec
    self.state = (*state[:4], spec.step_counter, *state[5:])  # energy at full, see DOES
    self.rng = np.random.default_rng(seed)
    self.explore = explore
    self.visited = set()

  def choose_action(self, frame, available_actions):
    """
    GETS:    frame -- the current grid (unused: the walk knows the state); available_actions -- unused, ls20 always has 1..4.
    DOES:    marks the tracked situation as visited, draws the next move and advances the tracked state by it.
             A situation is only marked when the walk is asked for its next move there, i.e. when the engine really got
             there: the move that uses up the last energy is never marked, since the life is lost and the level restarts.
    RETURNS: (GameAction, {}) like the other policies.
    """
    self.visited.add(self.state[:4] + self.state[5:])
    outcomes = {action: transition(self.spec, self.state, action) for action in ACTIONS}  # None = wall or blocked goal
    useful = [action for action, outcome in outcomes.items() if outcome is not None]
    action = int(self.rng.choice(list(ACTIONS))) if not useful or self.rng.random() < self.explore else int(self.rng.choice(useful))
    if outcomes[action] not in (None, "SOLVED"):  # "SOLVED" ends the run (level completed), None changes nothing
      self.state = (*outcomes[action][:4], self.spec.step_counter, *outcomes[action][5:])
    return GameAction.from_id(action), {}
