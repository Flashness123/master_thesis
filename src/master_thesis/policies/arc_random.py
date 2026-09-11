from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
from arcengine import GameAction


class ArcRandomPolicy:
  """
  Uniform random ARC policy.

  Samples only from the action types currently exposed by the
  environment. ACTION6 additionally receives a random (x, y)
  coordinate in the 64x64 ARC grid.

  RESET is excluded from random policy actions because episode
  resets are handled by the collection loop.
  """

  def __init__(self, seed: int = 42, ):
    self.rng = random.Random(seed)

  def choose_action(self, frame: np.ndarray, available_actions: Sequence[GameAction], ) -> tuple[GameAction, dict[str, int], ]:
    # Random policy currently does not inspect the frame.
    # It remains part of the interface so future ARC policies can.
    del frame

    actions = [action for action in available_actions if action is not GameAction.RESET]

    if not actions:
      raise RuntimeError("ARC environment exposes no usable actions")

    action = self.rng.choice(actions)

    if action.is_complex():
      return (action, {"x": self.rng.randint(0, 63, ), "y": self.rng.randint(0, 63, ), }, )

    return action, {}
