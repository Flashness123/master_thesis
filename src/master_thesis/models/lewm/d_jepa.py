"""
D-JEPA's relational ranker (arXiv:2609.24749; code github.com/NEBULIS-Lab/D-JEPA, src/djepa/models/relational.py and
ordinal.py) on top of a trained LeWM: only the relational branch, no predictive plasticity.

For one situation (current state + goal) the planner scores every candidate plan with LeWM. The ranker looks at K of
them together and corrects each plan's score by at most +-max_correction, so that plans that really reach the goal come
first. The score it corrects is the plan's rank under LeWM's cost, scaled to 0..1 (rank / (K - 1)): a correction of 0.2
can move a plan by a fifth of the list. A gate decides whether the corrected choice is used at all (gate_choice); its
threshold is calibrated on held-out situations, so a ranker that is not confident enough leaves the plain choice alone.

What differs from their PushT version:
  - one world model instead of two (they combine LeWM and TD-JEPA), so a candidate is 192 + 1 numbers instead of 386
  - a plan is described by its imagined state closest to the goal instead of its last one, because our planner's cost
    and our success labels count reaching the goal after any move
  - their candidate pools are perturbations of a recorded segment; ours are all 1,024 plans of the exhaustive planner,
    of which select_candidates keeps K = 63
"""

import torch
import torch.nn.functional as F
from torch import nn


class RelationalRanker(nn.Module):
  """
  GETS:    latent_dim     -- size of a LeWM latent (192).
           hidden, heads, feedforward, layers, low_rank, max_correction -- their sizes: 64, 4, 128, 2, 8, 0.2.
  DOES:    describes every candidate by LayerNorm(z_closest - z_goal) (where its imagined path gets closest to the
           goal, relative to the goal, without scale) and its base score; encodes each candidate on its own (Linear +
           LayerNorm + GELU); lets the candidates of one situation look at each other (2 pre-norm Transformer layers,
           no positional encoding: the order of the list carries no meaning); turns each into a bounded correction
           max_correction * tanh(up(tanh(down(h)))). `up` starts at zero, so an untrained ranker changes nothing.
  forward: closest [B, K, D] -- each candidate's imagined latent closest to the goal.
           goal    [B, D]    -- the goal's latent.
           base    [B, K]    -- base scores: rank under LeWM's cost / (K - 1), 0 = cheapest.
           -> (scores [B, K] = base + correction, correction [B, K]); lower score = better plan.
  """

  def __init__(self, latent_dim: int = 192, hidden: int = 64, heads: int = 4, feedforward: int = 128, layers: int = 2, low_rank: int = 8, max_correction: float = 0.2):
    super().__init__()
    self.max_correction = max_correction
    self.candidate_encoder = nn.Sequential(nn.Linear(latent_dim + 1, hidden), nn.LayerNorm(hidden), nn.GELU())
    self.set_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, heads, feedforward, dropout=0.0, activation="gelu", batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
    self.correction_down = nn.Linear(hidden, low_rank, bias=False)
    self.correction_up = nn.Linear(low_rank, 1, bias=False)
    nn.init.zeros_(self.correction_up.weight)  # no correction at the start: the ranker begins as the plain planner

  def forward(self, closest, goal, base):
    descriptor = F.layer_norm(closest - goal[:, None], (closest.shape[-1],))  # where each plan gets closest to the goal, relative to the goal  [B, K, D]
    hidden = self.set_encoder(self.candidate_encoder(torch.cat([descriptor, base[..., None]], dim=-1)))  # every candidate, after looking at the others  [B, K, hidden]
    correction = self.max_correction * torch.tanh(self.correction_up(torch.tanh(self.correction_down(hidden)))).squeeze(-1)  # bounded to +-max_correction  [B, K]
    return base + correction, correction


def select_candidates(order, lowest: int = 16, total: int = 63):
  """
  GETS:    order  -- every plan's index, sorted from cheapest to most expensive under the planner's cost [P] (P = 1,024).
           lowest -- how many of the cheapest plans are always kept (theirs: 16).
           total  -- how many plans the ranker sees (theirs: 63).
  DOES:    keeps the `lowest` cheapest plans (where the decision is made) and spreads the remaining picks evenly over
           the rest of the order, so the ranker also sees middling and bad plans to compare with. Their pools keep the 16
           cheapest and then go round-robin over families of perturbations of a recorded segment; we have no families,
           so the even spread over the cost order takes their place.
  RETURNS: the kept plans' indices [total], still sorted by cost, so position k / (total - 1) is a plan's base score.
  """
  rest = order[lowest:]
  return torch.cat([order[:lowest], rest[torch.linspace(0, len(rest) - 1, total - lowest, device=order.device).round().long()]])


def gate_choice(base, scores, threshold: float):
  """
  GETS:    base      [B, K] -- base scores (rank / (K - 1)).
           scores    [B, K] -- the ranker's corrected scores (lower = better).
           threshold        -- the gate's tau, calibrated on held-out situations (training/d_jepa.py).
  DOES:    their gate ("base_minus_refined"): advantage = base score of the base winner - corrected score of the
           corrected winner; the corrected winner is taken only where the advantage is larger than tau, elsewhere the
           base winner (the plain planner's choice) stays. tau = +inf always keeps the plain planner.
  RETURNS: the chosen candidate of every situation [B].
  """
  base_winner, corrected_winner = base.argmin(dim=1), scores.argmin(dim=1)
  advantage = base.gather(1, base_winner[:, None])[:, 0] - scores.gather(1, corrected_winner[:, None])[:, 0]
  return torch.where(advantage > threshold, corrected_winner, base_winner)
