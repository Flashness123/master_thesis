"""
Orthogonal Predictive Factorization (OPF) for LeWM.

JEPA-Anything, Cui et al. 2026, arXiv:2609.20800, section "Orthogonal Predictive Factorization".
Adapted from the authors' jepa_anything_core (https://github.com/Gen-Verse/JEPA-Anything, Apache License 2.0),
keeping their names, signatures, defaults and formulas. Where their code lives:
  their opf.py    -> OrthogonalFactorProjection below. Their raw_basis / fixed_basis + analysis_basis() is our
                     single `basis` attribute; their module functions decompose_state / compose_state are the
                     bodies of our decompose / compose.
  their losses.py -> the "Losses" section.
  their audit.py  -> the "Geometry audit" section (numeric fields only, no tolerances / pass-fail).
Changes:
  - geometry and losses run in float32 with autocast off (training is bf16-mixed; the original requires the
    latents and the basis to share a dtype).
  - orthogonality_mode supports "soft_gram" and "qr_retraction"; "qr_init" is omitted because our basis already
    starts orthonormal (the identity split). The retraction's QR runs in float64 on CPU (exact, and available on
    every device); the original runs it in the basis' own dtype and device.
  - omitted: initial_basis, valid_mask, reduction options, the streaming variance tracker, decorrelation diagnostics
    and jepa_anything_objective (lewm_forward assembles the terms itself, because our prediction term is discounted
    over a rollout).
  - Our default is orthogonality_mode="qr_retraction" (configs/model/lewm_opf.yaml): the basis is re-orthonormalized
    after every optimizer step and lewm_forward uses the paper's L_pred (factor_prediction_loss) unchanged. With
    "soft_gram" alone the factor-space loss let our basis collapse; see the OPF note in training/lewm.py.
FactorHeads is ours: the release defines no predictor module.
"""

import torch
from torch import nn

from .modules import MLP

##############
## Geometry ##
##############

ORTHOGONALITY_MODES = ("soft_gram", "qr_retraction")


def orthonormalize_basis(raw_basis):
  """
  The authors' QR orthonormalization of a full-rank basis (used by the qr_retraction mode).

  GETS:    raw_basis [K, r, d] with K * r = d, full rank.
  DOES:    QR of the stacked rows (Gram-Schmidt order: row 1 keeps its direction, row 2 is made orthogonal to row 1,
           and so on), with the authors' sign fix: the diagonal of R is made positive, so a retraction never flips the
           sign of a coordinate. Computed in float64 on CPU. Not differentiable (only used between optimizer steps).
  RETURNS: the orthonormal basis [K, r, d], in the dtype and on the device of raw_basis.
  """
  num_factors, factor_dim, state_dim = raw_basis.shape
  work = raw_basis.detach().to(device="cpu", dtype=torch.float64).reshape(state_dim, state_dim)
  if torch.linalg.matrix_rank(work).item() < state_dim:
    raise ValueError("raw_basis must be full rank")
  q, upper = torch.linalg.qr(work.transpose(0, 1), mode="reduced")
  diagonal = torch.diagonal(upper)
  signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
  q = q * signs.unsqueeze(0)
  return q.transpose(0, 1).reshape_as(raw_basis).to(device=raw_basis.device, dtype=raw_basis.dtype)


class OrthogonalFactorProjection(nn.Module):
  """
  Complete orthogonal factorization of a latent state: the K analysis projections P_1..P_K, stored as one
  basis [K, r, d] whose row block k is P_k^T.

  GETS:    state_dim   -- complete state width d (192).
           num_factors -- number of predictive coordinate groups K.
           factor_dim  -- coordinates per group r; inferred as d / K if omitted. K * r must equal d.
           learnable   -- True: the basis is trained; False: the basis stays fixed (the original's default).
           orthogonality_mode -- how a learnable basis stays orthogonal:
                          "soft_gram": only through projector_orthogonality_loss (the original's default);
                          "qr_retraction": additionally re-orthonormalized by QR after optimizer steps
                                           (after_optimizer_step), so it is exactly orthogonal at every forward pass.
           qr_retraction_frequency -- retract every this many optimizer steps (qr_retraction only).
  DOES:    starts the basis at the identity split, so with learnable=False factor k is simply the latent
           coordinates [k*r, (k+1)*r). decompose() analyses latents into factors, compose() synthesizes
           latents from factors.
  """

  def __init__(self, state_dim: int, num_factors: int, factor_dim: int | None = None, *, learnable: bool = False, orthogonality_mode: str = "soft_gram", qr_retraction_frequency: int = 1):  # the `*` makes `learnable` keyword-only: OrthogonalFactorProjection(192, 4, learnable=True), never positional by accident
    super().__init__()
    if orthogonality_mode not in ORTHOGONALITY_MODES:
      raise ValueError(f"orthogonality_mode must be one of {ORTHOGONALITY_MODES}, got {orthogonality_mode!r}")
    if isinstance(qr_retraction_frequency, bool) or not isinstance(qr_retraction_frequency, int) or qr_retraction_frequency <= 0:
      raise ValueError("qr_retraction_frequency must be a positive integer")
    if orthogonality_mode == "qr_retraction" and not learnable:
      raise ValueError("qr_retraction requires a learnable projection")
    if orthogonality_mode != "qr_retraction" and qr_retraction_frequency != 1:
      raise ValueError("qr_retraction_frequency is configurable only in qr_retraction mode")
    if factor_dim is None:  # our config never passes it, so r is always derived: 192 / 4 = 48
      if num_factors <= 0 or state_dim % num_factors != 0:
        raise ValueError("factor_dim cannot be inferred unless state_dim is divisible by K")
      factor_dim = state_dim // num_factors
    if num_factors * factor_dim != state_dim:
      raise ValueError(f"complete OPF state requires num_factors * factor_dim == state_dim ({num_factors} * {factor_dim} != {state_dim})")

    self.state_dim = state_dim      # d = 192
    self.num_factors = num_factors  # K = 4
    self.factor_dim = factor_dim    # r = 48
    self.learnable = learnable
    self.orthogonality_mode = orthogonality_mode
    self.qr_retraction_frequency = qr_retraction_frequency


    # Identity matrix [192, 192] cut into 4 row blocks of 48 rows -> [4, 48, 192].
    # Row block k is P_k^T: its 48 rows are the 48 directions that factor k "looks along".
    # At the start these are just the unit vectors e_{48k} .. e_{48k+47}, i.e. factor k = latent coordinates 48k..48k+47.
    basis = torch.eye(state_dim).reshape(num_factors, factor_dim, state_dim)
    if learnable:
      self.basis = nn.Parameter(basis)
    else:
      self.register_buffer("basis", basis)

  def decompose(self, state):
    """
    Analysis: z^(k) = P_k^T z.

    GETS:    state   [..., d].
    DOES:    projects the latent onto each factor's basis.
    RETURNS: factors [..., K, r] in float32.
    """
    with torch.autocast(device_type=state.device.type, enabled=False):
      return torch.einsum("...d,krd->...kr", state.float(), self.basis.float())

  def compose(self, factors):
    """
    Synthesis: z_hat = (P^T)^+ u_hat  (paper Eq. "synthesis").

    GETS:    factors [..., K, r].
    DOES:    concatenates the factors and maps them back through the Moore--Penrose pseudoinverse of the
             analysis matrix P^T (which is just P when P is exactly orthogonal).
    RETURNS: states [..., d] in float32.
    """
    with torch.autocast(device_type=factors.device.type, enabled=False):
      synthesis = torch.linalg.pinv(self.basis.float().reshape(self.state_dim, self.state_dim))
      return torch.einsum("...a,da->...d", factors.float().flatten(start_dim=-2), synthesis)

  def retract_orthogonal_(self):
    """
    QR-retract the learnable basis in place (the authors' qr_retraction mode).

    GETS:    nothing; acts on self.basis.
    DOES:    replaces the basis by orthonormalize_basis(basis). Runs only between optimizer steps, never inside
             forward(), as the authors require. The optimizer state (Adam's moments) is deliberately left as it is.
    RETURNS: self.
    """
    if self.orthogonality_mode != "qr_retraction":
      raise RuntimeError("orthogonal retraction requires qr_retraction mode")
    with torch.no_grad():
      self.basis.copy_(orthonormalize_basis(self.basis))
    return self

  def after_optimizer_step(self, step: int) -> bool:
    """
    The scheduled retraction; called by the OrthogonalRetraction callback (training/lewm.py) after every training batch.

    GETS:    step -- one-based number of optimizer steps done so far (trainer.global_step).
    DOES:    retracts if the mode is qr_retraction and step is a multiple of qr_retraction_frequency.
    RETURNS: whether a retraction ran.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
      raise ValueError("step must be a positive integer")
    if self.orthogonality_mode != "qr_retraction" or step % self.qr_retraction_frequency != 0:
      return False
    self.retract_orthogonal_()
    return True

  def forward(self, state):
    """Alias for decompose."""
    return self.decompose(state)

  def extra_repr(self) -> str:
    return (f"state_dim={self.state_dim}, num_factors={self.num_factors}, factor_dim={self.factor_dim}, learnable={self.learnable}, "
            f"orthogonality_mode={self.orthogonality_mode!r}, qr_retraction_frequency={self.qr_retraction_frequency}")


class FactorHeads(nn.Module):
  """
  The factor predictors q_k, as "a shared trunk followed by branch-specific heads" (paper): the trunk is the
  ARPredictor, this module holds one MLP head per factor.

  GETS:    input_dim   -- width of the predictor output (192).
           state_dim   -- complete state width d (192); each head outputs r = d / num_factors coordinates.
           num_factors -- K, must match the OrthogonalFactorProjection.
           hidden_dim  -- hidden width of every head.
           norm_fn     -- normalization inside each head (BatchNorm1d, as in pred_proj).
  DOES:    forward(): head k predicts the r coordinates of factor k from the same predictor output.
  RETURNS: forward(): predicted factors [N, K, r] for predictor outputs [N, input_dim].
  """

  def __init__(self, input_dim: int, state_dim: int, num_factors: int, hidden_dim: int, norm_fn=nn.LayerNorm):
    super().__init__()
    if state_dim % num_factors:  # same contract as OrthogonalFactorProjection: 192 must split evenly into K factors
      raise ValueError(f"{state_dim} is not divisible by {num_factors}")
    self.num_factors = num_factors              # K = 4; JEPA.__init__ checks these two against the projection
    self.factor_dim = state_dim // num_factors  # r = 48: how many numbers each head outputs
    # One MLP per factor, each:  Linear(192 -> 816) -> BatchNorm1d(816) -> GELU -> Linear(816 -> 48)   (MLP in modules.py)
    # norm_fn arrives from the config as a "partial" (functools.partial(BatchNorm1d)); MLP calls norm_fn(816), so
    # every head gets its OWN BatchNorm. The default LayerNorm only applies if no norm_fn is passed.
    # nn.ModuleList, not a plain Python list: only then are the heads' parameters registered (optimizer, .to(device), weights.pt).
    # Size: 4 heads x 198,336 = 793,344 parameters, vs 792,768 in pred_proj -> both variants have the same capacity.
    self.heads = nn.ModuleList([MLP(input_dim, hidden_dim, self.factor_dim, norm_fn=norm_fn) for _ in range(num_factors)])

  def forward(self, x):
    # x: [N, 192] = predictor outputs with batch and time already flattened (N = B * T), since BatchNorm1d needs 2-D input.
    # Every head reads the SAME full 192-dim vector (not only "its" 48 coordinates) and writes its own 48 numbers.
    # 4 tensors [N, 48] -> stack(dim=-2) -> [N, 4, 48] = (sample, factor, coordinate within factor).
    return torch.stack([head(x) for head in self.heads], dim=-2)


############
## Losses ##
############


def _coordinate_variance(samples):
  """
  GETS:    samples [N, ...] (the first axis is the sample axis).
  DOES:    population variance (correction=0) of every coordinate across the samples.
  RETURNS: variances [...].
  """
  centered = samples - samples.mean(dim=0, keepdim=True)
  return centered.square().mean(dim=0)


def factor_prediction_loss(predicted_factors, target_factors):
  """
  L_pred: directly regress factor direction and magnitude.
  Used by lewm_forward when loss.opf.prediction_space == "factor" (the default, together with qr_retraction).

  GETS:    predicted_factors, target_factors -- [..., K, r], same shape.
  DOES:    mean squared error over every sample, factor and coordinate.
  RETURNS: scalar.
  """
  difference = predicted_factors.float() - target_factors.float()
  return difference.square().mean()


def projector_orthogonality_loss(basis):
  """
  L_orth = sum_k ||P_k^T P_k - I||_F^2 + sum_{i<j} ||P_i^T P_j||_F^2.

  GETS:    basis [K, r, d] (row block k is P_k^T).
  DOES:    the first term keeps each factor's own basis orthonormal, the second keeps different factors in
           different directions.
  RETURNS: scalar; 0 for an exactly orthogonal basis (always for a fixed identity basis).
  """
  with torch.autocast(device_type=basis.device.type, enabled=False):
    work = basis.float()
    num_factors, factor_dim, _ = work.shape
    identity = torch.eye(factor_dim, device=work.device, dtype=work.dtype)
    within_grams = torch.einsum("kad,kbd->kab", work, work)
    within_penalties = (within_grams - identity).square().sum(dim=(-2, -1))

    cross_penalties = []
    for first in range(num_factors):
      for second in range(first + 1, num_factors):
        cross_gram = work[first] @ work[second].transpose(0, 1)
        cross_penalties.append(cross_gram.square().sum())
    penalties = torch.cat((within_penalties, torch.stack(cross_penalties))) if cross_penalties else within_penalties
    return penalties.sum()


def factor_activity_loss(factors, *, min_std: float = 0.1, eps: float = 1e-6):
  """
  L_fac: require every predictive factor coordinate to remain active across samples.

  GETS:    factors [..., K, r] (all leading axes are samples); min_std = gamma_fac.
  DOES:    hinge on each of the K * r coordinate standard deviations: max(0, min_std - sqrt(Var + eps)).
  RETURNS: scalar, the mean hinge; 0 while every coordinate has std >= min_std.
  """
  samples = factors.reshape(-1, *factors.shape[-2:]).float()
  standard_deviation = torch.sqrt(_coordinate_variance(samples) + eps)
  return torch.relu(min_std - standard_deviation).mean()


def encoder_variance_loss(context_states, *, min_std: float = 0.1, eps: float = 1e-6):
  """
  L_enc: the online-encoder activity floor.

  GETS:    context_states [..., d] (all leading axes are samples); min_std = gamma_enc.
  DOES:    hinge on each of the d coordinate standard deviations: max(0, min_std - sqrt(Var + eps)).
  RETURNS: scalar, the mean hinge; 0 while every coordinate has std >= min_std.
  """
  samples = context_states.reshape(-1, context_states.shape[-1]).float()
  standard_deviation = torch.sqrt(_coordinate_variance(samples) + eps)
  return torch.relu(min_std - standard_deviation).mean()


####################
## Geometry audit ##
####################


@torch.no_grad()
def audit_basis_geometry(basis):
  """
  Numeric geometry of the analysis basis (for logging, not for the loss).

  GETS:    basis [K, r, d].
  DOES:    singular values and cross-factor overlaps in float64 on CPU, as in the original.
  RETURNS: {"minimum_singular_value":     1 when orthogonal; small = synthesis amplifies prediction errors,
            "condition_number":           1 when orthogonal; inf when rank deficient (the original reports None),
            "max_cross_subspace_overlap": max over i<j of ||P_i^T P_j||_F; 0 when factors are orthogonal}
  """
  num_factors, factor_dim, state_dim = basis.shape
  work = basis.detach().to(device="cpu", dtype=torch.float64)
  singular_values = torch.linalg.svdvals(work.reshape(num_factors * factor_dim, state_dim))
  minimum_singular_value = float(singular_values.min())
  condition_number = float(singular_values.max()) / minimum_singular_value if minimum_singular_value > 0 else float("inf")

  cross_overlaps = [torch.linalg.matrix_norm(work[first] @ work[second].T, ord="fro") for first in range(num_factors) for second in range(first + 1, num_factors)]
  max_cross_subspace_overlap = float(torch.stack(cross_overlaps).max()) if cross_overlaps else 0.0
  return {"minimum_singular_value": minimum_singular_value, "condition_number": condition_number, "max_cross_subspace_overlap": max_cross_subspace_overlap}


@torch.no_grad()
def audit_factor_geometry(factors, *, min_standard_deviation: float = 0.1):
  """
  Per-coordinate factor activity (for logging, not for the loss).

  GETS:    factors [..., K, r]; min_standard_deviation is the activity threshold.
  DOES:    population standard deviation of every coordinate across samples (no eps, as in the original).
  RETURNS: {"inactive_coordinates": how many of the K * r coordinates fall below min_standard_deviation
            (the original returns the (factor, coordinate) pairs themselves)}
  """
  samples = factors.detach().reshape(-1, *factors.shape[-2:]).float()
  coordinate_standard_deviations = torch.sqrt(_coordinate_variance(samples).clamp_min(0.0))
  return {"inactive_coordinates": int((coordinate_standard_deviations < min_standard_deviation).sum())}
