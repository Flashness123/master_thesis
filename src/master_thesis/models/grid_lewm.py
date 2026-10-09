"""
Grid-LeWM, stage 1: LeWM with a categorical full-resolution Grid Transformer as observation encoder instead of the ViT.

  ARC grid [B, 4096]  colour IDs 0..15 (no RGB, no resizing; ArcGridToCells in training/lewm.py)
    -> GridEncoder       one learned vector per colour + one learned vector per cell position,
                         6 Transformer blocks with global bidirectional self-attention        -> [B, 4096, 192]
    -> AttentionPooling  one learned query attends over the 4096 cell tokens                 -> [B, 192]
    -> projector, ARPredictor, pred_proj, action encoder/decoder: LeWM's own modules (models/lewm/modules.py)

GridEncoder returns the complete spatial representation [B, 4096, 192]. Stage 2 (a spatial-temporal predictor on
[B, T, 4096, 192]) drops AttentionPooling and keeps GridEncoder unchanged.

Adapted from Grid-JEPA (huggingface.co/guychuk/arc-agi-3-grid-jepa, src/models/encoder.py):
  their GridPatchEmbed (color_embed, pos_embed) + ViTEncoder (blocks, norm) + GridJEPAEncoder  -> our GridEncoder
  their MultiHeadAttention, TransformerBlock                                                   -> same names and layout here
Deviations from Grid-JEPA:
  - attention runs through F.scaled_dot_product_attention instead of their explicit softmax(q k^T) v: the same computation,
    but on the GPU a fused kernel that never stores the 4096 x 4096 attention matrix
  - embed_dim 192 / depth 6 / 3 heads instead of their 384 / 12 / 6 (192 = LeWM's latent size)
  - no context masking and no EMA target encoder: LeWM trains the encoder end-to-end, SIGReg prevents collapse
  - AttentionPooling is ours: Grid-JEPA keeps all cell tokens
GridLeWM is a standalone copy of LeWM's JEPA (models/lewm/jepa.py); only __init__ (the pool) and encode() differ.
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
  return v.detach().clone() if torch.is_tensor(v) else v


class MultiHeadAttention(nn.Module):
  """
  Global bidirectional multi-head self-attention over the cell tokens (Grid-JEPA's MultiHeadAttention).

  GETS:    x -- tokens [N, L, 192]; N = frames, L = tokens per frame (4096 cells).
  DOES:    one linear layer makes the queries, keys and values of all heads; every token attends to EVERY token
           (no causal mask, no window); the heads are concatenated and projected back.
  RETURNS: [N, L, 192].
  """

  def __init__(self, dim, num_heads=3, qkv_bias=True, dropout=0.0):
    super().__init__()
    assert dim % num_heads == 0, "dim must split evenly into the heads"
    self.num_heads = num_heads
    self.head_dim = dim // num_heads  # 192 / 3 = 64
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.proj = nn.Linear(dim, dim, bias=qkv_bias)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    N, L, D = x.shape
    q, k, v = self.qkv(x).reshape(N, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # each [N, heads, L, head_dim]
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)  # softmax(q k^T / sqrt(head_dim)) v; is_causal=False: every cell sees all 4096
    x = x.transpose(1, 2).reshape(N, L, D)  # concatenate the heads again
    return self.dropout(self.proj(x))


class TransformerBlock(nn.Module):
  """
  Standard pre-norm Transformer block (Grid-JEPA's TransformerBlock).

  GETS:    x -- tokens [N, L, 192].
  DOES:    x + attention(LayerNorm(x)), then x + MLP(LayerNorm(x)); the MLP widens to mlp_ratio * 192 = 768 with GELU.
  RETURNS: [N, L, 192].
  """

  def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, dropout=0.0):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim, eps=1e-6)
    self.attn = MultiHeadAttention(dim, num_heads, qkv_bias, dropout)
    self.norm2 = nn.LayerNorm(dim, eps=1e-6)
    mlp_hidden = int(dim * mlp_ratio)
    self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, dim), nn.Dropout(dropout))

  def forward(self, x):
    x = x + self.attn(self.norm1(x))  # every cell gathers information from all cells
    x = x + self.mlp(self.norm2(x))  # every cell processes what it gathered, on its own
    return x


class GridEncoder(nn.Module):
  """
  The observation encoder: a Transformer over all 4096 cells of an ARC grid, one token per cell.

  GETS:    grid -- colour IDs [N, 4096] (or [N, 64, 64]), integers 0..15; N = frames.
  DOES:    every cell becomes the learned vector of its colour plus the learned vector of its position, then `depth`
           blocks of global self-attention + MLP and a final LayerNorm. No patches, no downsampling: 4096 tokens throughout.
  RETURNS: the spatial representation [N, 4096, 192], one token per cell (GridLeWM pools it; stage 2 will use it directly).
  """

  def __init__(self, num_colors=16, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0, max_grid_size=64):
    super().__init__()
    self.color_embed = nn.Embedding(num_colors, embed_dim)  # one learned vector per colour 0..15
    self.pos_embed = nn.Parameter(torch.zeros(1, max_grid_size * max_grid_size, embed_dim))  # one learned vector per cell of the flattened 64x64 grid (row by row), as in Grid-JEPA
    nn.init.trunc_normal_(self.pos_embed, std=0.02)
    self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
    self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

  def forward(self, grid):
    x = self.color_embed(grid.long().flatten(1))  # [N, 4096] colour IDs -> [N, 4096, 192]
    x = x + self.pos_embed[:, :x.size(1)]  # where each cell is
    for block in self.blocks:
      x = block(x)
    return self.norm(x)


class AttentionPooling(nn.Module):
  """
  Learned global aggregation of the cell tokens into one vector per frame (ours; LeWM's ViT uses its CLS token instead).

  GETS:    tokens -- [N, 4096, 192] from GridEncoder; N = frames.
  DOES:    one learned query scores every cell token (multi-head attention) and returns their weighted sum, so the model
           learns which cells matter for the summary (e.g. the player, the goal) instead of averaging all 4096 equally.
  RETURNS: one vector per frame [N, 192], the input of LeWM's projector.
  """

  def __init__(self, embed_dim=192, num_heads=3):
    super().__init__()
    self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
    nn.init.trunc_normal_(self.query, std=0.02)
    self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

  def forward(self, tokens):
    query = self.query.expand(tokens.size(0), -1, -1)  # the same learned query for every frame  [N, 1, 192]
    pooled, _ = self.attn(query, tokens, tokens, need_weights=False)  # [N, 1, 192]; need_weights=False lets PyTorch use its fused attention kernel
    return pooled[:, 0]


class GridLeWM(nn.Module):
  """
  Grid-LeWM: LeWM's Joint-Embedding Predictive Architecture with a categorical Grid Transformer as encoder:
    encoder        -- GridEncoder; turns a 64x64 grid of colour IDs into 4096 cell tokens [., 4096, 192].
    pool           -- AttentionPooling; turns the 4096 tokens into one vector [., 192].
    action_encoder -- Embedder; turns the 9-D action into an action latent [., 192].
    predictor      -- ARPredictor; causal transformer, predicts the next state latent given history+actions.
    projector      -- MLP 192->2048->192 (BatchNorm); post-processes the pooled vector into the state latent.
    pred_proj      -- MLP 192->2048->192; post-processes the predictor's output.

  Training uses only encode() + predict()
  Inference uses CEM = rollout()/criterion()/get_cost()
  Everything except __init__ and encode() is copied from LeWM's JEPA (models/lewm/jepa.py).
  """
  def __init__(self, encoder, pool, predictor, action_encoder, projector=None, pred_proj=None, action_decoder=None, ):
    super().__init__()

    self.encoder = encoder
    self.pool = pool
    self.predictor = predictor
    self.action_encoder = action_encoder
    self.action_decoder = action_decoder  # Delta-JEPA's LDAD (optional): names the action behind a latent difference. None = plain LeWM
    self.projector = projector or nn.Identity()  # 192->2048->192
    self.pred_proj = pred_proj or nn.Identity()  # 192->2048->192

  def encode(self, info):
    """
    Encode a batch of grid sequences (and their actions) into latents.

    GETS:    info -- a dict with at least "pixels" [B, T, 4096]: the colour IDs (still called "pixels" as in LeWM, so the
                     training step and the planner pass observations the same way); optionally "action" [B, T, 9].
    DOES:    flatten (B, T) so every frame is encoded on its own, GridEncoder -> 4096 cell tokens, AttentionPooling ->
             one vector, projector -> the state latent; reshape back to [B, T, 192]; if actions are present, encode them too.
    RETURNS: the SAME dict, with "emb" [B, T, 192] added (and "act_emb" [B, T, 192] if actions given).
             Mutates and returns info in place.
    """
    grids = info["pixels"]
    b = grids.size(0)
    grids = rearrange(grids, "b t ... -> (b t) ...")  # [B, T, 4096] -> [B*T, 4096]: every frame is encoded independently
    tokens = self.encoder(grids)  # the full spatial representation [B*T, 4096, 192]
    emb = self.projector(self.pool(tokens))  # attention pooling -> [B*T, 192] -> projector -> the state latent
    info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

    if "action" in info:
      info["act_emb"] = self.action_encoder(info["action"])  # [B, T, 9] -> [B, T, 192]
    return info

  def predict(self, emb, act_emb):
    """
    Predict the next-state latent at every position of a history.

    GETS:    emb     -- state latents   [B, T, 192]  (in training: the 3 context latents z0,z1,z2).
             act_emb -- action latents  [B, T, 192]  (the 3 action latents e(a0),e(a1),e(a2)).
    DOES:    run the causal ARPredictor (position t attends to 0..t, conditioned on actions via AdaLN),
             then post-project each output through pred_proj.
    RETURNS: predicted next-state latents [B, T, 192]. Position t = the prediction of state t+1.
    """
    preds = self.predictor(emb, act_emb)
    preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))  # pred_proj is an MLP over the feature dim only, so flatten (B,T) to apply it per-token, then restore
    preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
    return preds

  def decode_action(self, emb, next_emb):
    """
    Delta-JEPA's Latent Difference Action Decoder (Zhang et al. 2026, arXiv:2606.31232).

    GETS:    emb      -- latents of the states the actions were taken in  [B, T, 192]  (z_t).
             next_emb -- latents of the states they led to                [B, T, 192]  (z_t+1).
    DOES:    form the displacement z_t+1 - z_t and classify the action from THAT ALONE, so different actions have to move
             the latent in distinguishable directions.
    RETURNS: action logits [B, T, num_actions] (for ARC: 7 = ACTION1..ACTION7).
    """
    delta = next_emb - emb  # the encoder gets this gradient: it is what shapes the transition geometry
    logits = self.action_decoder(rearrange(delta, "b t d -> (b t) d"))  # the decoder is an MLP over the feature dim, so flatten (B,T) as in predict()
    return rearrange(logits, "(b t) a -> b t a", b=emb.size(0))

  def predict_rollout(self, emb, act_emb, history_size: int = 3):
    """
    Imagine the future autoregressively, WITH gradients (the training counterpart of rollout(), which the planner uses).

    GETS:    emb          -- the real starting latents [B, T0, 192] (in training: only z0, as the planner starts from one frame).
             act_emb      -- action latents [B, T0 + K - 1, 192]: e(a_i) is the action taken in state i, one per latent to feed in.
             history_size -- the predictor sees at most this many latents; older ones drop out, exactly as in rollout().
    DOES:    predict the next latent from the last `history_size` latents and their actions, append it, repeat K times.
    RETURNS: the K imagined latents [B, K, 192] (in training: ẑ1..ẑK).
    """
    start = emb.size(1)
    for _ in range(act_emb.size(1) - start + 1):
      n = emb.size(1)  # latents so far: real ones, then our own predictions
      next_emb = self.predict(emb[:, -history_size:], act_emb[:, :n][:, -history_size:])[:, -1:]  # the last position predicts the next state
      emb = torch.cat([emb, next_emb], dim=1)  # feed the prediction back in
    return emb[:, start:]

  ####################
  ## Inference only ##
  ####################

  def rollout(self, info, action_sequence, history_size: int = 3):
    """
    Autoregressively imagine the future latents for many candidate action plans at once.

    GETS:    info -- dict with "pixels" [B, S, T_hist, 4096]: the colour IDs of the SAME initial history repeated
                     across S candidates
             action_sequence -- [B, S, T, action_dim]: for each of S candidates, a full plan of T
                     actions. Its first T_hist entries are the actions already taken in the history;
                     the rest are the proposed future actions.
             history_size -- how many past latents the predictor is allowed to attend to per step
                     (HS below), a different thing from T_hist (how many real frames we were handed).
    DOES:    encode the initial history ONCE (shared across candidates), then step forward T-T_hist
             times feeding predictions back in as context, consuming one planned action per step.
    RETURNS: info with "predicted_emb" [B, S, T+1, D] added: the real history latents followed by the
             imagined latents, ending at the latent AFTER the last planned action.
    """
    assert "pixels" in info, "pixels not in info_dict"
    H = info["pixels"].size(2)  # T_hist: number of REAL history frames we were given
    B, S, T = action_sequence.shape[:3]   # B batch, S candidates, T total plan length
    act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
    info["action"] = act_0
    n_steps = T - H

    # All S plans share the same start, so encoding sample 0 and expanding avoids S redundant encoder passes.
    _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}  # drop the S axis: [B, T_hist, ...]
    _init = self.encode(_init)
    emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)   # [B, 1, T_hist, D] -> [B, S, T_hist, D]   introduce S again
    _init = {k: detach_clone(v) for k, v in _init.items()}  # no gradients during planning

    # flatten (B, S) into one axis so the predictor treats each candidate as an independent sequence
    emb = rearrange(emb, "b s ... -> (b s) ...").clone()
    act = rearrange(act_0, "b s ... -> (b s) ...")
    act_future = rearrange(act_future, "b s ... -> (b s) ...")

    # rollout predictor autoregressively for n_steps
    HS = history_size
    for t in range(n_steps):
      act_emb = self.action_encoder(act)
      emb_trunc = emb[:, -HS:]  # (BS, HS, D)
      act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
      pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)  the one new action
      emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D) history grows by +1

      next_act = act_future[:, t:t + 1, :]   # the next planned action [BS, 1, action_dim]
      act = torch.cat([act, next_act], dim=1)  # queue it so the NEXT iteration predicts through it

    # The loop appends a future action AFTER each prediction, so after the last iteration there is one queued action that has not been "applied" yet. This trailing predict consumes it -> the true final state
    act_emb = self.action_encoder(act)  # (BS, T, A_emb)
    emb_trunc = emb[:, -HS:]  # (BS, HS, D)
    act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
    pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
    emb = torch.cat([emb, pred_emb], dim=1)

    # unflatten batch and sample dimensions
    pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
    info["predicted_emb"] = pred_rollout
    return info

  def criterion(self, info_dict: dict):
    """
    Score each candidate by how close its FINAL imagined latent is to the goal latent.
    GETS:    info_dict with "predicted_emb" [B, S, *, D] (from rollout) and "goal_emb" [B, S, *, D].
    DOES:    take the goal's last frame, compare it to the rollout's last frame by summed MSE.
    RETURNS: cost [B, S] -- one scalar per candidate (lower = ends closer to the goal). CEM minimizes this.
    """
    pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
    goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

    goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

    # return last-step cost per action candidate
    cost = F.mse_loss(pred_emb[..., -1:, :], goal_emb[..., -1:, :].detach(), reduction="none", ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)
    return cost

  def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
    """
    CEM's cost function: given a goal and a batch of candidate action plans, return each plan's cost.
    GETS:    info_dict -- must contain "goal" (the goal grid's colour IDs) plus the current "pixels"/"action" history;
                          goal-side extras are passed as "goal_*" keys.
             action_candidates -- [B, S, T, action_dim]: S candidate plans to evaluate.
    DOES:    encode the goal grid into a goal latent, roll every candidate forward, score with criterion.
    RETURNS: cost [B, S]. This is the single call the CEM planner in arc_policies.py optimizes over.
    """
    assert "goal" in info_dict, "goal not in info_dict"

    device = next(self.parameters()).device
    for k in list(info_dict.keys()):
      if torch.is_tensor(info_dict[k]):
        info_dict[k] = info_dict[k].to(device)

    goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
    goal["pixels"] = goal["goal"]

    for k in info_dict:
      if k.startswith("goal_"):
        goal[k[len("goal_"):]] = goal.pop(k)

    goal.pop("action")
    goal = self.encode(goal)

    info_dict["goal_emb"] = goal["emb"]
    info_dict = self.rollout(info_dict, action_candidates)

    cost = self.criterion(info_dict)

    return cost
