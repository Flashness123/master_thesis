#TODO: DO we really want to keep only the CLS token?
#      Should we measure the affect of the history size of CEM?
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
  return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):
  """
  Joint-Embedding Predictive Architecture:
    encoder        -- ViT-tiny; turns a 224x224 image into a CLS token [.,192].
    action_encoder -- Embedder; turns the 9-D action into an action latent [.,192].
    predictor      -- ARPredictor; causal transformer, predicts the next state latent given history+actions.
    projector      -- MLP 192->2048->192 (BatchNorm); post-processes the CLS token into the state latent.
    pred_proj      -- MLP 192->2048->192; post-processes the predictor's output.

  Training uses only encode() + predict() 
  Inference uses CEM = rollout()/criterion()/get_cost()
  """
  def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None, action_decoder=None, ):
    super().__init__()

    self.encoder = encoder
    self.predictor = predictor
    self.action_encoder = action_encoder
    self.action_decoder = action_decoder  # Delta-JEPA's LDAD (optional): names the action behind a latent difference. None = plain LeWM, so older runs still load.
    self.projector = projector or nn.Identity()  # 192->2048->192  optional: fall back to a no-op if not configured, but they are configured by ARC config
    self.pred_proj = pred_proj or nn.Identity()  # 192->2048->192

  def encode(self, info):
    """
    Encode a batch of image sequences (and their actions) into latents.

    GETS:    info -- a dict with at least "pixels" [B, T, 3, 224, 224]; optionally "action" [B, T, 9].
             In training T=4; the dict is the DataLoader batch itself.
    DOES:    flatten (B,T) into one axis so the ViT sees B*T independent images, run encoder+projector,
             reshape back to [B, T, D]; if actions are present, encode them too.
    RETURNS: the SAME dict, with "emb" [B, T, 192] added (and "act_emb" [B, T, 192] if actions given).
             Mutates and returns info in place.
    """

    pixels = info['pixels'].float()
    b = pixels.size(0)
    pixels = rearrange(pixels, "b t ... -> (b t) ...")  # [B, T, 3, H, W] -> [B*T, 3, H, W]. Every frame is encoded independently no temporal mixing here
    output = self.encoder(pixels, interpolate_pos_encoding=True)  # interpolate_pos_encoding resizes the positional encoding to work with 224x224
    pixels_emb = output.last_hidden_state[:, 0]  # take the CLS token (index 0) as the whole-image summary
    emb = self.projector(pixels_emb)  # MLP projection -> the state latent, still [B*T, 192]
    info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)  #unflatten, remember b

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
    DOES:    form the displacement z_t+1 - z_t and classify the action from THAT ALONE. The decoder never
             sees z_t itself, so the action cannot be read off state-specific cues: the only way to solve the
             task is for different actions to move the latent in distinguishable directions (the paper's point,
             and its ablation shows this beats decoding from the two endpoints).
    RETURNS: action logits [B, T, num_actions] (for ARC: 7 = ACTION1..ACTION7).
    """
    delta = next_emb - emb  # the encoder gets this gradient: it is what shapes the transition geometry
    logits = self.action_decoder(rearrange(delta, "b t d -> (b t) d"))  # the decoder is an MLP over the feature dim, so flatten (B,T) as in predict()
    return rearrange(logits, "(b t) a -> b t a", b=emb.size(0))

  ####################
  ## Inference only ##
  ####################

  def rollout(self, info, action_sequence, history_size: int = 3):
    """
    Autoregressively imagine the future latents for many candidate action plans at once.

    GETS:    info -- dict with "pixels" [B, S, T_hist, 3, H, W]: the SAME initial history repeated
                     across S candidates
             action_sequence -- [B, S, T, action_dim]: for each of S candidates, a full plan of T
                     actions. Its first T_hist entries are the actions already taken in the history;
                     the rest are the proposed future actions.
             history_size -- how many past latents the predictor is allowed to attend to per step
                     (HS below). NOTE: this is the PREDICTOR's context window, a DIFFERENT thing from
                     T_hist (how many real frames we were handed). HS caps attention; T_hist is data.
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

    # All S plans share the same start, so encoding sample 0 and expanding avoids S redundant ViT passes.
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
      emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D) hisotry grows by +1

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
    cost = F.mse_loss(pred_emb[..., -1:, :], goal_emb[..., -1:, :].detach(), reduction="none", ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S
    return cost

  def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
    """
    CEM's cost function: given a goal and a batch of candidate action plans, return each plan's cost.
    GETS:    info_dict -- must contain "goal" (the goal image) plus the current "pixels"/"action" history;
                          goal-side extras are passed as "goal_*" keys.
             action_candidates -- [B, S, T, action_dim]: S candidate plans to evaluate.
    DOES:    encode the goal image into a goal latent, roll every candidate forward, score with criterion.
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
