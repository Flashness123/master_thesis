import torch


@torch.no_grad()
def world_model_metrics(model, embeddings, context_embeddings, context_actions, targets, predictions, ):
  """
  Diagnostics on recorded histories; call with the model in eval mode.
  context_embeddings │ embeddings[:, :3]         │ z0 z1 z2
  context_actions    │ action_embeddings[:, :3]  │ e(a0) e(a1) e(a2)
  targets            │ embeddings[:, 1:]         │ z1 z2 z3
  predictions        │ predict(context, actions) │ ẑ1 ẑ2 ẑ3
  """
  if model.training:  # the predictor has dropout 0.1; in training mode the extra forward pass below would be random
    raise ValueError("Evaluate world-model metrics with the model in eval mode")

  prediction_loss = (predictions - targets).square().mean()  #MSE of ẑ1..ẑ3 vs z1..z3
  copy_loss = (context_embeddings - targets).square().mean()

  metrics = {"copy_loss": copy_loss, "latent_std": (embeddings.flatten(0, 1).std(dim=0, unbiased=False).mean()), }  # baseline "nothing changes"; spread of the embeddings (0 = collapse)
  # (action_gap, the extra error when a2 was swapped for another sample's action, is replaced by next_state/action_spread: see next_state_metrics)

  rollout = context_embeddings[:, :1]  # start with the real z0 only
  for step in range(context_embeddings.shape[1]):  # 3 steps: predict ẑ1, then ẑ2, then ẑ3
    next_state = model.predict(rollout, context_actions[:, :step + 1])[:, -1:]  # the last position predicts the next state
    rollout = torch.cat([rollout, next_state], dim=1)  # feed the prediction back in as context
  metrics["rollout_loss"] = (rollout[:, -1] - targets[:, -1]).square().mean()  # ẑ3 (3 steps from z0, own predictions) vs real z3
  metrics["rollout_copy_loss"] = (context_embeddings[:, 0] - targets[:, -1]).square().mean()  # baseline: "z3 = z0"

  return metrics


@torch.no_grad()
def next_state_metrics(model, states, actions, outcomes, goals):
  """
  Next-state prediction for EVERY action along recorded solutions; call with the model in eval mode.
  Unlike world_model_metrics, which runs on every validation batch and can only check the one action that was recorded there,
  this runs with the planning evaluation (PlanningEvaluation in training/lewm.py) on a fixed set of states in which the game was played with every action (arc_policies.next_state_targets),
  so it also checks the actions the planner weighs up but nobody pressed. The set stays the same, so the curves only move when the model does.
  S = states along the solutions, A = actions the planner can press (4 in ls20), 192 = latent size
  states   │ [S, 192]     │ z(s)     the states of the solutions
  actions  │ [A, 192]     │ e(a)     every action
  outcomes │ [S, A, 192]  │ z(s, a)  the frame the game really shows after a
  goals    │ [S, 192]     │ z(g)     the solution's state a few steps later (planning.waypoint_every), the planner's goal
  """
  num_states, num_actions = outcomes.shape[:2]
  predictions = model.predict(states.repeat_interleave(num_actions, dim=0)[:, None], actions.repeat(num_states, 1)[:, None])[:, -1].reshape(outcomes.shape)  # ẑ(s, a) for every state and action, from one frame like the planner's first step  [S, A, 192]

  # error: does the model predict the next state, for every action? Relative to predicting "nothing changes": 0 = perfect, 1 = no better than that
  error = (predictions - outcomes).square().sum() / (states[:, None] - outcomes).square().sum()

  # spearman: do the predicted outcomes order the actions by distance to the goal (the planner's cost) like the real ones? 1 = same order, 0 = unrelated, -1 = reversed.
  # Only the order counts: tiny errors still score low when the actions' real distances differ by even less (multi-level model: ~2 against ~300, experiments.txt)
  distance = lambda latents: (latents - goals[:, None]).square().sum(dim=-1)  # [S, A]
  rank = lambda d: (d[:, None, :] < d[:, :, None]).sum(dim=-1) + ((d[:, None, :] == d[:, :, None]).sum(dim=-1) + 1) / 2  # rank 1..A within each state; ties share their mean rank (two blocked moves show the same frame): 1, 2.5, 2.5, 4
  predicted, real = [r - r.mean(dim=-1, keepdim=True) for r in (rank(distance(predictions)), rank(distance(outcomes)))]  # centered ranks  [S, A]
  ordered = real.square().sum(dim=-1) > 0  # states whose real distances are not all equal; the others have no order to predict
  spearman = ((predicted * real).sum(dim=-1) / (predicted.square().sum(dim=-1) * real.square().sum(dim=-1)).sqrt().clamp_min(1e-12))[ordered].mean()  # correlation of the ranks per state, mean over states; all predictions equal -> 0

  # action_spread: how much does one move change the real distance to the goal? Encoder only, no predictor. ~0 = all moves look alike, the planner has no direction (level 2: single-level model 37, multi-level 2.5)
  action_spread = distance(outcomes).std(dim=-1, unbiased=False).mean()

  return {"error": error, "spearman": spearman, "action_spread": action_spread}


def planning_metrics(rows):
  """Summarize the attempts of evaluate_planning (evaluation/arc_policies.py) into scalars for TensorBoard."""
  import numpy as np

  distances = [row["final_distance"] for row in rows if row["final_distance"] is not None]
  metrics = {"planning/success_rate": float(np.mean([row["reached"] for row in rows])), "planning/mean_steps": float(np.mean([row["steps"] for row in rows])), "planning/final_distance": float(np.mean(distances)) if distances else float("nan")}

  for goal in sorted({row["goal_steps"] for row in rows}):  # one success rate per goal distance
    metrics[f"planning/success_rate_goal{goal}"] = float(np.mean([row["reached"] for row in rows if row["goal_steps"] == goal]))

  return metrics


def waypoint_metrics(rows):
  """Summarize the attempts of evaluate_waypoints (evaluation/arc_policies.py) into scalars for TensorBoard."""
  import numpy as np

  reached = lambda selected: float(np.mean([row["waypoints_reached"] / row["waypoints_total"] for row in selected]))  # share of waypoints reached
  metrics = {"waypoints/completion_rate": float(np.mean([row["completed"] for row in rows])), "waypoints/reached_fraction": reached(rows)}

  for level in sorted({row["level"] for row in rows}):  # one curve per level (e.g. the held-out level next to the training levels)
    metrics[f"waypoints/level{level}_reached_fraction"] = reached([row for row in rows if row["level"] == level])

  return metrics
