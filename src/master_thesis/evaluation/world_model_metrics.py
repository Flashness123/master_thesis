import torch


@torch.no_grad()
def world_model_metrics(model, embeddings, context_embeddings, context_actions, targets, predictions, actions, ):
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

  if embeddings.shape[0] > 1:  # the swap needs a second sample in the batch
    swapped_actions = context_actions.clone()  # real actions e(a0), e(a1), e(a2)
    swapped_actions[:, -1] = context_actions[:, -1].roll(shifts=1, dims=0)  # replace only a2 with the neighbour's a2; a0, a1 stay consistent with the states
    swapped_prediction = model.predict(context_embeddings, swapped_actions)[:, -1]  # ẑ3 under the wrong a2
    changed = (actions[:, -2] != actions[:, -2].roll(shifts=1, dims=0)).any(dim=-1)  # did the swap really change a2? (actions holds a0..a3, so a2 is index -2)
    extra_error = (swapped_prediction - targets[:, -1]).square().mean(dim=-1) - (predictions[:, -1] - targets[:, -1]).square().mean(dim=-1)  # per sample: error with wrong a2 minus error with real a2
    if changed.any():
      metrics["action_gap"] = extra_error[changed].mean()  # > 0: the prediction depends on the action; only samples whose a2 really changed

  rollout = context_embeddings[:, :1]  # start with the real z0 only
  for step in range(context_embeddings.shape[1]):  # 3 steps: predict ẑ1, then ẑ2, then ẑ3
    next_state = model.predict(rollout, context_actions[:, :step + 1])[:, -1:]  # the last position predicts the next state
    rollout = torch.cat([rollout, next_state], dim=1)  # feed the prediction back in as context
  metrics["rollout_loss"] = (rollout[:, -1] - targets[:, -1]).square().mean()  # ẑ3 (3 steps from z0, own predictions) vs real z3
  metrics["rollout_copy_loss"] = (context_embeddings[:, 0] - targets[:, -1]).square().mean()  # baseline: "z3 = z0"

  return metrics


def planning_metrics(rows):
  """Summarize the attempts of evaluate_planning (evaluation/arc_policies.py) into scalars for TensorBoard."""
  import numpy as np

  distances = [row["final_distance"] for row in rows if row["final_distance"] is not None]
  metrics = {"planning/success_rate": float(np.mean([row["reached"] for row in rows])), "planning/mean_steps": float(np.mean([row["steps"] for row in rows])), "planning/final_distance": float(np.mean(distances)) if distances else float("nan")}

  for goal in sorted({row["goal_steps"] for row in rows}):  # one success rate per goal distance
    metrics[f"planning/success_rate_goal{goal}"] = float(np.mean([row["reached"] for row in rows if row["goal_steps"] == goal]))

  return metrics
