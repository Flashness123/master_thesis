import torch


@torch.no_grad()
def world_model_metrics(model, embeddings, context_embeddings, context_actions, targets, predictions, actions, ):
  """Diagnostics on recorded histories; call with the model in eval mode."""
  if model.training:
    raise ValueError("Evaluate world-model metrics with the model in eval mode")

  prediction_loss = (predictions - targets).square().mean()
  copy_loss = (context_embeddings - targets).square().mean()

  metrics = {"copy_loss": copy_loss, "copy_gap": copy_loss - prediction_loss, "batch_latent_std": (embeddings.flatten(0, 1).std(dim=0, unbiased=False).mean()), }

  if embeddings.shape[0] > 1:
    # Permute action histories between samples without changing their order.
    mismatched_predictions = model.predict(context_embeddings, context_actions.roll(shifts=1, dims=0), )
    mismatched_loss = (mismatched_predictions - targets).square().mean()
    history_size = context_actions.shape[1]
    action_histories = actions[:, :history_size]
    changed_contexts = (action_histories != action_histories.roll(shifts=1, dims=0)).flatten(start_dim=1).any(dim=1)
    metrics.update({"mismatched_action_loss": mismatched_loss, "action_gap": mismatched_loss - prediction_loss, "changed_action_context_fraction": (changed_contexts.float().mean()), })

  return metrics
