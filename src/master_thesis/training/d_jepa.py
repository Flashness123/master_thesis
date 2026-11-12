"""
Fit D-JEPA's relational ranker (models/lewm/d_jepa.py) on top of one trained LeWM checkpoint, from a collection of
labelled situations (environments/d_jepa_collection.py: per situation a start, a goal 5 moves ahead and the real outcome
of all 1,024 five-move plans; the same collection serves every checkpoint).

Steps:
  1. The checkpoint imagines all 1,024 plans of every situation and scores them as the exhaustive planner does
     (lewm_analysis.ExhaustivePlanner: distance to the goal latent, closest approach or last state); select_candidates
     keeps 63 per situation, each described by its imagined latent closest to the goal and its rank.
  2. The ranker is trained on the situations of the fit levels with their loss and settings (relational_loss in their
     code, pusht-relational.yaml): 1,000 AdamW updates of 8 situations, lr 3e-4, weight decay 1e-4, gradient clip 1.
     Only situations where some of the 63 kept plans reach the goal and some do not can teach it.
  3. Every 50 updates the gate threshold is calibrated on the situations of the calibration levels (the threshold with
     the most successes); the update with the most calibration successes is kept (then the smallest corrections, then
     the earliest update), as their checkpoint choice.
  4. Report on the calibration and the test levels: how often the plain planner's choice (the cheapest plan), the
     gated ranker's choice, and at least one of the 63 kept plans reach the goal. The test levels decide nothing.

Output: models/d_jepa/<checkpoint id>/ranker.pt (state_dict) and config.json (tau, the chosen update, the settings, the
results and the calibration history).

Usage:
  python -m master_thesis.training.d_jepa --name syn1000 --checkpoint <run>/weights.pt
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from master_thesis.paths import d_jepa_data_dir, model_dir

GAME = "ls20"


def fit(args):
  """
  GETS:    args -- name, checkpoint, device, cost_on, updates, batch, lr, weight_decay, clip, temperature, margin, tail,
           pair_weight, trust_weight, evaluate_every, seed, output (see main).
  DOES:    the steps of the module docstring.
  RETURNS: the output folder with ranker.pt and config.json.
  """
  from master_thesis.evaluation.lewm_analysis import ACTION_DIM, ACTIONS, load_checkpoint
  from master_thesis.models.lewm.d_jepa import RelationalRanker, gate_choice, select_candidates
  from master_thesis.training.lewm import ArcGridToPixels

  folder = d_jepa_data_dir(GAME, args.name)
  parts = [np.load(path) for path in sorted(folder.glob("part*.npz"))]
  settings = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("part*.json"))]
  if not parts or len({setting["horizon"] for setting in settings}) > 1:
    raise ValueError(f"{folder}: no collected parts, or parts with different horizons")
  data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0].files}
  horizon = settings[0]["horizon"]

  # 1. every plan of every situation imagined and scored as ExhaustivePlanner does; 63 kept, described by the imagined latent closest to the goal
  model, info = load_checkpoint(args.checkpoint, args.device)
  to_pixels = ArcGridToPixels(info["img_size"])
  pixels = lambda grid: to_pixels(torch.as_tensor(np.asarray(grid).reshape(1, -1), device=args.device))[None]  # with a batch and a time axis of 1
  sequences = torch.tensor(list(itertools.product(range(len(ACTIONS)), repeat=horizon)), device=args.device)  # the plans, in the order of `reached`  [P, horizon]
  plans = F.pad(F.one_hot(sequences, len(ACTIONS)).float(), (0, ACTION_DIM - len(ACTIONS)))[None]  # as LeWM's 9 numbers per move  [1, P, horizon, 9]
  dummy_action = torch.zeros(1, 1, ACTION_DIM, device=args.device)  # JEPA.get_cost needs an action entry; the plans replace it
  closest, goals, success = [], [], []
  with torch.no_grad():
    for start, goal, reached in zip(data["start"], data["goal"], data["reached"]):
      batch = {name: value.unsqueeze(1).expand(1, len(sequences), *value.shape[1:]) for name, value in (("pixels", pixels(start)), ("goal", pixels(goal)), ("action", dummy_action))}
      model.get_cost(batch, plans)  # imagines every plan; fills batch["predicted_emb"] [1, P, 1 + horizon, D] and batch["goal_emb"] [1, 1, D]
      imagined, goal_latent = batch["predicted_emb"][0, :, 1:], batch["goal_emb"][0, -1]  # [P, horizon, D], [D]
      distance = (imagined - goal_latent).square().sum(dim=-1)  # after every move  [P, horizon]
      cost = distance.min(dim=-1).values if args.cost_on == "path" else distance[:, -1]
      by_end = torch.argsort(distance[:, -1], stable=True)
      order = by_end[torch.argsort(cost[by_end], stable=True)]  # the planner's preference: cost, then end distance, then enumeration order
      kept = select_candidates(order)  # [63], cheapest first
      step = distance[kept].argmin(dim=-1) if args.cost_on == "path" else torch.full((len(kept),), horizon - 1, device=args.device)  # where each kept plan gets closest to the goal
      closest.append(imagined[kept, step])
      goals.append(goal_latent)
      success.append(torch.as_tensor(reached, device=args.device)[kept] > 0)
  closest, goals, success = torch.stack(closest), torch.stack(goals), torch.stack(success)  # [N, 63, D], [N, D], [N, 63]
  base = (torch.arange(success.shape[1], device=args.device, dtype=torch.float32) / (success.shape[1] - 1)).expand(success.shape).contiguous()  # rank / (K - 1): kept is sorted by cost
  part = {name: torch.as_tensor(data["part"] == name, device=args.device) for name in ("fit", "calibration", "test")}
  fit_rows = torch.nonzero(part["fit"] & success.any(dim=1) & ~success.all(dim=1))[:, 0]  # their loss needs both outcomes among the 63
  rows_of = {name: torch.nonzero(part[name])[:, 0] for name in ("calibration", "test")}
  print(f"{len(data['start'])} situations: {len(fit_rows)} to fit on, {len(rows_of['calibration'])} to calibrate on, {len(rows_of['test'])} to test on")

  ranker = RelationalRanker(closest.shape[-1]).to(args.device)
  optimizer = torch.optim.AdamW(ranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  generator = torch.Generator().manual_seed(args.seed)

  def choices(rows, threshold):
    """The gated choice and the ranker's corrections on some situations."""
    with torch.no_grad():
      scores, correction = ranker(closest[rows], goals[rows], base[rows])
    return gate_choice(base[rows], scores, threshold), scores, correction

  def calibrate():
    """Their gate calibration: on the calibration situations, the threshold with the most successes (then the most changed choices, then the smallest threshold); also the mean squared correction there."""
    rows = rows_of["calibration"]
    _, scores, correction = choices(rows, float("inf"))
    advantage = torch.unique(base[rows].min(dim=1).values - scores.min(dim=1).values)  # sorted; every threshold between two advantages decides the same way
    thresholds = [float("inf"), *((advantage[1:] + advantage[:-1]) / 2).tolist(), float(advantage[0]) - 1e-6]
    record = lambda threshold: (int(success[rows].gather(1, gate_choice(base[rows], scores, threshold)[:, None]).sum()), int((gate_choice(base[rows], scores, threshold) != 0).sum()), -threshold)
    successes, _, negative_threshold = max(record(threshold) for threshold in thresholds)
    return successes, -negative_threshold, float(correction.square().mean())

  # 2. + 3. train, calibrate the gate every evaluate_every updates, keep the best update
  best, history = None, []
  for update in range(1, args.updates + 1):
    ranker.train()
    rows = fit_rows[torch.randperm(len(fit_rows), generator=generator)[:args.batch].to(fit_rows.device)]  # `batch` fit situations, none twice
    scores, correction = ranker(closest[rows], goals[rows], base[rows])
    # their loss (relational_loss): listwise -log P(a successful plan), P = softmax over -score / T; a margin on the
    # success/failure pairs among the `tail` lowest current scores (all pairs of the situation if those hold only one
    # outcome); a penalty on large corrections
    scaled = -scores / args.temperature
    listwise = (torch.logsumexp(scaled, dim=1) - torch.logsumexp(scaled.masked_fill(~success[rows], float("-inf")), dim=1)).mean()
    pairs = []
    for row in range(len(rows)):
      outcome, row_scores = success[rows[row]], scores[row]
      tail = torch.argsort(row_scores.detach())[:args.tail]
      if outcome[tail].all() or not outcome[tail].any():
        tail = torch.arange(len(row_scores), device=row_scores.device)
      good, bad = row_scores[tail][outcome[tail]], row_scores[tail][~outcome[tail]]
      pairs.append(F.softplus((args.margin + good[:, None] - bad[None, :]) / args.temperature).mean())
    loss = listwise + args.pair_weight * torch.stack(pairs).mean() + args.trust_weight * correction.square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ranker.parameters(), args.clip)
    optimizer.step()
    if update == 1 or update % args.evaluate_every == 0 or update == args.updates:
      ranker.eval()
      successes, threshold, trust = calibrate()
      history.append({"update": update, "loss": loss.item(), "calibration_successes": successes, "tau": threshold, "trust": trust})
      print(json.dumps(history[-1]))
      if best is None or (successes, -trust, -update) > best["key"]:  # their checkpoint choice: most calibration successes, then the smallest corrections, then the earliest update
        best = {"key": (successes, -trust, -update), "update": update, "tau": threshold, "state": copy.deepcopy(ranker.state_dict())}

  # 4. report on the calibration and the test levels
  ranker.load_state_dict(best["state"])
  results = {}
  for name, rows in rows_of.items():
    chosen, _, _ = choices(rows, best["tau"])
    results[name] = {"situations": len(rows), "plain_planner": float(success[rows, 0].float().mean()),  # the exhaustive planner's choice = the cheapest kept plan
                     "d_jepa": float(success[rows].gather(1, chosen[:, None]).float().mean()), "changed": float((chosen != 0).float().mean()),  # how often the gate took the ranker's choice
                     "some_candidate": float(success[rows].any(dim=1).float().mean())}  # the best any reordering of the 63 could do
  output = model_dir("d_jepa", info["id"]) if args.output is None else args.output
  output.mkdir(parents=True, exist_ok=True)
  torch.save(best["state"], output / "ranker.pt")
  config = {"checkpoint": info["id"], "weights": args.checkpoint, "dataset": args.name, "horizon": horizon, "cost_on": args.cost_on, "lowest": 16, "total": success.shape[1], "tau": best["tau"], "update": best["update"],
            "fit_situations": len(fit_rows), "results": results, "settings": {key: str(value) for key, value in vars(args).items()}, "history": history}
  (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
  for name, result in results.items():
    print(f"{name} situations reaching the goal: plain planner {result['plain_planner']:.3f}, D-JEPA {result['d_jepa']:.3f} (choice changed in {result['changed']:.1%}), some candidate {result['some_candidate']:.3f}")
  print(f"-> {output}")
  return output


def main() -> None:
  parser = argparse.ArgumentParser(description="Fit D-JEPA's ranker and its gate on top of one LeWM checkpoint (GPU)")
  parser.add_argument("--name", required=True, help="the collection: datasets/ls20/d_jepa_<name>/ (environments/d_jepa_collection.py)")
  parser.add_argument("--checkpoint", required=True, help="LeWM weights file, below models/lewm/ or absolute")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--cost-on", choices=["path", "end"], default="path", help="the planner's cost: closest imagined state (path) or the last one (end); must match the planner the ranker is used with")
  parser.add_argument("--updates", type=int, default=1000)
  parser.add_argument("--batch", type=int, default=8, help="situations per update")
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--clip", type=float, default=1.0)
  parser.add_argument("--temperature", type=float, default=0.05)
  parser.add_argument("--margin", type=float, default=0.02)
  parser.add_argument("--tail", type=int, default=16, help="the pair loss looks at the success/failure pairs among this many lowest scores")
  parser.add_argument("--pair-weight", type=float, default=0.25)
  parser.add_argument("--trust-weight", type=float, default=0.1)
  parser.add_argument("--evaluate-every", type=int, default=50, help="updates between gate calibrations")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--output", type=Path, default=None, help="default: models/d_jepa/<checkpoint id>/")
  fit(parser.parse_args())


if __name__ == "__main__":
  main()
