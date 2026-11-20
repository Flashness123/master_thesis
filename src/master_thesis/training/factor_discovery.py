"""
Blind-spot discovery: which screen regions hold outcome-relevant information that a trained LeWM's latent has lost?
Found from frames, actions and the episode ends alone (level completed, or a lost life), without the game engine and
without anyone naming what matters. The regions, and every look each of them takes in the data, become the targets of
factor heads (training/lewm.py extensions.factor_heads).

Windows of recorded play from the validation episodes of the checkpoint's own split (the model never trained on them):
a start frame o_t and the next 25 actions, with outcome targets for k = 1, 5, 10, 25 moves that do not depend on the
encoder's blind spots (25 because some details matter only much later, e.g. a refill when the energy runs out):
  change    which 4x4 patches of the 64x64 grid differ between o_t and o_t+k (256 bits; the episode's last frame when it
            ends earlier)
  events    level completed / game over within k moves (an episode's last row: completed if it succeeded, else a lost
            life; on syn1000 every unsuccessful episode ends with one)
  residual  the model's own k-step error E(o_t+k) - rollout_k(z_t, a), standardised per dimension (only while the
            episode lasts k moves)
Event sampling: rare things are rare in uniform windows (a refill pickup lies in 7% of them on syn1000), so a share of
the windows (event_share) starts within 25 moves before a large change: a transition inside an episode that changes more
pixels than the event_quantile of all of them (pickups, key changes, ...; the transition into an episode's last row is
not counted, episode ends are frequent anyway). The rest start anywhere.
Two small networks predict all of them, one from the latent z_t, one from z_t and the one-hot grid. Where the second
does better, the pixels hold outcome information the latent lost. Each 4x4 patch of o_t is then replaced in turn by the
same patch of another frame, on the test windows with the largest gap; how much worse the second network gets per
patch is a heat map (z_t stays the latent of the real frame, so the map shows reliance on pixels beyond the latent).
Neighbouring patches with a high share form regions; per region, every distinct look in the sampled frames that occurs
often enough is one code.

Output (--output, default models/lewm/<checkpoint id>/factors/):
  factors.npz   heat [3, 16, 16] (change, events, residual), share [16, 16], per region i its codebook codebook_<i>
                [K_i, rows, columns] uint8, and with --objects per object family k its exact shapes object_<k>_template_<j>
                [height, width] uint8 (1 = the object's pixels)
  factors.json  settings, test losses, per region its pixel box [row0, row1, col0, col1] (end excluded), share of the
                heat map, codebook size and importance, per object family its colour, size, number of shapes, importance
                (t-value), room (fraction of one move) and max_count / max_area (its largest count and area in the
                windows' frames; the object heads in training divide by them), and per round the remaining gap and the
                weights of the found factors' futures

With --objects a second stage finds objects that sit in different places (refills, goal symbols, pushers, ...), one family
after another. Each round retrains the two networks with everything found so far as known input (the regions' looks and
the found families' occupancy maps in o_t), so what they explain no longer counts, and with the found factors' values in
o_t+k as further targets (a region's look; a family's four numbers as the object heads in training: count, area, centre
row and column): whatever changes an important factor is important itself (a refill makes the energy bar jump, which the
outcome targets alone do not show: the bar changes with every move anyway). Each found factor has an importance
in outcome units (a region: how much worse the latent + pixels network predicts with it covered; a family: its mean
importance when found), and its future counts with weight importance / (the latent network's loss on that future).
Every object of the windows with the largest remaining gap is removed in turn, from the frame and its latent; per family
of one colour and height: importance = how much worse the latent + pixels network predicts without it (outcome targets +
weighted factor futures), room = how far the latent moves without it. Two tests per family, over all its objects and over
those touched within the window (its pixels change: picked up, pushed, ...), each at the Bonferroni threshold for two
tests; the family with the highest importance / (room + 0.01) that passes one (t >= threshold, score >= min_score) is
added; else it stops. The remaining gap (mean over all test windows) is only logged, not a stop: rare details (a refill)
hardly move a mean, but they show in the windows where they occur.

Usage: python -m master_thesis.training.factor_discovery --checkpoint <weights.pt> [--windows 20000] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import NormalDist

import lance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from master_thesis.paths import dataset_path, model_dir

GAME = "ls20"
K = (1, 5, 10, 25)  # moves ahead
SPAN = 25  # actions per window (the largest k)
PATCH = 4  # the 64x64 grid as 16x16 patches
KINDS = ("change", "events", "residual")


class Outcome(nn.Module):
  """
  GETS:    pixels / latent -- whether the network sees the one-hot grid and / or the latent; extra -- values given with
           the latent (the current values of factors found before, so that they are explained away; 0 = none);
           targets -- outputs per k for the found factors' future values (0 = none).
  DOES:    a small CNN on the grid (16 colours x 64 x 64 -> 512) and / or a dense layer on the latent and the extra values
           (192 + extra -> 512), joined with the 25 actions (one-hot, 0 = none after the episode's end) and an MLP.
  RETURNS: (change logits [n, 4, 256], event logits [n, 4, 2], residual [n, 4, 192], factor logits [n, 4, targets]) for
           k = 1, 5, 10, 25.
  """

  def __init__(self, pixels: bool, latent: bool, latent_dim: int, extra: int = 0, targets: int = 0):
    super().__init__()
    self.pixels, self.latent, self.latent_dim = pixels, latent, latent_dim
    width = 0
    if pixels:
      self.cnn = nn.Sequential(nn.Conv2d(16, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 4, stride=4), nn.GELU(), nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
                               nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GELU(), nn.Flatten(), nn.Linear(128 * 16, 512), nn.GELU())
      width += 512
    if latent:
      self.dense = nn.Sequential(nn.Linear(latent_dim + extra, 512), nn.LayerNorm(512), nn.GELU())
      width += 512
    self.act = nn.Linear(SPAN * 5, 64)
    self.trunk = nn.Sequential(nn.Linear(width + 64, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU())
    self.heads = nn.Linear(512, len(K) * (256 + 2 + latent_dim + targets))

  def forward(self, grid, z, action):
    parts = []
    if self.pixels:
      parts.append(self.cnn(F.one_hot(grid.long(), 16).reshape(-1, 64, 64, 16).permute(0, 3, 1, 2).float()))
    if self.latent:
      parts.append(self.dense(z))
    parts.append(self.act(F.one_hot(action.long(), 5).reshape(len(action), -1).float()))
    out = self.heads(self.trunk(torch.cat(parts, dim=1))).reshape(len(action), len(K), -1)
    return out[..., :256], out[..., 256:258], out[..., 258:258 + self.latent_dim], out[..., 258 + self.latent_dim:]


def discover(checkpoint: str, windows_wanted: int, device: str, output: Path | None, seed: int, epochs: int, threshold: float, min_share: float, min_count: int, max_codes: int,
             objects: bool = False, min_t: float = 2.0, min_score: float = 1.0, max_templates: int = 16, max_families: int = 16,
             event_quantile: float = 0.98, event_share: float = 0.5):
  """
  GETS:    checkpoint -- a LeWM weights file; windows_wanted -- windows sampled from its validation episodes;
           device; output -- folder for factors.npz / .json (None = models/lewm/<checkpoint id>/factors/); seed;
           epochs -- training epochs of the two networks; threshold -- a patch belongs to a region when its heat share
           is at least threshold / 256; min_share -- a region needs this much of the heat map; min_count -- a look of a
           region becomes a code when it occurs this often in the sampled frames; max_codes -- at most this many codes
           per region (the most frequent); objects -- also find object families (below); min_t / min_score -- a family is kept
           when its importance has at least this t-value and importance / (room + 0.01) is at least min_score;
           max_templates -- at most this many exact shapes per family (the most frequent); max_families -- at most this many
           rounds / families; event_quantile -- a transition is a large change when it changes more pixels than this
           quantile of all transitions; event_share -- share of the windows that start within 25 moves before one.
  DOES:    the module docstring's steps: windows and targets, latents and the model's k-step errors, the latent and the
           latent + pixels networks, the gap, the occlusion heat map, regions and codebooks. With objects, one family after
           another (module docstring): retrain with everything found as known input and the found factors' futures as
           targets (weighted by their importance), add the best family, stop when none is reliably important and small
           in the latent.
  RETURNS: the output folder.
  """
  from master_thesis.evaluation import lewm_analysis as la

  rng = np.random.default_rng(seed)
  torch.manual_seed(seed)
  started = time.time()
  say = lambda text: print(f"[{time.time() - started:6.0f} s] {text}", flush=True)
  model, info = la.load_checkpoint(checkpoint, device)
  episodes, version, _ = la.rebuild_training_split(info)
  dataset = info["dataset"]
  val = episodes[(episodes.part == "val") & (episodes.length >= 2)].reset_index(drop=True)
  order = rng.permutation(len(val))
  fit_episodes = set(order[: int(0.8 * len(order))].tolist())

  # large changes: changed pixels of every transition of the validation episodes (all of them read once, in chunks; one row of overlap per chunk)
  lengths_all = val.length.to_numpy()
  episode_start = np.concatenate([[0], np.cumsum(lengths_all)[:-1]])  # position of each episode's first row in all_rows
  episode_last = episode_start + lengths_all - 1
  all_rows = np.concatenate([np.arange(offset, offset + length) for offset, length in zip(val.offset, lengths_all)])
  changed = np.zeros(len(all_rows), np.int32)  # pixels that differ from row i to row i + 1
  for chunk in range(0, len(all_rows) - 1, 100000):
    part = la.read_recorded_rows(dataset, version, all_rows[chunk:chunk + 100001])[0]
    changed[chunk:chunk + len(part) - 1] = (part[:-1] != part[1:]).sum(1)
  inner = np.ones(len(all_rows), bool)
  inner[episode_last] = False  # across two episodes
  inner[episode_last - 1] = False  # into an episode's last row (episode ends are events of their own, and frequent)
  limit = np.quantile(changed[inner], event_quantile)
  large = inner & (changed > limit)
  episode_of = np.repeat(np.arange(len(val)), lengths_all)
  start_positions = np.flatnonzero(np.isin(np.arange(len(all_rows)), episode_last, invert=True))  # every row with a next move
  counted = np.concatenate([[0], np.cumsum(large)])
  ends = np.minimum(start_positions + SPAN, episode_last[episode_of[start_positions]])
  event_positions = start_positions[counted[ends] - counted[start_positions] > 0]  # a large change within the next 25 moves

  # windows: event_share of them start before a large change, the rest anywhere (every move of every episode equally likely)
  events_wanted = int(rng.binomial(windows_wanted, event_share)) if len(event_positions) else 0
  positions = rng.permutation(np.concatenate([rng.choice(event_positions, events_wanted), rng.choice(start_positions, windows_wanted - events_wanted)]))
  episode_index = episode_of[positions]
  offsets, lengths, starts = val.offset.to_numpy()[episode_index], lengths_all[episode_index], positions - episode_start[episode_index]
  success, fit_mask = val.success.to_numpy()[episode_index].astype(bool), np.isin(episode_index, list(fit_episodes))
  windows = positions
  left = lengths - 1 - starts  # moves until the episode's last row
  valid = np.stack([left >= k for k in K], axis=1)  # [N, 4]: the window has the k-th frame (needed by the residual)
  frame_rows = offsets[:, None] + np.minimum(starts[:, None] + np.array((0,) + K)[None], lengths[:, None] - 1)  # frames t, t+1, t+5, t+10, t+25 (clipped to the episode's last row)
  grids = la.read_recorded_rows(dataset, version, frame_rows.reshape(-1))[0].reshape(len(windows), 1 + len(K), 4096)
  action_rows = offsets[:, None] + np.minimum(starts[:, None] + np.arange(SPAN)[None], lengths[:, None] - 1)
  table = lance.dataset(str(dataset_path(GAME, dataset)), version=version)
  actions = table.take(action_rows.reshape(-1), columns=["action"]).column("action").combine_chunks().flatten().to_numpy().reshape(len(windows), SPAN, 3)[:, :, 0].astype(int)
  actions = np.where((np.arange(SPAN)[None] < left[:, None]) & (actions >= 1) & (actions <= 4), actions, 0)  # no action after the episode's end
  completed = np.stack([success & (left <= k) for k in K], axis=1).astype(np.float32)
  game_over = np.stack([~success & (left <= k) for k in K], axis=1).astype(np.float32)
  change = np.stack([(grids[:, 0] != grids[:, 1 + i]).reshape(-1, 16, PATCH, 16, PATCH).any(axis=(2, 4)).reshape(-1, 256) for i in range(len(K))], axis=1).astype(np.float32)
  say(f"{len(windows)} windows from {len(val)} validation episodes of {dataset} ({events_wanted} before a large change: > {limit:.0f} changed pixels, "
      f"{large.sum()} of {inner.sum()} transitions); completed / game over within 25 moves: {int(completed[:, -1].sum())} / {int(game_over[:, -1].sum())}")

  # latents and the model's own k-step errors (rollouts from one real frame, as the planner)
  latents = la.encode_grids(model, info["img_size"], grids.reshape(-1, 4096), device).reshape(len(windows), 1 + len(K), la.D)
  with torch.no_grad():
    moves = model.action_encoder(torch.eye(len(la.ACTIONS), la.ACTION_DIM, device=device)[:, None])[:, 0]
    rolled = np.concatenate([model.predict_rollout(torch.as_tensor(latents[c:c + 1024, 0][:, None], device=device), moves[torch.as_tensor(np.maximum(actions[c:c + 1024], 1) - 1, device=device)],
                                                   info["history_size"]).float().cpu().numpy()[:, [k - 1 for k in K]] for c in range(0, len(windows), 1024)])
  residual = latents[:, 1:] - rolled  # [N, 4, 192]
  residual = residual / (np.stack([residual[fit_mask & valid[:, i], i].std(axis=0) for i in range(len(K))])[None] + 1e-6)  # per k and dimension, over the windows that last k moves
  say("latents and rollouts done")

  def losses(net, index, grid=None):  # per window: change, events (every k), residual (the k the episode lasts) loss summed over k  -> [n, 3]
    g = torch.as_tensor(grids[index, 0] if grid is None else grid, device=device)
    change_logits, event_logits, residual_prediction, _ = net(g, torch.as_tensor(latents[index, 0], device=device), torch.as_tensor(actions[index], device=device))
    m = torch.as_tensor(valid[index], device=device).float()
    change_loss = F.binary_cross_entropy_with_logits(change_logits, torch.as_tensor(change[index], device=device), reduction="none").mean(-1)
    event_loss = F.binary_cross_entropy_with_logits(event_logits, torch.as_tensor(np.stack([completed[index], game_over[index]], axis=-1), device=device), reduction="none").sum(-1)
    residual_loss = (residual_prediction - torch.as_tensor(residual[index], device=device)).pow(2).mean(-1)
    return torch.stack([change_loss.sum(1), event_loss.sum(1), (residual_loss * m).sum(1)], dim=1)

  fit, test = np.flatnonzero(fit_mask), np.flatnonzero(~fit_mask)
  nets = {}
  for name, pixels in (("latent", False), ("both", True)):
    net = Outcome(pixels, True, la.D).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
      net.train()
      for batch in np.array_split(rng.permutation(fit), max(1, len(fit) // 256)):
        optimizer.zero_grad()
        losses(net, batch).sum(1).mean().backward()
        optimizer.step()
    nets[name] = net.eval()
    say(f"trained the {name} network")
  with torch.no_grad():
    test_loss = {name: torch.cat([losses(net, b) for b in np.array_split(test, max(1, len(test) // 512))]).cpu().numpy() for name, net in nets.items()}
  gap = test_loss["latent"] - test_loss["both"]  # [n_test, 3]
  say("test loss " + " | ".join(f"{kind}: latent {test_loss['latent'][:, i].mean():.4f}, both {test_loss['both'][:, i].mean():.4f}" for i, kind in enumerate(KINDS)))

  # occlusion heat map on the test windows with the largest gap (change + events)
  top = test[np.argsort(-(gap[:, 0] + gap[:, 1]))[:512]]
  donors = grids[rng.permutation(test)[: len(top)], 0].reshape(-1, 64, 64)
  heat = np.zeros((len(KINDS), 16, 16))
  with torch.no_grad():
    base = losses(nets["both"], top).cpu().numpy()
    for pr in range(16):
      for pc in range(16):
        occluded = grids[top, 0].reshape(-1, 64, 64).copy()
        rows, columns = slice(pr * PATCH, (pr + 1) * PATCH), slice(pc * PATCH, (pc + 1) * PATCH)
        occluded[:, rows, columns] = donors[:, rows, columns]
        heat[:, pr, pc] = (losses(nets["both"], top, occluded.reshape(-1, 4096)).cpu().numpy() - base).mean(0)
  positive = np.clip(heat, 0, None)
  share = (positive / (positive.sum(axis=(1, 2), keepdims=True) + 1e-12)).mean(0)  # the three kinds' heat maps, each normalised to 1, averaged

  # regions: connected patches (4-neighbourhood) with share >= threshold / 256, keeping those with >= min_share in total
  selected, seen, regions = share >= threshold / 256, np.zeros((16, 16), bool), []
  for r in range(16):
    for c in range(16):
      if not selected[r, c] or seen[r, c]:
        continue
      stack, cells = [(r, c)], []
      seen[r, c] = True
      while stack:
        cr, cc = stack.pop()
        cells.append((cr, cc))
        for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
          if 0 <= nr < 16 and 0 <= nc < 16 and selected[nr, nc] and not seen[nr, nc]:
            seen[nr, nc] = True
            stack.append((nr, nc))
      if sum(share[cr, cc] for cr, cc in cells) >= min_share:
        rows, columns = [cell[0] for cell in cells], [cell[1] for cell in cells]
        regions.append({"box": [min(rows) * PATCH, (max(rows) + 1) * PATCH, min(columns) * PATCH, (max(columns) + 1) * PATCH], "share": float(sum(share[cr, cc] for cr, cc in cells)), "patches": len(cells)})

  # codebooks: every look of a region in the sampled frames that occurs at least min_count times (at most max_codes, the most frequent)
  frames = grids.reshape(-1, 64, 64)
  arrays = {"heat": heat, "share": share}
  for index, region in enumerate(regions):
    r0, r1, c0, c1 = region["box"]
    looks, counts = np.unique(frames[:, r0:r1, c0:c1].reshape(len(frames), -1), axis=0, return_counts=True)
    keep = np.argsort(-counts)[:max_codes]
    keep = keep[counts[keep] >= min_count]
    arrays[f"codebook_{index}"] = looks[keep].reshape(-1, r1 - r0, c1 - c0).astype(np.uint8)
    occluded = grids[top, 0].reshape(-1, 64, 64).copy()
    occluded[:, r0:r1, c0:c1] = donors[:, r0:r1, c0:c1]
    with torch.no_grad():  # importance in outcome units: how much worse the latent + pixels network predicts with the whole region covered (as an object family's importance)
      region_importance = float((losses(nets["both"], top, occluded.reshape(-1, 4096)).cpu().numpy() - base).sum(1).mean())
    region.update(codes=int(len(keep)), frames_covered=float(counts[keep].sum() / len(frames)), importance=region_importance)
    say(f"region {index}: pixel rows {r0}-{r1 - 1}, columns {c0}-{c1 - 1}, {region['share']:.0%} of the heat map, {len(keep)} codes covering {region['frames_covered']:.1%} of the sampled frames, importance {region_importance:.4f}")
  if not regions:
    say("no region found")

  # objects: blind spots that sit in different places (refills, goal symbols, pushers, ...), with the regions just found explained away
  families = []
  if objects:
    from scipy import ndimage

    def region_code(index, frame):  # code of region `index` in frame `frame` of every window (0 = o_t, 1.. = o_t+k); len(codebook) = a look not in it  -> [N]
      row0, row1, col0, col1 = regions[index]["box"]
      book = arrays[f"codebook_{index}"].reshape(len(arrays[f"codebook_{index}"]), -1)
      code = np.full(len(windows), len(book))
      for chunk in range(0, len(windows), 2048):
        content = grids[chunk:chunk + 2048, frame].reshape(-1, 64, 64)[:, row0:row1, col0:col1].reshape(-1, book.shape[1])
        match = (content[:, None] == book[None]).all(-1)
        code[chunk:chunk + 2048] = np.where(match.any(1), match.argmax(1), len(book))
      return code

    # known input: the found factors' values in o_t (one-hot per region; occupancy maps of families, added below), so the gap measures what is missing beyond them.
    # targets: the found factors' values in o_t+k, so whatever changes an important factor becomes important itself (a refill: the energy bar jumps)
    known, factor_targets = [], []
    for index, region in enumerate(regions):
      classes = len(arrays[f"codebook_{index}"]) + 1
      known.append(np.eye(classes, dtype=np.float32)[region_code(index, 0)])
      factor_targets.append({"name": f"region {index}", "kind": "code", "classes": classes, "future": np.stack([region_code(index, 1 + i) for i in range(len(K))], axis=1), "importance": region["importance"]})
    known = np.concatenate(known, axis=1) if known else np.zeros((len(windows), 0), np.float32)

    def known_losses(net, index, grid=None, z=None):  # losses() with the found factors given next to the latent, plus one loss per found factor's future value  -> [n, 3 + factors]
      g = torch.as_tensor(grids[index, 0] if grid is None else grid, device=device)
      zk = torch.as_tensor(np.concatenate([latents[index, 0] if z is None else z, known[index]], axis=1), device=device)
      change_logits, event_logits, residual_prediction, factor_logits = net(g, zk, torch.as_tensor(actions[index], device=device))
      m = torch.as_tensor(valid[index], device=device).float()
      change_loss = F.binary_cross_entropy_with_logits(change_logits, torch.as_tensor(change[index], device=device), reduction="none").mean(-1)
      event_loss = F.binary_cross_entropy_with_logits(event_logits, torch.as_tensor(np.stack([completed[index], game_over[index]], axis=-1), device=device), reduction="none").sum(-1)
      residual_loss = (residual_prediction - torch.as_tensor(residual[index], device=device)).pow(2).mean(-1)
      parts, start = [change_loss.sum(1), event_loss.sum(1), (residual_loss * m).sum(1)], 0
      for target in factor_targets:
        logits, future = factor_logits[..., start:start + target["classes"]], torch.as_tensor(target["future"][index], device=device)
        if target["kind"] == "code":  # the region's look after k moves
          parts.append(F.cross_entropy(logits.reshape(-1, target["classes"]), future.reshape(-1).long(), reduction="none").reshape(len(future), len(K)).sum(1))
        else:  # the family's four numbers after k moves (count, area, centre row, centre column; normalised as the object heads; the centre only where it has objects)
          has_objects = (future[..., :1] > 0).float()
          counted = torch.cat([torch.ones_like(has_objects), torch.ones_like(has_objects), has_objects, has_objects], dim=-1)
          parts.append(((logits - future).pow(2) * counted).sum(-1).sum(1))
        start += target["classes"]
      return torch.stack(parts, dim=1)

    # one family after another: each round retrains the two networks with everything found so far as known input (the regions' codes and the
    # found families' occupancy maps) and adds the one family that is most important for the outcome relative to its room in the latent, until no
    # family is reliably important and small in the latent any more
    from master_thesis.training.lewm import find_objects

    moved = np.sum((latents[fit, 1] - latents[fit, 0]) ** 2, axis=1)
    one_move = float(np.median(moved[valid[fit, 0] & (moved > 1e-6)]))
    region_known, found_keys, rounds = known, [], []
    for round_index in range(max_families + 1):
      known_nets = {}
      for name, pixels in (("latent", False), ("both", True)):
        torch.manual_seed(seed + 1)  # every round the same start values and batch order: rounds differ only in the known input, not in training noise
        order_rng = np.random.default_rng(seed + 1)
        net = Outcome(pixels, True, la.D, known.shape[1], sum(target["classes"] for target in factor_targets)).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
          net.train()
          for batch in np.array_split(order_rng.permutation(fit), max(1, len(fit) // 256)):
            optimizer.zero_grad()
            known_losses(net, batch).sum(1).mean().backward()
            optimizer.step()
        known_nets[name] = net.eval()
      with torch.no_grad():
        test_latent = torch.cat([known_losses(known_nets["latent"], b) for b in np.array_split(test, max(1, len(test) // 512))]).cpu().numpy()
        test_both = torch.cat([known_losses(known_nets["both"], b) for b in np.array_split(test, max(1, len(test) // 512))]).cpu().numpy()
      # a found factor's future counts by the factor's importance (outcome units) per unit of the latent network's loss on it: a candidate that makes the
      # future of an important factor as hard to predict as the latent alone finds it inherits the factor's whole importance
      factor_weights = np.array([max(target["importance"], 0.0) / (test_latent[:, 3 + i].mean() + 1e-6) for i, target in enumerate(factor_targets)])
      remaining = (test_latent - test_both) @ np.concatenate([[1.0, 1.0, 0.0], factor_weights])  # change + events + the found factors' futures, per test window
      gap_t = remaining.mean() / (remaining.std() / np.sqrt(len(remaining)) + 1e-9)
      rounds.append({"round": round_index, "known_families": len(families), "remaining_gap": float(remaining.mean()), "gap_t": float(gap_t),
                     "factor_weights": {target["name"]: float(weight) for target, weight in zip(factor_targets, factor_weights)}})
      say(f"round {round_index}: known {len(regions)} regions + {len(families)} families; remaining gap (change + events + factor futures) {remaining.mean():.4f} (t {gap_t:.1f}); "
          + ", ".join(f"{target['name']} weight {weight:.4f}" for target, weight in zip(factor_targets, factor_weights)))
      if round_index == max_families:  # no stop on the remaining gap itself: it is a mean over all windows, and rare details (a refill) hardly move it
        say(f"stop: {max_families} families")
        break

      # every object (connected area of one colour, up to 200 pixels) of the start frames of the windows with the largest remaining gap, removed one at a time
      top_objects = test[np.argsort(-remaining)[:1024]]
      signatures, members, removed = {}, [], []
      for window in top_objects:
        grid = grids[window, 0].reshape(64, 64)
        for colour in np.unique(grid):
          labels, _ = ndimage.label(grid == colour)
          for label_index, box in enumerate(ndimage.find_objects(labels), start=1):
            mask = labels[box] == label_index
            if mask.sum() > 200:
              continue  # floor, walls, frames
            type_id = signatures.setdefault((int(colour), mask.shape[0], mask.shape[1], mask.tobytes()), len(signatures))
            full = np.zeros((64, 64), bool)
            full[box] = mask
            ring = ndimage.binary_dilation(full) & ~full
            without = grid.copy()
            without[full] = np.bincount(grid[ring]).argmax() if ring.any() else grid.flat[0]
            inside_known = any(box[0].start < r1 and box[0].stop > r0 and box[1].start < c1 and box[1].stop > c0 for r0, r1, c0, c1 in (region["box"] for region in regions))
            touched = bool((grids[window, 1:].reshape(-1, 64, 64)[:, full] != grid[full]).any())  # its pixels change within the window (picked up, pushed, drawn over, ...)
            members.append((window, type_id, inside_known, touched))
            removed.append(without.reshape(-1))
      removed = np.stack(removed)
      owner = np.array([m[0] for m in members])
      removed_latents = la.encode_grids(model, info["img_size"], removed, device)
      with torch.no_grad():
        base = {int(w): b for w, b in zip(top_objects, torch.cat([known_losses(known_nets["both"], c) for c in np.array_split(top_objects, max(1, len(top_objects) // 512))]).cpu().numpy())}
        by_kind = np.concatenate([known_losses(known_nets["both"], owner[c:c + 512], removed[c:c + 512], removed_latents[c:c + 512]).cpu().numpy()
                                  - np.stack([base[int(w)] for w in owner[c:c + 512]]) for c in range(0, len(members), 512)])  # [members, 3 + factors]: per target how much worse the prediction gets without the object (frame and latent)
      via_factors = by_kind[:, 3:] @ factor_weights  # importance passed on by the found factors whose future the object changes
      importance = by_kind[:, :3].sum(1) + via_factors
      touched = np.array([m[3] for m in members])
      room = np.sum((removed_latents - latents[owner, 0]) ** 2, axis=1) / one_move  # how far the latent moves without the object, relative to one move

      # families: types of one colour and height, whatever their width or cutout (every goal symbol; every length of a bar, so a bar that is only partly
      # inside a found region does not come back as one family per length); the best one that is reliably important (t >= min_t) and small in the latent (score >= min_score)
      signature_of = {type_id: signature for signature, type_id in signatures.items()}
      family_of = {type_id: (colour, height) for type_id, (colour, height, width, _) in signature_of.items()}
      # two tests per family, over all its objects and over those touched within the window (an object acts only where the player gets to it; the
      # untouched ones dilute the mean); both use the Bonferroni threshold for two tests (t 2.27 for min_t 2)
      t_needed = NormalDist().inv_cdf(1 - (1 - NormalDist().cdf(min_t)) / 2)
      best, candidates = None, []
      for key in sorted({family_of[m[1]] for m in members} - set(found_keys)):
        chosen = np.array([family_of[m[1]] == key for m in members])
        if chosen.sum() < 5 or np.mean([m[2] for m, c in zip(members, chosen) if c]) >= 0.5:
          continue  # too rare, or inside a region found before
        tests = {}
        for test_name, selected in (("all", chosen), ("touched", chosen & touched)):
          if selected.sum() >= 5:
            values = importance[selected]
            tests[test_name] = {"t": float(values.mean() / (values.std() / np.sqrt(len(values)) + 1e-9)), "score": float(values.mean() / (np.median(room[selected]) + 0.01)), "objects": int(selected.sum()),
                                "importance": float(values.mean()), "via_factors": float(via_factors[selected].mean())}
        passed = [name for name, test_result in tests.items() if test_result["t"] >= t_needed and test_result["score"] >= min_score]
        rank = max((tests[name]["score"] for name in passed), default=max(test_result["score"] for test_result in tests.values()))
        candidates.append((rank, key, tests, passed))
        if passed and (best is None or rank > best[0]):
          best = (rank, key, chosen, tests, max(passed, key=lambda name: tests[name]["score"]))
      for rank, key, tests, passed in sorted(candidates, key=lambda c: -c[0])[:12]:  # the best candidates, also the rejected ones
        say(f"  candidate colour {key[0]}, height {key[1]}: " + " | ".join(f"{name}: score {r['score']:.2f}, t {r['t']:.1f}, {r['objects']} objects, importance {r['importance']:.4f} (via factors {r['via_factors']:.4f})" for name, r in tests.items())
            + ("" if passed else "  -> rejected"))
      if best is None:
        say("stop: no family is reliably important and small in the latent any more")
        break
      score, key, chosen, tests, test_name = best
      t_value, values = tests[test_name]["t"], importance[chosen]
      type_counts = {}
      for m, c in zip(members, chosen):
        if c:
          type_counts[m[1]] = type_counts.get(m[1], 0) + 1
      templates = sorted(type_counts, key=lambda type_id: -type_counts[type_id])[:max_templates]
      index = len(families)
      shapes = []
      for j, type_id in enumerate(templates):
        colour, height, width, packed = signature_of[type_id]
        arrays[f"object_{index}_template_{j}"] = np.frombuffer(packed, dtype=bool).reshape(height, width).astype(np.uint8)
        shapes.append(torch.as_tensor(arrays[f"object_{index}_template_{j}"], device=device))
      families.append({"colour": int(key[0]), "height": int(key[1]), "widths": sorted({int(signature_of[type_id][2]) for type_id in templates}), "templates": len(templates), "objects": int(chosen.sum()), "round": round_index,
                       "windows": int(len(set(owner[chosen].tolist()))), "importance": float(values.mean()), "t": float(t_value), "test": test_name, "tests": tests, "room": float(np.median(room[chosen])), "score": float(score)})
      found_keys.append(key)
      say(f"object family {index}: colour {key[0]}, height {key[1]}, widths {sorted({signature_of[type_id][2] for type_id in templates})}, {len(templates)} shapes, {int(chosen.sum())} objects | "
          f"passed the {test_name} test (t {t_value:.1f}, score {score:.2f}), importance over all its objects {values.mean():.4f}, room {np.median(room[chosen]):.4f} of a move")
      # from now on known: where this family's objects are in every start frame (16x16 occupancy: the position, so a pickup ahead can be foreseen); and a
      # target: its four numbers after k moves, as the object heads in training (find_objects; count and area divided by their largest value in these
      # windows, the centre by 64). Its importance for passing on is the mean over all its objects (not the touched ones only: what it is worth on average)
      occupancy, descriptor = [], []
      with torch.no_grad():
        for frame in range(1 + len(K)):
          parts = [find_objects(torch.as_tensor(grids[c:c + 4096, frame].reshape(-1, 64, 64), device=device), [int(key[0])], [shapes]) for c in range(0, len(windows), 4096)]
          occupancy.append(np.concatenate([part["occupancy"][:, 0].cpu().numpy() for part in parts]))
          descriptor.append(np.concatenate([part["descriptor"][:, 0].cpu().numpy() for part in parts]))
      descriptor = np.stack(descriptor, axis=1)  # [N, 1 + len(K), 4]
      max_count, max_area = float(descriptor[..., 0].max()), float(descriptor[..., 1].max())
      families[-1].update(max_count=max_count, max_area=max_area)
      known = np.concatenate([known, occupancy[0].astype(np.float32)], axis=1)
      factor_targets.append({"name": f"family {index}", "kind": "descriptor", "classes": 4, "importance": float(values.mean()),
                             "future": (descriptor[:, 1:] / np.array([max(max_count, 1), max(max_area, 1), 64.0, 64.0])).astype(np.float32)})
    if not families:
      say("no object family found")
    discovery_rounds = rounds

  folder = Path(output) if output else model_dir("lewm", info["id"]) / "factors"
  folder.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(folder / "factors.npz", **arrays)
  (folder / "factors.json").write_text(json.dumps({"checkpoint": info["id"], "dataset": dataset, "version": version, "windows": len(windows), "seed": seed, "epochs": epochs, "threshold": threshold,
                                                   "min_share": min_share, "min_count": min_count, "max_codes": max_codes, "k": list(K), "event_quantile": event_quantile, "event_share": event_share,
                                                   "large_change_pixels": float(limit), "event_windows": events_wanted,
                                                   "test_loss": {name: {kind: float(loss[:, i].mean()) for i, kind in enumerate(KINDS)} for name, loss in test_loss.items()},
                                                   "regions": regions, "objects": families, "rounds": discovery_rounds if objects else []}, indent=2), encoding="utf-8")
  say(f"written {folder}")
  return folder


def main():
  parser = argparse.ArgumentParser(description="Find the screen regions whose outcome-relevant information a LeWM's latent lost (no game engine), with a codebook of their looks")
  parser.add_argument("--checkpoint", required=True, help="LeWM weights file (absolute or below models/lewm/)")
  parser.add_argument("--windows", type=int, default=20000, help="windows sampled from the checkpoint's validation episodes")
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
  parser.add_argument("--output", default=None, help="output folder (default models/lewm/<checkpoint id>/factors/)")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--epochs", type=int, default=15, help="training epochs of the two outcome networks")
  parser.add_argument("--threshold", type=float, default=4.0, help="a patch joins a region when its heat share is at least threshold / 256 (4 = four times an even spread)")
  parser.add_argument("--min-share", type=float, default=0.10, help="a region needs at least this share of the heat map")
  parser.add_argument("--min-count", type=int, default=3, help="a look of a region becomes a code when it occurs at least this often in the sampled frames")
  parser.add_argument("--max-codes", type=int, default=256, help="at most this many codes per region (the most frequent)")
  parser.add_argument("--objects", action="store_true", help="also find object families (objects that sit in different places), with the regions explained away")
  parser.add_argument("--min-t", type=float, default=2.0, help="an object family needs at least this t-value of its importance")
  parser.add_argument("--min-score", type=float, default=1.0, help="an object family needs importance / (room + 0.01) of at least this")
  parser.add_argument("--max-templates", type=int, default=16, help="at most this many exact shapes per object family")
  parser.add_argument("--max-families", type=int, default=16, help="at most this many rounds / object families")
  parser.add_argument("--event-quantile", type=float, default=0.98, help="a transition is a large change when it changes more pixels than this quantile of all transitions")
  parser.add_argument("--event-share", type=float, default=0.5, help="share of the windows that start within 25 moves before a large change")
  args = parser.parse_args()
  discover(args.checkpoint, args.windows, args.device, args.output, args.seed, args.epochs, args.threshold, args.min_share, args.min_count, args.max_codes,
           args.objects, args.min_t, args.min_score, args.max_templates, args.max_families, args.event_quantile, args.event_share)


if __name__ == "__main__":
  main()
