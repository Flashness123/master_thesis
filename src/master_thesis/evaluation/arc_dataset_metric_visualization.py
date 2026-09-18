"""
Statistics of an ARC Lance dataset, grouped by the level each episode plays (start_level column).

Writes to $STABLEWM_HOME/evaluation/<game>/:
  <dataset>_metrics.csv    one row per level
  <dataset>_heatmap.png    one panel per level: share of steps in which each cell changed
  <dataset>_gifs/          optional: randomly chosen episodes as GIFs (--gifs N)

Usage:
  uv run python -m master_thesis.evaluation.arc_dataset_metric_visualization --dataset goose_l1-7_v2
  uv run python -m master_thesis.evaluation.arc_dataset_metric_visualization --dataset human_l1-7 --gifs 5 --gif-level 2 --gif-success yes
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files only (no window needed)
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
from PIL import Image, ImageDraw

from master_thesis.paths import dataset_path, evaluation_dir
from master_thesis.training.lewm import ArcGridToPixels  # only for its ARC colour palette

NUM_ACTIONS = 7  # ARC action types ACTION1..ACTION7 (0 is the dummy action on an episode's last state)


def load_columns(game: str, dataset_name: str):
  """Read the columns needed here from datasets/<game>/<dataset_name>.lance."""
  dataset = swm.data.load_dataset(str(dataset_path(game, dataset_name)), num_steps=1)  # windows are not used; we only need columns and episode boundaries
  grids = dataset.get_col_data("grid").astype(np.uint8)  # [rows, 4096] colour indices (stored as float32 in Lance)
  actions = dataset.get_col_data("action")[:, 0].astype(int)  # action id taken in each state (0 = dummy on the last state)
  levels = dataset.get_col_data("start_level")[:, 0].astype(int)  # level the row's episode plays
  successes = dataset.get_col_data("episode_success")[:, 0].astype(bool)  # did the row's episode complete its level?
  episodes = list(zip(dataset.offsets.tolist(), dataset.lengths.tolist()))  # (first row, number of rows) per episode
  return grids, actions, levels, successes, episodes  # the reader itself (with its float32 copies) is dropped here to save memory


def grids_to_gif(grids, path, labels=None, side_grid=None, side_grids=None, scale: int = 8, duration: int = 150):
  """
  Write ARC grids as a GIF (also used by the planning evaluation).

  grids:      [frames, 4096] or [frames, 64, 64] colour indices
  labels:     optional one text per frame, drawn top left
  side_grid:  optional single grid shown to the right of every frame (e.g. the planning goal)
  side_grids: optional one grid per frame shown to the right (e.g. the waypoint that was active)
  """
  palette = (ArcGridToPixels(64).palette.numpy() * 255).round().astype(np.uint8)  # [16, 3] colour index -> RGB
  enlarge = lambda grid: Image.fromarray(palette[np.asarray(grid).reshape(64, 64)]).resize((64 * scale, 64 * scale), Image.NEAREST)  # without blending colours
  fixed_side = enlarge(side_grid) if side_grid is not None else None

  frames = []
  for index, grid in enumerate(grids):
    frame = enlarge(grid)
    side = enlarge(side_grids[index]) if side_grids is not None else fixed_side  # per-frame side image, the fixed one, or none
    if side is not None:  # place the two images next to each other with a small gap
      both = Image.new("RGB", (frame.width * 2 + 8, frame.height), "black")
      both.paste(frame, (0, 0))
      both.paste(side, (frame.width + 8, 0))
      frame = both
    if labels is not None:
      ImageDraw.Draw(frame).text((4, 4), labels[index], fill=(255, 0, 255))  # magenta text, top left
    frames.append(frame)

  Path(path).parent.mkdir(parents=True, exist_ok=True)
  frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)  # duration ms per frame, repeat forever
  return path


def save_episode_gifs(game: str, dataset_name: str, count: int, level: int | None = None, success: bool | None = None, seed: int = 0, scale: int = 8, columns=None):
  """
  Save `count` randomly chosen episodes as GIFs to evaluation/<game>/<dataset_name>_gifs/.
  level / success filter the episodes (None = any); pass `columns` from load_columns() to avoid reading the dataset twice.
  """
  grids, actions, levels, successes, episodes = columns or load_columns(game, dataset_name)
  candidates = [index for index, (start, _) in enumerate(episodes) if (level is None or levels[start] == level) and (success is None or successes[start] == success)]  # episodes matching the filter
  chosen = np.random.default_rng(seed).choice(candidates, size=min(count, len(candidates)), replace=False)  # same seed -> same episodes

  output_dir = evaluation_dir(game) / f"{dataset_name}_gifs"

  for index in chosen:
    start, length = episodes[index]
    labels = [f"step {step}  action {actions[start + step]}" if step < length - 1 else f"step {step}  end" for step in range(length)]  # action taken in this state (the last state has none)
    outcome = "success" if successes[start] else "fail"
    path = output_dir / f"{dataset_name}_level{levels[start]}_episode{index}_{outcome}.gif"  # episode index = episode_idx in the dataset
    grids_to_gif(grids[start:start + length], path, labels=labels, scale=scale)
    print(f"Saved {path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Per-level statistics and change heatmaps of an ARC Lance dataset")
  parser.add_argument("--dataset", required=True, help="dataset name without .lance, e.g. goose_l1-7_v2")
  parser.add_argument("--game", default="ls20", help="game folder under datasets/")
  parser.add_argument("--gifs", type=int, default=0, help="also save this many random episodes as GIFs")
  parser.add_argument("--gif-level", type=int, default=None, help="only episodes of this level (default: any)")
  parser.add_argument("--gif-success", choices=["any", "yes", "no"], default="any", help="only successful / failed episodes")
  parser.add_argument("--seed", type=int, default=0, help="seed for choosing the GIF episodes")
  args = parser.parse_args()

  columns = load_columns(args.game, args.dataset)
  grids, actions, levels, successes, episodes = columns

  per_level = {}  # level -> accumulated statistics
  for start, length in episodes:
    level = levels[start]  # all rows of an episode share its level
    stats = per_level.setdefault(level, {"episodes": 0, "successful": 0, "actions": 0, "action_counts": np.zeros(NUM_ACTIONS + 1, dtype=int), "changed_cells": [], "cell_changes": np.zeros(grids.shape[1], dtype=int), "grid_hashes": set()})

    episode_grids = grids[start:start + length]  # [length, 4096]
    changed = episode_grids[1:] != episode_grids[:-1]  # [length - 1, 4096]: which cells each action changed

    stats["episodes"] += 1
    stats["successful"] += int(successes[start])
    stats["actions"] += length - 1  # the last state has no real action
    stats["action_counts"] += np.bincount(actions[start:start + length - 1], minlength=NUM_ACTIONS + 1)  # how often each action id was used
    stats["changed_cells"].extend(changed.sum(axis=1).tolist())  # number of changed cells per action
    stats["cell_changes"] += changed.sum(axis=0)  # per cell: in how many steps it changed
    stats["grid_hashes"].update(hash(grid.tobytes()) for grid in episode_grids)  # fingerprints of the screens seen

  rows = []  # one CSV row per level
  for level, stats in sorted(per_level.items()):
    changed_cells = np.asarray(stats["changed_cells"])  # changed cells of every action on this level
    steps = max(stats["actions"], 1)  # avoid division by zero for episodes without actions
    row = {
      "level": level,
      "episodes": stats["episodes"],
      "successful_pct": round(100 * stats["successful"] / stats["episodes"], 1),
      "actions": stats["actions"],
      "mean_episode_length": round(stats["actions"] / stats["episodes"], 1),
      "mean_changed_cells": round(float(changed_cells.mean()), 1) if changed_cells.size else 0.0,
      "no_change_pct": round(100 * float((changed_cells == 0).mean()), 1) if changed_cells.size else 0.0,
      "distinct_grids": len(stats["grid_hashes"]),
      "coverage_pct": round(100 * float((stats["cell_changes"] > 0).mean()), 1),  # share of cells that changed at least once
    }
    for action_id in range(1, NUM_ACTIONS + 1):  # share of each action type
      row[f"action{action_id}_pct"] = round(100 * stats["action_counts"][action_id] / steps, 1)
    rows.append(row)
    stats["heatmap"] = (stats["cell_changes"] / steps).reshape(64, 64)  # per cell: share of steps in which it changed

  output_dir = evaluation_dir(args.game)  # $STABLEWM_HOME/evaluation/<game>
  output_dir.mkdir(parents=True, exist_ok=True)

  csv_path = output_dir / f"{args.dataset}_metrics.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))  # column names from the first row
    writer.writeheader()
    writer.writerows(rows)

  for row in rows:  # short summary in the terminal
    print(", ".join(f"{key}={value}" for key, value in row.items() if not key.startswith("action") or key == "actions"))

  figure, axes = plt.subplots(1, len(per_level), figsize=(3 * len(per_level), 3.4), squeeze=False)  # one panel per level
  for axis, (level, stats) in zip(axes[0], sorted(per_level.items())):
    image = axis.imshow(stats["heatmap"], cmap="magma", vmin=0)  # bright = changed often
    axis.set_title(f"level {level} ({stats['episodes']} episodes)")
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)  # share of steps
  figure.suptitle(f"{args.dataset}: share of steps in which each cell changed")
  heatmap_path = output_dir / f"{args.dataset}_heatmap.png"
  figure.savefig(heatmap_path, dpi=120, bbox_inches="tight")

  print(f"Saved {csv_path}")
  print(f"Saved {heatmap_path}")

  if args.gifs > 0:
    success = {"any": None, "yes": True, "no": False}[args.gif_success]  # text option -> filter value
    save_episode_gifs(args.game, args.dataset, args.gifs, level=args.gif_level, success=success, seed=args.seed, columns=columns)


if __name__ == "__main__":
  main()
