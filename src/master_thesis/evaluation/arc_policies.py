"""
Acting in the real ARC game with a trained world model, and evaluating it.

  LewmRolloutPlanner  plans with LeWM's own latent rollout (JEPA.rollout, called by JEPA.get_cost) towards a goal
                      state; stable-worldmodel's CategoricalCEMSolver searches over action sequences
  run_episode         plays one episode with such a policy and optionally records every ARC grid (for GIFs)
  follow_waypoints    completes a level by planning from waypoint to waypoint (frames of a recorded solution)
  evaluate_planning   goal states come from recorded episodes; writes one row per attempt and a GIF
  evaluate_waypoints  tries to complete levels along successful recorded solutions; one row + GIF per attempt

Usage:
  uv run python -m master_thesis.evaluation.arc_policies --lewm-run ls20_goose-ppo-l1_ft130ep_0914-2034 --dataset human_l1-7
  uv run python -m master_thesis.evaluation.arc_policies --mode waypoints --lewm-run ls20_l1-human-goose_150k_0918-0120 --dataset l1_human-goose --levels 1
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

ARC_ACTION_DIM = 9  # LeWM action vector: 7 action types + x + y


class LewmRolloutPlanner:
  """
  Model-predictive control with LeWM: at every step CEM samples action sequences, LeWM's internal rollout
  imagines where they lead, and the first action of the best sequence is played in the real game.
  """

  def __init__(self, model, env, goal_grid, img_size: int = 224, horizon: int = 5, samples: int = 300, iterations: int = 10, topk: int = 30, replan_every: int = 1, cost_on: str = "path", device: str = "cuda", seed: int = 0, ignore_cells=None):
    import gymnasium as gym
    from stable_worldmodel.planning.solver.categorical_cem import CategoricalCEMSolver

    from master_thesis.training.lewm import ArcGridToPixels

    action_ids = [action.value for action in env.actions]
    assert action_ids == list(range(1, len(action_ids) + 1)), "the padding below assumes the actions ACTION1..n without gaps"

    self.device = device
    self.to_pixels = ArcGridToPixels(img_size)  # same preprocessing as in training
    self.goal_grid = np.asarray(goal_grid).reshape(-1)
    self.goal_pixels = self._pixels(self.goal_grid)
    self.compare = ~ignore_cells if ignore_cells is not None else np.ones(self.goal_grid.size, dtype=bool)  # cells that decide "goal reached"
    self.dummy_action = torch.zeros(1, 1, ARC_ACTION_DIM, device=device)  # JEPA.get_cost expects an action entry; the rollout replaces it
    self.distances = []  # the model's distance to the goal, one per plan
    self.replan_every = min(replan_every, horizon)  # 1 = replan after every played action; horizon = play a whole plan (open loop)
    self.queue = deque()  # actions of the current plan that are still to be played

    pad = lambda candidates: torch.nn.functional.pad(candidates, (0, ARC_ACTION_DIM - candidates.shape[-1]))  # one-hot over ACTION1..n -> 9-D LeWM action (x, y stay 0)

    def path_cost(info, candidates):  # a plan is as good as its closest approach to the goal (not only its last state)
      model.get_cost(info, pad(candidates))  # runs LeWM's rollout; fills info["predicted_emb"] and info["goal_emb"]
      path = info["predicted_emb"][:, :, 1:]  # [batch, samples, horizon, D]: imagined states after move 1, 2, ... (index 0 = current state)
      distance = (path - info["goal_emb"][:, None, -1:]).square().sum(dim=-1)  # [batch, samples, horizon]: distance to the goal after every move
      return distance.min(dim=-1).values  # [batch, samples]

    assert cost_on in ("path", "end"), "cost_on must be 'path' (closest approach) or 'end' (last imagined state, LeWM's own criterion)"
    cost = SimpleNamespace(get_cost=path_cost if cost_on == "path" else lambda info, candidates: model.get_cost(info, pad(candidates)))  # the solver only needs get_cost

    self.solver = CategoricalCEMSolver(cost=cost, num_samples=samples, n_steps=iterations, topk=topk, device=device, seed=seed)
    self.solver.configure(action_space=gym.spaces.Discrete(len(action_ids)), n_envs=1, config=SimpleNamespace(horizon=horizon, action_block=1))  # one action per planned step

  def _pixels(self, grid):
    return self.to_pixels(torch.as_tensor(np.asarray(grid).reshape(1, -1), device=self.device))[None]  # [1, 1, 3, img, img] = (batch, time, ...)

  @torch.no_grad()
  def choose_action(self, grid) -> int:
    if not self.queue:  # plan again only when the current plan is used up
      plan = self.solver.solve({"pixels": self._pixels(grid), "goal": self.goal_pixels, "action": self.dummy_action})  # CEM over action sequences
      self.distances.append(float(plan["costs"][0]))  # elite mean cost of the last CEM round
      self.queue.extend(int(action) for action in plan["actions"][0, :self.replan_every, 0])  # play this many actions of the best plan
    return self.queue.popleft()

  def stop(self, grid) -> bool:
    """Goal reached: every cell that is not ignored matches the goal."""
    return bool((np.asarray(grid).reshape(-1)[self.compare] == self.goal_grid[self.compare]).all())


def run_episode(env, policy, max_steps: int = 500, record: bool = False, options=None):
  """
  Play one episode with `policy` (needs choose_action, may provide stop).
  `record` keeps every ARC grid, also when the policy sees something else (images, LeWM embeddings).
  """
  observation, info = env.reset(options=options)
  grid = env.unwrapped.grid  # the real ARC grid, independent of what the policy sees
  stopped = hasattr(policy, "stop") and policy.stop(grid)  # the goal can already be reached at the start
  grids = [grid] if record else []
  actions = []

  while not stopped and len(actions) < max_steps:
    action = policy.choose_action(observation)
    observation, _, terminated, truncated, info = env.step(action)  # the reward is not used here
    grid = env.unwrapped.grid

    actions.append(action)
    if record:
      grids.append(grid)

    stopped = hasattr(policy, "stop") and policy.stop(grid)
    if terminated or truncated:  # level completed / game over / step limit of the environment
      break

  return {"grids": grids, "actions": actions, "steps": len(actions), "stopped": stopped, "info": info}


def follow_waypoints(model, env, waypoints, level: int, max_steps_per_waypoint: int = 15, ignore_cells=None, solution=None, **planner_kwargs):
  """
  Complete a level by planning from waypoint to waypoint (e.g. every 5th frame of a recorded human solution).
  Each waypoint is a short planning problem; the game keeps running between them (no reset).
  Stops at the first waypoint that is not reached within `max_steps_per_waypoint` actions.
  Pass `solution` (the frames of the recorded run) to also measure how far the played game stays on that run.
  """
  env.reset(options={"start_level": level})
  grid = env.unwrapped.grid
  grids, actions, targets, numbers = [grid], [], [waypoints[0]], [1]  # played grids, actions, and the waypoint (grid, number) active at each frame (for the GIF)
  reached = 0

  compare = ~ignore_cells if ignore_cells is not None else np.ones(np.asarray(grid).size, dtype=bool)  # cells that decide "waypoint reached" (the step counter is ignored)
  differing = lambda played, target: int((np.asarray(played).reshape(-1)[compare] != np.asarray(target).reshape(-1)[compare]).sum())  # how far the played state still is from the waypoint
  closest = []  # per waypoint: fewest differing cells reached (0 = reached exactly)

  for number, waypoint in enumerate(waypoints, start=1):
    planner = LewmRolloutPlanner(model, env, waypoint, ignore_cells=ignore_cells, **planner_kwargs)
    steps, ended, nearest = 0, False, differing(grid, waypoint)
    while not planner.stop(grid) and steps < max_steps_per_waypoint and not ended:
      action = planner.choose_action(grid)  # plan 5 steps ahead, play the first action
      _, _, terminated, truncated, _ = env.step(action)
      grid = env.unwrapped.grid
      grids.append(grid)
      actions.append(action)
      targets.append(waypoint)
      numbers.append(number)
      steps += 1
      nearest = min(nearest, differing(grid, waypoint))  # closest the played game came to this waypoint
      ended = terminated or truncated  # game over / win / step limit of the environment

    closest.append(nearest)
    if not planner.stop(grid):  # this waypoint was missed: give up
      break
    reached += 1

  on_path = [_human_step(grid, solution, compare) for grid in grids] if solution is not None else []  # which frame of the recorded run each played state is (-1 = none)

  return {"grids": grids, "actions": actions, "targets": targets, "numbers": numbers, "waypoints_reached": reached, "waypoints_total": len(waypoints), "completed": bool(env.unwrapped.success), "steps": len(actions), "closest_cells": closest,
          "on_path_fraction": round(float(np.mean([step >= 0 for step in on_path])), 2) if on_path else None, "furthest_human_step": max(on_path) if on_path else None}


def _human_step(grid, solution, compare):
  """Index of the frame of `solution` that the played state matches (ignoring the counter cells), or -1 if it matches none."""
  matches = (np.asarray(solution)[:, compare] == np.asarray(grid).reshape(-1)[compare]).all(axis=1)
  return int(matches.argmax()) if matches.any() else -1


def counter_cells(grids, episodes, max_changed: int = 10):
  """
  Cells that belong to an on-screen counter, found from the data: cells that change during steps in which
  almost nothing else changes (fewer than `max_changed` cells, e.g. a blocked move). Such cells show elapsed
  steps rather than the game state, so they are ignored when comparing a played state with a goal state.
  For ls20 this finds exactly the bar at the bottom (80 cells in rows 61-62).
  """
  mask = np.zeros(grids.shape[1], dtype=bool)
  for start, length in episodes:
    episode_grids = grids[start:start + length]
    changed = episode_grids[1:] != episode_grids[:-1]  # [steps, 4096]
    mask |= changed[changed.sum(axis=1) < max_changed].any(axis=0)  # cells changed by "nothing happened" steps
  return mask  # [4096] boolean


def load_lewm(run_name: str, weights: str = "weights.pt", device: str = "cuda"):
  """Rebuild a trained LeWM from models/lewm/<run_name> (as in training/ppo.py)."""
  from hydra.utils import instantiate

  from master_thesis.paths import model_dir

  run = model_dir("lewm", run_name)
  model = instantiate(json.loads((run / "model_config.json").read_text(encoding="utf-8")))  # architecture
  model.load_state_dict(torch.load(run / weights, map_location="cpu", weights_only=True), strict=True)  # trained weights
  return model.to(device).eval()


def evaluate_planning(model, game: str = "ls20", game_id: str = "ls20-9607627b", dataset: str = "human_l1-7", goal_steps=(5, 10, 25), levels=None, episodes_per_level: int = 2, gif_dir=None, max_steps_factor: int = 3, device: str = "cuda", seed: int = 0, **planner_kwargs):
  """
  Take successful recorded episodes, use the state `k` steps after their start as the goal and let the planner
  reach it in the real game. Returns one row per attempt and writes a GIF (played game | goal) for each.
  """
  from master_thesis.environments.arc_ppo import ArcPPOEnv
  from master_thesis.evaluation.arc_dataset_metric_visualization import grids_to_gif, load_columns

  grids, _, episode_levels, successes, episodes = load_columns(game, dataset)  # the recorded actions are not needed here
  ignore = counter_cells(grids, episodes)  # the on-screen step counter is not part of the goal

  rows = []
  for level in (sorted(set(episode_levels.tolist())) if levels is None else levels):
    chosen = [index for index, (start, _) in enumerate(episodes) if episode_levels[start] == level and successes[start]][:episodes_per_level]
    if not chosen:
      continue

    env = ArcPPOEnv(game_id=game_id, seed=seed + level, levels=[level], max_steps=500, stop_on_success=False)

    for index in chosen:
      start, length = episodes[index]
      for steps_ahead in goal_steps:
        if steps_ahead >= length:  # episode too short for this goal
          continue

        planner = LewmRolloutPlanner(model, env, grids[start + steps_ahead], ignore_cells=ignore, device=device, seed=seed, **planner_kwargs)
        played = run_episode(env, planner, max_steps=max_steps_factor * steps_ahead, record=True, options={"start_level": level})

        row = {"level": level, "episode": index, "goal_steps": steps_ahead, "replan_every": planner.replan_every, "reached": played["stopped"], "steps": played["steps"], "final_distance": planner.distances[-1] if planner.distances else None}
        rows.append(row)
        print(", ".join(f"{key}={value}" for key, value in row.items()))

        if gif_dir is not None:
          labels = [f"step {step}" + (f"  action {played['actions'][step]}" if step < len(played["actions"]) else "  end") for step in range(len(played["grids"]))]
          grids_to_gif(played["grids"], gif_dir / f"{dataset}_level{level}_episode{index}_goal{steps_ahead}_replan{planner.replan_every}_{'reached' if played['stopped'] else 'missed'}.gif", labels=labels, side_grid=grids[start + steps_ahead])

    env.close()

  return rows


def evaluate_waypoints(model, game: str = "ls20", game_id: str = "ls20-9607627b", dataset: str = "human_l1-7", levels=None, episodes_per_level: int = 2, episode_ids=None, waypoint_every: int = 5, max_steps_factor: int = 3, gif_dir=None, device: str = "cuda", seed: int = 0, **planner_kwargs):
  """
  Try to complete levels by following successful recorded solutions: every `waypoint_every`-th frame
  (and the final, completing frame) becomes a waypoint. Returns one row per attempt and writes a GIF (played game | active waypoint).
  """
  from master_thesis.environments.arc_ppo import ArcPPOEnv
  from master_thesis.evaluation.arc_dataset_metric_visualization import grids_to_gif, load_columns

  grids, _, episode_levels, successes, episodes = load_columns(game, dataset)
  ignore = counter_cells(grids, episodes)  # the on-screen step counter is not part of a waypoint

  rows = []
  for level in (sorted(set(episode_levels.tolist())) if levels is None else levels):
    successful = [index for index, (start, _) in enumerate(episodes) if episode_levels[start] == level and successes[start]]  # solutions of this level
    chosen = [index for index in successful if index in episode_ids] if episode_ids is not None else successful[:episodes_per_level]  # given episodes (e.g. the test split) or the first ones
    if not chosen:
      continue

    env = ArcPPOEnv(game_id=game_id, seed=seed + level, levels=[level], max_steps=500, stop_on_success=False)

    for index in chosen:
      start, length = episodes[index]
      solution = grids[start:start + length]  # the recorded human run; its last frame is the completion frame
      waypoints = list(solution[waypoint_every::waypoint_every])  # every 5th frame ...
      if len(waypoints) == 0 or not np.array_equal(waypoints[-1], solution[-1]):
        waypoints.append(solution[-1])  # ... and always the completion frame as the last waypoint

      played = follow_waypoints(model, env, waypoints, level, max_steps_per_waypoint=max_steps_factor * waypoint_every, ignore_cells=ignore, solution=solution, device=device, seed=seed, **planner_kwargs)

      row = {"level": level, "episode": index, "completed": played["completed"], "waypoints_reached": played["waypoints_reached"], "waypoints_total": played["waypoints_total"], "steps": played["steps"], "human_steps": length - 1,
             "closest_cells": played["closest_cells"],  # per attempted waypoint the fewest cells still differing (0 = reached)
             "on_path_fraction": played["on_path_fraction"], "furthest_human_step": played["furthest_human_step"]}  # how much of the played game lies on the recorded run
      rows.append(row)
      print(", ".join(f"{key}={value}" for key, value in row.items()))

      if gif_dir is not None:
        labels = [f"step {step}  waypoint {number}/{played['waypoints_total']}" for step, number in enumerate(played["numbers"])]  # which waypoint the planner was heading for
        grids_to_gif(played["grids"], gif_dir / f"{dataset}_level{level}_episode{index}_waypoints{waypoint_every}_{'completed' if played['completed'] else 'failed'}.gif", labels=labels, side_grids=played["targets"])

    env.close()

  return rows


def main() -> None:
  from master_thesis.paths import evaluation_dir

  parser = argparse.ArgumentParser(description="Plan with a trained LeWM towards goal states from recorded episodes")
  parser.add_argument("--lewm-run", required=True, help="model folder name under models/lewm")
  parser.add_argument("--weights", default="weights.pt", help="weights file inside the run, e.g. checkpoints/weights_step10000.pt")
  parser.add_argument("--dataset", default="human_l1-7", help="dataset the start and goal states come from")
  parser.add_argument("--game", default="ls20", help="game folder under datasets/")
  parser.add_argument("--game-id", default="ls20-9607627b", help="versioned game id for the real environment")
  parser.add_argument("--mode", choices=["goals", "waypoints"], default="goals", help="goals: reach single goal states; waypoints: complete levels by following human solutions")
  parser.add_argument("--goal-steps", type=int, nargs="+", default=[5, 10, 25], help="goals mode: how many steps after the episode start the goal lies")
  parser.add_argument("--waypoint-every", type=int, default=5, help="waypoints mode: every how many frames of the human solution a waypoint is placed")
  parser.add_argument("--levels", type=int, nargs="+", default=None, help="levels to evaluate (default: all in the dataset)")
  parser.add_argument("--episodes", type=int, default=2, help="successful episodes per level")
  parser.add_argument("--episode-ids", type=int, nargs="+", default=None, help="waypoints mode: use exactly these episodes (e.g. from the test split) instead of the first --episodes")
  parser.add_argument("--horizon", type=int, default=5, help="planning horizon in steps")
  parser.add_argument("--samples", type=int, default=300, help="action sequences per CEM round")
  parser.add_argument("--iterations", type=int, default=10, help="CEM rounds per step")
  parser.add_argument("--cost-on", choices=["path", "end"], default="path", help="path: a plan counts by its closest approach to the goal; end: only its last imagined state (LeWM's own criterion)")
  parser.add_argument("--replan-every", type=int, default=1, help="how many actions of a plan are played before replanning (1 = MPC, >= horizon = play the whole plan open loop)")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
  parser.add_argument("--tensorboard", action="store_true", help="also write planning/* into the run's TensorBoard")
  parser.add_argument("--step", type=int, default=None, help="step to log at (default: taken from the weights file name)")
  args = parser.parse_args()

  device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
  model = load_lewm(args.lewm_run, weights=args.weights, device=device)

  planner_settings = {"horizon": args.horizon, "samples": args.samples, "iterations": args.iterations, "replan_every": args.replan_every, "cost_on": args.cost_on}
  common = {"game": args.game, "game_id": args.game_id, "dataset": args.dataset, "levels": args.levels, "episodes_per_level": args.episodes, "device": device, "seed": args.seed}

  if args.mode == "waypoints":  # complete levels along human solutions
    gif_dir = evaluation_dir(args.game) / f"{args.lewm_run}_waypoints"
    rows = evaluate_waypoints(model, waypoint_every=args.waypoint_every, episode_ids=args.episode_ids, gif_dir=gif_dir, **common, **planner_settings)
    output = evaluation_dir(args.game) / f"{args.lewm_run}_waypoints.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\ncompleted {sum(row['completed'] for row in rows)}/{len(rows)} levels; saved {output} and the GIFs in {gif_dir}")
    return

  gif_dir = evaluation_dir(args.game) / f"{args.lewm_run}_planning"
  rows = evaluate_planning(model, goal_steps=args.goal_steps, gif_dir=gif_dir, **common, **planner_settings)

  output = evaluation_dir(args.game) / f"{args.lewm_run}_planning.json"
  output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
  print(f"\nreached {sum(row['reached'] for row in rows)}/{len(rows)} goals; saved {output} and the GIFs in {gif_dir}")

  if args.tensorboard and rows:  # write the same numbers the training callback logs into the run's TensorBoard
    from torch.utils.tensorboard import SummaryWriter

    from master_thesis.evaluation.world_model_metrics import planning_metrics
    from master_thesis.paths import model_dir

    step = args.step if args.step is not None else int("".join(character for character in args.weights if character.isdigit()) or 0)  # weights_step30000.pt -> 30000, weights.pt -> 0
    writer = SummaryWriter(log_dir=str(model_dir("lewm", args.lewm_run) / "tensorboard" / "version_0"))  # same folder the training wrote to
    for name, value in planning_metrics(rows).items():
      writer.add_scalar(name, value, step)
    writer.close()
    print(f"wrote planning/* into the TensorBoard of {args.lewm_run} at step {step}")


if __name__ == "__main__":
  main()
