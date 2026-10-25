"""
One HTML page to look at and compare the results of lewm_analysis.py: every analysed checkpoint, its metrics in the
dashboard groups (encoder, predictor, world model, planning, generalization), and its latent maps (PCA and UMAP, in 3D
to rotate). Every metric has an info icon that explains what it measures and how, with the formula.

Reads $STABLEWM_HOME/evaluation/ls20/lewm_analysis/<checkpoint>/results.json and latent_maps.json and writes
$STABLEWM_HOME/evaluation/ls20/lewm_analysis/dashboard.html. The page itself (layout, charts) is lewm_dashboard.html
next to this file; this script only fills in the data, the explanations below and the Plotly library, so the page needs
no server. The formulas are typeset with KaTeX from cdnjs: without internet they show as LaTeX source.

Usage:
  python -m master_thesis.evaluation.lewm_dashboard                                    # every analysed checkpoint
  python -m master_thesis.evaluation.lewm_dashboard --checkpoints <id> <id> --output compare.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from master_thesis.paths import evaluation_dir

GAME = "ls20"
COLUMNS = ["group", "experiment", "metric", "split", "level", "subset", "value", "n"]  # the fields of a metric row (lewm_analysis.metric_row)

# What the dashboard groups ask (lewm_analysis' module docstring)
GROUPS = {"encoder": "Does the representation keep the information needed to tell possible futures apart?",
          "predictor": "Given a useful representation, does the predictor know how actions change the future?",
          "world_model": "How well do encoder and predictor together predict the next state?",
          "planning": "Can the model steer the game?",
          "generalization": "What changes between known and new data?"}

SPLITS = {"seen_train": "Episodes the model was trained on.",
          "seen_val": "Other episodes of the same levels, which the model never saw during training (the training's own validation episodes).",
          "unseen": "Levels that were not in the training data at all (syn1000_unseen: every 50th synthetic level and real level 3). Split per level in two "
                    "halves: probes are fine-tuned on the 'adapt' half and tested on the 'eval' half."}

# Per metric (experiment/metric; "*" stands for a state_probes fact): a title, what it measures and how (condensed from the
# experiments' docstrings in lewm_analysis.py), and the formula. z = a latent (192 numbers), p_theta = the predictor,
# s = a state, a = a move, z(s, a) = the encoded real outcome of move a in state s.
INFO = {
  "latent_variance/latent_variance": {
    "title": "Latent variance",
    "text": "How much do the latents of the picked states differ at all? For each of the 192 latent numbers its variance over the states, averaged "
            "over the 192. Near 0 means collapse: every state gets almost the same latent. Per level: the same over the states of one level.",
    "latex": r"\mathrm{Var}(S)=\frac{1}{192}\sum_{d=1}^{192}\frac{1}{|S|}\sum_{s\in S}\big(z_d(s)-\bar z_d\big)^2"},
  "latent_variance/within_level_share": {
    "title": "Variance kept within levels",
    "text": "Does the latent mostly store which level we are in? The average variance inside a level divided by the variance over all states. "
            "Near 1: the latents vary as much inside a level as across levels. Near 0: almost all variation comes from the level, and inside a "
            "level everything looks alike, which is bad for planning, because the planner works inside one level.",
    "latex": r"\text{share}=\frac{\frac{1}{|L|}\sum_{\ell\in L}\mathrm{Var}(S_\ell)}{\mathrm{Var}(S)}\qquad(\text{levels with at least 2 states})"},
  "state_probes/*/balanced_accuracy": {
    "title": "Probe: balanced accuracy",
    "text": "Which facts about the game can be read from the latent? A small MLP (192 → 256 → classes) gets the frozen, standardized latent of a state "
            "and predicts one fact, e.g. lives, the key's shape, or whether a move would work. Trained on seen_train. Balanced accuracy: every "
            "class counts equally, however common it is. probe: on the seen splits. zero_shot: the same probe on the unseen eval half. adapted: "
            "a copy fine-tuned on the unseen adapt half. global_baseline: always the most common answer. level_baseline: the most common answer "
            "of the state's level; if the probe is not better, the latent only tells the level, not the fact.",
    "latex": r"\tilde z=\frac{z-\mu_{\text{train}}}{\sigma_{\text{train}}},\quad \hat y=\arg\max_c\, g_\phi(\tilde z)_c,\quad"
             r"\text{BA}=\frac{1}{C}\sum_{c=1}^{C}\frac{|\{s:\ y_s=c,\ \hat y_s=c\}|}{|\{s:\ y_s=c\}|}"},
  "state_probes/*/mae": {
    "title": "Probe: mean absolute error",
    "text": "The same probes for facts that are numbers (player_x, energy, goal_distance, ...): the average distance between the predicted and the "
            "real value, in the fact's own unit (cells, energy units). Lower is better. Same subsets and baselines as for the classes.",
    "latex": r"\text{MAE}=\frac{1}{N}\sum_{s}\big|\,y_s-\hat y_s\big|,\qquad \hat y_s=\sigma_y\, g_\phi(\tilde z_s)+\mu_y"},
  "state_probes/*/varying_levels": {
    "title": "Levels in which the fact changes",
    "text": "In what share of the levels does the fact change at all? A fact that never changes inside a level (e.g. rotator_present) can be "
            "solved by knowing the level, so a good probe score on it proves little.",
    "latex": r"\frac{\big|\{\ell:\ y \text{ takes more than one value in } S_\ell\}\big|}{|L|}"},
  "action_decoding/balanced_accuracy": {
    "title": "Move read from the latent change",
    "text": "Can the move be read from how the latent changes? This is what Delta-JEPA trains its action decoder for, measured the same way for "
            "every model. From every picked state all 4 moves were played in the real game; only moves that changed the game count (a blocked "
            "move looks the same in every direction). A small MLP gets the standardized latent change and names the move; 0.25 = chance. probe: "
            "trained on seen_train, tested on the seen splits. zero_shot: the same probe on the unseen eval half. adapted: a copy fine-tuned on "
            "the unseen adapt half. If zero_shot is low, a high adapted means the move is in the latent change but stored differently in new "
            "levels; a low adapted means it is missing.",
    "latex": r"\Delta z=z(s,a)-z(s),\quad \hat a=\arg\max_{a'} g_\phi\Big(\frac{\Delta z-\mu}{\sigma}\Big)_{a'},\quad"
             r"\text{BA}=\frac{1}{4}\sum_{a=1}^{4}\frac{|\{\hat a=a,\ a^{*}=a\}|}{|\{a^{*}=a\}|}"},
  "state_aliasing/effect_mismatch": {
    "title": "Close states that behave differently",
    "text": "Do states that are close in latent space really behave the same in the game? For each state its 10 nearest states (distance between "
            "latents, identical grids left out) in the same level and in other levels, and as baseline 10 random states of the same group. For "
            "each pair and move: does the move have a different effect (player movement, symbol change, refill, life lost, level completed)? "
            "Nearest neighbours should disagree much less often than random ones; disagreements among nearest neighbours are situations the "
            "planner would confuse. effect_mismatch_<move>: the same for one move.",
    "latex": r"\text{mismatch}=\frac{1}{|P|}\sum_{(s,n)\in P}\frac{1}{4}\sum_{a=1}^{4}\mathbf{1}\big[e(s,a)\neq e(n,a)\big],\qquad"
             r"P=\{(s,n):\ n\in N_{10}(s)\}"},
  "latent_maps/explained_variance": {
    "title": "PCA: explained variance",
    "text": "PCA over the latents of all picked states (all splits together): the share of the variance each of the first 10 directions explains. "
            "A very large pc1 means the representation is basically one-dimensional. The maps themselves are in the Latent maps tab; UMAP "
            "distorts distances and cluster sizes, so read it as a picture only.",
    "latex": r"\text{explained}_i=\frac{\lambda_i}{\sum_j\lambda_j},\qquad \lambda_i:\ \text{eigenvalues of } \operatorname{Cov}(z)"},
  "checkpoint_similarity/cka": {
    "title": "Similarity to the compared checkpoint (CKA)",
    "text": "How similar are the representations of two checkpoints (lewm_analysis --compare-to)? Both encode exactly the same states. Linear CKA "
            "centres both latent sets and compares which states are similar to which: 1 = the same representation (possibly rotated or "
            "scaled), 0 = unrelated. It shows how much something changed the representation, not whether it got better.",
    "latex": r"\operatorname{CKA}(X,Y)=\frac{\lVert Y_c^{\top}X_c\rVert_F^{2}}{\lVert X_c^{\top}X_c\rVert_F\,\lVert Y_c^{\top}Y_c\rVert_F}"},
  "one_step_prediction/prediction_error": {
    "title": "One-step prediction error",
    "text": "How well do encoder and predictor together predict the next state from one real frame and the recorded action? That is the "
            "planner's first step: no history. Squared difference to the encoded real next state, averaged over the 192 numbers and the states. "
            "Subsets: all steps, and by what the step really did: mechanics (the game changed), energy_only (a blocked move, only the energy bar "
            "changed), none (nothing changed).",
    "latex": r"\hat z_{t+1}=p_\theta(z_t,\ a_t),\qquad e_{\text{pred}}=\frac{1}{192}\big\lVert \hat z_{t+1}-z_{t+1}\big\rVert^2"},
  "one_step_prediction/copy_error": {
    "title": "Copy error (\"nothing changes\")",
    "text": "The baseline: the current latent compared with the real next one, as if nothing changed. It also shows how much one step moves the "
            "latent at all. For the kind 'none' it is always 0, because the grid did not change.",
    "latex": r"e_{\text{copy}}=\frac{1}{192}\big\lVert z_t-z_{t+1}\big\rVert^2"},
  "one_step_prediction/error_ratio": {
    "title": "One-step error ratio",
    "text": "The prediction error divided by the copy error: the averages over the states, not an average of per-state ratios, so steps that barely "
            "move the latent cannot blow it up. It can be compared between models whose latents have different scales. Below 1 the model is "
            "better than \"nothing changes\".",
    "latex": r"\text{ratio}=\frac{\sum_s e_{\text{pred}}(s)}{\sum_s e_{\text{copy}}(s)}"},
  "counterfactual_actions/prediction_error": {
    "title": "Counterfactual prediction error",
    "text": "Does the predictor know what every move would do, not just the recorded one? From every picked state all 4 moves were played in the "
            "real game. The predictor gets the state and up to 2 real states before it (fewer at an episode start) with the actions really "
            "taken there; only the last action varies. The error per move (up, down, left, right) and over all moves.",
    "latex": r"\hat z(s,a)=p_\theta\big(z_{t-2:t},\ (a_{t-2},a_{t-1},a)\big),\qquad e(s,a)=\frac{1}{192}\big\lVert \hat z(s,a)-z(s,a)\big\rVert^2"},
  "counterfactual_actions/error_ratio": {
    "title": "Counterfactual error ratio",
    "text": "The counterfactual prediction error divided by \"nothing changes\" (the state's own latent against the real outcome), as ratio of the "
            "averages. Below 1 the model knows what the move does better than assuming nothing happens.",
    "latex": r"\text{ratio}=\frac{\sum_{s,a} e(s,a)}{\sum_{s,a}\frac{1}{192}\lVert z_t-z(s,a)\rVert^2}"},
  "counterfactual_actions/prediction_error_by_change": {
    "title": "Counterfactual error by kind of change",
    "text": "The counterfactual prediction error, sorted by what the move really did: mechanics (the game changed), energy_only (blocked, only the "
            "energy bar changed), none (nothing changed). A model that always predicts \"nothing happens\" looks good on blocked moves only.",
    "latex": r"e(s,a)=\frac{1}{192}\big\lVert \hat z(s,a)-z(s,a)\big\rVert^2\ \text{ for all } (s,a) \text{ of one kind}"},
  "counterfactual_actions/error_ratio_by_change": {
    "title": "Counterfactual error ratio by kind of change",
    "text": "The counterfactual error ratio, sorted by what the move really did (mechanics, energy_only; for 'none' the baseline is 0 and the ratio "
            "is undefined).",
    "latex": r"\text{ratio}_k=\frac{\sum_{(s,a)\in k} e(s,a)}{\sum_{(s,a)\in k}\frac{1}{192}\lVert z_t-z(s,a)\rVert^2}"},
  "counterfactual_actions/effect_error": {
    "title": "Effect error between two moves",
    "text": "For each pair of moves (6 per state): does the predicted difference between their outcomes match the real difference? identical: the "
            "real game shows the same grid for both (e.g. two moves into walls), so the real difference is 0 and this is how far apart the model "
            "wrongly predicts them. different: the moves lead to different grids.",
    "latex": r"e_{\text{eff}}(a,b)=\frac{1}{192}\Big\lVert\big(\hat z(s,a)-\hat z(s,b)\big)-\big(z(s,a)-z(s,b)\big)\Big\rVert^2"},
  "counterfactual_actions/effect_error_ratio": {
    "title": "Effect error ratio",
    "text": "The effect error of pairs with different outcomes divided by the size of the real difference: 1 = no better than predicting no "
            "difference at all, 0 = every difference right.",
    "latex": r"\frac{\sum e_{\text{eff}}(a,b)}{\sum\frac{1}{192}\lVert z(s,a)-z(s,b)\rVert^2}"},
  "counterfactual_actions/predicted_difference": {
    "title": "Predicted difference between two moves",
    "text": "How different the model predicts the outcomes of two moves to be (pairs with different real outcomes). Much smaller than the real "
            "difference means the model mostly ignores the action.",
    "latex": r"\frac{1}{192}\big\lVert \hat z(s,a)-\hat z(s,b)\big\rVert^2"},
  "counterfactual_actions/real_difference": {
    "title": "Real difference between two moves",
    "text": "How different the encoded real outcomes of two moves are (pairs with different real outcomes): the target for the predicted difference.",
    "latex": r"\frac{1}{192}\big\lVert z(s,a)-z(s,b)\big\rVert^2"},
  "multi_step_rollout/prediction_error": {
    "title": "Rollout prediction error",
    "text": "Does the error grow when the model keeps predicting from its own predictions, and does real history help? Windows of 13 recorded "
            "states; the rollout starts at the 3rd. The model gets the start state alone (k1 … k10: the planner's situation) or also the 1 or "
            "2 real states before it (…_real2, …_real3), plus the recorded actions, and predicts step by step. It always looks at its last H "
            "latents: the real ones first, then its own predictions. Compared after k = 1, 3, 5 and 10 steps, all on the same windows.",
    "latex": r"\hat z_{t+j}=p_\theta\big(\text{last } H \text{ of } [\,z_{t-r+1},\dots,z_t,\ \hat z_{t+1},\dots,\hat z_{t+j-1}\,]\big),\qquad"
             r"e_k=\frac{1}{192}\big\lVert \hat z_{t+k}-z_{t+k}\big\rVert^2"},
  "multi_step_rollout/copy_error": {
    "title": "Rollout copy error",
    "text": "The baseline: the start state never changed. How far the real state has moved away from it after k steps.",
    "latex": r"\frac{1}{192}\big\lVert z_t-z_{t+k}\big\rVert^2"},
  "multi_step_rollout/error_ratio": {
    "title": "Rollout error ratio",
    "text": "The rollout error divided by the copy error, as ratio of the averages. A ratio that grows with k means the errors pile up; a lower "
            "ratio with 3 real start frames than with 1 means real history helps.",
    "latex": r"\text{ratio}_k=\frac{\sum_w e_k(w)}{\sum_w \frac{1}{192}\lVert z_t-z_{t+k}\rVert^2}\quad\text{(over the windows } w)"},
  "cem_planning/success_rate": {
    "title": "CEM: goals reached",
    "text": "Can the model steer the real game to a goal? Per evaluated level a recorded successful episode that starts at the level start; the "
            "recorded state 5, 10 or 25 steps in is the goal. CEM plans with the model: 300 candidate sequences of 5 moves, the 30 best kept, "
            "10 rounds; a sequence costs the distance of its closest imagined state to the goal. It plays the 5 moves, then plans again, with at "
            "most 3 × the goal's steps moves. Reached = the game shows the goal situation (only the energy bar may differ).",
    "latex": r"J(a_{1:5})=\min_{h=1,\dots,5}\big\lVert \hat z_h(a_{1:h})-z_g\big\rVert^2,\qquad"
             r"\text{success}=\frac{|\{\text{attempts that reach } g\}|}{|\{\text{attempts}\}|}"},
  "cem_planning/steps": {
    "title": "CEM: moves played",
    "text": "The average number of moves an attempt played until it reached the goal or ran out of moves.",
    "latex": r"\frac{1}{|A|}\sum_{\text{attempts}}\text{moves}"},
  "cem_planning/waypoints_reached_fraction": {
    "title": "CEM: waypoints reached",
    "text": "Every 5th recorded state of the solution and its final state are waypoints the planner has to reach one after another, with 3 × 5 "
            "moves for each; it stops at the first one it misses. The share of waypoints reached.",
    "latex": r"\frac{1}{|A|}\sum_{\text{attempts}}\frac{\text{waypoints reached}}{\text{waypoints}}"},
  "cem_planning/level_completion_rate": {
    "title": "CEM: levels completed",
    "text": "The share of waypoint attempts that completed the level.",
    "latex": r"\frac{|\{\text{waypoint attempts that complete the level}\}|}{|\{\text{waypoint attempts}\}|}"},
  "action_ranking/spearman": {
    "title": "Action ranking (Spearman)",
    "text": "Does the predictor order the 4 moves by \"which brings me closer to the goal\" like the real game does? That is what the planner "
            "relies on, but cheaper to measure. From one real frame, like the planner. The goal is the recorded state 5 steps later. Per state: "
            "the distance of each predicted and each real outcome to the goal, and the correlation of the two orders (equal distances share "
            "their rank), averaged over states whose real distances are not all equal. 1 = same order, 0 = unrelated, −1 = reversed.",
    "latex": r"d_a=\lVert \hat z(s,a)-z_g\rVert^2,\ \ d^{*}_a=\lVert z(s,a)-z_g\rVert^2,\qquad"
             r"\rho=\frac{\sum_a (r_a-\bar r)(r^{*}_a-\bar r^{*})}{\sqrt{\sum_a (r_a-\bar r)^2\,\sum_a (r^{*}_a-\bar r^{*})^2}}"},
  "action_ranking/action_spread": {
    "title": "Action spread",
    "text": "How different do the 4 moves look at all, measured on the real outcomes? The spread of their distances to the goal, averaged over the "
            "states. Near 0: every move looks the same to the planner, so it has no direction.",
    "latex": r"\text{spread}=\frac{1}{N}\sum_s \operatorname{std}_a\big(d^{*}_a(s)\big)"},
  "split_gaps/*": {
    "title": "Split gaps",
    "text": "What changes between known and new data? For a metric m: seen_val − seen_train (does the model do worse on episodes it was not trained "
            "on? memorization) and unseen − seen_val (how well it transfers to new levels). For errors a positive gap is worse, for accuracies "
            "a negative one. Only over all levels.",
    "latex": r"\Delta_{\text{val}-\text{train}}=m_{\text{seen\_val}}-m_{\text{seen\_train}},\qquad \Delta_{\text{unseen}-\text{val}}=m_{\text{unseen}}-m_{\text{seen\_val}}"},
}


def main() -> None:
  parser = argparse.ArgumentParser(description="Write one HTML page with the results of lewm_analysis for every analysed checkpoint")
  parser.add_argument("--checkpoints", nargs="+", default=None, help="checkpoint ids (folders below evaluation/ls20/lewm_analysis/); default: all")
  parser.add_argument("--output", type=Path, default=None, help="default: evaluation/ls20/lewm_analysis/dashboard.html")
  args = parser.parse_args()
  from plotly.offline import get_plotlyjs

  root = evaluation_dir(GAME) / "lewm_analysis"
  folders = sorted(path.parent for path in root.glob("*/results.json")) if args.checkpoints is None else [root / name for name in args.checkpoints]
  if not folders:
    raise SystemExit(f"No results.json below {root}: run lewm_analysis first")

  checkpoints = []
  for folder in folders:
    results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    maps_file = folder / "latent_maps.json"
    maps = json.loads(maps_file.read_text(encoding="utf-8")) if maps_file.exists() else {}
    rows = results.pop("metrics")
    checkpoints.append({"id": folder.name, **results,
                        "metrics": [[row[column] for column in COLUMNS] for row in rows],  # rows as lists: the page is a third of the size
                        "maps": {name: {column: [state[column] for state in states] for column in states[0]} for name, states in maps.items() if states}})  # per map one list per column
    print(f"{folder.name}: {len(rows)} metrics, maps {sorted(maps)}")

  data = {"columns": COLUMNS, "groups": GROUPS, "splits": SPLITS, "info": INFO, "checkpoints": checkpoints}
  template = Path(__file__).with_name("lewm_dashboard.html").read_text(encoding="utf-8")
  page = template.replace("__LEWM_DATA__", json.dumps(data).replace("</", "<\\/")).replace("__PLOTLY_JS__", get_plotlyjs())  # "</" escaped: the data sits inside a <script>
  output = args.output or root / "dashboard.html"
  output.write_text(page, encoding="utf-8")
  print(f"{len(checkpoints)} checkpoints -> {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
