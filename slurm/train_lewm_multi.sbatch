#!/bin/bash
# One LeWM on levels 1,2,4,5,6,7 (human + Goose + branched Goose); level 3 is held out. 1 H100, 10 h.
# Submit from ~/master_thesis:   sbatch slurm/train_lewm_multi.sbatch
# Smoke test (planning early): sbatch --time=00:30:00 --export=ALL,STEPS=400,PLANNING_EVERY=200,RUN_NAME=ls20_multi_smoke slurm/train_lewm_multi.sbatch
#SBATCH --job-name=lewm_multi
#SBATCH --account=p_atari26  # p_scads_lv_llm is locked ("Lecture ended")
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14  # Capella allows at most 14 cores per requested GPU
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail  # stop on errors and unset variables
export PYTHONUNBUFFERED=1  # print training output immediately into the log

cd "$HOME/master_thesis"
source hpc_env.sh  # STABLEWM_HOME, venv, uv

DATASET="human-goose-branch_l1.2.4.5.6.7"  # datasets/ls20/<DATASET>.lance
RUN_NAME="${RUN_NAME:-ls20_multi-no-l3_220k_succ0.1}"  # models/lewm/<RUN_NAME>_<MMDD-HHMM>
STEPS="${STEPS:-220000}"  # optimizer steps (~7.6 it/s -> ~8 h, leaves time for validation and planning)
BATCH_SIZE="${BATCH_SIZE:-128}"
SUCCESS_SHARE="${SUCCESS_SHARE:-0.1}"  # share of training samples from successful episodes, per level (the data has ~3 %)
PLANNING_EVERY="${PLANNING_EVERY:-10000}"  # waypoint rollout in the real game every N steps (GIFs + waypoints/* in TensorBoard)
OVERRIDES="${OVERRIDES:-}"  # extra Hydra overrides, e.g. --export=ALL,OVERRIDES="checkpoint_every_steps=100 planning.levels=[1,3]"

echo "host=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD) dataset=$DATASET run=$RUN_NAME steps=$STEPS batch=$BATCH_SIZE"
nvidia-smi --query-gpu=name,memory.total --format=csv

srun "$UV_PROJECT_ENVIRONMENT/bin/python" -m master_thesis.training.lewm --config-name train/arc_lewm \
  data.dataset="$DATASET" \
  run_name="$RUN_NAME" \
  trainer.max_steps="$STEPS" \
  trainer.max_epochs=-1 \
  trainer.log_every_n_steps=50 \
  trainer.check_val_every_n_epoch=10 \
  loader.batch_size="$BATCH_SIZE" \
  loader.num_workers=12 \
  sampling.success_probability_sampling="$SUCCESS_SHARE" \
  planning.every_steps="$PLANNING_EVERY" \
  planning.mode=waypoints \
  planning.samples=300 \
  planning.iterations=10 \
  $OVERRIDES
