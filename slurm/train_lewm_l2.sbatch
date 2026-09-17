#!/bin/bash
# LeWM on level 2 (dataset l2_human-goose), 1 H100.
# Submit from ~/master_thesis:   sbatch slurm/train_lewm_l2.sbatch
# Short benchmark:               sbatch --time=00:30:00 --export=ALL,STEPS=3000,RUN_NAME=ls20_l2_benchmark slurm/train_lewm_l2.sbatch
#SBATCH --job-name=lewm_l2
#SBATCH --account=p_scads_lv_llm
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail  # stop on errors and unset variables
export PYTHONUNBUFFERED=1  # print training output immediately into the log

cd "$HOME/master_thesis"
source hpc_env.sh  # STABLEWM_HOME, venv, uv

DATASET="l2_human-goose"  # datasets/ls20/<DATASET>.lance
RUN_NAME="${RUN_NAME:-ls20_l2-human-goose_150k}"  # models/lewm/<RUN_NAME>_<MMDD-HHMM>
STEPS="${STEPS:-150000}"  # optimizer steps
BATCH_SIZE="${BATCH_SIZE:-128}"

echo "host=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD) dataset=$DATASET run=$RUN_NAME steps=$STEPS batch=$BATCH_SIZE"
nvidia-smi --query-gpu=name,memory.total --format=csv

srun "$UV_PROJECT_ENVIRONMENT/bin/python" -m master_thesis.training.lewm --config-name train/arc_lewm \
  data.dataset="$DATASET" \
  run_name="$RUN_NAME" \
  trainer.max_steps="$STEPS" \
  trainer.max_epochs=-1 \
  trainer.log_every_n_steps=50 \
  trainer.check_val_every_n_epoch=20 \
  loader.batch_size="$BATCH_SIZE" \
  loader.num_workers=14 \
  sampling.success_probability_sampling=0.5
