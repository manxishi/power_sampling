#!/bin/bash
#SBATCH --job-name=psamp_train
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta
#SBATCH --cpus-per-task=20
#SBATCH -t 0-23:59
#SBATCH --mem=200000
#SBATCH --array=0-74     # 75 batches x 100 problems = 7500 (full train set, 1 seed)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# To run with multiple seeds for more samples per problem:
#   sbatch --array=0-599 scripts/power_samp_math_train.sh
#   (75 batches x 8 seeds = 600 jobs; uses SEED_OFFSET to avoid collisions on re-runs)
#
# To run just a subset of batches:
#   sbatch --array=0-9 scripts/power_samp_math_train.sh

NUM_BATCHES=75
NUM_SEEDS=1           # increase to 8 for multiple samples per problem
SEED_OFFSET=${SEED_OFFSET:-0}
SEED=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS + SEED_OFFSET ))
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))

module load anaconda/Python-ML-2025a

export HF_HOME=/home/gridsan/mshi/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_TRITON_FLASH_ATTN=0

# Remap CUDA_VISIBLE_DEVICES from UUIDs to integer indices
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NEW_IDS=$(python3 -c "
import os
devs = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
print(','.join(str(i) for i in range(len(devs))))
")
    export CUDA_VISIBLE_DEVICES="$NEW_IDS"
fi

WORK_DIR=/home/gridsan/mshi/reasoning-with-sampling/llm_experiments/train_distill
export PYTHONPATH="$PYTHONPATH:/home/gridsan/mshi/reasoning-with-sampling/llm_experiments"

source activate psamp
cd "$WORK_DIR"

echo "Running BATCH_IDX=${BATCH_IDX} SEED=${SEED} (task ${SLURM_ARRAY_TASK_ID})"
python power_samp_math_train.py \
  --batch_idx="${BATCH_IDX}" \
  --batch_size=100 \
  --mcmc_steps=10 \
  --temperature=0.25 \
  --seed="${SEED}" \
  --model=qwen_math \
  --save_str=results/
