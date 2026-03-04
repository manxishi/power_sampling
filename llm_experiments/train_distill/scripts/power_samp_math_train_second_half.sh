#!/bin/bash
#SBATCH --job-name=psamp_train_b
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta
#SBATCH --cpus-per-task=20
#SBATCH -t 0-23:59
#SBATCH --mem=200000
#SBATCH --array=0-36     # 37 tasks covering batches 38-74 (problems 3800-7499)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

BATCH_OFFSET=38   # start from batch 38 (problem 3800)
SEED_OFFSET=${SEED_OFFSET:-0}
SEED=$(( SEED_OFFSET ))
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID + BATCH_OFFSET ))

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
