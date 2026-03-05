#!/bin/bash
#SBATCH --job-name=psamp_train_b
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta
#SBATCH --cpus-per-task=20
#SBATCH -t 2-00:00
#SBATCH --mem=200000
#SBATCH --array=25-49    # batches 25-49 (problems 2500-4999)
#SBATCH --output=/home/gridsan/mshi/reasoning-with-sampling/llm_experiments/train_distill/logs/%x_%A_%a.out
#SBATCH --error=/home/gridsan/mshi/reasoning-with-sampling/llm_experiments/train_distill/logs/%x_%A_%a.err

NUM_SEEDS=1
SEED_OFFSET=${SEED_OFFSET:-0}
SEED=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS + SEED_OFFSET ))
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))

module load anaconda/Python-ML-2025a

export HF_HOME=/home/gridsan/mshi/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_TRITON_FLASH_ATTN=0

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
