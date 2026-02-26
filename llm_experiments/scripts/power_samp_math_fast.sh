#!/bin/bash
# Same as power_samp_math.sh but runs the fast version (KV cache + optimizations).
# Uses power_samp_math_fast.py; saves to *_fast* files.
#SBATCH --job-name=psamp_math_fast
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta
#SBATCH --cpus-per-task=20
#SBATCH -t 0-23:59
#SBATCH --mem=200000
#SBATCH --array=0-39
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

NUM_SHARDS=5
NUM_SEEDS=8
SEED=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))

module load anaconda/Python-ML-2025a

export HF_HOME=/home/gridsan/mshi/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_TRITON_FLASH_ATTN=0

# SLURM may set CUDA_VISIBLE_DEVICES to GPU UUIDs (e.g. "GPU-3c11fbc2-..."),
# which vLLM cannot parse. Remap to integer indices (0, 1, ...).
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NEW_IDS=$(python3 -c "
import os
devs = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
print(','.join(str(i) for i in range(len(devs))))
")
    export CUDA_VISIBLE_DEVICES="$NEW_IDS"
fi

WORK_DIR=/home/gridsan/mshi/reasoning-with-sampling/llm_experiments
export PYTHONPATH="$PYTHONPATH:$WORK_DIR"

source activate psamp
cd "$WORK_DIR"

echo "Running FAST shard BATCH_IDX=${BATCH_IDX} SEED=${SEED} (task ${SLURM_ARRAY_TASK_ID})"
python power_samp_math_fast.py \
  --batch_idx="${BATCH_IDX}" \
  --mcmc_steps=10 \
  --temperature=0.25 \
  --seed="${SEED}" \
  --model=qwen_math
# Add --no-flash-attn if Flash Attention 2 is not installed
# Add --compile-model to try torch.compile (experimental)
# Add --verbose for extra MCMC prints
