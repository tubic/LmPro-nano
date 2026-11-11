#!/bin/bash
# Controlled corrected-loss finetuning entry point. This is intentionally
# separate from the paper-reproduction launcher and never overwrites its runs.

set -euo pipefail

# PyTorch 1.12/CUDA 11.3 cannot JIT CUDA reduction kernels for RTX 4090
# (sm_89). Keep corrected finetuning in the dedicated CUDA 12.4 environment.
export LMPRO_ENV_BIN=${LMPRO_ENV_BIN:-/path/to/conda/envs/lmpro-nano-torch26-cu124/bin}
if [ ! -x "$LMPRO_ENV_BIN/python" ]; then
    echo "ERROR: corrected finetuning environment is unavailable: $LMPRO_ENV_BIN" >&2
    exit 1
fi

# Fairseq checkpoints contain argparse.Namespace metadata. PyTorch 2.6 defaults
# torch.load() to weights_only=True, so trusted local checkpoints need full load.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}

export PRETRAIN_EPOCH=${PRETRAIN_EPOCH:-8}
export SPLIT=${SPLIT:-nanobody_clustered}
export BPTT_ITERATIONS=${BPTT_ITERATIONS:--1}
export MAX_SENTENCES=${MAX_SENTENCES:-4}
export UPDATE_FREQ=${UPDATE_FREQ:-2}
export PEAK_LR=${PEAK_LR:-0.00003}
export MAX_EPOCH=${MAX_EPOCH:-120}
export PATIENCE=${PATIENCE:-40}
export CLIP_NORM=${CLIP_NORM:-10.0}
export VALID_SUBSET=valid
export LOSS_ENC_S=${LOSS_ENC_S:-1}
export LOSS_DEC_S=${LOSS_DEC_S:-1}
# Removing the batch-token denominator makes the raw geometry term about one
# average effective batch larger. 0.02 is a calibrated starting scale, not a
# value to compare with legacy loss=2 runs.
export LOSS_DEC_X=${LOSS_DEC_X:-0.02}
export LOSS_NORMALIZATION=token_mean
export PRETRAINED_EMBEDDING_NORM=layernorm
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}

exec bash "$(dirname "$0")/run_finetune_exp2.sh"
