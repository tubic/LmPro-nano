#!/usr/bin/env bash

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$SCRIPT_ROOT}
cd "$ROOT"

SEQ_DATA_DIR=${SEQ_DATA_DIR:-abgnn/final_3_data/seq}
TAG_DATA_DIR=${TAG_DATA_DIR:-abgnn/final_3_data/tag}
FAIRSEQ_MODELS_DIR=${FAIRSEQ_MODELS_DIR:-fairseq_models}

# Keep the launcher usable from a fresh non-interactive shell, matching the
# finetuning entry points below.  The active workspace's Fairseq installation
# is tied to this Python environment.
LMPRO_ENV_BIN=${LMPRO_ENV_BIN:-/home/dataDisk_7T/bo_sun/conda_envs/lmpro-nano-rtx4090/bin}
if [ -d "$LMPRO_ENV_BIN" ]; then
    export PATH="$LMPRO_ENV_BIN:$PATH"
fi

TOTAL_UPDATES=${TOTAL_UPDATES:-2000000}    # Total number of training steps
WARMUP_UPDATES=${1:-${WARMUP_UPDATES:-10000}}    # Warmup the learning rate
PEAK_LR=${2:-${PEAK_LR:-0.0006}}                  # Peak learning rate
TOKENS_PER_SAMPLE=${TOKENS_PER_SAMPLE:-256}
MAX_POSITIONS=${MAX_POSITIONS:-$TOKENS_PER_SAMPLE}
MAX_SENTENCES=${3:-${MAX_SENTENCES:-128}}  # fixed-length token blocks per GPU
SAMPLE_MODE=${5:-${SAMPLE_MODE:-none}}
CLIP_NORM=${6:-${CLIP_NORM:-0.0}}
FULL_CDR3_MASK_PROB=${FULL_CDR3_MASK_PROB:-0.0}
MASK_CONTEXT_MODE=${MASK_CONTEXT_MODE:-legacy_pad}
PRETRAIN_WEIGHT_DECAY=${PRETRAIN_WEIGHT_DECAY:-0.01}
PRETRAIN_SEED=${PRETRAIN_SEED:-1}
KEEP_LAST_EPOCHS=${KEEP_LAST_EPOCHS:-100}
SAVE_INTERVAL_UPDATES=${SAVE_INTERVAL_UPDATES:-0}
KEEP_INTERVAL_UPDATES=${KEEP_INTERVAL_UPDATES:--1}
NUM_WORKERS=${NUM_WORKERS:-1}
LOG_INTERVAL=${LOG_INTERVAL:-20}
# The effective batch counts 256-token blocks, not raw VHH records. With
# sample-break-mode=none, one block contains about two source VHH sequences.

# Train only on GPUs that are present, responsive and not blocklisted. PCI 0000:86:00.0 on
# this host has repeatedly fallen off the bus (Xid 79) and is excluded by default; see
# tools/healthy_gpus.sh to change or clear the blocklist.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(bash "$(dirname "$0")/tools/healthy_gpus.sh")}"
NGPU=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ "$NGPU" -lt 1 ]; then
    echo "ERROR: no healthy GPU found (all blocklisted or off the bus?)" >&2; exit 1
fi

# Keep the effective batch at ~2048 blocks/update however many GPUs survive, so the LR
# schedule stays comparable across restarts on a different GPU count.
TARGET_BATCH=${TARGET_BATCH:-2048}
UPDATE_FREQ=${4:-$(( (TARGET_BATCH + MAX_SENTENCES * NGPU / 2) / (MAX_SENTENCES * NGPU) ))}
[ "$UPDATE_FREQ" -lt 1 ] && UPDATE_FREQ=1
echo "GPUs=[$CUDA_VISIBLE_DEVICES] (n=$NGPU)  blocks/gpu=$MAX_SENTENCES  update-freq=$UPDATE_FREQ" \
     " -> effective block batch $((MAX_SENTENCES * NGPU * UPDATE_FREQ)) (target $TARGET_BATCH)"

# Save dir is keyed on upfreq4 regardless of the actual update-freq, so that a run resumed on
# a different GPU count keeps writing to (and resuming from) the same checkpoint directory.
MASK_SUFFIX=""
if { [ "$FULL_CDR3_MASK_PROB" != "0" ] && [ "$FULL_CDR3_MASK_PROB" != "0.0" ]; } \
    || [ "$MASK_CONTEXT_MODE" != "legacy_pad" ]; then
    MASK_TAG=${FULL_CDR3_MASK_PROB//./p}
    MASK_SUFFIX=_cdr3span${MASK_TAG}_context${MASK_CONTEXT_MODE}
fi
if [ -z "$MASK_SUFFIX" ]; then
    # Preserve the historical path so the legacy run can still resume losslessly.
    MODEL_NAME=abbert_warmup${WARMUP_UPDATES}_lr${PEAK_LR}_maxsen${MAX_SENTENCES}_upfreq4_sample${SAMPLE_MODE}_clip${CLIP_NORM}
else
    MODEL_NAME=abbert_warmup${WARMUP_UPDATES}_lr${PEAK_LR}_bsz${MAX_SENTENCES}_uf${UPDATE_FREQ}_sample${SAMPLE_MODE}_clip${CLIP_NORM}_updates${TOTAL_UPDATES}_seed${PRETRAIN_SEED}${MASK_SUFFIX}
fi
# Fresh save dir for the FIXED architecture (corrected vocabulary, real RoPE replacing the
# learned absolute PE, functional residual MLP). The older Finanl_new_RoPE_decoder_* /
# new_RoPE_decoder_* dirs hold the pre-fix weights that produced the paper's Table 1 and are
# NOT loadable by this code — keep them for provenance, do not resume from them.
MODEL_PATH=${MODEL_PATH:-checkpoints/pretrained/fixed_RoPE_$MODEL_NAME}

# The interactive shell's LD_LIBRARY_PATH shadows torch's bundled cuBLAS and makes every
# matmul fail with CUBLAS_STATUS_INVALID_VALUE. Clear it before touching the GPU.
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
echo "$(which fairseq-train)"
echo "sample-mode=${SAMPLE_MODE} mask-context=${MASK_CONTEXT_MODE} full-cdr3-prob=${FULL_CDR3_MASK_PROB} total-updates=${TOTAL_UPDATES}"
echo "seed=${PRETRAIN_SEED} save-dir=${MODEL_PATH} keep-last-epochs=${KEEP_LAST_EPOCHS}"
echo "save-interval-updates=${SAVE_INTERVAL_UPDATES} keep-interval-updates=${KEEP_INTERVAL_UPDATES}"

if [ "${PRINT_CONFIG_ONLY:-0}" = 1 ]; then
    exit 0
fi

exec fairseq-train "$SEQ_DATA_DIR" \
    --tag-data "$TAG_DATA_DIR" \
    --user-dir "$FAIRSEQ_MODELS_DIR" \
    --task antibody_masked_lm \
    --criterion antibody_masked_lm \
    --arch antibody_roberta_base \
    --sample-break-mode "$SAMPLE_MODE" \
    --mask-prob 0.5 \
    --full-cdr3-mask-prob "$FULL_CDR3_MASK_PROB" \
    --mask-context-mode "$MASK_CONTEXT_MODE" \
    --leave-unmasked-prob 0.0 \
    --random-token-prob 0.0 \
    --tokens-per-sample "$TOKENS_PER_SAMPLE" \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --adam-eps 1e-6 \
    --clip-norm "$CLIP_NORM" \
    --lr-scheduler polynomial_decay \
    --lr "$PEAK_LR" \
    --warmup-updates "$WARMUP_UPDATES" \
    --total-num-update "$TOTAL_UPDATES" \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay "$PRETRAIN_WEIGHT_DECAY" \
    --batch-size "$MAX_SENTENCES" \
    --update-freq "$UPDATE_FREQ" \
    --max-update "$TOTAL_UPDATES" \
    --seed "$PRETRAIN_SEED" \
    --fixed-validation-seed "$PRETRAIN_SEED" \
    --num-workers "$NUM_WORKERS" \
    --log-format simple \
    --log-interval "$LOG_INTERVAL" \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates "$SAVE_INTERVAL_UPDATES" \
    --keep-interval-updates "$KEEP_INTERVAL_UPDATES" \
    --keep-last-epochs "$KEEP_LAST_EPOCHS" \
    --save-dir "$MODEL_PATH" \
    --tensorboard-logdir "$MODEL_PATH/tensorboard"
