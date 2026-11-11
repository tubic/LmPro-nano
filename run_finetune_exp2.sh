#!/bin/bash
# Fine-tune from a selected encoder checkpoint.
# Set PRETRAINED_FILE, or provide CKDIR and PRETRAIN_EPOCH.
# run_finetune_corrected_exp2.sh selects full BPTT and a 4 x 2 batch configuration.
# This base launcher defaults to two-step gradient probes; BPTT_ITERATIONS=-1
# backpropagates through every refinement cycle. Reduce the micro-batch if needed
# to avoid fairseq skipping batches after CUDA out-of-memory errors.
BPTT_ITERATIONS=${BPTT_ITERATIONS:-2}
set -u

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$SCRIPT_ROOT}
PRETRAIN_EPOCH=${PRETRAIN_EPOCH:-}
PRETRAINED_FILE=${PRETRAINED_FILE:-}
PRETRAIN_LABEL=${PRETRAIN_LABEL:-}
SPLIT=${SPLIT:-nanobody_clustered}
USE_PRETRAINED=${USE_PRETRAINED:-1}
FINETUNE_BERT_SCHEME=${FINETUNE_BERT_SCHEME:-prefix_tuning}
USE_ROPE=${USE_ROPE:-1}
USE_RESIDUAL_MLP=${USE_RESIDUAL_MLP:-1}

# Use BPTT_ITERATIONS=-1 MAX_SENTENCES=4 UPDATE_FREQ=2 for the corrected full-BPTT run.
# Override CUDA_VISIBLE_DEVICES when scheduling onto a different card.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

LMPRO_ENV_BIN=${LMPRO_ENV_BIN:-/path/to/conda/envs/lmpro-nano-rtx4090/bin}
if [ -d "$LMPRO_ENV_BIN" ]; then
  export PATH="$LMPRO_ENV_BIN:$PATH"
fi
unset LD_LIBRARY_PATH                 # else every cuBLAS matmul fails on this host
export MKL_THREADING_LAYER=GNU

cd "$ROOT" || exit 1                 # data/seq_dict.txt is a relative path

case "$USE_PRETRAINED" in
  0|1) ;;
  *) echo "ERROR: USE_PRETRAINED must be 0 or 1" >&2; exit 1 ;;
esac
case "$FINETUNE_BERT_SCHEME" in
  prefix_tuning|fixed|all_tuning) ;;
  *) echo "ERROR: invalid FINETUNE_BERT_SCHEME=$FINETUNE_BERT_SCHEME" >&2; exit 1 ;;
esac
case "$USE_ROPE:$USE_RESIDUAL_MLP" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "ERROR: USE_ROPE and USE_RESIDUAL_MLP must each be 0 or 1" >&2; exit 1 ;;
esac

INIT_ARGS=()
INIT_LABEL=randominit
if [ "$USE_PRETRAINED" = 1 ]; then
  if [ -n "$PRETRAINED_FILE" ]; then
    INIT_LABEL=${PRETRAIN_LABEL:-pt-$(basename "$PRETRAINED_FILE" .pt)}
  else
    : "${PRETRAIN_EPOCH:?set PRETRAIN_EPOCH or PRETRAINED_FILE when USE_PRETRAINED=1}"
    CKDIR=${CKDIR:-checkpoints/pretrained/fixed_RoPE_abbert_warmup10000_lr0.0006_maxsen128_upfreq4_samplenone_clip0.0}
    PRETRAINED_FILE=$CKDIR/checkpoint${PRETRAIN_EPOCH}.pt
    INIT_LABEL=${PRETRAIN_LABEL:-pt-ep${PRETRAIN_EPOCH}}
  fi
  [ -f "$PRETRAINED_FILE" ] || { echo "ERROR: $PRETRAINED_FILE not written yet" >&2; exit 1; }
  INIT_ARGS=(--finetune-from-model "$PRETRAINED_FILE")
elif [ "$FINETUNE_BERT_SCHEME" != all_tuning ] && [ "${ALLOW_FROZEN_RANDOM_ENCODER:-0}" != 1 ]; then
  echo "ERROR: a random encoder must use FINETUNE_BERT_SCHEME=all_tuning" >&2
  echo "Set ALLOW_FROZEN_RANDOM_ENCODER=1 only for an explicit negative control." >&2
  exit 1
fi

# Environment overrides make checkpoint/seed probes reproducible without cloning this script.
# Defaults remain the paper configuration.
PEAK_LR=${PEAK_LR:-0.0001}; MAX_EPOCH=${MAX_EPOCH:-120}; MAX_UPDATE=${MAX_UPDATE:-0}; PATIENCE=${PATIENCE:-40}
MAX_SENTENCES=${MAX_SENTENCES:-8}; UPDATE_FREQ=${UPDATE_FREQ:-1}; MAX_POSITIONS=${MAX_POSITIONS:-256}
LOG_INTERVAL=${LOG_INTERVAL:-20}
VALID_SUBSET=${VALID_SUBSET:-valid,test}
PREFIX_LEN=${PREFIX_LEN:-5}; ITER_NUM=${ITER_NUM:-5}
LOSS_ENC_S=${LOSS_ENC_S:-1}; LOSS_DEC_S=${LOSS_DEC_S:-1}; LOSS_DEC_X=${LOSS_DEC_X:-2}
LOSS_NORMALIZATION=${LOSS_NORMALIZATION:-legacy}
PRETRAINED_EMBEDDING_NORM=${PRETRAINED_EMBEDDING_NORM:-none}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
CLIP_NORM=${CLIP_NORM:-1.0}
SEED=${SEED:-128}

# Encode gradient depth and accumulation in the directory name so runs cannot be confused.
NORMALIZATION_SUFFIX=""
if [ "$LOSS_NORMALIZATION" != legacy ] || [ "$PRETRAINED_EMBEDDING_NORM" != none ]; then
  NORMALIZATION_SUFFIX=_lossnorm${LOSS_NORMALIZATION}_embnorm${PRETRAINED_EMBEDDING_NORM}
fi
if [ "$CLIP_NORM" != 1.0 ]; then
  NORMALIZATION_SUFFIX=${NORMALIZATION_SUFFIX}_clip${CLIP_NORM}
fi
if [ "$USE_ROPE" = 0 ]; then
  NORMALIZATION_SUFFIX=${NORMALIZATION_SUFFIX}_norope
fi
if [ "$USE_RESIDUAL_MLP" = 0 ]; then
  NORMALIZATION_SUFFIX=${NORMALIZATION_SUFFIX}_nomlp
fi
SCHEME_SUFFIX=""
if [ "$USE_PRETRAINED" = 0 ] || [ "$FINETUNE_BERT_SCHEME" != prefix_tuning ]; then
  SCHEME_SUFFIX=_init${INIT_LABEL}_scheme${FINETUNE_BERT_SCHEME}
fi
DEFAULT_SAVE_DIR=checkpoints/exp2_nanobody/${INIT_LABEL}_${SPLIT}_pflen${PREFIX_LEN}_iter${ITER_NUM}_bptt${BPTT_ITERATIONS}_lr${PEAK_LR}_bsz${MAX_SENTENCES}_uf${UPDATE_FREQ}_seed${SEED}${NORMALIZATION_SUFFIX}${SCHEME_SUFFIX}
SAVE_DIR=${SAVE_DIR:-$DEFAULT_SAVE_DIR}
mkdir -p "$SAVE_DIR"
echo "initialization: $INIT_LABEL"
if [ "$USE_PRETRAINED" = 1 ]; then
  echo "pretrained: $PRETRAINED_FILE"
fi
echo "save dir  : $SAVE_DIR"
echo "encoder finetuning scheme: $FINETUNE_BERT_SCHEME"
echo "bptt-iterations: $BPTT_ITERATIONS $([ "$BPTT_ITERATIONS" = -1 ] || echo '(PROBE ONLY -- not a publishable AAR)')"
echo "micro-batch/update-freq/effective-batch: $MAX_SENTENCES/$UPDATE_FREQ/$((MAX_SENTENCES * UPDATE_FREQ))"
echo "loss normalization / embedding norm / weight decay: $LOSS_NORMALIZATION / $PRETRAINED_EMBEDDING_NORM / $WEIGHT_DECAY"
echo "gradient clip norm: $CLIP_NORM"
echo "RoPE / residual MLP: $USE_ROPE / $USE_RESIDUAL_MLP"

SAVE_ARGS=(--save-interval 1 --save-dir "$SAVE_DIR")
if [ "${NO_SAVE:-0}" = 1 ]; then
  SAVE_ARGS=(--no-save --save-dir "$SAVE_DIR")
fi

exec fairseq-train --sabdab-data abgnn/finetune/exp2-hern/$SPLIT \
  --user-dir fairseq_models --finetune \
  --task antibody_generation_task \
  --criterion antibody_generation_loss \
  --arch antibody_roberta_base \
  --use-rope "$USE_ROPE" --use-residual-mlp "$USE_RESIDUAL_MLP" \
  --finetune-bert-scheme "$FINETUNE_BERT_SCHEME" --pre-seq-len $PREFIX_LEN \
  --pretrained-embedding-norm $PRETRAINED_EMBEDDING_NORM \
  --refine-iteration $ITER_NUM --bptt-iterations $BPTT_ITERATIONS --block_size 8 \
  --loss-scale-enc $LOSS_ENC_S --loss-scale-dec-sloss $LOSS_DEC_S --loss-scale-dec-xloss $LOSS_DEC_X --loss-normalization $LOSS_NORMALIZATION \
  --optimizer adam --clip-norm $CLIP_NORM \
  --lr-scheduler fixed --lr $PEAK_LR --weight-decay $WEIGHT_DECAY \
  --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ --max-positions $MAX_POSITIONS --max-epoch $MAX_EPOCH --max-update $MAX_UPDATE \
  --log-format simple --log-interval $LOG_INTERVAL \
  --valid-subset $VALID_SUBSET --skip-invalid-size-inputs-valid-test \
  "${SAVE_ARGS[@]}" \
  "${INIT_ARGS[@]}" \
  --patience $PATIENCE --tensorboard-logdir "$SAVE_DIR/tensorboard" \
  --seed $SEED --num-workers 0
