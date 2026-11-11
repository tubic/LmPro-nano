#!/usr/bin/env bash

# Multi-seed LmPro-Nano variability chain (Reviewer 2.3): for each seed, run a
# full-BPTT finetune from the validation-selected u33000 pretraining
# checkpoint, sweep every retained epoch on the exact-disjoint validation
# control, then evaluate the selected epoch once on the strict 258-record
# test control.  Test is evaluated once per seed, only after its own
# validation selection; no test number is used for any selection.

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-$SCRIPT_ROOT}
GPU_INDEX=${GPU_INDEX:-0}
LOG_DIR=${LOG_DIR:-logs/multiseed_lmpro}
STATUS_FILE=${STATUS_FILE:-$LOG_DIR/status.log}
SEEDS=${SEEDS:-"256 512"}
PRETRAINED_FILE=${PRETRAINED_FILE:-checkpoints/pretrained/corrected_v2_eos_cdr3span0p5_contextkeep_seed1_u100k_eb4032/candidates/checkpoint_u33000.pt}
PRETRAIN_LABEL=${PRETRAIN_LABEL:-u33000}
SPLIT=${SPLIT:-nanobody_clustered}
VALID_RECORDS=${VALID_RECORDS:-baselines/nanobody_clustered/valid_records_pretrain_exact_disjoint.txt}
TEST_RECORDS=${TEST_RECORDS:-baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt}

cd "$ROOT"
mkdir -p "$LOG_DIR"
[ -f "$PRETRAINED_FILE" ] || { echo "missing pretraining checkpoint: $PRETRAINED_FILE" >&2; exit 1; }

timestamp() {
    date '+%F %T %Z'
}

status() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"
}

on_exit() {
    rc=$?
    status "multi-seed LmPro queue exit=$rc"
}
trap on_exit EXIT

gpu_memory_used() {
    nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
}

gpu_utilization() {
    nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
}

wait_for_gpu() {
    local ready=0 used util
    while true; do
        used=$(gpu_memory_used || true)
        util=$(gpu_utilization || true)
        if [[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] \
            && [ "$used" -lt 500 ] && [ "$util" -lt 10 ]; then
            ready=$((ready + 1))
            if [ "$ready" -ge 2 ]; then
                return 0
            fi
        else
            ready=0
            status "waiting for GPU $GPU_INDEX: memory < 500 MiB, util < 10% (used=${used:-unknown}, util=${util:-unknown}, ready=${ready}/2)"
        fi
        sleep 30
    done
}

for SEED in $SEEDS; do
    RUN=corrected_v2_${PRETRAIN_LABEL}_${SPLIT}_fullbptt_seed${SEED}_torch26
    SAVE_DIR=checkpoints/exp2_nanobody/$RUN
    SWEEP_DIR=logs/checkpoint_sweeps/${RUN}_exactdisjoint

    if grep -Fq "seed $SEED finetune complete" "$STATUS_FILE" 2>/dev/null; then
        status "seed $SEED finetune already complete"
    else
        wait_for_gpu
        status "seed $SEED finetune start: $RUN"
        env \
            CUDA_VISIBLE_DEVICES=$GPU_INDEX \
            PRETRAINED_FILE="$PRETRAINED_FILE" \
            PRETRAIN_LABEL="$PRETRAIN_LABEL" \
            SPLIT="$SPLIT" \
            SEED=$SEED \
            SAVE_DIR="$SAVE_DIR" \
            bash run_finetune_corrected_exp2.sh \
            > "$LOG_DIR/${RUN}.log" 2>&1
        status "seed $SEED finetune complete"
    fi

    if grep -Fq "checkpoint sweep complete" "$SWEEP_DIR/status.log" 2>/dev/null; then
        status "seed $SEED validation sweep and one test already complete"
    else
        wait_for_gpu
        status "seed $SEED validation sweep and one strict test start"
        env \
            SAVE_DIR="$SAVE_DIR" \
            RUN_NAME=${RUN}_exactdisjoint \
            VALID_RESTRICT_RECORDS="$VALID_RECORDS" \
            RESTRICT_RECORDS="$TEST_RECORDS" \
            SEED=$SEED \
            GPU_INDEX=$GPU_INDEX \
            GPU_MODE=full \
            RUN_TEST=1 \
            bash tools/sweep_finetune_checkpoints.sh \
            > "$LOG_DIR/${RUN}_sweep.log" 2>&1
        status "seed $SEED validation sweep and one strict test complete"
    fi
done

status "multi-seed LmPro queue complete"
