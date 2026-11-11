#!/usr/bin/env bash

# Finetune the validation-best completed pretraining checkpoint with full BPTT,
# then select a finetuning epoch on the exact-disjoint validation subset and
# evaluate the selected epoch once on the matching test subset.

set -euo pipefail

ROOT=${ROOT:-.}
CUDA_UUID=${CUDA_UUID:-MIG-10b9b777-a776-56de-9f7c-efde6c584f71}
GPU_INDEX=${GPU_INDEX:-7}
GPU_MODE=${GPU_MODE:-mig}
GPU_INSTANCE_ID=${GPU_INSTANCE_ID:-2}
COMPUTE_INSTANCE_ID=${COMPUTE_INSTANCE_ID:-0}
MIG_DEVICE_INDEX=${MIG_DEVICE_INDEX:-1}
GPU_MEMORY_LIMIT_MIB=${GPU_MEMORY_LIMIT_MIB:-500}
UPSTREAM_STATUS=${UPSTREAM_STATUS:-logs/clean_reviewer_baselines/status.log}
UPSTREAM_PATTERN=${UPSTREAM_PATTERN:-clean reviewer baseline queue complete}
WAIT_FOR_UPSTREAM=${WAIT_FOR_UPSTREAM:-1}

PRETRAIN_EPOCH=${PRETRAIN_EPOCH:-22}
EXPECTED_PRETRAIN_SHA256=${EXPECTED_PRETRAIN_SHA256:-aae45dbc700910848977aebe0a4e3759d8127feb941924c8f9960c6f2fec35cb}
PRETRAIN_DIR=${PRETRAIN_DIR:-checkpoints/pretrained/fixed_RoPE_abbert_warmup10000_lr0.0006_maxsen128_upfreq4_samplenone_clip0.0}
PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-$PRETRAIN_DIR/checkpoint${PRETRAIN_EPOCH}.pt}
MAX_SENTENCES=${MAX_SENTENCES:-4}
UPDATE_FREQ=${UPDATE_FREQ:-2}

SAVE_DIR=${SAVE_DIR:-checkpoints/exp2_nanobody/pt-ep${PRETRAIN_EPOCH}_nanobody_clustered_pflen5_iter5_bptt-1_lr0.0001_bsz${MAX_SENTENCES}_uf${UPDATE_FREQ}_seed128}
RUN_NAME=${RUN_NAME:-fullbptt_pt-ep${PRETRAIN_EPOCH}_seed128_pretrain_exact_disjoint}
LOG_DIR=${LOG_DIR:-logs/fullbptt_pt-ep${PRETRAIN_EPOCH}_seed128}
STATUS_FILE=${STATUS_FILE:-$LOG_DIR/status.log}
FINETUNE_LOG=${FINETUNE_LOG:-$LOG_DIR/finetune.log}
PROVENANCE_FILE=${PROVENANCE_FILE:-$LOG_DIR/provenance.tsv}
SWEEP_DIR=${SWEEP_DIR:-logs/checkpoint_sweeps/$RUN_NAME}

cd "$ROOT"
mkdir -p "$LOG_DIR"

timestamp() {
    date '+%F %T %Z'
}

status() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"
}

on_exit() {
    rc=$?
    status "pt${PRETRAIN_EPOCH} full-BPTT queue exit=$rc"
}
trap on_exit EXIT

[ -f "$PRETRAIN_CHECKPOINT" ] || {
    status "missing pinned pretraining checkpoint: $PRETRAIN_CHECKPOINT"
    exit 1
}
actual_sha256=$(sha256sum "$PRETRAIN_CHECKPOINT" | awk '{print $1}')
[ "$actual_sha256" = "$EXPECTED_PRETRAIN_SHA256" ] || {
    status "pretraining checkpoint checksum mismatch: expected=$EXPECTED_PRETRAIN_SHA256 actual=$actual_sha256"
    exit 1
}

if [ -f "$PROVENANCE_FILE" ]; then
    recorded_sha256=$(awk -F '\t' '$1 == "pretrain_sha256" {print $2}' "$PROVENANCE_FILE")
    [ "$recorded_sha256" = "$actual_sha256" ] || {
        status "existing run provenance does not match the pinned checkpoint"
        exit 1
    }
else
    {
        printf 'field\tvalue\n'
        printf 'pretrain_epoch\t%s\n' "$PRETRAIN_EPOCH"
        printf 'pretrain_checkpoint\t%s\n' "$PRETRAIN_CHECKPOINT"
        printf 'pretrain_sha256\t%s\n' "$actual_sha256"
        printf 'bptt_iterations\t-1\n'
        printf 'micro_batch\t%s\n' "$MAX_SENTENCES"
        printf 'update_freq\t%s\n' "$UPDATE_FREQ"
        printf 'effective_batch\t%s\n' "$((MAX_SENTENCES * UPDATE_FREQ))"
        printf 'seed\t128\n'
    } > "$PROVENANCE_FILE"
fi
status "pinned pretraining checkpoint verified (epoch=$PRETRAIN_EPOCH sha256=$actual_sha256)"

if [ "$WAIT_FOR_UPSTREAM" = 1 ]; then
    status "waiting for '$UPSTREAM_PATTERN' in $UPSTREAM_STATUS"
    while ! grep -Fq "$UPSTREAM_PATTERN" "$UPSTREAM_STATUS" 2>/dev/null; do
        if grep -Eq 'clean reviewer baseline queue exit=[1-9][0-9]*' "$UPSTREAM_STATUS" 2>/dev/null; then
            status "clean reviewer baseline queue failed; refusing to consume its GPU"
            exit 1
        fi
        sleep 30
    done
else
    status "upstream wait disabled"
fi

export CUDA_VISIBLE_DEVICES="$CUDA_UUID"
gpu_memory_used() {
    if [ "$GPU_MODE" = full ]; then
        nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used \
            --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
    else
        nvidia-smi -i "$GPU_INDEX" 2>/dev/null | awk \
            -v gpu="$GPU_INDEX" \
            -v gi="$GPU_INSTANCE_ID" \
            -v ci="$COMPUTE_INSTANCE_ID" \
            -v dev="$MIG_DEVICE_INDEX" \
            '$1 == "|" && $2 == gpu && $3 == gi && $4 == ci && $5 == dev {
                gsub(/MiB/, "", $7); print $7; exit
            }'
    fi
}
while true; do
    used=$(gpu_memory_used || true)
    if [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt "$GPU_MEMORY_LIMIT_MIB" ]; then
        break
    fi
    status "waiting for target MIG memory below ${GPU_MEMORY_LIMIT_MIB} MiB (used=${used:-unknown})"
    sleep 30
done

if grep -Fq 'full-BPTT finetuning complete' "$STATUS_FILE" 2>/dev/null; then
    status "full-BPTT finetuning already complete"
else
    status "full-BPTT finetuning start"
    if env \
        CUDA_VISIBLE_DEVICES="$CUDA_UUID" \
        PRETRAIN_EPOCH="$PRETRAIN_EPOCH" \
        BPTT_ITERATIONS=-1 \
        MAX_SENTENCES="$MAX_SENTENCES" \
        UPDATE_FREQ="$UPDATE_FREQ" \
        SAVE_DIR="$SAVE_DIR" \
        bash run_finetune_exp2.sh \
        >> "$FINETUNE_LOG" 2>&1; then
        finetune_rc=0
    else
        finetune_rc=$?
    fi
    status "full-BPTT finetuning command exit=$finetune_rc"
    [ "$finetune_rc" -eq 0 ] || exit "$finetune_rc"

    oom_count=$(grep -Fca 'OOM: Ran out of memory' "$FINETUNE_LOG" 2>/dev/null || true)
    status "full-BPTT finetuning OOM count=$oom_count"
    [ "$oom_count" -eq 0 ] || {
        status "refusing to evaluate a run that silently dropped OOM batches"
        exit 1
    }
    status "full-BPTT finetuning complete"
fi

if grep -Fq 'checkpoint sweep complete' "$SWEEP_DIR/status.log" 2>/dev/null; then
    status "strict checkpoint sweep already complete"
else
    status "strict checkpoint sweep start"
    env \
        CUDA_VISIBLE_DEVICES="$CUDA_UUID" \
        GPU_MODE="$GPU_MODE" \
        GPU_INDEX="$GPU_INDEX" \
        GPU_INSTANCE_ID="$GPU_INSTANCE_ID" \
        COMPUTE_INSTANCE_ID="$COMPUTE_INSTANCE_ID" \
        MIG_DEVICE_INDEX="$MIG_DEVICE_INDEX" \
        SAVE_DIR="$SAVE_DIR" \
        RUN_NAME="$RUN_NAME" \
        OUT_DIR="$SWEEP_DIR" \
        VALID_RESTRICT_RECORDS=baselines/nanobody_clustered/valid_records_pretrain_exact_disjoint.txt \
        RESTRICT_RECORDS=baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt \
        SEED=128 \
        NUM_DECODE=10000 \
        TOPK=100 \
        REPORT_TOPKS='1 10 100 1000' \
        RUN_TEST=1 \
        bash tools/sweep_finetune_checkpoints.sh \
        > "$LOG_DIR/checkpoint_sweep.driver.log" 2>&1
    status "strict checkpoint sweep complete"
fi

status "pt${PRETRAIN_EPOCH} full-BPTT queue complete"
