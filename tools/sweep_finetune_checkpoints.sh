#!/usr/bin/env bash

# Rank every retained finetuning epoch on validation AAR, then evaluate the
# validation-selected checkpoint once on the matched test subset.  Individual
# inference logs make the sweep resumable after an interruption.

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-$SCRIPT_ROOT}
SAVE_DIR=${SAVE_DIR:?set SAVE_DIR to the finetuning checkpoint directory}
RUN_NAME=${RUN_NAME:-$(basename "$SAVE_DIR")}
DATA_PATH=${DATA_PATH:-abgnn/finetune/exp2-hern/nanobody_clustered}
RESTRICT_RECORDS=${RESTRICT_RECORDS:-baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt}
VALID_RESTRICT_RECORDS=${VALID_RESTRICT_RECORDS:-baselines/nanobody_clustered/valid_records_pretrain_exact_disjoint.txt}
SEED=${SEED:-128}
NUM_DECODE=${NUM_DECODE:-10000}
TOPK=${TOPK:-100}
REPORT_TOPKS=${REPORT_TOPKS:-"1 10 100 1000"}
WAIT_FOR_STATUS=${WAIT_FOR_STATUS:-}
WAIT_FOR_PATTERN=${WAIT_FOR_PATTERN:-test inference exit=0}
RUN_TEST=${RUN_TEST:-0}
CHECKPOINT_EPOCHS=${CHECKPOINT_EPOCHS:-}
GPU_MEMORY_LIMIT_MIB=${GPU_MEMORY_LIMIT_MIB:-500}
GPU_UTIL_LIMIT_PERCENT=${GPU_UTIL_LIMIT_PERCENT:-101}
GPU_READY_SAMPLES=${GPU_READY_SAMPLES:-1}
GPU_INDEX=${GPU_INDEX:-0}
GPU_MODE=${GPU_MODE:-full}
GPU_INSTANCE_ID=${GPU_INSTANCE_ID:-2}
COMPUTE_INSTANCE_ID=${COMPUTE_INSTANCE_ID:-0}
MIG_DEVICE_INDEX=${MIG_DEVICE_INDEX:-1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_INDEX}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
LMPRO_ENV_BIN=${LMPRO_ENV_BIN:-/path/to/conda/envs/lmpro-nano-torch26-cu124/bin}
if [ -d "$LMPRO_ENV_BIN" ]; then
    export PATH="$LMPRO_ENV_BIN:$PATH"
fi
PYTHON=${PYTHON:-$LMPRO_ENV_BIN/python}
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU

cd "$ROOT"
[ -d "$SAVE_DIR" ] || { echo "ERROR: missing checkpoint directory: $SAVE_DIR" >&2; exit 1; }
[ -x "$PYTHON" ] || { echo "ERROR: missing Python interpreter: $PYTHON" >&2; exit 1; }
if [ "$RUN_TEST" = 1 ] && [ ! -f "$RESTRICT_RECORDS" ]; then
    echo "ERROR: missing test record subset: $RESTRICT_RECORDS" >&2
    exit 1
fi
if [ -n "$VALID_RESTRICT_RECORDS" ] && [ ! -f "$VALID_RESTRICT_RECORDS" ]; then
    echo "ERROR: missing validation record subset: $VALID_RESTRICT_RECORDS" >&2
    exit 1
fi

OUT_DIR=${OUT_DIR:-logs/checkpoint_sweeps/$RUN_NAME}
RESULTS_FILE=${RESULTS_FILE:-$OUT_DIR/validation_results.tsv}
SELECTION_FILE=${SELECTION_FILE:-$OUT_DIR/selected_checkpoint.txt}
STATUS_FILE=${STATUS_FILE:-$OUT_DIR/status.log}
mkdir -p "$OUT_DIR"

timestamp() {
    date '+%F %T %Z'
}

status() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"
}

if [ -n "$WAIT_FOR_STATUS" ]; then
    status "waiting for '$WAIT_FOR_PATTERN' in $WAIT_FOR_STATUS"
    while ! grep -Fq "$WAIT_FOR_PATTERN" "$WAIT_FOR_STATUS" 2>/dev/null; do
        if grep -Eq 'finetune exit=[1-9][0-9]*|validation inference exit=[1-9][0-9]*|test inference exit=[1-9][0-9]*' "$WAIT_FOR_STATUS" 2>/dev/null; then
            status "upstream run failed; refusing to start checkpoint sweep"
            exit 1
        fi
        sleep 60
    done
fi

# Avoid racing the upstream process as it releases the MIG slice.
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
gpu_utilization() {
    if [ "$GPU_MODE" = full ]; then
        nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
    else
        # MIG utilization is not available reliably on this driver. Memory is
        # still an exclusive-enough readiness signal for the legacy MIG path.
        printf '0\n'
    fi
}
ready_samples=0
while true; do
    used=$(gpu_memory_used || true)
    util=$(gpu_utilization || true)
    if [[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] \
        && [ "$used" -lt "$GPU_MEMORY_LIMIT_MIB" ] \
        && [ "$util" -lt "$GPU_UTIL_LIMIT_PERCENT" ]; then
        ready_samples=$((ready_samples + 1))
        if [ "$ready_samples" -ge "$GPU_READY_SAMPLES" ]; then
            break
        fi
    else
        ready_samples=0
    fi
    status "waiting for GPU readiness: memory < ${GPU_MEMORY_LIMIT_MIB} MiB, util < ${GPU_UTIL_LIMIT_PERCENT}% (used=${used:-unknown}, util=${util:-unknown}, ready=${ready_samples}/${GPU_READY_SAMPLES})"
    sleep 60
done

checkpoints=()
if [ -n "$CHECKPOINT_EPOCHS" ]; then
    for epoch in $CHECKPOINT_EPOCHS; do
        checkpoint="$SAVE_DIR/checkpoint${epoch}.pt"
        [ -f "$checkpoint" ] || { status "missing requested checkpoint: $checkpoint"; exit 1; }
        checkpoints+=("$checkpoint")
    done
else
    mapfile -t checkpoints < <(
        find "$SAVE_DIR" -maxdepth 1 -type f -name 'checkpoint[0-9]*.pt' -print | sort -V
    )
fi
[ "${#checkpoints[@]}" -gt 0 ] || { status "no epoch checkpoints found"; exit 1; }
status "validation sweep start: ${#checkpoints[@]} checkpoints, seed=$SEED, top $TOPK of $NUM_DECODE"

read -r -a report_topks <<< "$REPORT_TOPKS"
validation_restrict_args=()
if [ -n "$VALID_RESTRICT_RECORDS" ]; then
    validation_restrict_args=(--restrict_records "$VALID_RESTRICT_RECORDS")
fi

for checkpoint in "${checkpoints[@]}"; do
    filename=$(basename "$checkpoint")
    epoch=${filename#checkpoint}
    epoch=${epoch%.pt}
    log="$OUT_DIR/valid_epoch${epoch}_seed${SEED}.log"

    if grep -q '^Amino acid recovery rate = ' "$log" 2>/dev/null; then
        status "epoch $epoch already complete"
        continue
    fi

    status "epoch $epoch validation inference start"
    "$PYTHON" -u inference.py \
        --cdr_type 3 \
        --cktpath "$checkpoint" \
        --data_path "$DATA_PATH" \
        --split valid \
        "${validation_restrict_args[@]}" \
        --num_decode "$NUM_DECODE" \
        --topk "$TOPK" \
        --seed "$SEED" \
        --report_topks "${report_topks[@]}" \
        > "$log" 2>&1
    status "epoch $epoch validation inference complete"
done

printf 'epoch\taar\trmsd\tppl\tcheckpoint\n' > "$RESULTS_FILE"
for checkpoint in "${checkpoints[@]}"; do
    filename=$(basename "$checkpoint")
    epoch=${filename#checkpoint}
    epoch=${epoch%.pt}
    log="$OUT_DIR/valid_epoch${epoch}_seed${SEED}.log"
    aar=$(awk -F'= ' '/^Amino acid recovery rate = / {v=$2} END {print v}' "$log")
    rmsd=$(awk -F'= ' '/^RMSD = / {v=$2} END {print v}' "$log")
    ppl=$(awk -F'= ' '/^PPL = / {v=$2} END {print v}' "$log")
    [ -n "$aar" ] && [ -n "$rmsd" ] && [ -n "$ppl" ] || {
        status "missing final metrics for epoch $epoch"
        exit 1
    }
    printf '%s\t%s\t%s\t%s\t%s\n' "$epoch" "$aar" "$rmsd" "$ppl" "$checkpoint" >> "$RESULTS_FILE"
done

best_line=$(awk -F '\t' 'NR > 1 && ($2 + 0) > best {best=$2 + 0; line=$0} END {print line}' \
    best=-1 "$RESULTS_FILE")
[ -n "$best_line" ] || { status "could not select a validation checkpoint"; exit 1; }
IFS=$'\t' read -r best_epoch best_aar best_rmsd best_ppl best_checkpoint <<< "$best_line"
best_checkpoint_sha256=$(sha256sum "$best_checkpoint" | awk '{print $1}')
{
    printf 'selection_metric=validation_micro_aar\n'
    printf 'seed=%s\n' "$SEED"
    printf 'protocol=top_%s_of_%s\n' "$TOPK" "$NUM_DECODE"
    printf 'epoch=%s\n' "$best_epoch"
    printf 'aar=%s\n' "$best_aar"
    printf 'rmsd=%s\n' "$best_rmsd"
    printf 'ppl=%s\n' "$best_ppl"
    printf 'checkpoint=%s\n' "$best_checkpoint"
    printf 'checkpoint_sha256=%s\n' "$best_checkpoint_sha256"
} > "$SELECTION_FILE"
status "selected epoch $best_epoch (validation AAR=$best_aar)"

if [ "$RUN_TEST" = 1 ]; then
    test_log="$OUT_DIR/test_selected_epoch${best_epoch}_seed${SEED}.log"
    if ! grep -q '^Amino acid recovery rate = ' "$test_log" 2>/dev/null; then
        status "selected epoch $best_epoch matched-test inference start"
        "$PYTHON" -u inference.py \
            --cdr_type 3 \
            --cktpath "$best_checkpoint" \
            --data_path "$DATA_PATH" \
            --split test \
            --restrict_records "$RESTRICT_RECORDS" \
            --num_decode "$NUM_DECODE" \
            --topk "$TOPK" \
            --seed "$SEED" \
            --report_topks "${report_topks[@]}" \
            > "$test_log" 2>&1
    fi
    test_aar=$(awk -F'= ' '/^Amino acid recovery rate = / {v=$2} END {print v}' "$test_log")
    [ -n "$test_aar" ] || { status "selected-checkpoint test metrics missing"; exit 1; }
    status "selected epoch $best_epoch matched-test complete (AAR=$test_aar)"
fi

if [ "$RUN_TEST" = 1 ]; then
    status "checkpoint sweep complete (validation selection and one matched test evaluation)"
else
    status "validation checkpoint sweep complete (test not evaluated)"
fi
