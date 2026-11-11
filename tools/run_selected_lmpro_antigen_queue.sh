#!/usr/bin/env bash

# Finalize the clustered-split candidate on validation only, evaluate its strict
# test subset once, then run the antigen-disjoint generalization experiment.

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-$SCRIPT_ROOT}
GPU_INDEX=${GPU_INDEX:-0}
GPU_MEMORY_LIMIT_MIB=${GPU_MEMORY_LIMIT_MIB:-500}
GPU_UTIL_LIMIT_PERCENT=${GPU_UTIL_LIMIT_PERCENT:-10}
GPU_READY_SAMPLES=${GPU_READY_SAMPLES:-2}
LMPRO_ENV_BIN=${LMPRO_ENV_BIN:-/path/to/conda/envs/lmpro-nano-torch26-cu124/bin}
PYTHON=${PYTHON:-$LMPRO_ENV_BIN/python}

SWEEP_ROOT=${SWEEP_ROOT:-logs/checkpoint_sweeps}
U33000_SWEEP=${U33000_SWEEP:-$SWEEP_ROOT/corrected_v2_u33000_nanobody_clustered_fullbptt_seed128_torch26_exactdisjoint}
U33420_SWEEP=${U33420_SWEEP:-$SWEEP_ROOT/corrected_v2_u33420_nanobody_clustered_fullbptt_seed128_torch26_exactdisjoint}
SWEEP_COMPLETE_PATTERN=${SWEEP_COMPLETE_PATTERN:-validation checkpoint sweep complete (test not evaluated)}
PRETRAIN_ROOT=${PRETRAIN_ROOT:-checkpoints/pretrained/corrected_v2_eos_cdr3span0p5_contextkeep_seed1_u100k_eb4032/candidates}

LOG_DIR=${LOG_DIR:-logs/selected_lmpro_antigen_queue}
STATUS_FILE=${STATUS_FILE:-$LOG_DIR/status.log}
SELECTION_JSON=${SELECTION_JSON:-$LOG_DIR/selected_candidate.json}
CLUSTERED_TEST_LOG=${CLUSTERED_TEST_LOG:-$LOG_DIR/clustered_strict_test.log}
CLUSTERED_DATA=${CLUSTERED_DATA:-abgnn/finetune/exp2-hern/nanobody_clustered}
CLUSTERED_TEST_RECORDS=${CLUSTERED_TEST_RECORDS:-baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt}

ANTIGEN_SPLIT=${ANTIGEN_SPLIT:-nanobody_antigen_disjoint_v2}
ANTIGEN_DATA=${ANTIGEN_DATA:-abgnn/finetune/exp2-hern/$ANTIGEN_SPLIT}
ANTIGEN_VALID_RECORDS=${ANTIGEN_VALID_RECORDS:-baselines/$ANTIGEN_SPLIT/valid_records_pretrain_exact_disjoint.txt}
ANTIGEN_TEST_RECORDS=${ANTIGEN_TEST_RECORDS:-baselines/$ANTIGEN_SPLIT/test_records_pretrain_exact_disjoint.txt}

SEED=${SEED:-128}
NUM_DECODE=${NUM_DECODE:-10000}
TOPK=${TOPK:-100}
REPORT_TOPKS=${REPORT_TOPKS:-"1 10 100 1000"}

cd "$ROOT"
mkdir -p "$LOG_DIR"
[ -x "$PYTHON" ] || { echo "missing LmPro Python: $PYTHON" >&2; exit 1; }
for required in \
    "$CLUSTERED_TEST_RECORDS" \
    "$ANTIGEN_VALID_RECORDS" \
    "$ANTIGEN_TEST_RECORDS" \
    "$PRETRAIN_ROOT/checkpoint_u33000.pt" \
    "$PRETRAIN_ROOT/checkpoint_u33420.pt"; do
    [ -f "$required" ] || { echo "missing required input: $required" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_INDEX}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}
export PATH="$LMPRO_ENV_BIN:$PATH"
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU

timestamp() {
    date '+%F %T %Z'
}

status() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"
}

on_exit() {
    rc=$?
    status "selected LmPro/antigen queue exit=$rc"
}
trap on_exit EXIT

wait_for_sweep() {
    sweep_dir=$1
    status_file="$sweep_dir/status.log"
    status "waiting for '$SWEEP_COMPLETE_PATTERN' in $status_file"
    while ! grep -Fq "$SWEEP_COMPLETE_PATTERN" "$status_file" 2>/dev/null; do
        if grep -Eq 'validation inference exit=[1-9][0-9]*|missing final metrics|could not select' "$status_file" 2>/dev/null; then
            status "validation sweep failed: $status_file"
            exit 1
        fi
        sleep 30
    done
}

gpu_memory_used() {
    nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
}

gpu_utilization() {
    nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
}

wait_for_gpu() {
    ready_samples=0
    while true; do
        used=$(gpu_memory_used || true)
        util=$(gpu_utilization || true)
        if [[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] \
            && [ "$used" -lt "$GPU_MEMORY_LIMIT_MIB" ] \
            && [ "$util" -lt "$GPU_UTIL_LIMIT_PERCENT" ]; then
            ready_samples=$((ready_samples + 1))
            if [ "$ready_samples" -ge "$GPU_READY_SAMPLES" ]; then
                return
            fi
        else
            ready_samples=0
        fi
        status "waiting for GPU readiness: memory < ${GPU_MEMORY_LIMIT_MIB} MiB, util < ${GPU_UTIL_LIMIT_PERCENT}% (used=${used:-unknown}, util=${util:-unknown}, ready=${ready_samples}/${GPU_READY_SAMPLES})"
        sleep 30
    done
}

wait_for_sweep "$U33000_SWEEP"
wait_for_sweep "$U33420_SWEEP"

status "validation-only cross-candidate selection start"
"$PYTHON" tools/select_finetune_candidate.py \
    --candidate u33000 "$U33000_SWEEP/selected_checkpoint.txt" "$PRETRAIN_ROOT/checkpoint_u33000.pt" \
    --candidate u33420 "$U33420_SWEEP/selected_checkpoint.txt" "$PRETRAIN_ROOT/checkpoint_u33420.pt" \
    --output "$SELECTION_JSON" \
    >> "$LOG_DIR/selection.log" 2>&1

mapfile -t selected_fields < <(
    "$PYTHON" - "$SELECTION_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    selected = json.load(handle)["selected"]
print(selected["label"])
print(selected["finetuning_checkpoint"])
print(selected["pretraining_checkpoint"])
print(selected["finetuning_epoch"])
print(selected["validation_aar"])
PY
)
[ "${#selected_fields[@]}" -eq 5 ] || { status "could not read selected candidate"; exit 1; }
selected_label=${selected_fields[0]}
selected_finetuning_checkpoint=${selected_fields[1]}
selected_pretraining_checkpoint=${selected_fields[2]}
selected_finetuning_epoch=${selected_fields[3]}
selected_validation_aar=${selected_fields[4]}
status "selected $selected_label fine-tuning epoch $selected_finetuning_epoch (validation AAR=$selected_validation_aar)"

wait_for_gpu
read -r -a report_topks <<< "$REPORT_TOPKS"
if grep -q '^Amino acid recovery rate = ' "$CLUSTERED_TEST_LOG" 2>/dev/null; then
    status "clustered strict test already complete"
else
    status "clustered strict test start (single evaluation)"
    "$PYTHON" -u inference.py \
        --cdr_type 3 \
        --cktpath "$selected_finetuning_checkpoint" \
        --data_path "$CLUSTERED_DATA" \
        --split test \
        --restrict_records "$CLUSTERED_TEST_RECORDS" \
        --num_decode "$NUM_DECODE" \
        --topk "$TOPK" \
        --seed "$SEED" \
        --report_topks "${report_topks[@]}" \
        > "$CLUSTERED_TEST_LOG" 2>&1
fi
clustered_test_aar=$(awk -F'= ' '/^Amino acid recovery rate = / {value=$2} END {print value}' "$CLUSTERED_TEST_LOG")
[ -n "$clustered_test_aar" ] || { status "clustered strict test metrics missing"; exit 1; }
status "clustered strict test complete (AAR=$clustered_test_aar)"

antigen_run=corrected_v2_${selected_label}_${ANTIGEN_SPLIT}_fullbptt_seed${SEED}_torch26
antigen_save_dir=checkpoints/exp2_nanobody/$antigen_run
antigen_train_log=$LOG_DIR/${antigen_run}.log
if grep -Fq "antigen-disjoint fine-tuning complete: $antigen_run" "$STATUS_FILE" 2>/dev/null; then
    status "antigen-disjoint fine-tuning already complete: $antigen_run"
else
    status "antigen-disjoint fine-tuning start: $antigen_run"
    env \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        PRETRAINED_FILE="$selected_pretraining_checkpoint" \
        PRETRAIN_LABEL="corrected_v2_${selected_label}" \
        SPLIT="$ANTIGEN_SPLIT" \
        SAVE_DIR="$antigen_save_dir" \
        SEED="$SEED" \
        bash run_finetune_corrected_exp2.sh \
        > "$antigen_train_log" 2>&1
    status "antigen-disjoint fine-tuning complete: $antigen_run"
fi

status "antigen-disjoint validation sweep and single strict test start"
env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    GPU_INDEX="$GPU_INDEX" \
    GPU_MODE=full \
    GPU_READY_SAMPLES=2 \
    SAVE_DIR="$antigen_save_dir" \
    RUN_NAME="$antigen_run" \
    DATA_PATH="$ANTIGEN_DATA" \
    VALID_RESTRICT_RECORDS="$ANTIGEN_VALID_RECORDS" \
    RESTRICT_RECORDS="$ANTIGEN_TEST_RECORDS" \
    OUT_DIR="$SWEEP_ROOT/${antigen_run}_exactdisjoint" \
    SEED="$SEED" \
    NUM_DECODE="$NUM_DECODE" \
    TOPK="$TOPK" \
    REPORT_TOPKS="$REPORT_TOPKS" \
    RUN_TEST=1 \
    bash tools/sweep_finetune_checkpoints.sh \
    >> "$LOG_DIR/${antigen_run}_sweep.log" 2>&1

status "selected LmPro/antigen queue complete"
