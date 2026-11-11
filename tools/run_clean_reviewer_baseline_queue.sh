#!/usr/bin/env bash

# Run leakage-clean reviewer baselines after the validation-only LmPro sweep
# releases a full GPU. Each stage has an independent log and completion check.

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-$SCRIPT_ROOT}
GPU_INDEX=${GPU_INDEX:-2}
GPU_MEMORY_LIMIT_MIB=${GPU_MEMORY_LIMIT_MIB:-500}
GPU_UTIL_LIMIT_PERCENT=${GPU_UTIL_LIMIT_PERCENT:-10}
GPU_READY_SAMPLES=${GPU_READY_SAMPLES:-2}
UPSTREAM_STATUS=${UPSTREAM_STATUS:-logs/checkpoint_sweeps/corrected_v2_u33420_nanobody_clustered_fullbptt_seed128_torch26_exactdisjoint/status.log}
UPSTREAM_PATTERN=${UPSTREAM_PATTERN:-validation checkpoint sweep complete (test not evaluated)}
LOG_DIR=${LOG_DIR:-logs/clean_reviewer_baselines}
STATUS_FILE=${STATUS_FILE:-$LOG_DIR/status.log}
REVIEWER_ENV=${REVIEWER_ENV:-$ROOT/../.venvs/reviewer-baselines}
PROTEINMPNN_PYTHON=${PROTEINMPNN_PYTHON:-$REVIEWER_ENV/bin/python}
DIFFAB_PYTHON=${DIFFAB_PYTHON:-$REVIEWER_ENV/bin/python}

PROTEINMPNN_TRAIN_DIR=${PROTEINMPNN_TRAIN_DIR:-baselines/nanobody_clustered/proteinmpnn_clean/training_seed128}
PROTEINMPNN_VALIDATION_DIR=${PROTEINMPNN_VALIDATION_DIR:-baselines/nanobody_clustered/proteinmpnn_clean/validation_pretrain_exact_disjoint_s100}
PROTEINMPNN_FORMAL_DIR=${PROTEINMPNN_FORMAL_DIR:-baselines/nanobody_clustered/proteinmpnn_clean/eval_pretrain_exact_disjoint_s10000}
DIFFAB_TRAIN_DIR=${DIFFAB_TRAIN_DIR:-baselines/diffab/clean_training_seed128}
DIFFAB_EVAL_DIR=${DIFFAB_EVAL_DIR:-baselines/diffab/clean_pretrain_exact_disjoint_seed128_s100}
VALID_JSONL=${VALID_JSONL:-abgnn/finetune/exp2-hern/nanobody_clustered/valid_data.jsonl}
VALID_RECORDS=${VALID_RECORDS:-baselines/nanobody_clustered/proteinmpnn_clean/valid_records_pretrain_exact_disjoint_lmpro_loadable.txt}
TEST_JSONL=${TEST_JSONL:-abgnn/finetune/exp2-hern/nanobody_clustered/test_data.jsonl}
TEST_RECORDS=${TEST_RECORDS:-baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt}

cd "$ROOT"
mkdir -p "$LOG_DIR"
[ -x "$PROTEINMPNN_PYTHON" ] || { echo "missing ProteinMPNN Python: $PROTEINMPNN_PYTHON" >&2; exit 1; }
[ -x "$DIFFAB_PYTHON" ] || { echo "missing DiffAb Python: $DIFFAB_PYTHON" >&2; exit 1; }
[ -f "$VALID_RECORDS" ] || { echo "missing validation records: $VALID_RECORDS" >&2; exit 1; }
[ -f "$TEST_RECORDS" ] || { echo "missing test records: $TEST_RECORDS" >&2; exit 1; }

timestamp() {
    date '+%F %T %Z'
}

status() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"
}

on_exit() {
    rc=$?
    status "clean reviewer baseline queue exit=$rc"
}
trap on_exit EXIT

status "waiting for '$UPSTREAM_PATTERN' in $UPSTREAM_STATUS"
while ! grep -Fq "$UPSTREAM_PATTERN" "$UPSTREAM_STATUS" 2>/dev/null; do
    if grep -Eq 'validation inference exit=[1-9][0-9]*|test inference exit=[1-9][0-9]*|upstream run failed' "$UPSTREAM_STATUS" 2>/dev/null; then
        status "LmPro checkpoint sweep failed; refusing to consume its GPU"
        exit 1
    fi
    sleep 30
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_INDEX}
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}

gpu_memory_used() {
    nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
}

gpu_utilization() {
    nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
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
    sleep 30
done

if [ -f "$PROTEINMPNN_TRAIN_DIR/best.pt" ] && grep -Fq 'ProteinMPNN clean training complete' "$STATUS_FILE" 2>/dev/null; then
    status "ProteinMPNN clean training already complete"
else
    status "ProteinMPNN clean training start"
    "$PROTEINMPNN_PYTHON" -u tools/train_proteinmpnn_cdr3.py \
        --train-input baselines/nanobody_clustered/proteinmpnn_clean/train_input \
        --valid-input baselines/nanobody_clustered/proteinmpnn_clean/valid_input \
        --output-dir "$PROTEINMPNN_TRAIN_DIR" \
        --device cuda \
        --seed 128 \
        --epochs 150 \
        --patience 30 \
        --batch-size 16 \
        --optimizer noam \
        --warmup-steps 4000 \
        --lr-factor 2 \
        --gradient-clip 0 \
        --cpu-threads 8 \
        --amp \
        > "$LOG_DIR/proteinmpnn_train.log" 2>&1
    status "ProteinMPNN clean training complete"
fi

if [ -f "$PROTEINMPNN_VALIDATION_DIR/metrics.json" ]; then
    status "ProteinMPNN 100-sample validation already complete"
else
    status "ProteinMPNN 100-sample validation start"
    env \
        PYTHON="$PROTEINMPNN_PYTHON" \
        SOURCE_JSONL="$VALID_JSONL" \
        RESTRICT_RECORDS="$VALID_RECORDS" \
        OUTPUT_DIR="$PROTEINMPNN_VALIDATION_DIR" \
        MODEL_WEIGHTS_DIR="$PROTEINMPNN_TRAIN_DIR/" \
        MODEL_NAME=best \
        NUM_SEQUENCES=100 \
        BATCH_SIZE=20 \
        TOPKS='1 10 100' \
        SEED=128 \
        bash tools/run_proteinmpnn_benchmark.sh \
        > "$LOG_DIR/proteinmpnn_validation_s100.log" 2>&1
    status "ProteinMPNN 100-sample validation complete"
fi

if [ -f "$DIFFAB_TRAIN_DIR/best.pt" ] && grep -Fq 'DiffAb clean training complete' "$STATUS_FILE" 2>/dev/null; then
    status "DiffAb clean training already complete"
else
    status "DiffAb clean training start"
    "$DIFFAB_PYTHON" -u tools/train_diffab_cdr3.py \
        --train-jsonl abgnn/finetune/exp2-hern/nanobody_clustered/train_data.jsonl \
        --train-records baselines/nanobody_clustered/proteinmpnn_clean/train_records_lmpro_loadable.txt \
        --valid-jsonl abgnn/finetune/exp2-hern/nanobody_clustered/valid_data.jsonl \
        --valid-records baselines/nanobody_clustered/proteinmpnn_clean/valid_records_pretrain_exact_disjoint_lmpro_loadable.txt \
        --architecture-checkpoint third_party/DiffAb/trained_models/fixbb.pt \
        --output-dir "$DIFFAB_TRAIN_DIR" \
        --device cuda \
        --seed 128 \
        --epochs 300 \
        --patience 40 \
        --batch-size 8 \
        --amp \
        > "$LOG_DIR/diffab_train.log" 2>&1
    status "DiffAb clean training complete"
fi

if [ -f "$DIFFAB_EVAL_DIR/metrics.json" ]; then
    status "DiffAb 100-sample evaluation already complete"
else
    status "DiffAb 100-sample evaluation start"
    "$DIFFAB_PYTHON" -u tools/run_diffab_benchmark.py \
        --source-jsonl "$TEST_JSONL" \
        --restrict-records "$TEST_RECORDS" \
        --checkpoint "$DIFFAB_TRAIN_DIR/best.pt" \
        --output-dir "$DIFFAB_EVAL_DIR" \
        --device cuda \
        --num-samples 100 \
        --batch-size 16 \
        --seed 128 \
        --resume \
        > "$LOG_DIR/diffab_eval_s100.log" 2>&1
    status "DiffAb 100-sample evaluation complete"
fi

# Match LmPro-Nano's formal selection budget: rank the top 100 of 10,000
# ProteinMPNN samples for every strict test record.
if [ -f "$PROTEINMPNN_FORMAL_DIR/metrics.json" ]; then
    status "ProteinMPNN 10000-sample evaluation already complete"
else
    status "ProteinMPNN 10000-sample evaluation start"
    env \
        PYTHON="$PROTEINMPNN_PYTHON" \
        SOURCE_JSONL="$TEST_JSONL" \
        RESTRICT_RECORDS="$TEST_RECORDS" \
        OUTPUT_DIR="$PROTEINMPNN_FORMAL_DIR" \
        MODEL_WEIGHTS_DIR="$PROTEINMPNN_TRAIN_DIR/" \
        MODEL_NAME=best \
        NUM_SEQUENCES=10000 \
        BATCH_SIZE=50 \
        TOPKS='1 10 100 1000' \
        SEED=128 \
        bash tools/run_proteinmpnn_benchmark.sh \
        > "$LOG_DIR/proteinmpnn_eval_s10000.log" 2>&1
    status "ProteinMPNN 10000-sample evaluation complete"
fi

status "clean reviewer baseline queue complete"
