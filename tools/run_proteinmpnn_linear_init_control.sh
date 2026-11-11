#!/usr/bin/env bash

# Train and evaluate ProteinMPNN with LmPro-Nano's linear CDR-H3 coordinate initialization.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-$ROOT/../.venvs/reviewer-baselines/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-baselines/nanobody_clustered/proteinmpnn_linear_init}
LOG_DIR=${LOG_DIR:-logs/proteinmpnn_linear_init}
TRAIN_DIR=$OUTPUT_ROOT/training_seed128
VALIDATION_DIR=$OUTPUT_ROOT/validation_pretrain_exact_disjoint_s100
FORMAL_DIR=$OUTPUT_ROOT/eval_pretrain_exact_disjoint_s10000
TEST_RECORDS=baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-2}

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
[ -x "$PYTHON" ] || { echo "missing Python environment: $PYTHON" >&2; exit 1; }

timestamp() { date '+%F %T %Z'; }
status() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOG_DIR/status.log"; }
on_exit() { rc=$?; status "ProteinMPNN linear-init control exit=$rc"; }
trap on_exit EXIT

export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

if [ ! -f "$OUTPUT_ROOT/train_input/manifest.tsv" ] || [ ! -f "$OUTPUT_ROOT/valid_input/manifest.tsv" ]; then
    status "prepare linear-init train/validation inputs"
    env PYTHON="$PYTHON" OUTPUT_DIR="$OUTPUT_ROOT" CDR_BACKBONE=linear_init \
        bash tools/prepare_proteinmpnn_clean_inputs.sh > "$LOG_DIR/prepare.log" 2>&1
fi

if [ ! -f "$TRAIN_DIR/best.pt" ]; then
    status "train linear-init ProteinMPNN on GPU $GPU_A"
    CUDA_VISIBLE_DEVICES=$GPU_A env -u LD_LIBRARY_PATH "$PYTHON" -u \
        tools/train_proteinmpnn_cdr3.py \
        --train-input "$OUTPUT_ROOT/train_input" \
        --valid-input "$OUTPUT_ROOT/valid_input" \
        --output-dir "$TRAIN_DIR" \
        --device cuda --seed 128 --epochs 150 --patience 30 --batch-size 16 \
        --optimizer noam --warmup-steps 4000 --lr-factor 2 --gradient-clip 0 \
        --cpu-threads 8 --amp > "$LOG_DIR/train.log" 2>&1
fi

if [ ! -f "$VALIDATION_DIR/metrics.json" ]; then
    status "run 100-sample linear-init validation on GPU $GPU_A"
    CUDA_VISIBLE_DEVICES=$GPU_A env -u LD_LIBRARY_PATH \
        PYTHON="$PYTHON" SOURCE_JSONL=abgnn/finetune/exp2-hern/nanobody_clustered/valid_data.jsonl \
        RESTRICT_RECORDS="$OUTPUT_ROOT/valid_records_pretrain_exact_disjoint_lmpro_loadable.txt" \
        OUTPUT_DIR="$VALIDATION_DIR" MODEL_WEIGHTS_DIR="$TRAIN_DIR/" MODEL_NAME=best \
        NUM_SEQUENCES=100 BATCH_SIZE=20 TOPKS='1 10 100' SEED=128 \
        CDR_BACKBONE=linear_init bash tools/run_proteinmpnn_benchmark.sh \
        > "$LOG_DIR/validation_s100.log" 2>&1
fi

"$PYTHON" tools/shard_record_list.py --input "$TEST_RECORDS" \
    --output-dir "$OUTPUT_ROOT/formal_record_shards" --num-shards 2 \
    > "$LOG_DIR/shard_records.log"

run_shard() {
    local shard_index=$1 gpu_index=$2
    local shard_name
    shard_name=$(printf 'shard_%02d' "$shard_index")
    local shard_dir=$FORMAL_DIR/$shard_name
    if [ -f "$shard_dir/metrics.json" ]; then
        status "$shard_name already complete"
        return
    fi
    status "run $shard_name on GPU $gpu_index"
    CUDA_VISIBLE_DEVICES=$gpu_index env -u LD_LIBRARY_PATH \
        PYTHON="$PYTHON" SOURCE_JSONL=abgnn/finetune/exp2-hern/nanobody_clustered/test_data.jsonl \
        RESTRICT_RECORDS="$OUTPUT_ROOT/formal_record_shards/$shard_name.txt" \
        OUTPUT_DIR="$shard_dir" MODEL_WEIGHTS_DIR="$TRAIN_DIR/" MODEL_NAME=best \
        NUM_SEQUENCES=10000 BATCH_SIZE=50 TOPKS='1 10 100 1000' SEED=128 \
        CDR_BACKBONE=linear_init bash tools/run_proteinmpnn_benchmark.sh \
        > "$LOG_DIR/${shard_name}_s10000.log" 2>&1
    status "$shard_name complete"
}

run_shard 0 "$GPU_A" &
pid_a=$!
run_shard 1 "$GPU_B" &
pid_b=$!
shard_rc=0
wait "$pid_a" || shard_rc=$?
wait "$pid_b" || shard_rc=$?
if [ "$shard_rc" -ne 0 ]; then
    status "one or more formal shards failed"
    exit "$shard_rc"
fi

"$PYTHON" tools/merge_proteinmpnn_metrics.py \
    --inputs "$FORMAL_DIR/shard_00/metrics.json" "$FORMAL_DIR/shard_01/metrics.json" \
    --output "$FORMAL_DIR/metrics.json" \
    --per-record-inputs "$FORMAL_DIR/shard_00/per_record.tsv" "$FORMAL_DIR/shard_01/per_record.tsv" \
    --per-record-output "$FORMAL_DIR/per_record.tsv" | tee "$LOG_DIR/merged_metrics.log"
"$PYTHON" tools/paired_aar_bootstrap.py \
    --lmpro-tsv logs/checkpoint_sweeps/corrected_v2_u33000_nanobody_clustered_fullbptt_seed128_torch26_exactdisjoint/per_record.tsv \
    --proteinmpnn-tsv "$FORMAL_DIR/per_record.tsv" \
    --topk 100 --resamples 10000 --seed 128 \
    --json-out "$FORMAL_DIR/paired_bootstrap_top100.json" \
    | tee "$LOG_DIR/paired_bootstrap_top100.log"
status "ProteinMPNN linear-init control complete"
