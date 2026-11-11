#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON=${PYTHON:-python}
SOURCE_JSONL=${SOURCE_JSONL:-abgnn/finetune/exp2-hern/nanobody_clustered/test_data.jsonl}
RESTRICT_RECORDS=${RESTRICT_RECORDS:-baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt}
OUTPUT_DIR=${OUTPUT_DIR:-baselines/nanobody_clustered/proteinmpnn_pretrain_exact_disjoint}
EPITOPE_SIZE=${EPITOPE_SIZE:-20}
CDR_BACKBONE=${CDR_BACKBONE:-native}
NUM_SEQUENCES=${NUM_SEQUENCES:-100}
BATCH_SIZE=${BATCH_SIZE:-10}
TEMPERATURE=${TEMPERATURE:-0.1}
SEED=${SEED:-128}
LIMIT=${LIMIT:-}
TOPKS=${TOPKS:-"1 10 100"}
MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:-}
MODEL_NAME=${MODEL_NAME:-v_48_020}

if (( NUM_SEQUENCES % BATCH_SIZE != 0 )); then
    echo "NUM_SEQUENCES must be divisible by BATCH_SIZE (ProteinMPNN drops the remainder)" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
export_args=(
    --source-jsonl "$SOURCE_JSONL"
    --restrict-records "$RESTRICT_RECORDS"
    --output-dir "$OUTPUT_DIR/input"
    --epitope-size "$EPITOPE_SIZE"
    --cdr-backbone "$CDR_BACKBONE"
)
if [[ -n "$LIMIT" ]]; then
    export_args+=(--limit "$LIMIT")
fi
"$PYTHON" tools/export_proteinmpnn_benchmark.py "${export_args[@]}"

model_args=(--model_name "$MODEL_NAME")
if [[ -n "$MODEL_WEIGHTS_DIR" ]]; then
    model_args+=(--path_to_model_weights "$MODEL_WEIGHTS_DIR")
fi
"$PYTHON" third_party/ProteinMPNN/protein_mpnn_run.py \
    --jsonl_path "$OUTPUT_DIR/input/parsed_pdbs.jsonl" \
    --chain_id_jsonl "$OUTPUT_DIR/input/assigned_chains.jsonl" \
    --fixed_positions_jsonl "$OUTPUT_DIR/input/fixed_positions.jsonl" \
    --out_folder "$OUTPUT_DIR/output" \
    --num_seq_per_target "$NUM_SEQUENCES" \
    --batch_size "$BATCH_SIZE" \
    --sampling_temp "$TEMPERATURE" \
    --seed "$SEED" \
    --suppress_print 1 \
    "${model_args[@]}"

read -r -a topk_args <<< "$TOPKS"
"$PYTHON" tools/evaluate_proteinmpnn.py \
    --manifest "$OUTPUT_DIR/input/manifest.tsv" \
    --seq-dir "$OUTPUT_DIR/output/seqs" \
    --topk "${topk_args[@]}" \
    --per-record-out "$OUTPUT_DIR/per_record.tsv" \
    --json-out "$OUTPUT_DIR/metrics.json"
