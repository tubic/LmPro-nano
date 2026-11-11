#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON=${PYTHON:-/home/af3_bosun/miniforge3/envs/lmpro-nano/bin/python}
SPLIT_DIR=${SPLIT_DIR:-abgnn/finetune/exp2-hern/nanobody_clustered}
BASELINE_DIR=${BASELINE_DIR:-baselines/nanobody_clustered}
OUTPUT_DIR=${OUTPUT_DIR:-$BASELINE_DIR/proteinmpnn_clean}
CDR_BACKBONE=${CDR_BACKBONE:-native}

train_records="$OUTPUT_DIR/train_records_lmpro_loadable.txt"
valid_records="$OUTPUT_DIR/valid_records_pretrain_exact_disjoint_lmpro_loadable.txt"

env -u LD_LIBRARY_PATH "$PYTHON" tools/make_lmpro_loadable_subset.py \
    --split-dir "$SPLIT_DIR" \
    --split train \
    --output "$train_records"

env -u LD_LIBRARY_PATH "$PYTHON" tools/make_lmpro_loadable_subset.py \
    --split-dir "$SPLIT_DIR" \
    --split valid \
    --input-records "$BASELINE_DIR/valid_records_pretrain_exact_disjoint.txt" \
    --output "$valid_records"

"$PYTHON" tools/export_proteinmpnn_benchmark.py \
    --source-jsonl "$SPLIT_DIR/train_data.jsonl" \
    --restrict-records "$train_records" \
    --output-dir "$OUTPUT_DIR/train_input" \
    --epitope-size 20 \
    --cdr-backbone "$CDR_BACKBONE"

"$PYTHON" tools/export_proteinmpnn_benchmark.py \
    --source-jsonl "$SPLIT_DIR/valid_data.jsonl" \
    --restrict-records "$valid_records" \
    --output-dir "$OUTPUT_DIR/valid_input" \
    --epitope-size 20 \
    --cdr-backbone "$CDR_BACKBONE"

echo "prepared leakage-clean ProteinMPNN train/valid inputs under $OUTPUT_DIR"
