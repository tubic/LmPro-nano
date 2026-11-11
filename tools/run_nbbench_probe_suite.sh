#!/usr/bin/env bash

# Train resumable NbBench probe runs from completed Lm-Nano exports.

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-$SCRIPT_ROOT}
PYTHON=${PYTHON:-python}
EMBEDDING_DIR=${EMBEDDING_DIR:-baselines/nbbench/lmnano_corrected_u33000}
OUTPUT_DIR=${OUTPUT_DIR:-baselines/nbbench/results/corrected_u33000}
LOG_DIR=${LOG_DIR:-logs/nbbench_probes/corrected_u33000}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-0}
CPU_THREADS=${CPU_THREADS:-8}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-2}
HEAD_STYLE=${HEAD_STYLE:-nbbench}
FEATURE_STORAGE=${FEATURE_STORAGE:-memory}
DATASETS=${DATASETS:-"VRClassification CDRInfilling thermo-seq thermo-tm vhh_affinity-seq vhh_affinity-score polyreaction nanobody_type SARS-CoV-2 hIL6 Paratope"}
SEEDS=${SEEDS:-42}
OVERLAP_POLICIES=${OVERLAP_POLICIES:-keep}
WAIT_FOR_STATUS=${WAIT_FOR_STATUS:-}
WAIT_FOR_PATTERN=${WAIT_FOR_PATTERN:-NbBench export suite complete}
WAIT_FOR_EXPORTS=${WAIT_FOR_EXPORTS:-0}

cd "$ROOT"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

timestamp() {
    date '+%F %T %Z'
}

status() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOG_DIR/status.log"
}

if [ -n "$WAIT_FOR_STATUS" ]; then
    status "waiting for '$WAIT_FOR_PATTERN' in $WAIT_FOR_STATUS"
    while ! grep -Fq "$WAIT_FOR_PATTERN" "$WAIT_FOR_STATUS" 2>/dev/null; do
        sleep 30
    done
fi

for dataset in $DATASETS; do
    export_metadata="$EMBEDDING_DIR/$dataset/metadata.json"
    if [ "$WAIT_FOR_EXPORTS" = 1 ]; then
        status "$dataset waiting for export metadata"
        while [ ! -f "$export_metadata" ]; do
            sleep 30
        done
    fi
    [ -f "$export_metadata" ] || {
        status "$dataset export is missing: $export_metadata"
        exit 1
    }
    for policy in $OVERLAP_POLICIES; do
        for seed in $SEEDS; do
            run_name="${HEAD_STYLE}_${policy}_seed${seed}"
            result="$OUTPUT_DIR/$dataset/$run_name/results.json"
            log="$LOG_DIR/${dataset}_${run_name}.log"
            if [ -f "$result" ]; then
                status "$dataset $run_name already complete"
                continue
            fi
            status "$dataset $run_name probe start (device=$DEVICE)"
            "$PYTHON" -u tools/train_nbbench_probe.py \
                --dataset "$dataset" \
                --embedding-dir "$EMBEDDING_DIR" \
                --output-dir "$OUTPUT_DIR" \
                --overlap-policy "$policy" \
                --head-style "$HEAD_STYLE" \
                --feature-storage "$FEATURE_STORAGE" \
                --batch-size "$BATCH_SIZE" \
                --cpu-threads "$CPU_THREADS" \
                --gradient-accumulation "$GRADIENT_ACCUMULATION" \
                --seed "$seed" \
                --device "$DEVICE" \
                --overwrite \
                > "$log" 2>&1
            status "$dataset $run_name probe complete"
        done
    done
done

status "NbBench probe suite complete"
