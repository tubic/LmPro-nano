#!/usr/bin/env bash

# Export a resumable set of frozen Lm-Nano NbBench representations.  A dataset
# is complete only after export_nbbench_embeddings.py writes metadata.json.

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-$SCRIPT_ROOT}
PYTHON=${PYTHON:-python}
CHECKPOINT=${CHECKPOINT:-checkpoints/pretrained/corrected_v2_eos_cdr3span0p5_contextkeep_seed1_u100k_eb4032/candidates/checkpoint_u33000.pt}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-32}
CPU_THREADS=${CPU_THREADS:-8}
DTYPE=${DTYPE:-float16}
OVERLAP_POLICY=${OVERLAP_POLICY:-keep}
POOLING=${POOLING:-cls}
OUTPUT_DIR=${OUTPUT_DIR:-baselines/nbbench/lmnano_corrected_u33000}
LOG_DIR=${LOG_DIR:-logs/nbbench_exports/corrected_u33000}
DATASETS=${DATASETS:-"VRClassification CDRInfilling thermo-seq thermo-tm vhh_affinity-seq vhh_affinity-score polyreaction nanobody_type SARS-CoV-2 hIL6 Paratope"}

cd "$ROOT"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}

timestamp() {
    date '+%F %T %Z'
}

status() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOG_DIR/status.log"
}

for dataset in $DATASETS; do
    output="$OUTPUT_DIR/$dataset"
    log="$LOG_DIR/$dataset.log"
    if [ -f "$output/metadata.json" ] && [ -f "$output/representations.npy" ]; then
        status "$dataset already complete"
        continue
    fi
    status "$dataset export start (device=$DEVICE, batch=$BATCH_SIZE)"
    "$PYTHON" -u tools/export_nbbench_embeddings.py \
        --dataset "$dataset" \
        --checkpoint "$CHECKPOINT" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --cpu-threads "$CPU_THREADS" \
        --dtype "$DTYPE" \
        --pooling "$POOLING" \
        --overlap-policy "$OVERLAP_POLICY" \
        --output-dir "$OUTPUT_DIR" \
        --overwrite \
        > "$log" 2>&1
    status "$dataset export complete"
done

status "NbBench export suite complete"
