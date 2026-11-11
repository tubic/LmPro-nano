#!/usr/bin/env bash

# Run the full-corpus MMseqs2 audit after the active CPU-heavy NbBench jobs.

set -euo pipefail

ROOT=${ROOT:-.}
PYTHON=${PYTHON:-/home/af3_bosun/miniforge3/envs/lmpro-nano/bin/python}
LOG_DIR=${LOG_DIR:-logs/pretrain_homology_audit}
STATUS_FILE=${STATUS_FILE:-$LOG_DIR/status.log}
OUTPUT_DIR=${OUTPUT_DIR:-baselines/nanobody_clustered/pretrain_homology_audit}
THREADS=${THREADS:-16}

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
    status "pretraining homology audit queue exit=$rc"
}
trap on_exit EXIT

wait_for() {
    local file=$1
    local pattern=$2
    status "waiting for '$pattern' in $file"
    while ! grep -Fq "$pattern" "$file" 2>/dev/null; do
        sleep 30
    done
}

wait_for logs/nbbench_exports_full_cpu/status.log 'NbBench export suite complete'
wait_for logs/nbbench_probes_linear_cpu/status.log 'NbBench probe suite complete'
wait_for logs/nbbench_probes_strict_cpu/status.log 'NbBench probe suite complete'
wait_for logs/nbbench_probes_paratope_cpu/status.log 'NbBench probe suite complete'
wait_for logs/nbbench_exports_interaction_mean_cpu/status.log 'NbBench export suite complete'
wait_for logs/nbbench_probes_interaction_mean_cpu/status.log 'NbBench probe suite complete'

if [ -f "$OUTPUT_DIR/summary.json" ] && \
   grep -Fq '"max_pretrain_spans": 0' "$OUTPUT_DIR/summary.json"; then
    status "full pretraining homology audit already complete"
else
    status "full pretraining homology audit start"
    "$PYTHON" -u tools/audit_pretrain_homology.py \
        --output-dir "$OUTPUT_DIR" \
        --threads "$THREADS" \
        --minimum-test-hit-records 13 \
        --minimum-valid-hit-records 11 \
        > "$LOG_DIR/audit.log" 2>&1
    status "full pretraining homology audit complete"
fi

status "pretraining homology audit queue complete"
