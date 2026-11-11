#!/usr/bin/env bash

# NbBench follow-on chain: after the GPU-2 ablation queue releases the card,
# export frozen Lm-Nano representations for all restored NbBench datasets with
# the validation-selected u33000 encoder, then train keep/exclude overlap
# probes from those exports. Both suites are resumable through their own
# metadata/result files.

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-$SCRIPT_ROOT}
GPU_INDEX=${GPU_INDEX:-2}
UPSTREAM_STATUS=${UPSTREAM_STATUS:-logs/ablation_queue_gpu2/status.log}
UPSTREAM_PATTERN=${UPSTREAM_PATTERN:-ablation queue (GPU 2) complete}
LOG_DIR=${LOG_DIR:-logs/nbbench_followup}
STATUS_FILE=${STATUS_FILE:-$LOG_DIR/status.log}
PYTHON=${PYTHON:-/path/to/conda/envs/lmpro-nano-torch26-cu124/bin/python}
OVERLAP_POLICIES=${OVERLAP_POLICIES:-"keep exclude-all"}
PROBE_DEVICE=${PROBE_DEVICE:-cuda}
SEEDS=${SEEDS:-42}

cd "$ROOT"
mkdir -p "$LOG_DIR"
[ -x "$PYTHON" ] || { echo "missing Python: $PYTHON" >&2; exit 1; }

timestamp() {
    date '+%F %T %Z'
}

status() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"
}

on_exit() {
    rc=$?
    status "NbBench follow-up queue exit=$rc"
}
trap on_exit EXIT

status "waiting for '$UPSTREAM_PATTERN' in $UPSTREAM_STATUS"
exit_lines_at_start=$(grep -cE 'queue exit=[0-9]+' "$UPSTREAM_STATUS" 2>/dev/null || true)
while ! grep -Fq "$UPSTREAM_PATTERN" "$UPSTREAM_STATUS" 2>/dev/null; do
    last_exit=$(grep -E 'queue exit=[0-9]+' "$UPSTREAM_STATUS" 2>/dev/null \
        | tail -n +$((exit_lines_at_start + 1)) | tail -1 \
        | grep -oE 'exit=[0-9]+' | cut -d= -f2 || true)
    if [ -n "${last_exit:-}" ] && [ "$last_exit" != "0" ]; then
        status "upstream queue failed (exit=$last_exit); refusing to start"
        exit 1
    fi
    sleep 60
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_INDEX}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset LD_LIBRARY_PATH || true
export MKL_THREADING_LAYER=GNU

status "NbBench embedding export suite start"
env PYTHON="$PYTHON" DEVICE=cuda \
    bash tools/run_nbbench_export_suite.sh \
    > "$LOG_DIR/export_suite.log" 2>&1
status "NbBench embedding export suite complete"

status "NbBench probe suite start (policies: $OVERLAP_POLICIES, device: $PROBE_DEVICE)"
env PYTHON="$PYTHON" DEVICE="$PROBE_DEVICE" OVERLAP_POLICIES="$OVERLAP_POLICIES" SEEDS="$SEEDS" \
    bash tools/run_nbbench_probe_suite.sh \
    > "$LOG_DIR/probe_suite.log" 2>&1
status "NbBench probe suite complete"

status "NbBench follow-up queue complete"
