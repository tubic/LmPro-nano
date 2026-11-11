#!/usr/bin/env bash

# Run the frozen four-checkpoint IgFold control and evaluate it on the strict
# pretraining-exact-disjoint structure subset. CPU mode is the default so this
# independent baseline can run while both training GPUs are occupied.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-$ROOT/../.venvs/structure-baselines/bin/python}
CPU_ONLY=${CPU_ONLY:-1}
CPU_THREADS=${CPU_THREADS:-16}
LOG_DIR=${LOG_DIR:-logs/structure_baselines/igfold_full_20260806}
STATUS_FILE=${STATUS_FILE:-$LOG_DIR/status.log}
PREDICTION_DIR=${PREDICTION_DIR:-baselines/structure_predictors/igfold_pretrain_exact_disjoint}
EVALUATION_DIR=${EVALUATION_DIR:-baselines/structure_predictors/igfold_pretrain_exact_disjoint_eval}

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
    status "IgFold structure queue exit=$rc"
}
trap on_exit EXIT

[ -x "$PYTHON" ] || { status "missing structure environment: $PYTHON"; exit 1; }

predict_args=(
    --predictor igfold
    --output-dir "$PREDICTION_DIR"
    --num-models 4
)
if [ "$CPU_ONLY" = 1 ]; then
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS="$CPU_THREADS"
    export MKL_NUM_THREADS="$CPU_THREADS"
    predict_args+=(--cpu)
    status "IgFold four-model CPU prediction start (threads=$CPU_THREADS)"
else
    status "IgFold four-model GPU prediction start (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
fi

if grep -Fq '"completed_records": 258' "$PREDICTION_DIR/prediction_summary.json" 2>/dev/null && \
   grep -Fq '"failed_records": 0' "$PREDICTION_DIR/prediction_summary.json" 2>/dev/null; then
    status "IgFold predictions already complete"
else
    "$PYTHON" -u tools/run_nanobody_structure_benchmark.py \
        "${predict_args[@]}" \
        > "$LOG_DIR/predict.log" 2>&1
    status "IgFold four-model prediction complete"
fi

status "IgFold structure evaluation start"
"$PYTHON" -u tools/evaluate_structure_predictions.py \
    --prediction-dir "$PREDICTION_DIR" \
    --output-dir "$EVALUATION_DIR" \
    > "$LOG_DIR/evaluate.log" 2>&1
status "IgFold structure evaluation complete"
status "IgFold structure queue complete"
