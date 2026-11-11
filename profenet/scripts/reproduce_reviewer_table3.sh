#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="python3"
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

if [[ $# -ne 1 ]]; then
    echo "Usage: PYTHON=/path/to/python $0 EMPTY_OUTPUT_DIRECTORY" >&2
    exit 2
fi

OUTDIR="$1"
if [[ -e "$OUTDIR" ]] && [[ -n "$(find "$OUTDIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty output directory: $OUTDIR" >&2
    exit 2
fi
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

# Recorded as ROOT-relative paths so regenerated reports are byte-identical
# no matter where the repository or handoff package is checked out.
PROFE="outputs/table3_runs/table3-profe_enhanced-rde3_2026_08_07__19_32_40_profenet_v3_96k_full/outer_predictions.csv"
RDE="outputs/table3_runs/table3-rde_profe-rde3_2026_08_02__21_22_39_best191k_full_bs16_acc2/outer_predictions.csv"
ENSEMBLE="outputs/table3_runs/ensemble_profenet_v3_96k_rde_fixed_equal_locked/outer_predictions.csv"

for path in "$PROFE" "$RDE" "$ENSEMBLE"; do
    if [[ ! -f "$ROOT/$path" ]]; then
        echo "Required locked prediction file is missing: $ROOT/$path" >&2
        exit 1
    fi
done
if ! PYTHON_PATH="$(command -v "$PYTHON")"; then
    echo "Python executable is unavailable: $PYTHON" >&2
    exit 1
fi
PYTHON="$PYTHON_PATH"

cd "$ROOT"
"$PYTHON" -m pytest -q \
    tests/test_table3_uncertainty.py \
    tests/test_skempi_metrics.py \
    tests/test_table3_ensemble.py \
    tests/test_table3_gate.py \
    | tee "$OUTDIR/test_results.txt"

"$PYTHON" summarize_table3_uncertainty.py \
    --predictions "$PROFE" \
    --reference "$RDE" \
    --mode single \
    --bootstrap-replicates 2000 \
    --seed 2022 \
    --outdir "$OUTDIR/table3_uncertainty_profenet_v3_vs_rde"

"$PYTHON" summarize_table3_uncertainty.py \
    --predictions "$ENSEMBLE" \
    --reference "$RDE" \
    --mode single \
    --bootstrap-replicates 2000 \
    --seed 2022 \
    --outdir "$OUTDIR/table3_uncertainty_fixed_ensemble_vs_rde"

for report in table3_uncertainty_profenet_v3_vs_rde table3_uncertainty_fixed_ensemble_vs_rde; do
    for artifact in fold_metrics.csv fold_variation.csv uncertainty.json uncertainty.md; do
        cmp \
            "$ROOT/outputs/reviewer_revision/$report/$artifact" \
            "$OUTDIR/$report/$artifact"
    done
done

"$PYTHON" scripts/build_reviewer_manifest.py \
    --output "$OUTDIR/reproducibility_manifest.json"

(
    cd "$OUTDIR"
    find . -type f ! -name REPORTS.sha256 -print0 \
        | sort -z \
        | xargs -0 sha256sum
) > "$OUTDIR/REPORTS.sha256"

echo "Reviewer Table 3 reproduction completed: $OUTDIR"
