#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 EMPTY_HANDOFF_DIRECTORY" >&2
    exit 2
fi

DEST="$1"
if [[ -e "$DEST" ]] && [[ -n "$(find "$DEST" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty handoff directory: $DEST" >&2
    exit 2
fi
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"

copy_path() {
    local relative="$1"
    if [[ ! -e "$ROOT/$relative" ]]; then
        echo "Required handoff artifact is missing: $ROOT/$relative" >&2
        exit 1
    fi
    mkdir -p "$DEST/$(dirname "$relative")"
    cp -a "$ROOT/$relative" "$DEST/$relative"
}

for relative in \
    README.md \
    LICENSE \
    requirements-reviewer.txt \
    env.yml \
    start.pdf \
    "Reviewer 1 2 4.pdf" \
    "Reviewer 3.pdf" \
    summarize_table3_uncertainty.py \
    ensemble_table3_predictions.py \
    check_table3_metrics.py \
    screen_inner_dev.py \
    train_table3.py; do
    copy_path "$relative"
done

for relative in docs scripts rde tests configs/train outputs/reviewer_revision \
    data/generated/pdbredo_skempi50_disjoint; do
    copy_path "$relative"
done

PROFE_RUN="outputs/table3_runs/table3-profe_enhanced-rde3_2026_08_07__19_32_40_profenet_v3_96k_full"
RDE_RUN="outputs/table3_runs/table3-rde_profe-rde3_2026_08_02__21_22_39_best191k_full_bs16_acc2"
ENSEMBLE_RUN="outputs/table3_runs/ensemble_profenet_v3_96k_rde_fixed_equal_locked"

for run in "$PROFE_RUN" "$RDE_RUN"; do
    for file in outer_predictions.csv run_manifest.json selection.lock.json \
        split_manifest.json table3_metrics.csv table3_metrics_per_fold.csv; do
        copy_path "$run/$file"
    done
done
copy_path "$PROFE_RUN/gate_check_single.json"
copy_path "$RDE_RUN/table3_acceptance.json"
copy_path "$ENSEMBLE_RUN"

mkdir -p "$DEST/external_audit/md_protocol"
MD_SOURCE="$ROOT/../gjxscript"
for file in step4.0_minimization.mdp step4.1_equilibration.mdp \
    step5_production.mdp submit_MD.sh; do
    if [[ -f "$MD_SOURCE/$file" ]]; then
        cp -a "$MD_SOURCE/$file" "$DEST/external_audit/md_protocol/$file"
    fi
done

cp "$ROOT/README.md" "$DEST/HANDOFF.md"

(
    cd "$DEST"
    find . -type f ! -name BUNDLE_MANIFEST.sha256 -print0 \
        | sort -z \
        | xargs -0 sha256sum
) > "$DEST/BUNDLE_MANIFEST.sha256"

echo "Reviewer handoff created: $DEST"
