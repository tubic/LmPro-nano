#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON:-python3}"
pipeline_dir="${1:-}"
processed_dir="${RDE_PROCESSED_DIR:-/tmp/af3_bosun_rde_cache}"
cuda_device="${FORCE_CUDA_DEVICE:-}"
resume_checkpoint="${RDE_RESUME_CHECKPOINT:-}"
lock_file="$repo/outputs/rde_pretrain/full_pretrain.lock"

if [[ -z "$pipeline_dir" ]]; then
    echo "Usage: FORCE_CUDA_DEVICE=<device> $0 <existing-pipeline-dir>" >&2
    exit 2
fi
pipeline_dir="$(readlink -f "$pipeline_dir")"
if [[ ! -d "$pipeline_dir" || ! -s "$pipeline_dir/stages.tsv" ]]; then
    echo "Invalid pipeline directory: $pipeline_dir" >&2
    exit 2
fi
if [[ -z "$cuda_device" ]]; then
    echo "FORCE_CUDA_DEVICE is required for a fallback continuation" >&2
    exit 2
fi

mkdir -p "$repo/outputs/rde_pretrain"
cd "$repo"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another full RDE run holds $lock_file" >&2
    exit 1
fi

stages_file="$pipeline_dir/stages.tsv"
record_stage() {
    printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" |
        tee -a "$stages_file"
}

finish_continuation() {
    status=$?
    if [[ "$status" -eq 0 ]]; then
        record_stage fallback_continuation_complete passed
    else
        record_stage fallback_continuation_failed "exit_status=$status"
    fi
}
trap finish_continuation EXIT

export CUDA_VISIBLE_DEVICES="$cuda_device"
continuation_id="$(date +%Y%m%d_%H%M%S)"
record_stage fallback_continuation_started \
    "cuda_device=$cuda_device processed_dir=$processed_dir continuation_id=$continuation_id"

enhanced_run_dir="$(
    awk -F '\t' '
        $2 == "enhanced_table3_passed" {
            value = $3
            sub(/^run_dir=/, "", value)
        }
        END { print value }
    ' "$stages_file"
)"
enhanced_run_dir="$(readlink -f "$enhanced_run_dir")"
enhanced_metrics="$enhanced_run_dir/table3_metrics.csv"
enhanced_predictions="$enhanced_run_dir/outer_predictions.csv"
if [[ ! -s "$enhanced_metrics" || ! -s "$enhanced_predictions" ]]; then
    record_stage fallback_prerequisite_failed \
        "missing_enhanced_metrics_or_predictions=$enhanced_run_dir"
    exit 1
fi

for cache_file in structures.lmdb keys.pkl sanitized_clusters.pkl; do
    if [[ ! -s "$processed_dir/$cache_file" ]]; then
        record_stage fallback_prerequisite_failed \
            "missing_cache_file=$processed_dir/$cache_file"
        exit 1
    fi
done

cache_hash_manifest="$pipeline_dir/local_cache_${continuation_id}.sha256"
sha256sum \
    "$processed_dir/structures.lmdb" \
    "$processed_dir/keys.pkl" \
    "$processed_dir/sanitized_clusters.pkl" > "$cache_hash_manifest"
record_stage local_cache_locked "manifest=$cache_hash_manifest"

hash_manifest="$pipeline_dir/fallback_resume_inputs_${continuation_id}.sha256"
sha256sum \
    train_rde.py \
    rde/datasets/pdbredo_chain.py \
    rde/utils/transforms/corrupt_chi.py \
    train_table3.py \
    ensemble_table3_predictions.py \
    check_table3_metrics.py \
    "$pipeline_dir/candidate_plan.json" \
    "$pipeline_dir/profe_enhanced_table3.yml" \
    "$pipeline_dir/profe_rde_table3.yml" \
    scripts/resume_table3_fallback.sh > "$hash_manifest"
record_stage fallback_inputs_locked "manifest=$hash_manifest"

pretrain_log="$pipeline_dir/full_pretrain_resume_${continuation_id}.log"
pretrain_args=(
    configs/train/rde.yml
    --device cuda
    --num-workers 8
    --batch-size 32
    --accumulation-steps 2
    --processed-dir "$processed_dir"
    --logdir ./outputs/rde_pretrain
    --tag full_rde_table3_local_cache
    --save-freq 1000
)
if [[ -n "$resume_checkpoint" ]]; then
    resume_checkpoint="$(readlink -f "$resume_checkpoint")"
    if [[ ! -s "$resume_checkpoint" ]]; then
        record_stage pretraining_resume_failed \
            "missing_checkpoint=$resume_checkpoint"
        exit 1
    fi
    pretrain_args+=(--resume "$resume_checkpoint")
    record_stage pretraining_resumed \
        "checkpoint=$resume_checkpoint batch_size=32 accumulation_steps=2 effective_batch_size=64"
else
    record_stage pretraining_restarted \
        "seed=2022 batch_size=32 accumulation_steps=2 effective_batch_size=64 max_iters=200000"
fi

pretrain_status=0
"$python_bin" -u train_rde.py "${pretrain_args[@]}" \
    2>&1 | tee "$pretrain_log" || pretrain_status=$?
if [[ "$pretrain_status" -ne 0 ]]; then
    record_stage pretraining_failed \
        "exit_status=$pretrain_status log=$pretrain_log"
    exit "$pretrain_status"
fi

rde_run_dir="$(
    awk -F= '/^RUN_DIR=/{value=substr($0, index($0, "=") + 1)} END{print value}' \
        "$pretrain_log"
)"
if [[ -z "$rde_run_dir" || ! -d "$rde_run_dir" ]]; then
    record_stage pretraining_failed "missing_or_invalid_RUN_DIR=$rde_run_dir"
    exit 1
fi
rde_run_dir="$(readlink -f "$rde_run_dir")"
best_checkpoint="$rde_run_dir/checkpoints/best.pt"
if [[ ! -s "$best_checkpoint" && -n "$resume_checkpoint" ]]; then
    prior_best="$(dirname "$resume_checkpoint")/best.pt"
    if [[ -s "$prior_best" ]]; then
        best_checkpoint="$prior_best"
    fi
fi
if [[ ! -s "$best_checkpoint" ]]; then
    record_stage pretraining_failed "missing_best_checkpoint=$best_checkpoint"
    exit 1
fi
record_stage pretraining_passed "run_dir=$rde_run_dir"

mkdir -p "$repo/trained_models"
best_checkpoint="$(readlink -f "$best_checkpoint")"
if [[ -e "$repo/trained_models/RDE.pt" || -L "$repo/trained_models/RDE.pt" ]]; then
    if [[ "$(readlink -f "$repo/trained_models/RDE.pt")" != "$best_checkpoint" ]]; then
        record_stage weight_link_failed "trained_models/RDE.pt already targets another file"
        exit 1
    fi
else
    ln -s "$best_checkpoint" "$repo/trained_models/RDE.pt"
fi
sha256sum "$best_checkpoint" > "$pipeline_dir/RDE.pt.sha256"
record_stage weight_locked "checkpoint=$best_checkpoint"

record_stage rde_table3_started \
    "protocol=published_rde_3fold subset_metrics=all_single_multiple"
table3_log="$pipeline_dir/rde_table3_resume_${continuation_id}.log"
table3_status=0
"$python_bin" -u train_table3.py "$pipeline_dir/profe_rde_table3.yml" \
    --device cuda \
    --num-workers 4 \
    --tag full_locked \
    2>&1 | tee "$table3_log" || table3_status=$?
if [[ "$table3_status" -ne 0 ]]; then
    record_stage rde_table3_failed "exit_status=$table3_status"
    exit "$table3_status"
fi

table3_run_dir="$(
    awk -F= '/^RUN_DIR=/{value=substr($0, index($0, "=") + 1)} END{print value}' \
        "$table3_log"
)"
if [[ -z "$table3_run_dir" || ! -d "$table3_run_dir" ]]; then
    record_stage rde_table3_failed "missing_or_invalid_RUN_DIR=$table3_run_dir"
    exit 1
fi
table3_run_dir="$(readlink -f "$table3_run_dir")"
metrics_file="$table3_run_dir/table3_metrics.csv"
if [[ ! -s "$metrics_file" || ! -s "$table3_run_dir/outer_predictions.csv" ]]; then
    record_stage rde_table3_failed "missing_metrics_or_predictions=$table3_run_dir"
    exit 1
fi
record_stage rde_table3_passed "run_dir=$table3_run_dir"

run_pdb5_supplement() {
    record_stage pdb5_supplement_started \
        "model=ProFE-Net batch_size=8 accumulation_steps=4 effective_batch_size=32"
    supplement_log="$pipeline_dir/enhanced_pdb5_resume_${continuation_id}.log"
    supplement_status=0
    "$python_bin" -u train_table3.py "$pipeline_dir/profe_enhanced_table3.yml" \
        --protocol pdb5 \
        --device cuda \
        --num-workers 4 \
        --batch-size 8 \
        --accumulation-steps 4 \
        --tag supplemental_locked \
        2>&1 | tee "$supplement_log" || supplement_status=$?
    if [[ "$supplement_status" -ne 0 ]]; then
        record_stage pdb5_supplement_failed "exit_status=$supplement_status"
        return "$supplement_status"
    fi
    supplement_run_dir="$(
        awk -F= '/^RUN_DIR=/{value=substr($0, index($0, "=") + 1)} END{print value}' \
            "$supplement_log"
    )"
    if [[ -z "$supplement_run_dir" || ! -d "$supplement_run_dir" ]]; then
        record_stage pdb5_supplement_failed \
            "missing_or_invalid_RUN_DIR=$supplement_run_dir"
        return 1
    fi
    supplement_run_dir="$(readlink -f "$supplement_run_dir")"
    if [[ ! -s "$supplement_run_dir/table3_metrics.csv" || \
          ! -s "$supplement_run_dir/table3_metrics_per_fold.csv" ]]; then
        record_stage pdb5_supplement_failed "missing_metrics=$supplement_run_dir"
        return 1
    fi
    printf 'run_dir=%s\n' "$supplement_run_dir" > "$pipeline_dir/pdb5_supplement.env"
    record_stage pdb5_supplement_passed "run_dir=$supplement_run_dir"
}

rde_gate_status=0
"$python_bin" -u check_table3_metrics.py "$metrics_file" \
    --mode single \
    --out "$table3_run_dir/table3_acceptance.json" \
    2>&1 | tee "$pipeline_dir/rde_acceptance_resume_${continuation_id}.log" || \
    rde_gate_status=$?
if [[ "$rde_gate_status" -eq 0 ]]; then
    printf 'candidate=ProFE-RDE-baseline\nrun_dir=%s\n' "$table3_run_dir" \
        > "$pipeline_dir/accepted_candidate.env"
    record_stage acceptance_passed \
        "candidate=ProFE-RDE-baseline report=$table3_run_dir/table3_acceptance.json"
    run_pdb5_supplement
    exit 0
fi
record_stage acceptance_failed \
    "candidate=ProFE-RDE-baseline continuing_to_declared_rank_3"

pipeline_id="${pipeline_dir##*/full_pipeline_}"
ensemble_dir="$repo/outputs/table3_runs/table3-fixed-equal-${pipeline_id}"
record_stage fixed_ensemble_started "weights=0.5,0.5 outdir=$ensemble_dir"
"$python_bin" -u ensemble_table3_predictions.py \
    "$enhanced_predictions" \
    "$table3_run_dir/outer_predictions.csv" \
    --outdir "$ensemble_dir" \
    2>&1 | tee "$pipeline_dir/fixed_ensemble_resume_${continuation_id}.log"

ensemble_gate_status=0
"$python_bin" -u check_table3_metrics.py "$ensemble_dir/table3_metrics.csv" \
    --mode single \
    --out "$ensemble_dir/table3_acceptance.json" \
    2>&1 | tee "$pipeline_dir/ensemble_acceptance_resume_${continuation_id}.log" || \
    ensemble_gate_status=$?
if [[ "$ensemble_gate_status" -eq 0 ]]; then
    printf 'candidate=ProFE-fixed-equal-ensemble\nrun_dir=%s\n' "$ensemble_dir" \
        > "$pipeline_dir/accepted_candidate.env"
    record_stage acceptance_passed \
        "candidate=ProFE-fixed-equal-ensemble report=$ensemble_dir/table3_acceptance.json"
    run_pdb5_supplement
    exit 0
fi

record_stage acceptance_failed "candidate=all_declared_candidates"
exit 1
