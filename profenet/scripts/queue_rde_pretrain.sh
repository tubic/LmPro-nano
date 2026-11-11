#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON:-python3}"
lock_file="$repo/outputs/rde_pretrain/full_pretrain.lock"
mkdir -p "$repo/outputs/rde_pretrain"
cd "$repo"

exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another full RDE pretraining queue or run holds $lock_file"
    exit 1
fi

candidate="${FORCE_CUDA_DEVICE:-}"
if [[ -n "$candidate" ]]; then
    echo "$(date --iso-8601=seconds) using explicitly selected CUDA device $candidate"
else
    stable_count=0
    while true; do
        available="$(
            nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits |
            awk -F',' '
                {
                    gsub(/ /, "", $1);
                    gsub(/ /, "", $2);
                    gsub(/ /, "", $3);
                    if (!found && ($1 + 0) <= 5 && $2 != "[N/A]" && ($2 + 0) <= 5 && ($3 + 0) >= 36000) {
                        print $1;
                        found = 1;
                    }
                }
            '
        )"

        if [[ -n "$available" && "$available" == "$candidate" ]]; then
            stable_count=$((stable_count + 1))
        elif [[ -n "$available" ]]; then
            candidate="$available"
            stable_count=1
        else
            candidate=""
            stable_count=0
        fi

        echo "$(date --iso-8601=seconds) candidate=${candidate:-none} stable=$stable_count/3"
        if [[ "$stable_count" -ge 3 ]]; then
            break
        fi
        sleep 60
    done
fi

echo "$(date --iso-8601=seconds) launching locked Table 3 pipeline on CUDA device $candidate"
export CUDA_VISIBLE_DEVICES="$candidate"

pipeline_id="$(date +%Y%m%d_%H%M%S)"
pipeline_dir="$repo/outputs/rde_pretrain/full_pipeline_$pipeline_id"
mkdir -p "$pipeline_dir"
stages_file="$pipeline_dir/stages.tsv"
printf 'timestamp\tstage\tdetail\n' > "$stages_file"

record_stage() {
    printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" |
        tee -a "$stages_file"
}

finish_pipeline() {
    status=$?
    if [[ "$status" -eq 0 ]]; then
        record_stage pipeline_complete passed
    else
        record_stage pipeline_failed "exit_status=$status"
    fi
}
trap finish_pipeline EXIT

printf 'cuda_device=%s\nvisible_cuda_device=0\n' "$candidate" > "$pipeline_dir/device.env"
record_stage gpu_locked "cuda_device=$candidate"

cp configs/train/table3_candidate_plan.json "$pipeline_dir/candidate_plan.json"
cp configs/train/profe_enhanced_table3.yml "$pipeline_dir/profe_enhanced_table3.yml"
cp configs/train/profe_rde_table3.yml "$pipeline_dir/profe_rde_table3.yml"
sha256sum \
    "$pipeline_dir/candidate_plan.json" \
    "$pipeline_dir/profe_enhanced_table3.yml" \
    "$pipeline_dir/profe_rde_table3.yml" \
    train_table3.py train_rde.py ensemble_table3_predictions.py \
    check_table3_metrics.py > "$pipeline_dir/locked_inputs.sha256"
record_stage candidate_plan_locked "plan=$pipeline_dir/candidate_plan.json"

extract_run_dir() {
    awk -F= '/^RUN_DIR=/{value=substr($0, index($0, "=") + 1)} END{print value}' "$1"
}

run_enhanced_preflight() {
    batch_size="$1"
    preflight_log="$pipeline_dir/enhanced_preflight_bs${batch_size}.log"
    preflight_status=0
    "$python_bin" -u train_table3.py "$pipeline_dir/profe_enhanced_table3.yml" \
        --device cuda \
        --num-workers 2 \
        --steps-per-fold 1 \
        --batch-size "$batch_size" \
        --accumulation-steps 1 \
        --checkpoint-freq 3 \
        --no-context-precompute \
        --max-val-batches 1 \
        --tag "enhanced_batch${batch_size}_preflight" \
        2>&1 | tee "$preflight_log" || preflight_status=$?
    return "$preflight_status"
}

record_stage enhanced_preflight_started batch_size=8
if run_enhanced_preflight 8; then
    enhanced_batch_size=8
    enhanced_accumulation_steps=4
    record_stage enhanced_preflight_passed batch_size=8
else
    if grep -Eqi 'out of memory|outofmemory|CUDNN_STATUS_ALLOC_FAILED' \
        "$pipeline_dir/enhanced_preflight_bs8.log"; then
        record_stage enhanced_preflight_retry "batch_size=4 reason=cuda_oom_at_8"
        if run_enhanced_preflight 4; then
            enhanced_batch_size=4
            enhanced_accumulation_steps=8
            record_stage enhanced_preflight_passed batch_size=4
        else
            record_stage enhanced_preflight_failed batch_size=4
            exit 1
        fi
    else
        record_stage enhanced_preflight_failed "batch_size=8 reason=non_oom_error"
        exit 1
    fi
fi

record_stage enhanced_table3_started "batch_size=$enhanced_batch_size accumulation_steps=$enhanced_accumulation_steps effective_batch_size=32 steps_per_fold=10000"
enhanced_log="$pipeline_dir/enhanced_table3.log"
enhanced_status=0
"$python_bin" -u train_table3.py "$pipeline_dir/profe_enhanced_table3.yml" \
    --device cuda \
    --num-workers 4 \
    --batch-size "$enhanced_batch_size" \
    --accumulation-steps "$enhanced_accumulation_steps" \
    --tag full_locked \
    2>&1 | tee "$enhanced_log" || enhanced_status=$?
if [[ "$enhanced_status" -ne 0 ]]; then
    record_stage enhanced_table3_failed "exit_status=$enhanced_status"
    exit "$enhanced_status"
fi

enhanced_run_dir="$(extract_run_dir "$enhanced_log")"
if [[ -z "$enhanced_run_dir" || ! -d "$enhanced_run_dir" ]]; then
    record_stage enhanced_table3_failed "missing_or_invalid_RUN_DIR=$enhanced_run_dir"
    exit 1
fi
enhanced_run_dir="$(readlink -f "$enhanced_run_dir")"
enhanced_metrics="$enhanced_run_dir/table3_metrics.csv"
enhanced_predictions="$enhanced_run_dir/outer_predictions.csv"
if [[ ! -s "$enhanced_metrics" || ! -s "$enhanced_predictions" ]]; then
    record_stage enhanced_table3_failed "missing_metrics_or_predictions=$enhanced_run_dir"
    exit 1
fi
record_stage enhanced_table3_passed "run_dir=$enhanced_run_dir"

run_pdb5_supplement() {
    record_stage pdb5_supplement_started "model=ProFE-Net batch_size=$enhanced_batch_size accumulation_steps=$enhanced_accumulation_steps effective_batch_size=32"
    supplement_log="$pipeline_dir/enhanced_pdb5.log"
    supplement_status=0
    "$python_bin" -u train_table3.py "$pipeline_dir/profe_enhanced_table3.yml" \
        --protocol pdb5 \
        --device cuda \
        --num-workers 4 \
        --batch-size "$enhanced_batch_size" \
        --accumulation-steps "$enhanced_accumulation_steps" \
        --tag supplemental_locked \
        2>&1 | tee "$supplement_log" || supplement_status=$?
    if [[ "$supplement_status" -ne 0 ]]; then
        record_stage pdb5_supplement_failed "exit_status=$supplement_status"
        return "$supplement_status"
    fi
    supplement_run_dir="$(extract_run_dir "$supplement_log")"
    if [[ -z "$supplement_run_dir" || ! -d "$supplement_run_dir" ]]; then
        record_stage pdb5_supplement_failed "missing_or_invalid_RUN_DIR=$supplement_run_dir"
        return 1
    fi
    supplement_run_dir="$(readlink -f "$supplement_run_dir")"
    if [[ ! -s "$supplement_run_dir/table3_metrics.csv" || ! -s "$supplement_run_dir/table3_metrics_per_fold.csv" ]]; then
        record_stage pdb5_supplement_failed "missing_metrics=$supplement_run_dir"
        return 1
    fi
    printf 'run_dir=%s\n' "$supplement_run_dir" > "$pipeline_dir/pdb5_supplement.env"
    record_stage pdb5_supplement_passed "run_dir=$supplement_run_dir"
}

enhanced_gate_status=0
"$python_bin" -u check_table3_metrics.py "$enhanced_metrics" \
    --mode single \
    --out "$enhanced_run_dir/table3_acceptance.json" \
    2>&1 | tee "$pipeline_dir/enhanced_acceptance.log" || enhanced_gate_status=$?
if [[ "$enhanced_gate_status" -eq 0 ]]; then
    printf 'candidate=ProFE-Net\nrun_dir=%s\n' "$enhanced_run_dir" > "$pipeline_dir/accepted_candidate.env"
    record_stage acceptance_passed "candidate=ProFE-Net report=$enhanced_run_dir/table3_acceptance.json"
    run_pdb5_supplement
    exit 0
fi
record_stage acceptance_failed "candidate=ProFE-Net continuing_to_declared_rank_2"

run_rde_preflight() {
    batch_size="$1"
    preflight_log="$pipeline_dir/rde_preflight_bs${batch_size}.log"
    preflight_status=0
    "$python_bin" -u train_rde.py configs/train/rde.yml \
        --device cuda \
        --num-workers 4 \
        --max-iters 2 \
        --batch-size "$batch_size" \
        --accumulation-steps 1 \
        --val-freq 2 \
        --max-val-batches 1 \
        --save-freq 1 \
        --logdir ./outputs/rde_pretrain \
        --tag "batch${batch_size}_preflight" \
        2>&1 | tee "$preflight_log" || preflight_status=$?
    return "$preflight_status"
}

record_stage rde_preflight_started "batch_candidates=64,32,16,8 effective_batch_size=64"
full_batch_size=""
for batch_candidate in 64 32 16 8; do
    if run_rde_preflight "$batch_candidate"; then
        full_batch_size="$batch_candidate"
        pretrain_accumulation_steps=$((64 / batch_candidate))
        record_stage rde_preflight_passed "batch_size=$full_batch_size accumulation_steps=$pretrain_accumulation_steps"
        break
    fi
    if grep -Eqi 'out of memory|outofmemory|CUDNN_STATUS_ALLOC_FAILED' \
        "$pipeline_dir/rde_preflight_bs${batch_candidate}.log"; then
        record_stage rde_preflight_retry "batch_size=$batch_candidate reason=cuda_oom"
    else
        record_stage rde_preflight_failed "batch_size=$batch_candidate reason=non_oom_error"
        exit 1
    fi
done
if [[ -z "$full_batch_size" ]]; then
    record_stage rde_preflight_failed "all_batch_candidates_oom"
    exit 1
fi

if [[ -e "$repo/trained_models/RDE.pt" || -L "$repo/trained_models/RDE.pt" ]]; then
    record_stage weight_link_failed "trained_models/RDE.pt already exists"
    exit 1
fi

record_stage pretraining_started "batch_size=$full_batch_size accumulation_steps=$pretrain_accumulation_steps effective_batch_size=64 max_iters=200000"
pretrain_log="$pipeline_dir/full_pretrain.log"
pretrain_status=0
"$python_bin" -u train_rde.py configs/train/rde.yml \
    --device cuda \
    --num-workers 8 \
    --batch-size "$full_batch_size" \
    --accumulation-steps "$pretrain_accumulation_steps" \
    --logdir ./outputs/rde_pretrain \
    --tag full_rde_table3 \
    --save-freq 1000 \
    2>&1 | tee "$pretrain_log" || pretrain_status=$?
if [[ "$pretrain_status" -ne 0 ]]; then
    record_stage pretraining_failed "exit_status=$pretrain_status"
    exit "$pretrain_status"
fi

rde_run_dir="$(awk -F= '/^RUN_DIR=/{value=substr($0, index($0, "=") + 1)} END{print value}' "$pretrain_log")"
if [[ -z "$rde_run_dir" || ! -d "$rde_run_dir" ]]; then
    record_stage pretraining_failed "missing_or_invalid_RUN_DIR=$rde_run_dir"
    exit 1
fi
rde_run_dir="$(readlink -f "$rde_run_dir")"
best_checkpoint="$rde_run_dir/checkpoints/best.pt"
if [[ ! -s "$best_checkpoint" ]]; then
    record_stage pretraining_failed "missing_best_checkpoint=$best_checkpoint"
    exit 1
fi
record_stage pretraining_passed "run_dir=$rde_run_dir"

mkdir -p "$repo/trained_models"
best_checkpoint="$(readlink -f "$best_checkpoint")"
ln -s "$best_checkpoint" "$repo/trained_models/RDE.pt"
sha256sum "$best_checkpoint" > "$pipeline_dir/RDE.pt.sha256"
record_stage weight_locked "checkpoint=$best_checkpoint"

record_stage rde_table3_started "protocol=published_rde_3fold subset_metrics=all_single_multiple"
table3_log="$pipeline_dir/rde_table3.log"
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

table3_run_dir="$(awk -F= '/^RUN_DIR=/{value=substr($0, index($0, "=") + 1)} END{print value}' "$table3_log")"
if [[ -z "$table3_run_dir" || ! -d "$table3_run_dir" ]]; then
    record_stage rde_table3_failed "missing_or_invalid_RUN_DIR=$table3_run_dir"
    exit 1
fi
table3_run_dir="$(readlink -f "$table3_run_dir")"
metrics_file="$table3_run_dir/table3_metrics.csv"
if [[ ! -s "$metrics_file" ]]; then
    record_stage rde_table3_failed "missing_metrics=$metrics_file"
    exit 1
fi
record_stage rde_table3_passed "run_dir=$table3_run_dir"

rde_gate_status=0
"$python_bin" -u check_table3_metrics.py "$metrics_file" \
    --mode single \
    --out "$table3_run_dir/table3_acceptance.json" \
    2>&1 | tee "$pipeline_dir/rde_acceptance.log" || rde_gate_status=$?
if [[ "$rde_gate_status" -eq 0 ]]; then
    printf 'candidate=ProFE-RDE-baseline\nrun_dir=%s\n' "$table3_run_dir" > "$pipeline_dir/accepted_candidate.env"
    record_stage acceptance_passed "candidate=ProFE-RDE-baseline report=$table3_run_dir/table3_acceptance.json"
    run_pdb5_supplement
    exit 0
fi
record_stage acceptance_failed "candidate=ProFE-RDE-baseline continuing_to_declared_rank_3"

ensemble_dir="$repo/outputs/table3_runs/table3-fixed-equal-${pipeline_id}"
record_stage fixed_ensemble_started "weights=0.5,0.5 outdir=$ensemble_dir"
"$python_bin" -u ensemble_table3_predictions.py \
    "$enhanced_predictions" \
    "$table3_run_dir/outer_predictions.csv" \
    --outdir "$ensemble_dir" \
    2>&1 | tee "$pipeline_dir/fixed_ensemble.log"

ensemble_gate_status=0
"$python_bin" -u check_table3_metrics.py "$ensemble_dir/table3_metrics.csv" \
    --mode single \
    --out "$ensemble_dir/table3_acceptance.json" \
    2>&1 | tee "$pipeline_dir/ensemble_acceptance.log" || ensemble_gate_status=$?
if [[ "$ensemble_gate_status" -eq 0 ]]; then
    printf 'candidate=ProFE-fixed-equal-ensemble\nrun_dir=%s\n' "$ensemble_dir" > "$pipeline_dir/accepted_candidate.env"
    record_stage acceptance_passed "candidate=ProFE-fixed-equal-ensemble report=$ensemble_dir/table3_acceptance.json"
    run_pdb5_supplement
    exit 0
fi
record_stage acceptance_failed "candidate=all_declared_candidates"
exit 1
