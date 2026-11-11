#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON:-python3}"
checkpoint="${1:-}"
eval_id="${2:-provisional}"
lock_file="$repo/outputs/rde_pretrain/provisional_table3.lock"
eval_dir="$repo/outputs/rde_pretrain/provisional_eval_${eval_id}"
forced_device="${FORCE_CUDA_DEVICE:-}"

if [[ -z "$checkpoint" ]]; then
    echo "Usage: $0 <rde-checkpoint> [evaluation-id]" >&2
    exit 2
fi
checkpoint="$(readlink -f "$checkpoint")"
if [[ ! -s "$checkpoint" ]]; then
    echo "Checkpoint does not exist: $checkpoint" >&2
    exit 2
fi

mkdir -p "$eval_dir"
cd "$repo"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another provisional Table 3 evaluation holds $lock_file" >&2
    exit 1
fi

stages_file="$eval_dir/stages.tsv"
if [[ ! -e "$stages_file" ]]; then
    printf 'timestamp\tstage\tdetail\n' > "$stages_file"
fi
record_stage() {
    printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" |
        tee -a "$stages_file"
}

finish_queue() {
    status=$?
    if [[ "$status" -eq 0 ]]; then
        record_stage queue_complete passed
    else
        record_stage queue_failed "exit_status=$status"
    fi
}
trap finish_queue EXIT

sha256sum \
    "$checkpoint" \
    configs/train/profe_rde_table3.yml \
    train_table3.py \
    check_table3_metrics.py > "$eval_dir/locked_inputs.sha256"
record_stage snapshot_locked \
    "checkpoint=$checkpoint selection=pretraining_validation_only outer_metrics_not_used_for_selection"

candidate=""
stable_count=0
if [[ -n "$forced_device" ]]; then
    candidate="$forced_device"
    record_stage gpu_forced "cuda_device=$candidate selection=operator_verified_idle"
else
    while true; do
        available="$(
            nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
                --format=csv,noheader,nounits |
            awk -F',' '
                {
                    gsub(/ /, "", $1)
                    gsub(/ /, "", $2)
                    gsub(/ /, "", $3)
                    if (!found && ($1 + 0) <= 5 && ($2 + 0) <= 5 && ($3 + 0) >= 30000) {
                        print $1
                        found = 1
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

        record_stage gpu_wait "candidate=${candidate:-none} stable=$stable_count/3"
        if [[ "$stable_count" -ge 3 ]]; then
            break
        fi
        sleep 60
    done
fi

export CUDA_VISIBLE_DEVICES="$candidate"
record_stage gpu_locked "cuda_device=$candidate snapshot=$checkpoint"

preflight_log="$eval_dir/preflight.log"
preflight_status=0
"$python_bin" -u train_table3.py configs/train/profe_rde_table3.yml \
    --device cuda \
    --rde-checkpoint "$checkpoint" \
    --num-workers 2 \
    --steps-per-fold 1 \
    --checkpoint-freq 3 \
    --no-context-precompute \
    --max-val-batches 1 \
    --tag "${eval_id}_preflight" \
    2>&1 | tee "$preflight_log" || preflight_status=$?
if [[ "$preflight_status" -ne 0 ]]; then
    record_stage preflight_failed "exit_status=$preflight_status"
    exit "$preflight_status"
fi
record_stage preflight_passed "cuda_device=$candidate"

full_log="$eval_dir/table3.log"
record_stage table3_started \
    "protocol=rde3 steps_per_fold=10000 subset_metrics=all_single_multiple"
"$python_bin" -u train_table3.py configs/train/profe_rde_table3.yml \
    --device cuda \
    --rde-checkpoint "$checkpoint" \
    --num-workers 4 \
    --tag "${eval_id}_full" \
    2>&1 | tee "$full_log"

run_dir="$(
    awk -F= '/^RUN_DIR=/{value=substr($0, index($0, "=") + 1)} END{print value}' \
        "$full_log"
)"
if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    record_stage table3_failed "missing_or_invalid_RUN_DIR=$run_dir"
    exit 1
fi
run_dir="$(readlink -f "$run_dir")"
if [[ ! -s "$run_dir/table3_metrics.csv" || ! -s "$run_dir/outer_predictions.csv" ]]; then
    record_stage table3_failed "missing_metrics_or_predictions=$run_dir"
    exit 1
fi
printf 'run_dir=%s\n' "$run_dir" > "$eval_dir/result.env"
record_stage table3_passed "run_dir=$run_dir"

gate_status=0
"$python_bin" -u check_table3_metrics.py "$run_dir/table3_metrics.csv" \
    --mode single \
    --out "$run_dir/table3_acceptance_provisional.json" \
    2>&1 | tee "$eval_dir/gate.log" || gate_status=$?
record_stage provisional_gate "exit_status=$gate_status report=$run_dir/table3_acceptance_provisional.json"
