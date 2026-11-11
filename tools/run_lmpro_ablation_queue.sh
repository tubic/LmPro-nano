#!/usr/bin/env bash

# Follow-on ablation / CI chain behind an upstream GPU queue. Each ablation
# reuses the corrected finetuning contract (token_mean, layernorm, clip 10,
# lr 3e-5, micro-batch 4, update-freq 2, seed 128) with exactly one component
# switched off, then runs the standard validation-only sweep and one strict
# test. An optional CI backfill re-executes the headline seeds' already
# reported tests with --per_record_out and refuses to keep the TSV unless the
# aggregates reproduce the original logs exactly.

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-$SCRIPT_ROOT}
GPU_INDEX=${GPU_INDEX:?set GPU_INDEX}
UPSTREAM_STATUS=${UPSTREAM_STATUS:?set UPSTREAM_STATUS}
UPSTREAM_PATTERN=${UPSTREAM_PATTERN:?set UPSTREAM_PATTERN}
ABLATIONS=${ABLATIONS:-""}
CI_SEEDS=${CI_SEEDS:-""}
LOG_DIR=${LOG_DIR:-logs/ablation_queue_gpu${GPU_INDEX}}
STATUS_FILE=${STATUS_FILE:-$LOG_DIR/status.log}
PRETRAINED_FILE=${PRETRAINED_FILE:-checkpoints/pretrained/corrected_v2_eos_cdr3span0p5_contextkeep_seed1_u100k_eb4032/candidates/checkpoint_u33000.pt}
PRETRAIN_LABEL=${PRETRAIN_LABEL:-u33000}
SPLIT=${SPLIT:-nanobody_clustered}
DATA_PATH=${DATA_PATH:-abgnn/finetune/exp2-hern/nanobody_clustered}
VALID_RECORDS=${VALID_RECORDS:-baselines/nanobody_clustered/valid_records_pretrain_exact_disjoint.txt}
TEST_RECORDS=${TEST_RECORDS:-baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt}
LMPRO_ENV_BIN=${LMPRO_ENV_BIN:-/path/to/conda/envs/lmpro-nano-torch26-cu124/bin}
PYTHON=$LMPRO_ENV_BIN/python

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
    status "ablation queue (GPU $GPU_INDEX) exit=$rc"
}
trap on_exit EXIT

status "waiting for '$UPSTREAM_PATTERN' in $UPSTREAM_STATUS"
# Exit lines may predate a manual restart of the upstream queue (stale failure
# markers). Only exit lines appended after this script starts waiting count.
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

gpu_memory_used() {
    nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
}

gpu_utilization() {
    nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
}

wait_for_gpu() {
    local ready=0 used util
    while true; do
        used=$(gpu_memory_used || true)
        util=$(gpu_utilization || true)
        if [[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] \
            && [ "$used" -lt 500 ] && [ "$util" -lt 10 ]; then
            ready=$((ready + 1))
            if [ "$ready" -ge 2 ]; then
                return 0
            fi
        else
            ready=0
            status "waiting for GPU $GPU_INDEX: memory < 500 MiB, util < 10% (used=${used:-unknown}, util=${util:-unknown}, ready=${ready}/2)"
        fi
        sleep 60
    done
}

run_ablation() {
    local name=$1
    shift
    local save_dir="checkpoints/exp2_nanobody/corrected_v2_ablation_${name}_${SPLIT}_seed128_torch26"
    local sweep_dir="logs/checkpoint_sweeps/corrected_v2_ablation_${name}_${SPLIT}_seed128_torch26_exactdisjoint"

    if grep -Fq "ablation $name finetune complete" "$STATUS_FILE" 2>/dev/null; then
        status "ablation $name finetune already complete"
    else
        wait_for_gpu
        status "ablation $name finetune start"
        env \
            CUDA_VISIBLE_DEVICES=$GPU_INDEX \
            PRETRAIN_LABEL="$PRETRAIN_LABEL" \
            SPLIT="$SPLIT" \
            SEED=128 \
            SAVE_DIR="$save_dir" \
            "$@" \
            bash run_finetune_corrected_exp2.sh \
            > "$LOG_DIR/ablation_${name}.log" 2>&1
        status "ablation $name finetune complete"
    fi

    if grep -Fq "checkpoint sweep complete" "$sweep_dir/status.log" 2>/dev/null; then
        status "ablation $name sweep and one test already complete"
    else
        wait_for_gpu
        status "ablation $name validation sweep and one strict test start"
        env \
            SAVE_DIR="$save_dir" \
            RUN_NAME="corrected_v2_ablation_${name}_${SPLIT}_seed128_torch26_exactdisjoint" \
            DATA_PATH="$DATA_PATH" \
            VALID_RESTRICT_RECORDS="$VALID_RECORDS" \
            RESTRICT_RECORDS="$TEST_RECORDS" \
            SEED=128 \
            GPU_INDEX=$GPU_INDEX \
            GPU_MODE=full \
            RUN_TEST=1 \
            bash tools/sweep_finetune_checkpoints.sh \
            > "$LOG_DIR/ablation_${name}_sweep.log" 2>&1
        status "ablation $name validation sweep and one strict test complete"
    fi
}

for ablation in $ABLATIONS; do
    case "$ablation" in
        no_pretrain)
            run_ablation no_pretrain USE_PRETRAINED=0 FINETUNE_BERT_SCHEME=all_tuning
            ;;
        fixed_encoder)
            run_ablation fixed_encoder PRETRAINED_FILE="$PRETRAINED_FILE" FINETUNE_BERT_SCHEME=fixed
            ;;
        bptt1)
            run_ablation bptt1 PRETRAINED_FILE="$PRETRAINED_FILE" BPTT_ITERATIONS=1
            ;;
        no_mlp)
            run_ablation no_mlp PRETRAINED_FILE="$PRETRAINED_FILE" USE_RESIDUAL_MLP=0
            ;;
        *)
            status "unknown ablation: $ablation"
            exit 1
            ;;
    esac
done

ci_backfill() {
    local seed=$1 sweep_dir=$2 baseline_log=$3
    local sel="$sweep_dir/selected_checkpoint.txt"
    [ -f "$sel" ] || { status "CI seed $seed: missing selection file $sel"; return 1; }
    [ -f "$baseline_log" ] || { status "CI seed $seed: missing baseline log $baseline_log"; return 1; }
    local ckpt epoch tsv rerun_log base_aar base_ppl base_rmsd new_aar new_ppl new_rmsd
    ckpt=$(awk -F= '$1=="checkpoint" {print $2}' "$sel")
    epoch=$(awk -F= '$1=="epoch" {print $2}' "$sel")
    [ -f "$ckpt" ] || { status "CI seed $seed: missing checkpoint $ckpt"; return 1; }
    tsv="$sweep_dir/per_record.tsv"
    rerun_log="$sweep_dir/test_ci_rerun_epoch${epoch}_seed${seed}.log"
    base_aar=$(awk -F'= ' '/^Amino acid recovery rate = / {v=$2} END {print v}' "$baseline_log")
    base_ppl=$(awk -F'= ' '/^PPL = / {v=$2} END {print v}' "$baseline_log")
    base_rmsd=$(awk -F'= ' '/^RMSD = / {v=$2} END {print v}' "$baseline_log")
    [ -n "$base_aar" ] && [ -n "$base_ppl" ] && [ -n "$base_rmsd" ] || {
        status "CI seed $seed: baseline aggregates unreadable in $baseline_log"; return 1; }

    if grep -q '^Amino acid recovery rate = ' "$rerun_log" 2>/dev/null && [ -f "$tsv" ]; then
        status "CI seed $seed: rerun already present, verifying only"
    else
        wait_for_gpu
        status "CI seed $seed: per-record rerun start (epoch $epoch)"
        env \
            CUDA_VISIBLE_DEVICES=$GPU_INDEX \
            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
            MKL_THREADING_LAYER=GNU \
            "$PYTHON" -u inference.py \
                --cdr_type 3 \
                --cktpath "$ckpt" \
                --data_path "$DATA_PATH" \
                --split test \
                --restrict_records "$TEST_RECORDS" \
                --num_decode 10000 \
                --topk 100 \
                --seed "$seed" \
                --report_topks 1 10 100 1000 \
                --per_record_out "$tsv" \
                > "$rerun_log" 2>&1
        status "CI seed $seed: per-record rerun complete"
    fi

    new_aar=$(awk -F'= ' '/^Amino acid recovery rate = / {v=$2} END {print v}' "$rerun_log")
    new_ppl=$(awk -F'= ' '/^PPL = / {v=$2} END {print v}' "$rerun_log")
    new_rmsd=$(awk -F'= ' '/^RMSD = / {v=$2} END {print v}' "$rerun_log")
    if [ "$base_aar" != "$new_aar" ] || [ "$base_ppl" != "$new_ppl" ] || [ "$base_rmsd" != "$new_rmsd" ]; then
        rm -f "$tsv"
        status "CI seed $seed: aggregate mismatch (baseline $base_aar/$base_ppl/$base_rmsd vs rerun $new_aar/$new_ppl/$new_rmsd); TSV discarded"
        return 1
    fi
    status "CI seed $seed: aggregates reproduce baseline exactly; per-record TSV kept at $tsv"
}

for seed in $CI_SEEDS; do
    case "$seed" in
        128)
            ci_backfill 128 \
                logs/checkpoint_sweeps/corrected_v2_u33000_nanobody_clustered_fullbptt_seed128_torch26_exactdisjoint \
                logs/selected_lmpro_antigen_queue/clustered_strict_test.log
            ;;
        256|512)
            sweep_dir=logs/checkpoint_sweeps/corrected_v2_u33000_nanobody_clustered_fullbptt_seed${seed}_torch26_exactdisjoint
            epoch=$(awk -F= '$1=="epoch" {print $2}' "$sweep_dir/selected_checkpoint.txt" 2>/dev/null || true)
            if [ -z "${epoch:-}" ]; then
                status "CI seed $seed: no validation selection found; skipping"
                exit 1
            fi
            ci_backfill "$seed" "$sweep_dir" "$sweep_dir/test_selected_epoch${epoch}_seed${seed}.log"
            ;;
        *)
            status "CI seed $seed: unsupported seed"
            exit 1
            ;;
    esac
done

status "ablation queue (GPU $GPU_INDEX) complete"
