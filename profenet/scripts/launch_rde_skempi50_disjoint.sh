#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON:-python3}"
source_cache="${RDE_SOURCE_CACHE:-/tmp/af3_bosun_rde_cache}"
clean_cache="${RDE_CLEAN_CACHE:-/tmp/af3_bosun_rde_cache_skempi50_disjoint}"
cuda_device="${FORCE_CUDA_DEVICE:?FORCE_CUDA_DEVICE must name an allocated GPU or MIG UUID}"
batch_size="${RDE_BATCH_SIZE:-16}"
accumulation_steps="${RDE_ACCUMULATION_STEPS:-4}"
num_workers="${RDE_NUM_WORKERS:-8}"
run_id="${RDE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
run_root="$repo/outputs/rde_pretrain/skempi50_disjoint_$run_id"
lock_file="$repo/outputs/rde_pretrain/skempi50_disjoint_pretrain.lock"

if (( batch_size * accumulation_steps != 64 )); then
    echo "Effective batch must remain 64, got $((batch_size * accumulation_steps))" >&2
    exit 2
fi

mkdir -p "$run_root" "$clean_cache"
cd "$repo"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another SKEMPI-disjoint RDE run holds $lock_file" >&2
    exit 1
fi

if [[ ! -s data/generated/pdbredo_skempi50_disjoint/audit.json ]]; then
    "$python_bin" scripts/build_pdbredo_skempi_disjoint.py \
        --output-dir data/generated/pdbredo_skempi50_disjoint \
        > "$run_root/data_build.log"
fi

for filename in structures.lmdb keys.pkl; do
    source_path="$source_cache/$filename"
    target_path="$clean_cache/$filename"
    if [[ ! -s "$source_path" ]]; then
        echo "Missing source cache file: $source_path" >&2
        exit 1
    fi
    if [[ -e "$target_path" || -L "$target_path" ]]; then
        if [[ "$(readlink -f "$target_path")" != "$(readlink -f "$source_path")" ]]; then
            echo "Clean cache path already targets another file: $target_path" >&2
            exit 1
        fi
    else
        ln -s "$source_path" "$target_path"
    fi
done

sha256sum \
    scripts/build_pdbredo_skempi_disjoint.py \
    configs/train/rde_skempi50_disjoint.yml \
    train_rde.py \
    data/generated/pdbredo_skempi50_disjoint/pdbredo_clusters.txt \
    data/generated/pdbredo_skempi50_disjoint/pdbredo_splits.txt \
    data/generated/pdbredo_skempi50_disjoint/audit.json \
    > "$run_root/locked_inputs.sha256"

export CUDA_VISIBLE_DEVICES="$cuda_device"
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$cuda_device batch_size=$batch_size accumulation_steps=$accumulation_steps num_workers=$num_workers clean_cache=$clean_cache"
set -o pipefail
"$python_bin" -u train_rde.py configs/train/rde_skempi50_disjoint.yml \
    --device cuda \
    --num-workers "$num_workers" \
    --batch-size "$batch_size" \
    --accumulation-steps "$accumulation_steps" \
    --processed-dir "$clean_cache" \
    --logdir ./outputs/rde_pretrain \
    --tag skempi50_disjoint_pdbsplit \
    --save-freq 1000 \
    2>&1 | tee "$run_root/full_train.log"
