#!/usr/bin/env bash

# Build a non-destructive, versioned LmPro-Nano release and checksum manifest.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VERSION=${VERSION:-1.0.0}
DEST_ROOT=${DEST_ROOT:-$ROOT/../releases}
NAME=LmPro-Nano-v${VERSION}
STAGE=$DEST_ROOT/$NAME
ARCHIVE=$DEST_ROOT/$NAME.tar.zst

cd "$ROOT"
if [ -e "$STAGE" ] || [ -e "$ARCHIVE" ]; then
    echo "ERROR: release target already exists; choose a new VERSION or DEST_ROOT" >&2
    echo "stage=$STAGE" >&2
    echo "archive=$ARCHIVE" >&2
    exit 2
fi

mkdir -p "$STAGE"

copy_path() {
    local source=$1
    [ -e "$source" ] || { echo "ERROR: missing release input: $source" >&2; exit 1; }
    mkdir -p "$STAGE/$(dirname "$source")"
    cp -a "$source" "$STAGE/$source"
}

for path in \
    inference.py \
    run_finetune_corrected_exp2.sh \
    run_finetune_exp2.sh \
    run_pretrain_corrected.sh \
    requirements.txt \
    data \
    environment \
    fairseq_models \
    tests \
    tools \
    third_party \
    abgnn/finetune/exp2-hern/nanobody_clustered \
    abgnn/finetune/exp2-hern/nanobody_antigen_disjoint_v2; do
    copy_path "$path"
done

# Copy the current documentation and workflow assets.
mkdir -p "$STAGE/docs"
cp -a docs/. "$STAGE/docs/"
cp README.md "$STAGE/README.md"
cp docs/LICENSE "$STAGE/LICENSE"
printf '%s\n' "$VERSION" > "$STAGE/VERSION"

mkdir -p "$STAGE/baselines/nanobody_clustered"
for source in baselines/nanobody_clustered/*; do
    [ -f "$source" ] && cp -a "$source" "$STAGE/baselines/nanobody_clustered/"
done
mkdir -p "$STAGE/baselines/nanobody_antigen_disjoint_v2"
for source in baselines/nanobody_antigen_disjoint_v2/*; do
    [ -f "$source" ] && cp -a "$source" "$STAGE/baselines/nanobody_antigen_disjoint_v2/"
done

mkdir -p "$STAGE/checkpoints/release"
cp -a checkpoints/pretrained/corrected_v2_eos_cdr3span0p5_contextkeep_seed1_u100k_eb4032/candidates/checkpoint_u33000.pt \
    "$STAGE/checkpoints/release/lm_nano_pretrain_u33000.pt"
cp -a checkpoints/exp2_nanobody/corrected_v2_u33000_nanobody_clustered_fullbptt_seed128_torch26/checkpoint51.pt \
    "$STAGE/checkpoints/release/lmpro_nano_clustered_seed128_epoch51.pt"
cp -a checkpoints/exp2_nanobody/corrected_v2_u33000_nanobody_antigen_disjoint_v2_fullbptt_seed128_torch26/checkpoint10.pt \
    "$STAGE/checkpoints/release/lmpro_nano_antigen_disjoint_v2_seed128_epoch10.pt"
mkdir -p "$STAGE/checkpoints/baselines"
cp -a baselines/nanobody_clustered/proteinmpnn_clean/training_seed128/best.pt \
    "$STAGE/checkpoints/baselines/proteinmpnn_native_backbone_seed128_best.pt"
cp -a baselines/nanobody_clustered/proteinmpnn_linear_init/training_seed128/best.pt \
    "$STAGE/checkpoints/baselines/proteinmpnn_linear_init_seed128_best.pt"
cp -a baselines/diffab/clean_training_seed128/best.pt \
    "$STAGE/checkpoints/baselines/diffab_clean_seed128_best.pt"

mkdir -p "$STAGE/release_evidence/lmpro" "$STAGE/release_evidence/baselines" \
    "$STAGE/release_evidence/nbbench" "$STAGE/release_evidence/structure" \
    "$STAGE/release_evidence/floors"

cp -a logs/selected_lmpro_antigen_queue/selected_candidate.json \
    logs/selected_lmpro_antigen_queue/clustered_strict_test.log \
    logs/checkpoint_sweeps/corrected_v2_u33000_nanobody_antigen_disjoint_v2_fullbptt_seed128_torch26_exactdisjoint/test_selected_epoch10_seed128.log \
    "$STAGE/release_evidence/lmpro/"

for seed in 128 256 512; do
    source_dir=logs/checkpoint_sweeps/corrected_v2_u33000_nanobody_clustered_fullbptt_seed${seed}_torch26_exactdisjoint
    cp -a "$source_dir/per_record.tsv" \
        "$STAGE/release_evidence/lmpro/seed${seed}_per_record.tsv"
    cp -a "$source_dir/cluster_bootstrap_top100.json" \
        "$STAGE/release_evidence/lmpro/seed${seed}_cluster_bootstrap_top100.json"
    cp -a "$source_dir/structure_metrics_numdecode1.tsv" \
        "$STAGE/release_evidence/lmpro/seed${seed}_structure_metrics.tsv"
    cp -a "$source_dir/structure_metrics_numdecode1_summary.json" \
        "$STAGE/release_evidence/lmpro/seed${seed}_structure_metrics_summary.json"
done

cp -a baselines/nanobody_clustered/proteinmpnn_clean/eval_pretrain_exact_disjoint_s10000/metrics.json \
    "$STAGE/release_evidence/baselines/proteinmpnn_native_backbone_metrics.json"
cp -a baselines/nanobody_clustered/proteinmpnn_clean/eval_pretrain_exact_disjoint_s10000/per_record.tsv \
    "$STAGE/release_evidence/baselines/proteinmpnn_native_backbone_per_record.tsv"
cp -a baselines/nanobody_clustered/proteinmpnn_clean/eval_pretrain_exact_disjoint_s10000/paired_bootstrap_top100.json \
    "$STAGE/release_evidence/baselines/proteinmpnn_native_backbone_paired_bootstrap_top100.json"
cp -a baselines/nanobody_clustered/proteinmpnn_linear_init/eval_pretrain_exact_disjoint_s10000/metrics.json \
    "$STAGE/release_evidence/baselines/proteinmpnn_linear_init_metrics.json"
cp -a baselines/nanobody_clustered/proteinmpnn_linear_init/eval_pretrain_exact_disjoint_s10000/per_record.tsv \
    "$STAGE/release_evidence/baselines/proteinmpnn_linear_init_per_record.tsv"
cp -a baselines/nanobody_clustered/proteinmpnn_linear_init/eval_pretrain_exact_disjoint_s10000/paired_bootstrap_top100.json \
    "$STAGE/release_evidence/baselines/proteinmpnn_linear_init_paired_bootstrap_top100.json"
cp -a baselines/diffab/clean_pretrain_exact_disjoint_seed128_s100/metrics.json \
    "$STAGE/release_evidence/baselines/diffab_fixbb_metrics.json"

while IFS= read -r source; do
    relative=${source#baselines/nbbench/results/corrected_u33000/}
    mkdir -p "$STAGE/release_evidence/nbbench/$(dirname "$relative")"
    cp -a "$source" "$STAGE/release_evidence/nbbench/$relative"
done < <(find baselines/nbbench/results/corrected_u33000 -type f -name results.json | sort)

for predictor in igfold_pretrain_exact_disjoint_eval nanobodybuilder2_pretrain_exact_disjoint_eval; do
    mkdir -p "$STAGE/release_evidence/structure/$predictor"
    cp -a baselines/structure_predictors/$predictor/{summary.json,per_record.tsv,failures.tsv} \
        "$STAGE/release_evidence/structure/$predictor/"
done
cp -a logs/reviewer_qa_20260809/antigen_disjoint_v2_lmpro313_single_shot_floors.log \
    logs/reviewer_qa_20260809/antigen_disjoint_v2_lmpro313_best_of_n_floor_seed128.log \
    "$STAGE/release_evidence/floors/"

printf 'role\trelease_path\tsource_path\tsha256\tbytes\n' > "$STAGE/ARTIFACTS.tsv"
add_artifact() {
    local role=$1 release_path=$2 source_path=$3
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$role" "$release_path" "$source_path" \
        "$(sha256sum "$STAGE/$release_path" | awk '{print $1}')" \
        "$(stat -c '%s' "$STAGE/$release_path")" >> "$STAGE/ARTIFACTS.tsv"
}
add_artifact pretraining_checkpoint checkpoints/release/lm_nano_pretrain_u33000.pt \
    checkpoints/pretrained/corrected_v2_eos_cdr3span0p5_contextkeep_seed1_u100k_eb4032/candidates/checkpoint_u33000.pt
add_artifact clustered_model checkpoints/release/lmpro_nano_clustered_seed128_epoch51.pt \
    checkpoints/exp2_nanobody/corrected_v2_u33000_nanobody_clustered_fullbptt_seed128_torch26/checkpoint51.pt
add_artifact antigen_disjoint_model checkpoints/release/lmpro_nano_antigen_disjoint_v2_seed128_epoch10.pt \
    checkpoints/exp2_nanobody/corrected_v2_u33000_nanobody_antigen_disjoint_v2_fullbptt_seed128_torch26/checkpoint10.pt
add_artifact proteinmpnn_native_backbone_checkpoint checkpoints/baselines/proteinmpnn_native_backbone_seed128_best.pt \
    baselines/nanobody_clustered/proteinmpnn_clean/training_seed128/best.pt
add_artifact proteinmpnn_linear_init_checkpoint checkpoints/baselines/proteinmpnn_linear_init_seed128_best.pt \
    baselines/nanobody_clustered/proteinmpnn_linear_init/training_seed128/best.pt
add_artifact diffab_checkpoint checkpoints/baselines/diffab_clean_seed128_best.pt \
    baselines/diffab/clean_training_seed128/best.pt

printf 'role\tsource_path\tsha256\tbytes\n' > "$STAGE/EXTERNAL_ARTIFACTS.tsv"
record_external() {
    local role=$1 source_path=$2
    [ -f "$source_path" ] || { echo "ERROR: missing external artifact: $source_path" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\n' \
        "$role" "$source_path" "$(sha256sum "$source_path" | awk '{print $1}')" \
        "$(stat -c '%s' "$source_path")" >> "$STAGE/EXTERNAL_ARTIFACTS.tsv"
}
record_external clustered_seed256 \
    checkpoints/exp2_nanobody/corrected_v2_u33000_nanobody_clustered_fullbptt_seed256_torch26/checkpoint52.pt
record_external clustered_seed512 \
    checkpoints/exp2_nanobody/corrected_v2_u33000_nanobody_clustered_fullbptt_seed512_torch26/checkpoint96.pt
record_external ablation_random_init_all_tuning \
    checkpoints/exp2_nanobody/corrected_v2_ablation_no_pretrain_nanobody_clustered_seed128_torch26/checkpoint29.pt
record_external ablation_no_prefix_fixed_abbert \
    checkpoints/exp2_nanobody/corrected_v2_ablation_fixed_encoder_nanobody_clustered_seed128_torch26/checkpoint11.pt
record_external ablation_bptt1 \
    checkpoints/exp2_nanobody/corrected_v2_ablation_bptt1_nanobody_clustered_seed128_torch26/checkpoint81.pt
record_external ablation_no_residual_mlp \
    checkpoints/exp2_nanobody/corrected_v2_ablation_no_mlp_nanobody_clustered_seed128_torch26/checkpoint97.pt

# Third-party repositories are distributed as pinned source snapshots. Their git history
# and upstream/example weights are not runtime inputs; clean retrained baseline weights are
# already frozen under checkpoints/baselines.
find "$STAGE/third_party" -depth -path '*/.git/*' -delete
find "$STAGE/third_party" -depth -type d -name .git -empty -delete
find "$STAGE/third_party" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -delete

# Generated bytecode is host-specific and is not part of the source release.
find "$STAGE" -type f -name '*.pyc' -delete
find "$STAGE" -depth -type d -name __pycache__ -empty -delete

(
    cd "$STAGE"
    find . -type f ! -name MANIFEST.sha256 -printf '%P\n' | LC_ALL=C sort | \
        while IFS= read -r relative; do sha256sum "$relative"; done > MANIFEST.sha256
)

python3 "$STAGE/tools/verify_release_bundle.py" --root "$STAGE"
tar --zstd -cf "$ARCHIVE" -C "$DEST_ROOT" "$NAME"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

echo "release stage: $STAGE"
echo "release archive: $ARCHIVE"
echo "archive checksum: $ARCHIVE.sha256"
