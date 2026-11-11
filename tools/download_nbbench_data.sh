#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

OUTPUT_DIR=${OUTPUT_DIR:-baselines/nbbench/data}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

declare -A COMMITS=(
    [VRClassification]=5277b263eff958c12fbe8eef4dc34214b4f1c930
    [CDRInfilling]=c1540b2be150639f67a8466b3fbd75fc451e79f8
    [thermo-seq]=9e0c8a48a3656fc327a22e0751ca97bcceea757e
    [thermo-tm]=1d40bb4b5dde5bb2ff14940a7a68c05ae59e8655
    [vhh_affinity-seq]=542aafdc196e133e6eaefb27438ced33690fba49
    [vhh_affinity-score]=e25be300aaeeac2e0959b38603fec374eb0b2276
    [polyreaction]=e05a1781efac7c894d64afecf74ccb41c33895a3
    [nanobody_type]=781c5c1ec691b04f532948f57a14d7c07520b926
    [SARS-CoV-2]=13474d198137a03acd9b9149688710333c2663b2
    [hIL6]=04f9922f9c8e1fc66faa62bd504057df1cf44c71
    [Paratope]=96d6586b4dee66842e3587cc8aa9bc6b299f6f8f
)

download_file() {
    local dataset=$1
    local file=$2
    local commit=${COMMITS[$dataset]}
    local directory="$OUTPUT_DIR/$dataset"
    local output="$directory/$file"
    mkdir -p "$directory"
    if [[ -s "$output" ]]; then
        return
    fi
    echo "downloading $dataset/$file (resuming ${output}.part when present)"
    curl --fail --location --http1.1 \
        --retry 20 --retry-all-errors --retry-delay 5 \
        --connect-timeout 20 --max-time 43200 \
        --continue-at - --output "$output.part" \
        "$HF_ENDPOINT/datasets/ZYMScott/$dataset/resolve/$commit/$file"
    mv "$output.part" "$output"
}

for dataset in "${!COMMITS[@]}"; do
    download_file "$dataset" README.md
    download_file "$dataset" train.csv
    download_file "$dataset" val.csv
    download_file "$dataset" test.csv
done

for dataset in SARS-CoV-2 hIL6 Paratope; do
    download_file "$dataset" antigen_embeddings.pt
done

manifest="$OUTPUT_DIR/SOURCE_COMMITS.tsv"
{
    echo -e "dataset\tcommit"
    for dataset in "${!COMMITS[@]}"; do
        echo -e "$dataset\t${COMMITS[$dataset]}"
    done | sort
} > "$manifest"

echo "downloaded pinned NbBench data under $OUTPUT_DIR"
echo "source manifest: $manifest"
