#!/usr/bin/env bash

# Download the four official NanoBodyBuilder2 weights and verify Zenodo MD5s.

set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-baselines/structure_predictors/ImmuneBuilder_weights}
BASE_URL=${BASE_URL:-https://zenodo.org/records/7258553/files}
mkdir -p "$OUTPUT_DIR"

models=(
    "nanobody_model_1 4591075f467ca9f76a37a5d1d3cfe591"
    "nanobody_model_2 620fc916720bc7068cd18c7afa1aea8d"
    "nanobody_model_3 f1ef0a66a54efb9d14ddbb97fdc30785"
    "nanobody_model_4 a7e14a33f4c00e96df896120e9cf8522"
)

for item in "${models[@]}"; do
    read -r name expected_md5 <<< "$item"
    destination="$OUTPUT_DIR/$name"
    if [ -s "$destination" ] && [ "$(md5sum "$destination" | cut -d' ' -f1)" = "$expected_md5" ]; then
        echo "$name already verified"
        continue
    fi
    temporary="$destination.part"
    curl --fail --location --retry 5 --retry-delay 5 \
        "$BASE_URL/$name?download=1" \
        --output "$temporary"
    observed_md5=$(md5sum "$temporary" | cut -d' ' -f1)
    if [ "$observed_md5" != "$expected_md5" ]; then
        echo "ERROR: $name MD5 $observed_md5 != $expected_md5" >&2
        exit 1
    fi
    mv "$temporary" "$destination"
    echo "$name downloaded and verified"
done

