#!/usr/bin/env bash

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$SCRIPT_ROOT}
cd "$ROOT"

LMPRO_ENV_BIN=${LMPRO_ENV_BIN:-/home/af3_bosun/miniforge3/envs/lmpro-nano/bin}
if [ -d "$LMPRO_ENV_BIN" ]; then
    export PATH="$LMPRO_ENV_BIN:$PATH"
fi

# These names reproduce the shipped final_3_data binaries. Override TEXT or
# DATA_VERSION only when intentionally building a different corpus revision.
TEXT=${TEXT:-abgnn/nano_data}
DEST=${DEST:-abgnn/final_3_data}
DATA_VERSION=${DATA_VERSION:-3}
mkdir -p "$DEST"

for kind in seq tag; do
    for split in train valid test; do
        input="$TEXT/${split}_${DATA_VERSION}.${kind}.tokens"
        [ -f "$input" ] || {
            echo "ERROR: missing raw pretraining input: $input" >&2
            exit 1
        }
    done
done

fairseq-preprocess \
    --only-source \
    --trainpref "$TEXT/train_${DATA_VERSION}.seq.tokens" \
    --validpref "$TEXT/valid_${DATA_VERSION}.seq.tokens" \
    --testpref "$TEXT/test_${DATA_VERSION}.seq.tokens" \
    --srcdict data/seq_dict.txt \
    --destdir "$DEST/seq" \
    --workers 24

fairseq-preprocess \
    --only-source \
    --trainpref "$TEXT/train_${DATA_VERSION}.tag.tokens" \
    --validpref "$TEXT/valid_${DATA_VERSION}.tag.tokens" \
    --testpref "$TEXT/test_${DATA_VERSION}.tag.tokens" \
    --srcdict data/tag_dict.txt \
    --destdir "$DEST/tag" \
    --workers 24
