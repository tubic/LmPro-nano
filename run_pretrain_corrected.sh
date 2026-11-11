#!/bin/bash
# Start a separate AbBERT pretraining run with downstream-style CDR3 spans.
# The default 100k-update schedule gives polynomial decay a realistic horizon;
# do not resume it into the legacy fixed_RoPE_* checkpoint directory.

set -euo pipefail

export TOTAL_UPDATES=${TOTAL_UPDATES:-100000}
export FULL_CDR3_MASK_PROB=${FULL_CDR3_MASK_PROB:-0.5}
export MASK_CONTEXT_MODE=keep_unmasked
export SAMPLE_MODE=${SAMPLE_MODE:-eos}
# An eos block is one antibody. This effective batch preserves the masked-token
# scale of the previous packed run while leaving headroom for 152-token outliers.
export MAX_SENTENCES=${MAX_SENTENCES:-112}
export TARGET_BATCH=${TARGET_BATCH:-4032}
export SAVE_INTERVAL_UPDATES=${SAVE_INTERVAL_UPDATES:-500}
export KEEP_INTERVAL_UPDATES=${KEEP_INTERVAL_UPDATES:-3}

exec bash "$(dirname "$0")/pretrain-abbert.sh" "$@"
