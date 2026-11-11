#!/bin/bash
# Block until pretraining has finished writing a given epoch checkpoint, then run a command.
#
#   nohup tools/await_checkpoint_then.sh 8 "PRETRAIN_EPOCH=8 bash run_finetune_exp2.sh" \
#         logs/finetune_pt-ep8.log &
#
# Waits for fairseq's own "saved checkpoint ... checkpointN.pt" line rather than for the
# file to appear: fairseq writes to a temp path and renames, and the log line is emitted
# only after the write completes (it even reports how long the write took). Watching for
# the file alone can catch a partially written checkpoint.
set -u
EPOCH="${1:?usage: await_checkpoint_then.sh <epoch> <cmd> [logfile]}"
CMD="${2:?}"
LOG="${3:-/dev/stdout}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PRETRAIN_LOG=logs/pretrain_fixed.log
CKDIR=checkpoints/pretrained/fixed_RoPE_abbert_warmup10000_lr0.0006_maxsen128_upfreq4_samplenone_clip0.0
TARGET="checkpoint${EPOCH}.pt"

echo "[$(date '+%F %T')] waiting for $TARGET ..."
while true; do
    if grep -qa "saved checkpoint.*/${TARGET} " "$PRETRAIN_LOG" 2>/dev/null; then
        break
    fi
    # Also accept the file if it is present and has been size-stable for a full poll cycle,
    # in case the log was rotated or the watchdog restarted the run.
    if [ -f "$CKDIR/$TARGET" ]; then
        a=$(stat -c %s "$CKDIR/$TARGET"); sleep 60; b=$(stat -c %s "$CKDIR/$TARGET")
        [ "$a" = "$b" ] && [ "$a" -gt 0 ] && break
    fi
    sleep 60
done

echo "[$(date '+%F %T')] $TARGET is complete; launching: $CMD"
bash -c "$CMD" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] command exited with $?"
