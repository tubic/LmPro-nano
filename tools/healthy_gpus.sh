#!/bin/bash
# Print a CUDA_VISIBLE_DEVICES list of GPUs that are present, responsive, and not blocklisted.
#
# Why by PCI address and not index: indices are assigned in PCI order at driver load, so they
# shift whenever a card disappears or is re-seated. The PCI bus id is stable, so that is what
# the blocklist keys on.
#
# GPU 0000:86:00.0 on this host has repeatedly hit `Xid 79 / GPU has fallen off the bus`
# (2026-07-24 killed a pretraining run after 4 h; its PCI audio function reads back rev ff,
# i.e. the device is electrically gone until a reboot). Override the default with:
#   BAD_GPU_PCI="0000:86:00.0 0000:3b:00.0" tools/healthy_gpus.sh
# or clear it entirely with BAD_GPU_PCI="" once the card is replaced.
#
#   usage:  export CUDA_VISIBLE_DEVICES=$(tools/healthy_gpus.sh)
#           tools/healthy_gpus.sh --count      # just how many are usable

# Default is EMPTY: the blocklist is host-specific. On the machine this package came from,
# the GPU at PCI 0000:86:00.0 repeatedly died with `Xid 79 / GPU has fallen off the bus` and
# was excluded with:  export BAD_GPU_PCI="0000:86:00.0"
# Set it on the new host only if a card there turns out to be faulty.
BAD_GPU_PCI="${BAD_GPU_PCI-}"

healthy=()
# Probe each index on its own: a single fallen-off-the-bus card makes a whole-table
# `nvidia-smi --query-gpu=...` fail, which would otherwise hide the healthy cards.
for i in $(seq 0 15); do
    line=$(nvidia-smi -i "$i" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null) || continue
    [ -z "$line" ] && continue
    bus=$(echo "$line" | tr -d ' ' | tr 'A-F' 'a-f')
    bus=${bus#0000000}          # nvidia-smi prints 00000000:86:00.0
    [ "${bus:0:5}" != "0000:" ] && bus="0000:${bus#*:}"

    skip=0
    for bad in $BAD_GPU_PCI; do
        bad=$(echo "$bad" | tr 'A-F' 'a-f')
        [ "$bus" = "$bad" ] && skip=1
    done
    [ "$skip" = "1" ] && continue

    healthy+=("$i")
done

if [ "$1" = "--count" ]; then
    echo "${#healthy[@]}"
else
    (IFS=,; echo "${healthy[*]}")
fi
