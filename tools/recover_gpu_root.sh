#!/bin/bash
# Try to recover from `Xid 79 / GPU has fallen off the bus` WITHOUT rebooting.
#
#   sudo bash tools/recover_gpu_root.sh --dry-run   # inspect only, changes nothing
#   sudo bash tools/recover_gpu_root.sh             # actually attempt the recovery
#
# NOTE: written for the ORIGINAL host, whose GPU at PCI 0000:86:00.0 kept dying with Xid 79.
# The procedure is generic, but before using it elsewhere set BAD_PCI for that host and
# re-check who else is using the machine (there, a colleague had multi-week CPU jobs that a
# reboot would have destroyed, which is why the driver-level route was tried first).
#
# What it does, stopping as soon as the GPUs are healthy:
#   1. safety checks (see below) and a state report
#   2. removes the DEAD GPU from the PCI tree - this is what may let the processes wedged in
#      uninterruptible D state abort and release the ~17.5 GiB/GPU they hold
#   3. rescans the PCI bus to try to bring the card back
#   4. reports whether training can be restarted
#
# Safety checks before touching anything:
#   * refuses unless the target device is genuinely unreachable (nvidia-smi cannot query it),
#     so it can never yank a healthy card out from under a running job;
#   * refuses if any OTHER user holds a GPU compute process;
#   * never kills other users' processes, never unloads modules under a live display server,
#     never reboots.
#
# RESIDUAL RISK, stated plainly: writing to a wedged device's `remove` node can itself block
# (the script would hang; harmless, but you would need another shell), and in rare cases the
# driver can oops while tearing down a device it has lost. If that happened the host would
# go down uncleanly - the same outcome as the reboot you are trying to avoid, but unplanned.
# Judge accordingly; this is why --dry-run exists.

set -u
BAD_PCI="${BAD_PCI:-0000:86:00.0}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

hr()  { printf '%s\n' "------------------------------------------------------------"; }
say() { echo "[recover] $*"; }
refcount() { awk '/^nvidia /{print $3}' /proc/modules 2>/dev/null; }
dstate()   { ps -eo stat= | grep -c '^D'; }

# Probe each index separately: a single fallen-off-the-bus card makes a whole-table
# `nvidia-smi --query-gpu=...` fail outright, which would otherwise hide every healthy card
# and make the final assessment below report a false "not recovered".
probe_gpus() {
    for i in $(seq 0 15); do
        line=$(nvidia-smi -i "$i" --query-gpu=index,pci.bus_id,memory.used,memory.total \
               --format=csv,noheader 2>/dev/null) || continue
        [ -n "$line" ] && echo "$line"
    done
}

if [ "$(id -u)" -ne 0 ]; then
    echo "must run as root: sudo bash $0" >&2; exit 1
fi

STATE=$(mktemp /tmp/recover_gpu.XXXXXX) || exit 1
trap 'rm -f "$STATE"' EXIT

hr; say "1/4  state and safety checks"
say "nvidia module refcount : $(refcount)   (non-zero => rmmod cannot work)"
say "processes in D state   : $(dstate)"
say "responsive GPUs (index, bus, used, total):"
probe_gpus | sed 's/^/          /'

# (a) the target must really be unreachable - never remove a card that still answers
if nvidia-smi -i "$BAD_PCI" --query-gpu=pci.bus_id --format=csv,noheader >/dev/null 2>&1; then
    say "ABORT: $BAD_PCI still responds to nvidia-smi, so it is NOT the fallen-off card."
    say "       Refusing to remove a live GPU. Re-check which device actually failed:"
    say "         dmesg | grep -iE 'xid|fallen off the bus'"
    exit 2
fi
say "check: $BAD_PCI is unreachable (consistent with Xid 79)  OK"

# (b) nobody else may be using the GPUs
others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
         while read -r p; do p=${p%,*}; [ -n "$p" ] && ps -o user= -p "$p" 2>/dev/null; done |
         sort -u | grep -v "^${SUDO_USER:-bo_sun}$" | tr '\n' ' ')
if [ -n "${others// /}" ]; then
    say "ABORT: other users hold GPU compute processes: $others"
    say "       Coordinate with them first."
    exit 2
fi
say "check: no other user holds a GPU process  OK"

if [ "$DRY" = "1" ]; then
    hr; say "--dry-run: all checks passed, nothing was changed."
    say "Re-run without --dry-run to attempt the recovery."
    exit 0
fi

hr; say "2/4  removing dead device $BAD_PCI from the PCI tree"
if [ -e "/sys/bus/pci/devices/$BAD_PCI/remove" ]; then
    # backgrounded with a timeout: the write can block forever on a wedged device
    ( echo 1 > "/sys/bus/pci/devices/$BAD_PCI/remove" ) &
    wpid=$!
    for _ in $(seq 1 30); do kill -0 "$wpid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$wpid" 2>/dev/null; then
        say "WARNING: the remove write is still blocked after 30s (driver is wedged)."
        say "         Leaving it; it may complete later. Do not kill it."
    else
        say "removed"
    fi
    sleep 8
else
    say "device node absent already, skipping"
fi
say "D-state processes now  : $(dstate)"
say "nvidia module refcount : $(refcount)"

hr; say "3/4  rescanning the PCI bus"
echo 1 > /sys/bus/pci/rescan
sleep 15
nvidia-smi -L >/dev/null 2>&1; sleep 5

hr; say "4/4  final state"
probe_gpus > "$STATE" 2>/dev/null
sed 's/^/          /' "$STATE"

ngpu=$(grep -c ',' "$STATE" 2>/dev/null || echo 0)
# field 3 is "NNN MiB"; a card whose leaked allocation was released sits well under 1 GiB
freed=$(awk -F, '{gsub(/[^0-9]/,"",$3); if ($3+0 < 1000) c++} END{print c+0}' "$STATE")

hr
say "GPUs visible: $ngpu | with memory freed: $freed | D-state procs: $(dstate)"
if [ "$freed" -ge 3 ]; then
    say "SUCCESS - memory released. Restart training (auto-resumes from checkpoint_last.pt):"
    say "    cd <the LmPro-Nano directory> && bash pretrain-abbert.sh"
    say "pretrain-abbert.sh excludes $BAD_PCI automatically and rescales --update-freq."
else
    say "NOT recovered - the wedged processes still hold their allocations."
    say "Next escalation, in order:"
    say "  a) stop the display server so nvidia_drm/nvidia_modeset are released, then"
    say "       rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia && modprobe nvidia"
    say "     (still fails while the refcount printed above is non-zero)"
    say "  b) reboot - but warn xin_feng first: CPU jobs running 14-23 days."
fi
hr
