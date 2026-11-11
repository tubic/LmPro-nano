#!/bin/bash
# Watch a training log; if it goes stale, work out why and act.
#
# Motivation: on 2026-07-24 a GPU hit `Xid 79 / GPU has fallen off the bus` and the job died
# *silently* — the log simply stopped mid-step with no error line, no traceback, non-zero
# process count — and that went unnoticed for 4 h 15 m. Staleness of the log is the only
# reliable liveness signal.
#
#   usage: tools/train_watchdog.sh <log> <restart-cmd> [stale-seconds]
#   e.g.   nohup tools/train_watchdog.sh logs/pretrain_fixed.log "bash pretrain-abbert.sh" 900 &
#   adopt: MANAGED_PID=<setsid-leader> tools/train_watchdog.sh <log> <restart-cmd> 900
#
# fairseq restores from checkpoint_last.pt in --save-dir automatically, so the restart
# command is just the original training command; at most one checkpoint interval is lost.
# The managed session leader and its /proc start time are recorded in
# <log-without-.log>.train.pid. A restart signals only that process group.
#
# Xid 79 is deliberately NOT auto-restarted: the wedged processes sit in uninterruptible
# D state, cannot be killed, and keep their ~17.5 GiB/GPU allocated, so a relaunch would
# only OOM in a loop. That case needs a host reboot and is reported instead.

set -u
LOG="${1:?usage: train_watchdog.sh <log> <restart-cmd> [stale-seconds]}"
CMD="${2:?}"
STALE="${3:-900}"                 # default: 15 min without a new log line = suspicious
CHECK_EVERY=${CHECK_EVERY:-120}
RELAUNCH_GRACE=${RELAUNCH_GRACE:-180}
STATE="${LOG%.log}.watchdog.log"
PIDFILE="${LOG%.log}.train.pid"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE"; }

process_start_ticks() {
    local pid=$1
    [ -r "/proc/$pid/stat" ] || return 1
    awk '{print $22}' "/proc/$pid/stat"
}

remember_managed_process() {
    local pid=$1 start_ticks
    start_ticks=$(process_start_ticks "$pid") || return 1
    printf '%s %s\n' "$pid" "$start_ticks" > "$PIDFILE"
}

managed_process_group() {
    local pid expected_ticks actual_ticks pgid sid
    [ -r "$PIDFILE" ] || return 1
    read -r pid expected_ticks < "$PIDFILE" || return 1
    [[ "$pid" =~ ^[0-9]+$ && "$expected_ticks" =~ ^[0-9]+$ ]] || return 1
    actual_ticks=$(process_start_ticks "$pid") || return 1
    [ "$actual_ticks" = "$expected_ticks" ] || return 1
    read -r pgid sid < <(ps -o pgid=,sid= -p "$pid") || return 1
    pgid=${pgid//[[:space:]]/}
    sid=${sid//[[:space:]]/}
    # Every managed launch uses setsid, so the recorded PID must remain both
    # session and process-group leader. Refuse to kill anything if that
    # identity no longer holds (for example after PID reuse).
    [ "$pgid" = "$pid" ] && [ "$sid" = "$pid" ] || return 1
    printf '%s\n' "$pgid"
}

stop_managed_process_group() {
    local pgid deadline stuck
    if ! pgid=$(managed_process_group); then
        log "no live managed process group found; nothing to kill"
        rm -f "$PIDFILE"
        return 0
    fi

    log "stopping managed process group pgid=$pgid"
    kill -TERM -- "-$pgid" 2>/dev/null || true
    deadline=$((SECONDS + 20))
    while kill -0 -- "-$pgid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
        sleep 1
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true

    stuck=$(ps -eo pid=,pgid=,stat= | awk -v pgid="$pgid" \
        '$2 == pgid && $3 ~ /D/ {print $1}' | head -5 | tr '\n' ' ')
    rm -f "$PIDFILE"
    if [ -n "$stuck" ]; then
        log "FATAL: managed processes stuck in D state ($stuck) -- unkillable."
        log "       THE HOST MUST BE REBOOTED. Afterwards rerun: $CMD"
        exit 3
    fi
}

launch_managed_process_group() {
    local pid
    log "relaunching managed process group: $CMD"
    setsid nohup bash -c "exec $CMD" >> "$LOG" 2>&1 < /dev/null &
    pid=$!
    remember_managed_process "$pid" || {
        log "FATAL: could not record managed process pid=$pid"
        return 1
    }
    log "managed process group started: pid=$pid"
}

log "watchdog up: log=$LOG stale_after=${STALE}s restart='$CMD'"

if [ -n "${MANAGED_PID:-}" ]; then
    remember_managed_process "$MANAGED_PID" || {
        log "FATAL: cannot adopt MANAGED_PID=$MANAGED_PID"
        exit 2
    }
    pgid=$(managed_process_group) || {
        log "FATAL: MANAGED_PID=$MANAGED_PID is not a setsid process-group leader"
        rm -f "$PIDFILE"
        exit 2
    }
    log "adopted existing managed process group pgid=$pgid"
elif pgid=$(managed_process_group 2>/dev/null); then
    log "tracking existing managed process group pgid=$pgid"
fi

restarts=0
while true; do
    sleep "$CHECK_EVERY"

    [ -f "$LOG" ] || { log "log missing, waiting"; continue; }
    age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    [ "$age" -lt "$STALE" ] && continue

    log "STALE: no output for ${age}s"

    # Hardware first — an Xid in the last hour means a reboot is required, not a restart.
    if dmesg 2>/dev/null | grep -qiE "fallen off the bus|Xid .*: 79"; then
        if ! nvidia-smi >/dev/null 2>&1; then
            log "FATAL: nvidia-smi reports a missing device and dmesg shows Xid 79."
            log "       Processes will be wedged in D state holding GPU memory; kill -9 will not"
            log "       work and a relaunch would OOM. THE HOST MUST BE REBOOTED."
            log "       Afterwards just rerun: $CMD   (auto-resumes from checkpoint_last.pt)"
            exit 3
        fi
        log "WARN: Xid 79 seen in dmesg but nvidia-smi is healthy (may be from an older boot)"
    fi

    if [ "$restarts" -ge 5 ]; then
        log "FATAL: already restarted ${restarts}x; refusing to loop. Investigate manually."
        exit 4
    fi

    restarts=$((restarts + 1))
    log "restart #$restarts: stopping only this watchdog's managed process group"
    stop_managed_process_group
    launch_managed_process_group || exit 2
    sleep "$RELAUNCH_GRACE"       # give it time to load data before judging staleness again
done
