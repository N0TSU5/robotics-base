#!/usr/bin/env bash
# Waits until <run>/checkpoint-<step> is completely saved, then restarts the run from it (stop the launcher's process
# group, relaunch scripts/launch.sh with the given arguments). No steps are redone, and the relaunch loads the code
# that is on disk at that moment. Run it detached:
#
#   setsid nohup scripts/restart_after_checkpoint.sh runs/sawyer 5000 --batch 4 --accum 16 > /dev/null 2>&1 &
#
# Progress lines go to <run>/train.log, prefixed [restart-watcher].
set -u
cd "$(dirname "$0")/.."
OUT=$1
STEP=$2
shift 2
CKPT="$OUT/checkpoint-$STEP"
log() { echo "[restart-watcher] $(date -Is) $*" >> "$OUT/train.log"; }

log "waiting for $CKPT, then relaunching with: $*"
until [ -f "$CKPT/trainer_state.json" ] && [ -f "$CKPT/scheduler.pt" ] && [ -f "$CKPT/rng_state.pth" ]; do
    sleep 30
done
sleep 60  # let the Trainer finish writing and rotate out the oldest checkpoint
log "$CKPT saved; stopping the run"
kill -- -"$(cat "$OUT/launcher.pid")"
while pgrep -f "scripts/train.py --output $OUT" > /dev/null; do
    sleep 5
done
setsid nohup scripts/launch.sh "$OUT" "$@" > /dev/null 2>&1 < /dev/null &
log "relaunched"
