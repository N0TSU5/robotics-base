#!/usr/bin/env bash
# Keeps a training run alive: starts it, and after a crash (Hub outage, OOM, ...) restarts it from the latest
# checkpoint, up to MAX_RESTARTS times. Run it detached in its own process group:
#
#   setsid nohup scripts/launch.sh runs/sawyer > /dev/null 2>&1 &
#   tail -f runs/sawyer/train.log                       # follow progress (also runs/sawyer/dashboard.png)
#   kill -- -"$(cat runs/sawyer/launcher.pid)"           # stop the launcher and the training it started
#
# Extra arguments after the output dir are passed to scripts/train.py.
set -u
cd "$(dirname "$0")/.."
OUT=${1:-runs/sawyer}
shift || true
MAX_RESTARTS=${MAX_RESTARTS:-20}
mkdir -p "$OUT"
echo $$ > "$OUT/launcher.pid"

for attempt in $(seq 0 "$MAX_RESTARTS"); do
    resume=()
    if compgen -G "$OUT/checkpoint-*" > /dev/null; then
        resume=(--resume latest)
    fi
    echo "[launcher] $(date -Is) attempt $attempt ${resume[*]}" >> "$OUT/train.log"
    .venv/bin/python scripts/train.py --output "$OUT" --workers 14 "${resume[@]}" "$@" >> "$OUT/train.log" 2>&1
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "[launcher] $(date -Is) training finished" >> "$OUT/train.log"
        exit 0
    fi
    echo "[launcher] $(date -Is) train.py exited with $status, restarting in 60 s" >> "$OUT/train.log"
    sleep 60
done
echo "[launcher] giving up after $MAX_RESTARTS restarts" >> "$OUT/train.log"
exit 1
