#!/usr/bin/env bash
# WP3: run teacher/track.py on each video as soon as its detect.done appears (CPU; runs alongside detect.py).
# usage (repo root): bash tools/wp3/track_loop.sh [id ...]   -> log in work/track.log
cd "$(dirname "$0")/../.."
IDS=${*:-"24SU_SPP 26BR_END 25SU_DIN 19SU_END 24SU_END 25BR_DIN 25BR_END 25SU_END 26BR_DIN"}
for id in $IDS; do
  until [ -f "work/$id/detect.done" ]; do
    [ -f PAUSE ] && { echo "$(date '+%F %T') PAUSE, stop" >> work/track.log; exit 3; }
    sleep 60
  done
  echo "$(date '+%F %T') track $id" >> work/track.log
  .venv/Scripts/python.exe teacher/track.py "$id" >> work/track.log 2>&1
done
echo "$(date '+%F %T') track loop finished" >> work/track.log
