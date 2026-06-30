#!/bin/bash
# Quick status check for the running pilot.
cd /home/y/whale_acoustic_library
PID=$(cat logs/pilot_run.pid 2>/dev/null)
echo "=== Pilot status ==="
if pgrep -f "run_pilot.py" > /dev/null; then
  echo "RUNNING (parent bash PID $PID)"
  pgrep -af "run_pilot.py"
else
  echo "NOT RUNNING (process exited)"
fi
echo
echo "=== Scratch buffer ==="
N=$(ls scratch/wav_chunks/ 2>/dev/null | wc -l)
echo "chunks downloaded: $N / 1439"
du -sh scratch/wav_chunks/ 2>/dev/null
echo
echo "=== Log tail ==="
tail -10 logs/pilot_run.log
