#!/usr/bin/env bash
# BP Q4 fill-in for the 80 days lost when bp_q4 was OOM-killed on 2026-08-18
# processing 2025-10-12 (386 positive segments → 257 detection events).
#
# Days lost: 2025-10-12 through 2025-12-31 (80 days).
#
# WARNING: same OOM risk as the original run. Some Bush Point days produce
# >250 detection events, and current pipeline holds them all in memory
# during annotation. If this fill-in dies again mid-day, restart with the
# --start date bumped past the crash day.
#
# DO NOT LAUNCH while the main expansion chain is still running — memory
# constraints prevent two concurrent pipelines. Wait until bin/run_expansion.sh
# has finished (or at least until no run_batch.py process exists). Check with:
#   pgrep -af run_batch.py || echo "safe to launch"
#
# Then:
#   nohup bash bin/bp_q4_fillin.sh > logs/bp_q4_fillin.out 2>&1 &
#   disown

set -u
cd "$(dirname "$0")/.."
export PATH="/home/y/miniconda3/envs/whales/bin:$PATH"
mkdir -p logs

echo "=== BP Q4 fill-in starting at $(date -Iseconds) ==="
echo "Processing 2025-10-12..2025-12-31 (80 days) at rpi_bush_point"
echo "  log -> logs/batch_bp_q4_fillin.log"
echo

python src/run_batch.py \
    --start 2025-10-12 --days 80 --hydrophone-id rpi_bush_point \
    > logs/batch_bp_q4_fillin.log 2>&1
rc=$?

echo "=== BP Q4 fill-in end at $(date -Iseconds) — exit=$rc ==="
if [ "$rc" -eq 137 ]; then
    echo
    echo "!!! Exit 137 = SIGKILL, likely OOM on a high-volume day."
    echo "!!! Check logs/batch_bp_q4_fillin.log for the last 'day complete'"
    echo "!!! then relaunch with --start = the DAY AFTER that."
fi
