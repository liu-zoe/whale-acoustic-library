#!/usr/bin/env bash
# BP Q3 fill-in for the 71 days lost when bp_q3 was OOM-killed on 2026-08-17
# while materializing 481 detection events on 2025-07-22.
#
# Days lost: 2025-07-22 through 2025-09-30 (71 days).
#
# DO NOT LAUNCH while the main expansion chain is still running — memory
# constraints prevent two concurrent pipelines. Wait until bin/run_expansion.sh
# has finished (or at least until no run_batch.py process exists). Check with:
#   pgrep -af run_batch.py || echo "safe to launch"
#
# Then:
#   nohup bash bin/bp_q3_fillin.sh > logs/bp_q3_fillin.out 2>&1 &
#   disown
#
# The mid-day-22 partial state in the DB is fine — INSERT OR REPLACE in
# pilot/catalog.py will overwrite any half-written 07-22 rows with clean data.

set -u
cd "$(dirname "$0")/.."
export PATH="/home/y/miniconda3/envs/whales/bin:$PATH"
mkdir -p logs

echo "=== BP Q3 fill-in starting at $(date -Iseconds) ==="
echo "Processing 2025-07-22..2025-09-30 (71 days) at rpi_bush_point"
echo "  log -> logs/batch_bp_q3_fillin.log"
echo

python src/run_batch.py \
    --start 2025-07-22 --days 71 --hydrophone-id rpi_bush_point \
    > logs/batch_bp_q3_fillin.log 2>&1
rc=$?

echo "=== BP Q3 fill-in end at $(date -Iseconds) — exit=$rc ==="
