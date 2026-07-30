#!/usr/bin/env bash
# Sequential runner for the c) + expand phases: Q4 Orcasound Lab, then
# 2025 Q3 + Q4 at three additional Orcasound network nodes.
#
# Runs in series (not parallel) because the pipeline loads OrcaHello + Perch
# + Multispecies (~5-8 GB) per invocation and this machine can't fit two
# copies without OOM (backfill runs showed "allocation exceeds 10% of free
# system memory" warnings under a single load).
#
# Each batch is tolerant of per-day failures (run_batch.py catches them and
# continues to the next day), and this script is tolerant of per-batch
# failures (continues to the next batch, non-zero exit codes logged).
#
# Kick off with:
#   nohup bash bin/run_expansion.sh >logs/expansion.out 2>&1 &
#   disown
#
# Monitor with:
#   tail -f logs/expansion.out
#   ls -la logs/batch_*.log | tail
#
# Estimated total wall time: ~10-15 days (7 batches × ~1.5-2 days each based
# on prior Q3 Lab runs).

set -u  # unset vars are errors; but no -e, so a batch failure doesn't stop us
cd "$(dirname "$0")/.."
export PATH="/home/y/miniconda3/envs/whales/bin:$PATH"
mkdir -p logs

# Same argparse defaults as the previous Q3 Lab runs: multispecies detection
# on, Perch annotations on. Confidence gate + shadow-mode flag baked into
# PerchService / catalog per D-044.
COMMON_ARGS=""

run_batch() {
    local tag="$1"; local start="$2"; local days="$3"; local hid="$4"
    local log="logs/batch_${tag}.log"
    echo
    echo "==================================================================="
    echo "[$(date -Iseconds)] START $tag  start=$start days=$days hid=$hid"
    echo "  -> $log"
    echo "==================================================================="
    python src/run_batch.py \
        --start "$start" --days "$days" --hydrophone-id "$hid" \
        $COMMON_ARGS \
        > "$log" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] END   $tag  exit=$rc"
    return $rc
}

echo "=== Expansion chain starting at $(date -Iseconds) ==="

# c) Q4 Orcasound Lab — extends 2025 coverage from Q3-only to full H2
run_batch "q4_lab"     "2025-10-01" 92 "rpi_orcasound_lab"

# Expansion: three nodes, each running Q3 + Q4 back-to-back.
# Order chosen for scientific value: K/L pods (Bush Point, Port Townsend)
# first, then southern node (Sunset Bay).
run_batch "bp_q3"      "2025-07-01" 92 "rpi_bush_point"
run_batch "bp_q4"      "2025-10-01" 92 "rpi_bush_point"
run_batch "pt_q3"      "2025-07-01" 92 "rpi_port_townsend"
run_batch "pt_q4"      "2025-10-01" 92 "rpi_port_townsend"
run_batch "sb_q3"      "2025-07-01" 92 "rpi_sunset_bay"
run_batch "sb_q4"      "2025-10-01" 92 "rpi_sunset_bay"

echo
echo "=== Expansion chain complete at $(date -Iseconds) ==="
echo "Batch log summary:"
for log in logs/batch_*.log; do
    echo "  $log: $(wc -l < "$log") lines"
done
