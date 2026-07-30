#!/usr/bin/env bash
# Snapshot of the expansion chain's state — safe to invoke at any time
# whether the chain is running, waiting, complete, or crashed.
#
#   bin/expansion_status.sh          # human-readable
#
# Prints, for each batch:
#   - present / missing log file
#   - if present: first + last INFO lines, and any FAILED lines
#   - DB clip counts per (hydrophone_location, quarter)
#
# Also lists any currently-running run_batch.py process.

set -u
cd "$(dirname "$0")/.."
LIB_DB="/media/y/hlabflash/whale_library/db/library.sqlite"
PYENV="/home/y/miniconda3/envs/whales/bin/python"

echo "==================================================================="
echo "Expansion status  $(date -Iseconds)"
echo "==================================================================="
echo
echo "-- Running processes --"
pgrep -af "run_batch.py|run_expansion.sh" || echo "  (none)"
echo
echo "-- Batch logs (bin/run_expansion.sh writes these) --"
declare -A BATCH_LABELS=(
    [q4_lab]="c) Q4 Orcasound Lab 2025-10..2025-12"
    [bp_q3]="Bush Point Q3 2025-07..2025-09"
    [bp_q4]="Bush Point Q4 2025-10..2025-12"
    [pt_q3]="Port Townsend Q3 2025-07..2025-09"
    [pt_q4]="Port Townsend Q4 2025-10..2025-12"
    [sb_q3]="Sunset Bay Q3 2025-07..2025-09"
    [sb_q4]="Sunset Bay Q4 2025-10..2025-12"
)
for tag in q4_lab bp_q3 bp_q4 pt_q3 pt_q4 sb_q3 sb_q4; do
    label="${BATCH_LABELS[$tag]}"
    log="logs/batch_${tag}.log"
    if [[ -f "$log" ]]; then
        first_batch=$(grep -m1 "batch:" "$log" | head -c 100)
        last_day=$(grep "day complete" "$log" | tail -1 | head -c 120)
        complete=$(grep -c "BATCH COMPLETE" "$log")
        failed=$(grep -c "FAILED\|day failed" "$log")
        echo "  [$tag] $label"
        echo "    log: $log ($(wc -l < "$log") lines, $(stat -c %s "$log") bytes)"
        [[ -n "$first_batch" ]] && echo "    start: $first_batch"
        [[ -n "$last_day"    ]] && echo "    last:  $last_day"
        [[ "$complete" -gt 0 ]] && echo "    STATUS: complete"
        [[ "$failed"   -gt 0 ]] && echo "    day-level failures: $failed"
    else
        echo "  [$tag] $label  --  (no log yet)"
    fi
done
echo
echo "-- DB clip counts by node × quarter --"
$PYENV - <<'PY'
import sqlite3
db = "/media/y/hlabflash/whale_library/db/library.sqlite"
c = sqlite3.connect(db)
sql = """
  SELECT hydrophone_location AS loc,
         substr(start_utc_iso, 1, 7) AS ym,
         species, review_status, COUNT(*) AS n
  FROM clips
  GROUP BY loc, ym, species, review_status
  ORDER BY loc, ym
"""
# Aggregate to (loc, quarter, species)
from collections import defaultdict
agg = defaultdict(lambda: {"keep":0, "reject":0, "uncertain":0, "pending":0})
for loc, ym, sp, rs, n in c.execute(sql):
    q = "Q3" if ym[5:7] in ("07","08","09") else ("Q4" if ym[5:7] in ("10","11","12") else "?")
    agg[(loc, q, sp)][rs] = agg[(loc, q, sp)].get(rs, 0) + n

hdr = f"  {'node':20s} {'Q':4s} {'sp':10s} {'keep':>6s} {'rej':>6s} {'unc':>6s} {'pend':>6s}"
print(hdr); print("  " + "-"*(len(hdr)-2))
for k in sorted(agg):
    (loc, q, sp) = k; d = agg[k]
    print(f"  {loc:20s} {q:4s} {sp:10s} {d['keep']:6d} {d['reject']:6d} "
          f"{d['uncertain']:6d} {d['pending']:6d}")

# Overall totals
total = c.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
srkw = c.execute("SELECT COUNT(*) FROM clips WHERE species='SRKW'").fetchone()[0]
labs = c.execute("SELECT COUNT(*) FROM clips WHERE hydrophone_location='orcasound_lab'").fetchone()[0]
others = total - labs
print(f"\n  totals: {total} clips ({srkw} SRKW, {total-srkw} humpback+other)")
print(f"          {labs} at Orcasound Lab, {others} at other nodes")
c.close()
PY
