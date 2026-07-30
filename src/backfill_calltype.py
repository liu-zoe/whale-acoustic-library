#!/usr/bin/env python3
"""Back-fill the D-044 supervised call-type prediction onto every SRKW clip
already in the catalog. One-off — for future clips run_batch.py handles this
automatically via the extended PerchService.annotate.

Also runs init_schema on connect so the three new columns
(`perch_predicted_calltype`, `perch_calltype_confidence`,
`cross_node_unvalidated`) exist before the UPDATEs.

Resumable: skips clips that already have a non-null perch_predicted_calltype
so re-running after an interruption just picks up where it left off.

    conda activate whales
    python src/backfill_calltype.py               # backfill all pending
    python src/backfill_calltype.py --force       # re-score every SRKW clip
    python src/backfill_calltype.py --limit 5     # dry-run on 5 clips
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot import catalog
from pilot.perch_service import PerchService

log = logging.getLogger("backfill_calltype")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-score clips that already have a prediction")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process this many clips (for testing)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    conn = catalog.get_conn()
    catalog.init_schema(conn)   # ensures the 3 new columns exist

    # Pick target clips. cross_node_unvalidated defaults to 0 in the schema
    # so all existing rows (all orcasound_lab) get 0 for free — we only need
    # to set it explicitly for future non-Lab nodes.
    where = "species='SRKW'"
    if not args.force:
        where += " AND perch_predicted_calltype IS NULL"
    sql = f"SELECT clip_id, raw_wav_path FROM {'clips'} WHERE {where}"
    if args.limit:
        sql += f" LIMIT {args.limit}"
    todo = conn.execute(sql).fetchall()
    n_all = conn.execute("SELECT COUNT(*) FROM clips WHERE species='SRKW'").fetchone()[0]
    log.info("SRKW clips total: %d; to score: %d (%s)",
             n_all, len(todo),
             "--force: all" if args.force else "missing predictions")
    if not todo:
        log.info("nothing to do")
        return 0

    svc = PerchService()

    # ---- Score them ----
    t0 = time.time()
    failures = 0
    for i, row in enumerate(todo, 1):
        cid, wav = row["clip_id"], row["raw_wav_path"]
        try:
            label, conf = svc.predict_calltype(wav)
        except Exception as exc:
            failures += 1
            log.warning("[%d/%d] %s: %s: %s", i, len(todo), cid,
                        type(exc).__name__, exc)
            continue
        conn.execute(
            "UPDATE clips SET perch_predicted_calltype=?, "
            "perch_calltype_confidence=? WHERE clip_id=?",
            (label, conf, cid),
        )
        if i % 20 == 0 or i == len(todo):
            conn.commit()
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (len(todo) - i)
            log.info("[%d/%d]  %s  ->  %-16s  P=%.2f  "
                     "(%.1fs/clip, ETA %.0f s)",
                     i, len(todo), cid[-22:], label, conf, rate, eta)
    conn.commit()

    # ---- Post-scoring distribution report ----
    log.info("done: %d scored, %d failed in %.0f s",
             len(todo) - failures, failures, time.time() - t0)
    log.info("current corpus distribution (SRKW clips, all hydrophones):")
    for row in conn.execute(
        "SELECT COALESCE(perch_predicted_calltype, '(null)') AS label, "
        "COUNT(*) AS n FROM clips WHERE species='SRKW' "
        "GROUP BY perch_predicted_calltype ORDER BY n DESC"):
        log.info("  %-20s  %d", row["label"], row["n"])
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
