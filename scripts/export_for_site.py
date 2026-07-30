#!/usr/bin/env python3
"""Export catalog SQLite -> JSON files for the static companion site.

Writes three artifacts under site/data/:

  catalog.json   — every clip's compact metadata: id, time, species, status,
                   is_curious, conf, segs, snr, Multispecies summary, Perch
                   summary, Acartia count. Excludes per-clip Multispecies
                   12-class JSON to keep size down.
  timeline.json  — per-day aggregates for the Vega-Lite activity chart:
                   {day, keep, reject, uncertain, hb_keep, hb_other, total}.
  diurnal.json   — per-UTC-hour aggregate of keep clips, for the diurnal
                   chart.
  showcase.json  — the 15-clip Tier-B showcase pick (already produced by
                   the selector; this script copies it in case it's stale).

All paths in catalog.json are relative — `audio_raw/...`, `audio_clean/...`,
`spectrograms/...` — same as the sanitised data/library.sqlite.

Re-runnable, idempotent. Run after any catalog update:
    conda activate whales
    python scripts/export_for_site.py
"""
from __future__ import annotations
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Read from the sanitised shipped copy so relative paths are already in place.
# Falls back to the live DB if data/library.sqlite is missing.
DB_PRIMARY = REPO / "data/library.sqlite"
DB_FALLBACK = Path("/media/y/hlabflash/whale_library/db/library.sqlite")
OUT = REPO / "site/data"
OUT.mkdir(parents=True, exist_ok=True)

# Local paths on the build machine that we don't want leaking into the
# shipped catalog.json. Anything with these prefixes gets stripped to the
# relative form (audio_raw/... etc.) so a downstream user cloning the
# repo sees the same relative layout the shipped data/library.sqlite
# was originally sanitised with.
_LOCAL_PREFIXES = ("/media/y/hlabflash/whale_library/",)


def _relpath(p: str | None) -> str | None:
    if not p:
        return p
    for prefix in _LOCAL_PREFIXES:
        if p.startswith(prefix):
            return p[len(prefix):]
    return p


def _db():
    if DB_PRIMARY.exists():
        return sqlite3.connect(DB_PRIMARY), True
    return sqlite3.connect(DB_FALLBACK), False


def export_catalog(conn) -> int:
    """Compact per-clip metadata for the catalog browser + showcase cards."""
    rows = []
    # D-044 columns and cross_node_unvalidated are read via COALESCE so that
    # rows from a DB dumped before those columns existed just get sensible
    # defaults on the site.
    for r in conn.execute("""
        SELECT clip_id, hydrophone_location, start_utc_iso, species, call_type,
               review_status, COALESCE(is_curious, 0) AS is_curious, review_note,
               peak_confidence, n_segments, snr_db,
               multispecies_top, multispecies_top_score,
               perch_p_humpback,
               nearest_ref_call, nearest_ref_pod, nearest_ref_similarity,
               acartia_sightings_within_24h_50km,
               raw_wav_path, clean_wav_path, spectrogram_path,
               perch_predicted_calltype, perch_calltype_confidence,
               COALESCE(cross_node_unvalidated, 0) AS cross_node
        FROM clips ORDER BY start_utc_iso
    """):
        (cid, loc, t, sp, ct, rs, ic, note, conf, segs, snr,
         ms_top, ms_top_score, perch_hb,
         nr_call, nr_pod, nr_sim, ac, rp, cp, spp,
         pct_label, pct_conf, cross_node) = r
        rows.append({
            "clip_id": cid, "loc": loc, "t": t, "species": sp,
            "call_type": ct, "status": rs, "curious": bool(ic),
            "note": note, "conf": round(conf, 3), "segs": segs,
            "snr_db": round(snr, 2),
            "ms_top": ms_top,
            "ms_top_score": round(ms_top_score, 3) if ms_top_score is not None else None,
            "perch_p_humpback": round(perch_hb, 3) if perch_hb is not None else None,
            "ref_call": nr_call, "ref_pod": nr_pod,
            "ref_sim": round(nr_sim, 3) if nr_sim is not None else None,
            "acartia_24h_50km": ac,
            # Sanitised to relative paths so the shipped repo doesn't leak
            # the build-machine's absolute paths (see _LOCAL_PREFIXES).
            "raw_path": _relpath(rp), "clean_path": _relpath(cp),
            "spec_path": _relpath(spp),
            # D-044 supervised call-type prediction:
            # calltype in {S01, not-S01, unknown-calltype} or null pre-backfill
            "calltype": pct_label,
            "calltype_conf": round(pct_conf, 3) if pct_conf is not None else None,
            # D-044 shadow-mode flag: 1 for non-Lab nodes, 0 otherwise.
            # The site hides call-type predictions when this is 1.
            "cross_node": int(cross_node),
        })
    (OUT / "catalog.json").write_text(json.dumps(rows, separators=(",", ":")))
    return len(rows)


def export_timeline(conn) -> int:
    """Per-day stacked counts for the activity-explorer chart."""
    by_day: dict[str, dict] = {}
    # Stable across both species + status combinations
    for d, sp, st, n in conn.execute("""
        SELECT substr(start_utc_iso, 1, 10) AS d, species, review_status, count(*)
        FROM clips
        GROUP BY d, species, review_status
        ORDER BY d
    """):
        by_day.setdefault(d, {"day": d, "keep": 0, "reject": 0, "uncertain": 0,
                              "pending": 0, "hb_keep": 0, "hb_other": 0, "total": 0})
        if sp == "humpback":
            if st == "keep":
                by_day[d]["hb_keep"] += n
            else:
                by_day[d]["hb_other"] += n
        else:
            by_day[d][st] = by_day[d].get(st, 0) + n
        by_day[d]["total"] += n
    rows = list(by_day.values())
    (OUT / "timeline.json").write_text(json.dumps(rows, separators=(",", ":")))
    return len(rows)


def export_diurnal(conn) -> int:
    """Per-UTC-hour keep counts (SRKW only) for the diurnal chart."""
    hours = Counter()
    for ts, in conn.execute(
        "SELECT start_unix FROM clips WHERE species='SRKW' AND review_status='keep'"):
        # Catalog stores start_unix as a real; need to coerce
        from datetime import datetime, timezone
        h = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        hours[h] += 1
    rows = [{"hour_utc": h, "hour_pdt": (h - 7) % 24, "n_keep": hours.get(h, 0)}
            for h in range(24)]
    (OUT / "diurnal.json").write_text(json.dumps(rows, separators=(",", ":")))
    return sum(r["n_keep"] for r in rows)


def export_showcase(conn) -> int:
    """Tier-B showcase: 1 best-of-day clip per day with confirmed SRKW.

    Mirrors the selection logic the build originally used. Idempotent; will
    re-pick if review labels have changed since the last run.
    """
    import math
    days: dict[str, dict] = {}
    for r in conn.execute("""
        SELECT clip_id, substr(start_utc_iso, 1, 10) AS day, start_utc_iso,
               peak_confidence, n_segments, snr_db,
               nearest_ref_call, nearest_ref_pod, nearest_ref_similarity,
               acartia_sightings_within_24h_50km,
               perch_predicted_calltype, perch_calltype_confidence,
               COALESCE(cross_node_unvalidated, 0) AS cross_node
        FROM clips
        WHERE species='SRKW' AND review_status='keep'
          AND (is_curious IS NULL OR is_curious = 0)
    """):
        (cid, day, t, conf, segs, snr, cc, pod, sim, ac,
         pct_label, pct_conf, cross_node) = r
        score = conf * math.log(segs + 1)
        if day not in days or days[day]["score"] < score:
            days[day] = dict(
                clip_id=cid, day=day, start_utc=t,
                peak_confidence=round(conf, 3),
                n_segments=segs, snr_db=round(snr, 2),
                nearest_ref_call=cc, nearest_ref_pod=pod,
                nearest_ref_similarity=round(sim, 3) if sim is not None else None,
                acartia_sightings=ac, score=round(score, 3),
                # D-044 supervised prediction — only shown on the showcase
                # card when the model was confident enough to not abstain
                # AND the clip is at the home node (shadow-mode predictions
                # at non-Lab nodes stay hidden in the UI, see D-044).
                calltype=pct_label if not cross_node else None,
                calltype_conf=round(pct_conf, 3) if pct_conf is not None else None,
            )
    picks = sorted(days.values(), key=lambda r: r["day"])
    for p in picks:
        p.pop("score", None)
    (OUT / "showcase.json").write_text(json.dumps(picks, indent=2))
    return len(picks)


def main() -> int:
    conn, primary = _db()
    src = "data/library.sqlite (shipped)" if primary else "live DB"
    print(f"reading from: {src}")
    n_catalog = export_catalog(conn)
    n_timeline = export_timeline(conn)
    n_diurnal = export_diurnal(conn)
    n_showcase = export_showcase(conn)
    conn.close()
    print(f"  catalog.json:  {n_catalog} clips")
    print(f"  timeline.json: {n_timeline} days")
    print(f"  diurnal.json:  {n_diurnal} keep clips across 24 hours")
    print(f"  showcase.json: {n_showcase} clips")
    print(f"\nfiles in {OUT}:")
    for f in sorted(OUT.glob("*.json")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
