"""Run the detection pipeline over a multi-day batch, one day at a time.

run_pilot.py processes a single day and downloads everything up front; a
multi-day batch done that way would hold ~16 GB of scratch per day all at
once. This driver instead processes each day fully — download, detect, clip,
classify, catalog — then deletes that day's chunk WAVs before moving to the
next, so peak scratch stays at roughly one day (the project plan's iterative
batch design). OrcaHello and Multispecies are loaded once and reused.

    conda activate whales
    python src/run_batch.py --start 2025-07-15 --days 7

Per-day failures are logged and skipped; the batch continues to the next day.
A smoke run first is recommended:  --start 2025-07-15 --days 1 --smoke-chunks 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make the pilot package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).parent))

from pilot import config as C
from pilot import download as dl
from pilot import detect as det
from pilot import cluster as cl
from pilot import clip as clipmod
from pilot import multispecies_detect as ms_detect
from pilot import catalog as cat
from pilot.perch_service import PerchService
from pilot import crossref as xr
from pilot import ui as uimod

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s :: %(message)s"


def _day_window(date: datetime) -> tuple[float, float]:
    """UTC midnight-to-midnight Unix window for one calendar day."""
    start = date.replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def process_day(date, *, model, ms_clf, perch, acartia, args, log) -> dict:
    """Run the full pipeline for one calendar day. Returns a summary dict."""
    label = date.strftime("%Y-%m-%d")
    start_unix, end_unix = _day_window(date)

    # Phase 1: download
    log.info("[%s] listing HLS segments", label)
    segments = dl.list_segments(start_unix, end_unix)
    log.info("[%s] %d segments to download", label, len(segments))
    chunks = dl.download_chunks(segments, C.WAV_CHUNKS_DIR, limit=args.smoke_chunks)
    if not chunks:
        log.warning("[%s] no chunks downloaded; skipping day", label)
        return {"date": label, "chunks": 0, "events": 0, "clips": 0}

    # Phase 2: detect
    detections = det.detect_all(model, chunks)
    pos = sum(1 for d in detections if d.confidence > C.LOCAL_DETECTION_THRESHOLD)
    log.info("[%s] %d segments scored, %d positive", label, len(detections), pos)
    det_path = C.LOGS_DIR / f"detections_{label}.jsonl"
    with det_path.open("w") as f:
        for d in detections:
            f.write(json.dumps({
                "chunk_index": d.chunk_index,
                "chunk_wav_path": d.chunk_wav_path,
                "chunk_start_unix": d.chunk_start_unix,
                "seg_start_in_chunk_s": d.seg_start_in_chunk_s,
                "seg_duration_s": d.seg_duration_s,
                "confidence": d.confidence,
            }) + "\n")

    # Phase 3: cluster
    events = cl.cluster_detections(detections)
    log.info("[%s] %d detection events", label, len(events))

    # Phase 4: clip + denoise + spectrogram (OrcaHello / SRKW events)
    clips = clipmod.make_all_clips(events, chunks)
    log.info("[%s] materialized %d / %d SRKW clips", label, len(clips), len(events))

    # Phase 4b: Multispecies primary detection — scan the audio for humpback
    hb_clips = []
    hb_max_score = 0.0
    if args.multispecies_detect and ms_clf is not None:
        hb_dets = ms_detect.detect_all(ms_clf, chunks, hop_s=args.ms_hop)
        hb_max_score = max((d.confidence for d in hb_dets), default=0.0)
        n_hi = sum(1 for d in hb_dets if d.confidence >= args.ms_threshold)
        hb_events = cl.cluster_detections(hb_dets, threshold=args.ms_threshold)
        hb_clips = clipmod.make_all_clips(
            hb_events, chunks, tag="hb_", species="humpback"
        )
        log.info("[%s] Multispecies humpback scan: max score %.3f, "
                 "%d windows >= %.2f, %d events, %d clips",
                 label, hb_max_score, n_hi, args.ms_threshold,
                 len(hb_events), len(hb_clips))

    all_clips = clips + hb_clips

    # Phase 4.5: secondary species / call-type scoring on every clip
    multispecies_by_clip: dict[str, dict[str, float]] = {}
    if ms_clf is not None:
        for c in all_clips:
            try:
                multispecies_by_clip[c.clip_id] = ms_clf.score_wav(c.raw_wav_path)
            except Exception as exc:
                log.warning("[%s] Multispecies scoring failed for %s: %s",
                            label, c.clip_id, exc)

    # Phase 4.6: Perch annotations — humpback-vs-vessel filter (D-027) + SRKW
    # tentative call-type / pod via Ford-Osborne nearest neighbor (D-031). Per-
    # clip failures are caught inside annotate() so a single bad clip never
    # aborts the day.
    perch_by_clip: dict[str, object] = {}
    if perch is not None:
        # SRKW clips get nearest-ref labels; humpback clips get P(humpback);
        # PerchService.annotate routes by species. We log a single max/min
        # summary at end rather than per-clip noise.
        for c in clips:
            perch_by_clip[c.clip_id] = perch.annotate(c.clip_id, c.raw_wav_path, "SRKW")
        for c in hb_clips:
            perch_by_clip[c.clip_id] = perch.annotate(c.clip_id, c.raw_wav_path, "humpback")
        srkw_sims = [a.nearest_ref_similarity for a in perch_by_clip.values()
                     if a.nearest_ref_similarity is not None]
        hb_probs = [a.perch_p_humpback for a in perch_by_clip.values()
                    if a.perch_p_humpback is not None]
        if srkw_sims:
            log.info("[%s] Perch SRKW nearest-ref: %d clips, sim max %.2f / median %.2f",
                     label, len(srkw_sims), max(srkw_sims),
                     sorted(srkw_sims)[len(srkw_sims)//2])
        if hb_probs:
            n_pass = sum(1 for p in hb_probs if p >= 0.5)
            log.info("[%s] Perch humpback filter: %d clips, P(hb) max %.2f, "
                     "%d / %d would pass at 0.5 threshold",
                     label, len(hb_probs), max(hb_probs), n_pass, len(hb_probs))

    # Phase 5: catalog + Acartia cross-reference (one insert per detector)
    sightings = xr.count_sightings_for_clips(all_clips, acartia)
    conn = cat.get_conn()
    cat.init_schema(conn)
    cat.insert_clips(
        conn, clips,
        detection_model=C.HF_REPO_ID.split("/")[-1],
        detection_threshold=C.LOCAL_DETECTION_THRESHOLD,
        species="SRKW",
        sightings_by_clip_id=sightings,
        multispecies_by_clip_id=multispecies_by_clip,
        perch_by_clip_id=perch_by_clip,
    )
    if hb_clips:
        cat.insert_clips(
            conn, hb_clips,
            detection_model="multispecies-whale",
            detection_threshold=args.ms_threshold,
            species="humpback",
            sightings_by_clip_id=sightings,
            multispecies_by_clip_id=multispecies_by_clip,
            perch_by_clip_id=perch_by_clip,
        )
    conn.close()

    # Phase 7: drop this day's chunk WAVs before the next day
    if not args.keep_chunks:
        removed = 0
        for c in chunks:
            try:
                Path(c.wav_path).unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
        log.info("[%s] cleaned %d chunk WAVs from scratch", label, removed)

    return {
        "date": label, "chunks": len(chunks), "segments": len(detections),
        "positive": pos, "events": len(events), "clips": len(clips),
        "hb_clips": len(hb_clips), "hb_max_score": hb_max_score,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-07-15", help="first day, YYYY-MM-DD (UTC)")
    ap.add_argument("--days", type=int, default=7, help="number of consecutive days")
    ap.add_argument("--smoke-chunks", type=int, default=None,
                    help="if set, only download/process the first N chunks per day")
    ap.add_argument("--skip-multispecies", action="store_true",
                    help="skip the Multispecies secondary scoring phase")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="don't delete chunk WAVs between days (uses ~16 GB/day)")
    ap.add_argument("--multispecies-detect",
                    default=True,
                    action=argparse.BooleanOptionalAction,
                    help="scan audio with the Multispecies model for humpback "
                         "(default: on; pass --no-multispecies-detect to skip)")
    ap.add_argument("--ms-hop", type=float, default=C.MULTISPECIES_SCAN_HOP_S,
                    help="window hop (s) for the Multispecies primary scan")
    ap.add_argument("--ms-threshold", type=float, default=C.MULTISPECIES_DETECT_THRESHOLD,
                    help="humpback score threshold for the Multispecies scan")
    ap.add_argument("--perch",
                    default=True,
                    action=argparse.BooleanOptionalAction,
                    help="annotate clips with Perch 2.0 outputs — "
                         "humpback-vs-vessel P(hb) for humpback clips, "
                         "Ford-Osborne nearest-ref call/pod for SRKW clips "
                         "(default: on; pass --no-perch to skip)")
    args = ap.parse_args()

    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format=LOG_FORMAT,
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(C.LOGS_DIR / "batch_run.log")],
    )
    log = logging.getLogger("batch")

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    log.info("batch: %d day(s) from %s (UTC calendar days)", args.days, args.start)

    log.info("loading OrcaHello model")
    model = det.load_model()

    ms_clf = None
    if not args.skip_multispecies:
        try:
            from pilot.multispecies import MultispeciesClassifier
            log.info("loading Multispecies model")
            ms_clf = MultispeciesClassifier()
        except (FileNotFoundError, ImportError) as exc:
            log.warning("Multispecies unavailable; continuing without it: %s", exc)

    perch = None
    if args.perch:
        # PerchService is lazy — model + artifacts load on first use, so an
        # error here generally means the artifacts are missing on disk.
        try:
            perch = PerchService()
            log.info("Perch annotations enabled (--no-perch to skip)")
        except Exception as exc:
            log.warning("Perch unavailable; continuing without it: %s", exc)

    acartia = xr.load_acartia_near_lab()
    log.info("Acartia: %d SRKW sightings near lab loaded", len(acartia))

    summaries = []
    t0 = time.time()
    for i in range(args.days):
        date = start_date + timedelta(days=i)
        label = date.strftime("%Y-%m-%d")
        try:
            summaries.append(process_day(
                date, model=model, ms_clf=ms_clf, perch=perch,
                acartia=acartia, args=args, log=log,
            ))
        except Exception:
            log.exception("[%s] day failed; continuing to next day", label)
            summaries.append({"date": label, "error": True})
        log.info("[%s] day complete (%.0f min elapsed)", label, (time.time() - t0) / 60)

    # Render the review page once over the whole accumulated catalog.
    n_cards = uimod.render(C.REVIEW_DIR / "index.html")
    log.info("rendered review page: %d clips", n_cards)

    print()
    print("=" * 64)
    print(f"BATCH COMPLETE — {args.days} day(s) from {args.start} "
          f"in {(time.time() - t0) / 3600:.1f} h")
    for s in summaries:
        if s.get("error"):
            print(f"  {s['date']}  FAILED (see logs/batch_run.log)")
        else:
            print(f"  {s['date']}  chunks {s.get('chunks', 0):4d}  "
                  f"SRKW clips {s.get('clips', 0):3d}  "
                  f"humpback clips {s.get('hb_clips', 0):3d}  "
                  f"(max humpback score {s.get('hb_max_score', 0.0):.2f})")
    print(f"  catalog: {C.DB_PATH}  ({n_cards} clips total)")
    print(f"  review:  python src/review_server.py  ->  http://127.0.0.1:5000")
    print("=" * 64)


if __name__ == "__main__":
    main()
