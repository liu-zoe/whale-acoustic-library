"""Run the one-day Orcasound Lab pilot end-to-end.

Reads the pilot day metadata written by pick_pilot_day.py, downloads the day
in 60 s WAV chunks, runs OrcaHello, clusters detections into events, cuts 30 s
clips with denoise + spectrogram, writes the SQLite catalog, cross-references
Acartia, and renders a static review page.

Pass ``--smoke-chunks N`` to run on the first N chunks only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Make pilot package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).parent))

from pilot import config as C
from pilot import download as dl
from pilot import detect as det
from pilot import cluster as cl
from pilot import clip as clipmod
from pilot import catalog as cat
from pilot import crossref as xr
from pilot import ui as uimod


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s :: %(message)s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-chunks", type=int, default=None,
                    help="if set, only download/process the first N chunks")
    ap.add_argument("--skip-download", action="store_true",
                    help="reuse existing chunk WAVs and manifest")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="don't delete chunk WAVs after pipeline finishes")
    ap.add_argument("--skip-multispecies", action="store_true",
                    help="skip the Google Multispecies secondary scoring phase")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    log = logging.getLogger("pilot")

    pilot_meta = json.loads(C.PILOT_DAY_JSON.read_text())
    log.info("pilot day: %s, prefix: %s",
             pilot_meta["selected_day_utc"], pilot_meta["selected_session_s3_prefix"])
    folder_epoch = pilot_meta["selected_session_epoch"]
    # Window: from the folder epoch + audio offset for the next 24 h.
    start_unix = folder_epoch + 2.0  # FOLDER_TO_AUDIO_OFFSET
    end_unix = start_unix + 24 * 3600.0

    manifest_path = C.LOGS_DIR / "chunks_manifest.json"

    # Phase 1: download
    if args.skip_download and manifest_path.exists():
        chunks = dl.load_manifest(manifest_path)
        log.info("loaded %d chunks from manifest", len(chunks))
    else:
        log.info("listing HLS segments...")
        segments = dl.list_segments(start_unix, end_unix)
        log.info("got %d segments to download", len(segments))
        chunks = dl.download_chunks(
            segments, C.WAV_CHUNKS_DIR,
            limit=args.smoke_chunks,
        )
        dl.save_manifest(chunks, manifest_path)
        log.info("downloaded %d chunks; manifest at %s", len(chunks), manifest_path)

    # Phase 2: detection
    log.info("loading OrcaHello model")
    model = det.load_model()

    log.info("running detection on %d chunks", len(chunks))
    t0 = time.time()
    detections = det.detect_all(model, chunks)
    log.info("detection done in %.1f s; %d total segments, %d positive (>%.2f)",
             time.time() - t0, len(detections),
             sum(1 for d in detections if d.confidence > C.LOCAL_DETECTION_THRESHOLD),
             C.LOCAL_DETECTION_THRESHOLD)
    detections_path = C.LOGS_DIR / "detections.jsonl"
    with detections_path.open("w") as f:
        for d in detections:
            f.write(json.dumps({
                "chunk_index": d.chunk_index,
                "chunk_wav_path": d.chunk_wav_path,
                "chunk_start_unix": d.chunk_start_unix,
                "seg_start_in_chunk_s": d.seg_start_in_chunk_s,
                "seg_duration_s": d.seg_duration_s,
                "confidence": d.confidence,
            }) + "\n")
    log.info("wrote per-segment detections to %s", detections_path)

    # Phase 3: cluster
    events = cl.cluster_detections(detections)
    log.info("clustered into %d detection events", len(events))

    # Phase 4: clip + denoise + spectrograms
    clips = clipmod.make_all_clips(events, chunks)
    log.info("materialized %d / %d clips", len(clips), len(events))

    # Phase 4.5: secondary species / call-type scoring (Google Multispecies)
    multispecies_by_clip: dict[str, dict[str, float]] = {}
    if args.skip_multispecies:
        log.info("skipping Multispecies scoring (--skip-multispecies)")
    elif clips:
        try:
            from pilot.multispecies import MultispeciesClassifier
            log.info("loading Multispecies model")
            ms = MultispeciesClassifier()
            log.info("scoring %d clips with Multispecies", len(clips))
            for c in clips:
                multispecies_by_clip[c.clip_id] = ms.score_wav(c.raw_wav_path)
        except (FileNotFoundError, ImportError) as exc:
            # Missing weights or TensorFlow: log and continue OrcaHello-only.
            log.warning("Multispecies scoring skipped: %s", exc)

    # Phase 5: catalog + Acartia cross-ref
    near = xr.load_acartia_near_lab()
    log.info("Acartia: %d SRKW sightings within %.0f km of lab (any time)",
             len(near), C.ACARTIA_RADIUS_KM)
    sightings_by_clip = xr.count_sightings_for_clips(clips, near)

    conn = cat.get_conn()
    cat.init_schema(conn)
    cat.insert_clips(
        conn, clips,
        detection_model=C.HF_REPO_ID.split("/")[-1],
        detection_threshold=C.LOCAL_DETECTION_THRESHOLD,
        sightings_by_clip_id=sightings_by_clip,
        multispecies_by_clip_id=multispecies_by_clip,
    )
    conn.close()
    log.info("wrote %d rows to %s", len(clips), C.DB_PATH)

    # Phase 6: review page
    n = uimod.render(C.REVIEW_DIR / "index.html")
    log.info("rendered review page (%d cards) at %s",
             n, C.REVIEW_DIR / "index.html")

    # Phase 7: cleanup
    if not args.keep_chunks:
        for c in chunks:
            try:
                Path(c.wav_path).unlink(missing_ok=True)
            except Exception:
                pass
        log.info("cleaned up %d chunk WAVs from scratch", len(chunks))

    print()
    print("=" * 60)
    print(f"PILOT COMPLETE")
    print(f"  detections (segments > {C.LOCAL_DETECTION_THRESHOLD}): "
          f"{sum(1 for d in detections if d.confidence > C.LOCAL_DETECTION_THRESHOLD)}")
    print(f"  events: {len(events)}")
    print(f"  clips materialized: {len(clips)}")
    print(f"  Multispecies scored: {len(multispecies_by_clip)} / {len(clips)}")
    print(f"  DB: {C.DB_PATH}")
    print(f"  review HTML: {C.REVIEW_DIR / 'index.html'}")
    print(f"  start the review server: python src/review_server.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
