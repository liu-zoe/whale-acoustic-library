#!/usr/bin/env python3
"""Positive-control test for the Multispecies humpback-detection path.

Runs the exact Phase 2b chain — MultispeciesClassifier.score_windows ->
multispecies_detect.detect_chunk -> cluster.cluster_detections — on known
humpback recordings (Watkins Marine Mammal Sound Database, in testdata/), to
prove the wiring fires when humpback is genuinely present. Orca clips from the
library serve as a negative control (should NOT register as humpback).

    conda activate whales
    python src/positive_control.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot.multispecies import MultispeciesClassifier  # noqa: E402
from pilot import multispecies_detect as msd  # noqa: E402
from pilot import cluster as cl  # noqa: E402
from pilot import config as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HUMPBACK_DIR = REPO / "testdata/humpback_whatkins/humpback_whale/sound"
ORCA_DIR = Path("/media/y/hlabflash/whale_library/audio_raw")


def run_one(clf, wav_path, label):
    """Return (duration_s, max_Mn_score, n_windows>=thr, n_events)."""
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    dur = len(audio) / float(sr)
    # the real detection path: fake chunk -> detect_chunk -> cluster_detections
    chunk = SimpleNamespace(wav_path=str(wav_path), index=0, start_unix=0.0, name=label)
    dets = msd.detect_chunk(clf, chunk)
    events = cl.cluster_detections(dets, threshold=C.MULTISPECIES_DETECT_THRESHOLD)
    max_mn = max((d.confidence for d in dets), default=0.0)
    n_hi = sum(1 for d in dets if d.confidence >= C.MULTISPECIES_DETECT_THRESHOLD)
    return dur, max_mn, n_hi, len(events)


def main() -> int:
    clf = MultispeciesClassifier()
    # sample a spread across the 128 Watkins recordings, not one session
    humpback = sorted(HUMPBACK_DIR.glob("*.wav"))[::8]
    orca = sorted(ORCA_DIR.glob("*.wav"))[:3]

    hdr = f"  {'file':28s} {'dur s':>7s} {'max Mn':>7s} {'win>=.5':>8s} {'events':>7s}"
    fired = 0
    print("POSITIVE CONTROL — known humpback recordings (Watkins database):")
    print(hdr)
    print("  " + "-" * 62)
    for w in humpback:
        try:
            dur, mx, nhi, nev = run_one(clf, w, w.stem)
        except Exception as exc:
            print(f"  {w.name:28s}  SKIPPED ({type(exc).__name__})")
            continue
        fired += nev > 0
        flag = "  <-- humpback detected" if nev > 0 else ""
        print(f"  {w.name:28s} {dur:7.1f} {mx:7.3f} {nhi:8d} {nev:7d}{flag}")

    print("\nNEGATIVE CONTROL — orca clips (should NOT register as humpback):")
    print(hdr)
    print("  " + "-" * 62)
    for w in orca:
        try:
            dur, mx, nhi, nev = run_one(clf, w, w.stem)
            print(f"  {w.name:28s} {dur:7.1f} {mx:7.3f} {nhi:8d} {nev:7d}")
        except Exception as exc:
            print(f"  {w.name:28s}  SKIPPED ({type(exc).__name__})")

    n = len(humpback)
    print("\n" + "=" * 66)
    print(f"RESULT: humpback detected in {fired} / {n} known humpback recordings")
    print("=" * 66)
    return 0 if fired > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
