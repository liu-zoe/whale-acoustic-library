#!/usr/bin/env python3
"""Path C subset picker — 30 diverse high-quality clips with OrcaHello-centered focus.

Fixes the v1 picker's loudest-window-energy selector (which user testing
showed missed the real call on ~28% of clips because vessel noise outweighed
the call in raw energy). v2 instead asks OrcaHello — the model that DECIDED
each clip was an SRKW detection in the first place — where the actual call
is, by running per-3s-segment inference on the clip and picking the highest-
confidence segment as the call moment.

Outputs:
  site/labeling_test/data_v2.json
  site/labeling_test/audio_v2/{clip_id}_focus.flac     (5s centered on call)
  site/labeling_test/spectrograms_v2/{clip_id}_focus.png

Re-runnable with --batch-size to grow the subset later (default 30; on a
re-run with --batch-size 50, picks 20 more clips beyond what's already there).
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot import config as C
from pilot import detect as det

PERCH_SR = 32000
FOCUS_WIN_S = 5.0
CLIP_SR = 48000  # native rate of catalog WAVs
INBAND_LO_HZ = 500.0
INBAND_HI_HZ = 12000.0

REPO = Path(__file__).resolve().parents[1]
LIB = Path("/media/y/hlabflash/whale_library")
OUT = REPO / "site/labeling_test"
(OUT / "audio_v2").mkdir(parents=True, exist_ok=True)
(OUT / "spectrograms_v2").mkdir(parents=True, exist_ok=True)


def find_call_center(model, clip_wav_path: str) -> tuple[float, float]:
    """Use OrcaHello to locate the highest-confidence 3s window in the clip.

    Returns (center_time_s, peak_segment_confidence).
    """
    result = model.detect_srkw_from_file(str(clip_wav_path))
    if not result.segment_predictions:
        # Defensive fallback: middle of the clip
        return (15.0, 0.0)
    # Find the highest-confidence segment
    best = max(result.segment_predictions, key=lambda s: float(s.confidence))
    center = float(best.start_time_s) + float(best.duration_s) / 2.0
    return (center, float(best.confidence))


def extract_focused_window(audio_full: np.ndarray, sr: int, center_s: float,
                           win_s: float = FOCUS_WIN_S) -> np.ndarray:
    """Pull a `win_s` window centered on `center_s`, clamped to clip bounds."""
    half = win_s / 2.0
    start = max(0.0, center_s - half)
    end = min(len(audio_full) / sr, center_s + half)
    # If we hit a boundary, shift the window to maintain win_s where possible
    if end - start < win_s:
        if start <= 0:
            end = min(len(audio_full) / sr, start + win_s)
        else:
            start = max(0.0, end - win_s)
    i0 = int(round(start * sr))
    i1 = int(round(end * sr))
    return audio_full[i0:i1]


def pick_diverse(quality_idxs: list[int], embeddings_n: np.ndarray,
                 n_target: int, already_picked: set[int]) -> list[int]:
    """Farthest-point sampling: from `quality_idxs`, pick `n_target` clips
    most-different from each other and from already_picked."""
    selected = list(already_picked)
    pool = [i for i in quality_idxs if i not in already_picked]
    if not selected and pool:
        selected.append(pool.pop(0))
    while len(selected) - len(already_picked) < n_target and pool:
        best_i, best_min_d = None, -np.inf
        for ci in pool:
            sims = embeddings_n[ci] @ embeddings_n[selected].T
            min_d = float(1 - sims.max())
            if min_d > best_min_d:
                best_min_d, best_i = min_d, ci
        if best_i is None:
            break
        selected.append(best_i)
        pool.remove(best_i)
    return [i for i in selected if i not in already_picked]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-total", type=int, default=30,
                    help="total number of clips to end up with (default 30)")
    args = ap.parse_args()

    # ------------------------- candidate pool -------------------------
    npz = np.load(REPO / "models/srkw_clusterer_v0/keep_embeddings.npz",
                  allow_pickle=True)
    E = npz["embeddings"]
    meta = list(npz["meta"])
    clip_ids = [m["clip_id"] for m in meta]

    c = sqlite3.connect("/media/y/hlabflash/whale_library/db/library.sqlite")
    row_by_id = {r[0]: {"peak_confidence": r[1], "snr_db": r[2],
                         "n_segments": r[3]}
                 for r in c.execute(
                     "SELECT clip_id, peak_confidence, snr_db, n_segments "
                     "FROM clips WHERE species='SRKW'")}
    c.close()

    # Quality gate — looser than the original 7-clip picker because we need
    # a bigger candidate pool to draw 30 diverse picks from.
    keep_snrs = sorted(row_by_id[cid]["snr_db"] for cid in clip_ids
                       if cid in row_by_id)
    median_snr = keep_snrs[len(keep_snrs) // 2]
    print(f"  median SNR: {median_snr:.1f} dB (gate)")

    candidates = []
    for i, cid in enumerate(clip_ids):
        r = row_by_id.get(cid)
        if r and r["peak_confidence"] >= 0.85 and r["snr_db"] >= median_snr:
            candidates.append((i, r))
    print(f"  candidates passing gate (conf>=0.85 & SNR>={median_snr:.1f}): "
          f"{len(candidates)}")

    # Sort by composite quality so the seed of farthest-point sampling
    # starts from the best clip
    candidates.sort(key=lambda x: -(x[1]["peak_confidence"] *
                                     math.log(x[1]["n_segments"] + 1) *
                                     max(1.0, x[1]["snr_db"])))
    quality_idxs = [c[0] for c in candidates]

    # Already-picked: any clip already in audio_v2/
    existing = {p.stem.replace("_focus", "")
                for p in (OUT / "audio_v2").glob("*.flac")}
    already_idxs = {i for i, cid in enumerate(clip_ids) if cid in existing}
    print(f"  already picked (existing audio_v2 files): {len(already_idxs)}")
    n_needed = args.target_total - len(already_idxs)
    print(f"  need {n_needed} more for target_total={args.target_total}")
    if n_needed <= 0:
        print("  nothing to do; target already met")
        return 0

    E_n = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    new_picks = pick_diverse(quality_idxs, E_n, n_needed, already_idxs)
    print(f"  new picks: {len(new_picks)}")

    # ------------------------- OrcaHello load -------------------------
    print("\n  loading OrcaHello model...")
    model = det.load_model()
    print("  done.")

    # ------------------------- extract focused windows -------------------------
    # If there's an existing data_v2.json, load it; we'll append
    data_path = OUT / "data_v2.json"
    if data_path.exists():
        existing_data = {d["clip_id"]: d for d in json.loads(data_path.read_text())}
    else:
        existing_data = {}

    t0 = time.time()
    for n, idx in enumerate(new_picks, 1):
        m = meta[idx]; cid = m["clip_id"]
        raw_path = m["raw_path"] if Path(m["raw_path"]).is_absolute() else LIB / m["raw_path"]
        center_s, peak_seg_conf = find_call_center(model, raw_path)

        audio, sr = sf.read(raw_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        seg = extract_focused_window(audio, sr, center_s)

        # Resample to 32 kHz for FLAC (matches Perch ingestion convention)
        if sr != PERCH_SR:
            n_out = int(round(len(seg) * PERCH_SR / sr))
            seg32 = scipy.signal.resample(seg, n_out).astype(np.float32)
        else:
            seg32 = seg
        out_audio = OUT / "audio_v2" / f"{cid}_focus.flac"
        sf.write(out_audio, seg32, PERCH_SR, format="FLAC", subtype="PCM_16")

        # Focused spectrogram
        fig, ax = plt.subplots(figsize=(10, 4))
        f, t, Sxx = scipy.signal.spectrogram(
            seg, fs=sr, nperseg=1024, noverlap=768, scaling="spectrum")
        db = 10 * np.log10(Sxx + 1e-12); db -= db.max()
        ax.pcolormesh(t, f, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
        ax.set_ylim(0, 12000); ax.set_ylabel("Hz"); ax.set_xlabel("Time (s)")
        ax.set_title(f"{cid}  (5s centered on OrcaHello peak @ {center_s:.1f}s of "
                     f"the 30s clip, segment conf={peak_seg_conf:.2f})",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "spectrograms_v2" / f"{cid}_focus.png", dpi=110)
        plt.close(fig)

        r = row_by_id.get(cid, {})
        existing_data[cid] = {
            "clip_id": cid, "start_utc": m["start_utc"],
            "n_segments": r.get("n_segments"),
            "peak_confidence": r.get("peak_confidence"),
            "snr_db": r.get("snr_db"),
            "focused_audio": f"audio_v2/{cid}_focus.flac",
            "focused_spectrogram": f"spectrograms_v2/{cid}_focus.png",
            "orcahello_call_center_s": center_s,
            "orcahello_peak_segment_confidence": peak_seg_conf,
            "full_clip_audio_url": f"http://127.0.0.1:5000/files/audio_clean/{cid}.wav",
            "full_clip_spec_url": f"http://127.0.0.1:5000/files/spectrograms/{cid}.png",
        }
        if n % 5 == 0 or n == len(new_picks):
            print(f"    [{n}/{len(new_picks)}]  {cid[-22:]}  call@{center_s:.1f}s  "
                  f"seg-conf={peak_seg_conf:.2f}  ({time.time()-t0:.0f}s)")

    # Write data sorted by start_utc for stable ordering
    data_list = sorted(existing_data.values(), key=lambda d: d["start_utc"])
    data_path.write_text(json.dumps(data_list, indent=2))
    print(f"\n  data -> {data_path.relative_to(REPO)} ({len(data_list)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
