#!/usr/bin/env python3
"""SRKW call-type labeler v2 — segment-level scoring with loudest-window pooling.

Differences from v1 (`srkw_call_labeler.py`):

  1. **Expanded reference pool.** Adds the BR2011Nora, CWR, and BR Scalls
     recordings harvested by `scripts/harvest_refs.py` on top of the 48
     Ford-Osborne + favorites references v1 used. Now ~hundreds of refs
     with per-call-type AND per-pod metadata (BR Scalls only — Nora and
     CWR aren't pod-tagged).

  2. **Loudest-window pooling.** v1 mean-pooled all 6 Perch windows from a
     30 s clip, diluting the 3-second call into 27 s of ambient. v2 picks
     the window with the highest call-band energy (500 Hz - 12 kHz) and
     uses only that window's 1536-dim embedding for nearest-neighbour
     matching. Also stores the chosen window's start time so we know
     WHERE in the 30 s the call is.

  3. **Per-window margin threshold.** Only assigns a label when the
     nearest reference's similarity beats the second-nearest by a margin.
     Clips that can't decide land on `call=None, similarity=<low>`,
     which the dashboard can display as "unknown" rather than confidently
     mislabelling.

Produces:
  - models/srkw_call_labeler_v2/reference_embeddings.npz   (cached)
  - models/srkw_call_labeler_v2/predictions.json
  - catalog updates: `nearest_ref_call`, `nearest_ref_pod`,
    `nearest_ref_similarity` (overwrites v1 values)

Re-runnable; cached refs make subsequent runs cheap.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
DB_PATH = Path("/media/y/hlabflash/whale_library/db/library.sqlite")
REF_ROOT = REPO / "testdata/srkw_reference"
ARTIFACTS = REPO / "models/srkw_call_labeler_v2"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

PERCH_SR = 32000
PERCH_WINDOW_S = 5.0
# In-band power is computed for the loudest-window selection. SRKW calls live
# mostly between 500 Hz and 12 kHz at the energy levels Perch listens to.
INBAND_LO_HZ = 500.0
INBAND_HI_HZ = 12000.0
# Margin threshold for accepting a label. If the top reference doesn't beat
# the runner-up by this much in cosine similarity, the label is suppressed.
MARGIN_THRESHOLD = 0.03

# Pod from individual prefix (BR Scalls): A** = J pod, C** = K pod, L** = L pod.
POD_FROM_INDIVIDUAL = {"A": "J", "C": "K", "L": "L"}

FO_PATTERN = re.compile(r"FO-S(\d+)(?:-([RJKL]+))?(?:-([a-z]))?\.flac")
FAVORITES_PATTERN = re.compile(r"([JKL])-pod-S(\d+)-stereo\.mp3")
NORA_PATTERN = re.compile(r"Nora S(\d+)(?:i+)?\.aiff", re.IGNORECASE)
CWR_PATTERN = re.compile(r"S(\d+)(i*)cwr\.wav", re.IGNORECASE)
# SFU filename: "S04__pods-J__seq00.mp3", "S02iii__pods-L__seq02.mp3"
# The first part is the full Ford code with sub-type letters (i, ii, iii)
# preserved — that's the key advantage of this source.
SFU_PATTERN = re.compile(
    r"S(\d+)(i*)__pods-([JKL,none]+)__seq(\d+)\.mp3", re.IGNORECASE)


@dataclass
class Reference:
    path: Path
    call_code: str        # "S01", "S33", "S02i", ...
    pods: str             # "J" / "K" / "L" / "R" / ""
    source: str           # "ford_osborne" / "favorites" / "nora" / "cwr" / "br_scalls"


def collect_references() -> list[Reference]:
    refs: list[Reference] = []

    # Ford-Osborne (existing)
    for f in sorted((REF_ROOT / "ford_osborne/flac").glob("*.flac")):
        m = FO_PATTERN.match(f.name)
        if m:
            refs.append(Reference(f, f"S{int(m.group(1)):02d}",
                                  m.group(2) or "", "ford_osborne"))

    # Pod favorites (existing)
    for f in sorted((REF_ROOT / "favorites").glob("*.mp3")):
        m = FAVORITES_PATTERN.match(f.name)
        if m:
            refs.append(Reference(f, f"S{int(m.group(2)):02d}", m.group(1),
                                  "favorites"))

    # Nora (new)
    for f in sorted((REF_ROOT / "nora").glob("*.aiff")):
        m = NORA_PATTERN.match(f.name)
        if m:
            refs.append(Reference(f, f"S{int(m.group(1)):02d}", "",
                                  "nora"))

    # CWR (new) — keeps the i/iii subtype tag in the call code
    for f in sorted((REF_ROOT / "cwr").glob("*.wav")):
        m = CWR_PATTERN.match(f.name)
        if m:
            code = f"S{int(m.group(1)):02d}{m.group(2).lower()}"
            refs.append(Reference(f, code, "", "cwr"))

    # SFU library (123 takes, 30 distinct call types, sub-types preserved).
    # This is the only source with proper S02i / S02ii / S02iii / S08i / S08ii
    # / S37i / S37ii distinctions.
    sfu_dir = REF_ROOT / "sfu"
    if sfu_dir.exists():
        for f in sorted(sfu_dir.glob("*.mp3")):
            m = SFU_PATTERN.match(f.name)
            if not m:
                continue
            num = int(m.group(1))
            subtype = m.group(2).lower()          # "" / "i" / "ii" / "iii"
            pods_part = m.group(3).replace("none", "")
            # Comma-separated pods become a single string "JL" etc. for vote weight
            pods_clean = "".join(p for p in pods_part if p in "JKL")
            code = f"S{num:02d}{subtype}"
            refs.append(Reference(f, code, pods_clean, "sfu"))

    # BR Scalls (new) — filename pattern {date}_{individual}_{call_type}_{orig}.wav
    br_dir = REF_ROOT / "br_scalls"
    if br_dir.exists():
        for f in sorted(br_dir.glob("*.wav")):
            parts = f.name.split("_", 3)
            if len(parts) < 4:
                continue
            date, individual, ct, _ = parts
            pod = POD_FROM_INDIVIDUAL.get(individual[:1].upper(), "")
            # Normalise call type — Ford codes vary; "Unk" stays as-is
            if re.match(r"S\d", ct):
                # e.g. "S1" -> "S01"
                num = re.match(r"S(\d+)", ct)
                if num:
                    ct = f"S{int(num.group(1)):02d}"
            refs.append(Reference(f, ct, pod, "br_scalls"))

    return refs


def _resample_to_perch(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != PERCH_SR:
        n_out = int(round(len(audio) * PERCH_SR / sr))
        audio = scipy.signal.resample(audio, n_out).astype(np.float32)
    return audio.astype(np.float32)


def _inband_window_energies(audio: np.ndarray, hop_samples: int,
                            win_samples: int) -> np.ndarray:
    """Energy in [INBAND_LO_HZ, INBAND_HI_HZ] for each non-overlapping window."""
    # Quick STFT then sum band power per window
    f, t, Z = scipy.signal.stft(audio, fs=PERCH_SR, nperseg=2048,
                                noverlap=1536)
    band = (f >= INBAND_LO_HZ) & (f <= INBAND_HI_HZ)
    frame_power = np.sum(np.abs(Z[band, :]) ** 2, axis=0)
    # Map STFT frames -> Perch windows
    hop_stft_per_perch = int(round(hop_samples / 512))   # nperseg-noverlap=512
    n_windows = max(1, len(audio) // hop_samples)
    energies = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        f0 = i * hop_stft_per_perch
        f1 = min(len(frame_power), f0 + hop_stft_per_perch)
        if f1 > f0:
            energies[i] = frame_power[f0:f1].mean()
    return energies


def embed_loudest_window(model, audio: np.ndarray, sr: int) -> tuple[np.ndarray, float, float]:
    """Embed the loudest in-band window of the clip; return (emb, win_start_s, win_power).

    For clips shorter than one Perch window this falls back to embedding the
    whole clip mean-pooled (degenerate; same as v1 behaviour for short clips).
    """
    audio = _resample_to_perch(audio, sr)
    win = int(PERCH_WINDOW_S * PERCH_SR)
    if len(audio) <= win:
        out = model.embed(audio)
        e = np.squeeze(out.embeddings, axis=1)
        return (e[0] if e.shape[0] else np.zeros(1536, dtype=np.float32),
                0.0, float(np.mean(audio ** 2)))

    # Slide 5 s windows with 1 s hop, find the loudest in-band one
    hop = int(1.0 * PERCH_SR)
    starts = list(range(0, len(audio) - win + 1, hop))
    energies = []
    for s in starts:
        seg = audio[s:s + win]
        # power in the in-band frequency range (cheap: STFT one segment)
        f, _, Z = scipy.signal.stft(seg, fs=PERCH_SR, nperseg=1024, noverlap=512)
        band = (f >= INBAND_LO_HZ) & (f <= INBAND_HI_HZ)
        energies.append(float(np.sum(np.abs(Z[band, :]) ** 2)))
    best_i = int(np.argmax(energies))
    seg = audio[starts[best_i]:starts[best_i] + win]
    out = model.embed(seg)
    e = np.squeeze(out.embeddings, axis=1)
    return (e[0] if e.shape[0] else np.zeros(1536, dtype=np.float32),
            starts[best_i] / PERCH_SR, energies[best_i])


def build_reference_embeddings(model, refs: list[Reference]) -> np.ndarray:
    """Embed each reference using loudest-window pooling. Cached on disk."""
    cache = ARTIFACTS / "reference_embeddings.npz"
    if cache.exists():
        print(f"  loading cached reference embeddings: {cache.name}")
        npz = np.load(cache, allow_pickle=True)
        return list(npz["refs"]), npz["embeddings"]
    print(f"  embedding {len(refs)} reference clips (loudest-window per ref)...")
    t0 = time.time()
    E = np.zeros((len(refs), 1536), dtype=np.float32)
    ok = []
    for i, r in enumerate(refs):
        try:
            audio, sr = sf.read(r.path, dtype="float32", always_2d=False)
            e, _, _ = embed_loudest_window(model, audio, sr)
            E[i] = e
            ok.append(True)
        except Exception as exc:
            print(f"    SKIP {r.path.name}: {type(exc).__name__}: {exc}")
            ok.append(False)
        if (i + 1) % 25 == 0 or i + 1 == len(refs):
            print(f"    [{i+1}/{len(refs)}] {time.time()-t0:.0f}s")
    # Drop failures
    keep_mask = np.array(ok)
    E = E[keep_mask]
    refs_ok = [r for r, k in zip(refs, ok) if k]
    payload = np.array([{"path": str(r.path), "call_code": r.call_code,
                          "pods": r.pods, "source": r.source} for r in refs_ok],
                       dtype=object)
    np.savez_compressed(cache, refs=payload, embeddings=E)
    print(f"  cached -> {cache.name} ({len(refs_ok)} refs)")
    return list(payload), E


def label_one_clip(model, raw_path, ref_dicts, E_ref_n):
    """Run v2 labeling on one clip. Returns (call, pod, sim, window_start_s,
    margin) — the margin between best and runner-up similarity."""
    audio, sr = sf.read(raw_path, dtype="float32", always_2d=False)
    e, win_start, _ = embed_loudest_window(model, audio, sr)
    e_n = e / (np.linalg.norm(e) + 1e-12)
    sim = E_ref_n @ e_n
    order = np.argsort(-sim)
    top = int(order[0])
    second = int(order[1]) if len(order) > 1 else top
    margin = float(sim[top] - sim[second])
    best_call = ref_dicts[top]["call_code"]
    best_sim = float(sim[top])
    # Pod vote: top-5 single-pod-labeled refs, similarity-weighted
    votes: dict[str, float] = {}
    for idx in order[:5]:
        pods = ref_dicts[idx]["pods"]
        if pods in ("J", "K", "L"):
            votes[pods] = votes.get(pods, 0.0) + float(sim[idx])
    pod = max(votes, key=votes.get) if votes else "?"

    if margin < MARGIN_THRESHOLD:
        # Ambiguous — suppress the label (keep similarity for transparency).
        return (None, "?", best_sim, win_start, margin)
    return (best_call, pod, best_sim, win_start, margin)


def main() -> int:
    print("=== loading Perch 2.0 ===")
    from perch_hoplite.zoo import model_configs
    model = model_configs.load_model_by_name("perch_v2_cpu")

    print("\n=== collecting references (expanded pool) ===")
    refs = collect_references()
    from collections import Counter
    by_source = Counter(r.source for r in refs)
    by_call = Counter(r.call_code for r in refs)
    pod_counts = Counter()
    for r in refs:
        for p in r.pods:
            pod_counts[p] += 1
    print(f"  {len(refs)} references")
    print(f"  by source: {dict(by_source)}")
    print(f"  distinct call types: {len(by_call)}")
    print(f"  top 5 by ref-count: {by_call.most_common(5)}")
    print(f"  pod tags (multi-pod entries counted per-pod): {dict(pod_counts)}")

    print("\n=== building reference embeddings (loudest-window per ref) ===")
    ref_dicts, E_ref = build_reference_embeddings(model, refs)
    E_ref_n = E_ref / (np.linalg.norm(E_ref, axis=1, keepdims=True) + 1e-12)

    # --- Add new catalog columns for window-start + margin (v2 metadata) ---
    conn = sqlite3.connect(DB_PATH)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(clips)")}
    for col, typ in (("nearest_ref_window_start_s", "REAL"),
                     ("nearest_ref_margin", "REAL"),
                     ("nearest_ref_labeler_version", "TEXT")):
        if col not in cols:
            conn.execute(f"ALTER TABLE clips ADD COLUMN {col} {typ}")
    conn.commit()

    rows = list(conn.execute(
        "SELECT clip_id, raw_wav_path FROM clips WHERE species='SRKW' "
        "ORDER BY peak_confidence DESC"))
    print(f"\n=== labeling {len(rows)} SRKW clips with v2 ===")
    t0 = time.time()
    predictions = []
    n_suppressed = 0
    for i, (clip_id, raw_path) in enumerate(rows):
        try:
            call, pod, sim, win_s, margin = label_one_clip(
                model, raw_path, ref_dicts, E_ref_n)
        except Exception as exc:
            print(f"  SKIP {clip_id}: {type(exc).__name__}: {exc}")
            continue
        if call is None:
            n_suppressed += 1
        conn.execute(
            "UPDATE clips SET nearest_ref_call=?, nearest_ref_pod=?, "
            "nearest_ref_similarity=?, nearest_ref_window_start_s=?, "
            "nearest_ref_margin=?, nearest_ref_labeler_version=? "
            "WHERE clip_id=?",
            (call, pod, sim, win_s, margin, "v2", clip_id))
        predictions.append({"clip_id": clip_id, "nearest_call": call,
                            "pod": pod, "similarity": sim,
                            "window_start_s": win_s, "margin": margin})
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            conn.commit()
            print(f"  [{i+1}/{len(rows)}] {time.time()-t0:.0f}s elapsed, "
                  f"{n_suppressed} suppressed so far")
    conn.commit()

    # Stats
    from collections import Counter
    by_call = Counter(p["nearest_call"] or "(suppressed)" for p in predictions)
    by_pod = Counter(p["pod"] for p in predictions)
    print(f"\n=== v2 nearest-call distribution (top 12) ===")
    for code, n in by_call.most_common(12):
        print(f"  {code}: {n}")
    print(f"\n=== v2 nearest-pod distribution ===")
    for pod, n in by_pod.most_common():
        print(f"  {pod}: {n}")
    sims = [p["similarity"] for p in predictions]
    print(f"\n=== v2 similarity stats ===")
    print(f"  min={min(sims):.3f}  median={sorted(sims)[len(sims)//2]:.3f}  max={max(sims):.3f}")
    print(f"  ≥0.7: {sum(1 for s in sims if s>=0.7)}")
    print(f"  ≥0.5: {sum(1 for s in sims if s>=0.5)}")
    print(f"  ≥0.3: {sum(1 for s in sims if s>=0.3)}")

    (ARTIFACTS / "predictions.json").write_text(json.dumps(predictions, indent=2))
    print(f"\n  predictions -> {ARTIFACTS/'predictions.json'}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
