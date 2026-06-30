#!/usr/bin/env python3
"""Tentative call-type (and pod, where derivable) labeling for SRKW clips
via Perch 2.0 nearest-neighbor lookup against the Ford-Osborne reference
catalogue.

Phase 2.5 of the project (task #26). The Ford-Osborne catalogue covers ~30
discrete SRKW call types (S01-S46). We embed each catalogue sample once
through Perch 2.0, then for every catalog SRKW clip in our library we find
the most similar reference embedding -> tentative call-type label.

Pod labeling caveat: only 16/45 Ford-Osborne samples have explicit pod tags
in their filenames (J:4, L:1, R:11 = mixed-resident). For pod inference we
also add the 3 Orcasound-favorites (J-S01, K-S16, L-S19) as one-shot pod
positives. Even with that, pod assignment is provisional — there are too
few labeled positives per pod for a confident classifier; this gives a
working hypothesis the human reviewer can validate or correct.

Outputs:
  - models/srkw_call_labeler_v0/reference_embeddings.npz  (cached)
  - models/srkw_call_labeler_v0/predictions.json
  - catalog columns:  nearest_ref_call, nearest_ref_pod, nearest_ref_similarity
"""
from __future__ import annotations

import json
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
RAW_DIR = Path("/media/y/hlabflash/whale_library/audio_raw")
REF_ROOT = REPO / "testdata/srkw_reference"
ARTIFACTS = REPO / "models/srkw_call_labeler_v0"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

PERCH_SR = 32000
PERCH_WINDOW_S = 5.0

FO_PATTERN = re.compile(r"FO-S(\d+)(?:-([RJKL]+))?(?:-([a-z]))?\.flac")
FAVORITES_PATTERN = re.compile(r"([JKL])-pod-S(\d+)-stereo\.mp3")


@dataclass
class Reference:
    path: Path
    call_code: str        # "S01", "S33", ...
    pods: str             # "J", "K", "L", "R", "JR", "" (unlabeled)
    source: str           # "ford_osborne" or "favorites"


def collect_references():
    refs: list[Reference] = []
    for f in sorted((REF_ROOT / "ford_osborne/flac").glob("*.flac")):
        m = FO_PATTERN.match(f.name)
        if m:
            refs.append(Reference(f, f"S{int(m.group(1)):02d}", m.group(2) or "", "ford_osborne"))
    for f in sorted((REF_ROOT / "favorites").glob("*.mp3")):
        m = FAVORITES_PATTERN.match(f.name)
        if m:
            refs.append(Reference(f, f"S{int(m.group(2)):02d}", m.group(1), "favorites"))
    return refs


def embed_audio(model, audio: np.ndarray, sr: int) -> np.ndarray:
    """Return mean-pooled Perch embedding for one audio clip."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != PERCH_SR:
        n_out = int(round(len(audio) * PERCH_SR / sr))
        audio = scipy.signal.resample(audio, n_out).astype(np.float32)
    out = model.embed(audio.astype(np.float32))
    e = np.squeeze(out.embeddings, axis=1)   # (windows, 1536)
    if e.shape[0] == 0:
        return np.zeros(1536, dtype=np.float32)
    return e.mean(axis=0)


def build_reference_embeddings(model, refs):
    cache = ARTIFACTS / "reference_embeddings.npz"
    if cache.exists():
        print(f"  loading cached reference embeddings: {cache.name}")
        npz = np.load(cache, allow_pickle=True)
        return list(npz["refs"]), npz["embeddings"]
    print(f"  embedding {len(refs)} reference clips...")
    t0 = time.time()
    E = np.zeros((len(refs), 1536), dtype=np.float32)
    for i, r in enumerate(refs):
        audio, sr = sf.read(r.path, dtype="float32", always_2d=False)
        E[i] = embed_audio(model, audio, sr)
        if (i+1) % 10 == 0 or i+1 == len(refs):
            print(f"    [{i+1}/{len(refs)}] {time.time()-t0:.0f}s")
    payload = np.array([{"path": str(r.path), "call_code": r.call_code,
                          "pods": r.pods, "source": r.source} for r in refs], dtype=object)
    np.savez_compressed(cache, refs=payload, embeddings=E)
    print(f"  cached -> {cache.name}")
    return list(payload), E


def nearest_pod(ref_dicts: list, sim: np.ndarray, k: int = 5) -> tuple[str, float]:
    """Return (best_pod, fraction_in_top_k) — voting among the top-k references
    that have a known single pod. Returns ('?', 0.0) if no pod-labeled refs in top-k."""
    order = np.argsort(-sim)
    votes: dict[str, float] = {}
    seen = 0
    for idx in order:
        if seen >= k:
            break
        seen += 1
        pods = ref_dicts[idx]["pods"]
        # Only count single-pod-labeled refs in the vote
        if pods in ("J", "K", "L"):
            votes[pods] = votes.get(pods, 0.0) + float(sim[idx])
    if not votes:
        return ("?", 0.0)
    best = max(votes, key=votes.get)
    return (best, votes[best] / sum(votes.values()))


def main() -> int:
    print("=== loading Perch 2.0 ===")
    from perch_hoplite.zoo import model_configs
    model = model_configs.load_model_by_name("perch_v2_cpu")

    print("\n=== collecting references ===")
    refs = collect_references()
    pod_counts = {}
    for r in refs:
        for p in r.pods:
            pod_counts[p] = pod_counts.get(p, 0) + 1
    print(f"  {len(refs)} references; pod tags (multi-pod entries counted per-pod): {pod_counts}")
    print(f"  distinct call types: {len(set(r.call_code for r in refs))}")

    print("\n=== building reference embeddings ===")
    ref_dicts, E_ref = build_reference_embeddings(model, refs)
    # Pre-normalize for cosine similarity
    E_ref_n = E_ref / (np.linalg.norm(E_ref, axis=1, keepdims=True) + 1e-12)

    # --- Pull catalog SRKW clips, embed each, find nearest neighbor ---
    conn = sqlite3.connect(DB_PATH)
    # Ensure columns exist
    cols = {row[1] for row in conn.execute("PRAGMA table_info(clips)")}
    for col, typ in (("nearest_ref_call", "TEXT"),
                     ("nearest_ref_pod", "TEXT"),
                     ("nearest_ref_similarity", "REAL")):
        if col not in cols:
            conn.execute(f"ALTER TABLE clips ADD COLUMN {col} {typ}")
    conn.commit()

    rows = list(conn.execute(
        "SELECT clip_id, raw_wav_path FROM clips WHERE species='SRKW' "
        "ORDER BY peak_confidence DESC"))
    print(f"\n=== labeling {len(rows)} SRKW clips via Perch nearest neighbor ===")
    predictions = []
    t0 = time.time()
    for i, (clip_id, raw_path) in enumerate(rows):
        try:
            audio, sr = sf.read(raw_path, dtype="float32", always_2d=False)
            e = embed_audio(model, audio, sr)
        except Exception as exc:
            print(f"  SKIP {clip_id}: {type(exc).__name__}: {exc}")
            continue
        e_n = e / (np.linalg.norm(e) + 1e-12)
        sim = E_ref_n @ e_n
        top = int(np.argmax(sim))
        best_call = ref_dicts[top]["call_code"]
        best_sim = float(sim[top])
        best_pod, pod_confidence = nearest_pod(ref_dicts, sim, k=5)
        conn.execute(
            "UPDATE clips SET nearest_ref_call=?, nearest_ref_pod=?, "
            "nearest_ref_similarity=? WHERE clip_id=?",
            (best_call, best_pod, best_sim, clip_id))
        predictions.append({"clip_id": clip_id, "nearest_call": best_call,
                            "similarity": best_sim, "pod": best_pod,
                            "pod_confidence": pod_confidence})
        if (i+1) % 50 == 0 or i+1 == len(rows):
            conn.commit()
            print(f"  [{i+1}/{len(rows)}]  ({time.time()-t0:.0f}s elapsed)")
    conn.commit()

    # Summary
    from collections import Counter
    by_call = Counter(p["nearest_call"] for p in predictions)
    by_pod = Counter(p["pod"] for p in predictions)
    print(f"\n=== nearest-call distribution (top 10) ===")
    for code, n in by_call.most_common(10):
        print(f"  {code}: {n}")
    print(f"\n=== nearest-pod distribution ===")
    for pod, n in by_pod.most_common():
        print(f"  {pod}: {n}")

    (ARTIFACTS / "predictions.json").write_text(json.dumps(predictions, indent=2))
    print(f"\n  predictions -> {ARTIFACTS/'predictions.json'}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
