#!/usr/bin/env python3
"""Surface candidate non-SRKW / non-humpback marine-mammal recordings in the
reject and curious piles via Perch 2.0 embedding similarity search.

Task #24. Limitation: porpoise echolocation lives at 110-150 kHz, above our
hydrophone's 24 kHz Nyquist; this method cannot find porpoises. It can find
in-band candidates — minke / fin / right whale (in-band low-frequency calls)
and the user's own curious-other-mammal seed clips.

Approach: build a "reference pool" combining
  - Watkins extras (minke, finback, N. Atlantic right whale)            -> 121 refs
  - User-confirmed humpback positives + Watkins humpback (D-027 cache)  -> 65 refs
  - Ford-Osborne SRKW catalogue (D-031 cache)                           -> 48 refs

Then for each *candidate* (every reviewed-reject + curious clip) compute the
mean-pooled Perch embedding and find its nearest neighbor across the entire
reference pool. If the nearest neighbor is a non-SRKW / non-humpback species
**and** the user hasn't yet tagged the clip as curious, surface it as a
candidate with the species hypothesis.

Outputs:
  - models/other_mammal_search_v0/candidates.json
  - prints the top candidates and proposes which to promote to `curious`.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
DB_PATH = Path("/media/y/hlabflash/whale_library/db/library.sqlite")
RAW_DIR = Path("/media/y/hlabflash/whale_library/audio_raw")
WATKINS_HB_DIR = REPO / "testdata/humpback_whatkins/humpback_whale/sound"
WATKINS_EXTRAS_DIR = REPO / "testdata/watkins_extras"
SRKW_REF_CACHE = REPO / "models/srkw_call_labeler_v0/reference_embeddings.npz"
HUMPBACK_CLF_CACHE = REPO / "models/perch_humpback_v0/embeddings.npz"
ARTIFACTS = REPO / "models/other_mammal_search_v0"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

PERCH_SR = 32000

# Promotion thresholds — we surface candidates whose nearest non-SRKW
# non-humpback reference has high enough similarity AND whose similarity to
# that reference clearly exceeds similarity to all SRKW + humpback refs.
PROMOTE_MIN_SIMILARITY = 0.30   # absolute minimum to even consider
PROMOTE_MARGIN = 0.05           # nearest 'other' must beat nearest SRKW/hb by this margin


@dataclass
class Ref:
    name: str
    species: str   # "minke", "finback", "n_atl_right", "humpback", "srkw"
    source: str    # "watkins_extras", "watkins_hb", "ford_osborne"


def collect_references(model):
    """Build and return (refs, embeddings_normalized) for the full pool."""
    refs: list[Ref] = []
    embeddings: list[np.ndarray] = []

    # SRKW Ford-Osborne references (cached from D-031)
    if SRKW_REF_CACHE.exists():
        npz = np.load(SRKW_REF_CACHE, allow_pickle=True)
        for r, e in zip(npz["refs"], npz["embeddings"]):
            refs.append(Ref(name=r["call_code"], species="srkw", source=r["source"]))
            embeddings.append(e)
        print(f"  + {sum(1 for r in refs if r.species=='srkw')} SRKW refs (cached)")

    # Humpback positives (Watkins + user-keep, cached from D-027)
    if HUMPBACK_CLF_CACHE.exists():
        npz = np.load(HUMPBACK_CLF_CACHE, allow_pickle=True)
        for r in npz["labeled_rows"]:
            if r["label"] != 1:
                continue
            refs.append(Ref(name=r["name"], species="humpback", source="watkins_hb"
                             if r["source"]=="watkins" else "user_review"))
            embeddings.append(r["embeddings"].mean(axis=0))
        print(f"  + {sum(1 for r in refs if r.species=='humpback')} humpback positives (cached)")

    # Watkins extras — embed fresh
    extras = sorted(WATKINS_EXTRAS_DIR.glob("**/*.wav"))
    print(f"  + embedding {len(extras)} Watkins extras (minke/finback/right)...")
    t0 = time.time()
    for i, wav_path in enumerate(extras, 1):
        species = wav_path.parent.parent.name  # e.g. "minke_whale"
        try:
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != PERCH_SR:
                audio = scipy.signal.resample(
                    audio, int(round(len(audio) * PERCH_SR / sr))).astype(np.float32)
            out = model.embed(audio.astype(np.float32))
            e = np.squeeze(out.embeddings, axis=1)
            if e.shape[0] == 0:
                continue
            refs.append(Ref(name=wav_path.stem, species=species, source="watkins_extras"))
            embeddings.append(e.mean(axis=0))
        except Exception as exc:
            print(f"    SKIP {wav_path.name}: {type(exc).__name__}: {exc}")
        if i % 20 == 0:
            print(f"    [{i}/{len(extras)}]  ({time.time()-t0:.0f}s)")
    print(f"  total references: {len(refs)} in {time.time()-t0:.0f}s")

    E = np.stack(embeddings).astype(np.float32)
    E_n = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return refs, E_n


def embed_candidate(model, wav_path) -> np.ndarray:
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != PERCH_SR:
        audio = scipy.signal.resample(
            audio, int(round(len(audio) * PERCH_SR / sr))).astype(np.float32)
    out = model.embed(audio.astype(np.float32))
    e = np.squeeze(out.embeddings, axis=1)
    return e.mean(axis=0) if e.shape[0] > 0 else np.zeros(1536, dtype=np.float32)


def main() -> int:
    print("=== loading Perch 2.0 ===")
    from perch_hoplite.zoo import model_configs
    model = model_configs.load_model_by_name("perch_v2_cpu")

    print("\n=== building reference pool ===")
    refs, E_ref = collect_references(model)
    by_species = Counter(r.species for r in refs)
    print(f"  species composition: {dict(by_species)}")

    # Candidate set: every reject + curious clip
    conn = sqlite3.connect(DB_PATH)
    cands = list(conn.execute(
        "SELECT clip_id, species, review_status, review_note, raw_wav_path "
        "FROM clips WHERE review_status IN ('reject','curious') ORDER BY start_unix"))
    conn.close()
    print(f"\n=== {len(cands)} candidate clips (reject + curious) ===")

    NON_TARGET = {"minke_whale", "finback_whale", "northern_right_whale"}

    t0 = time.time()
    results = []
    for i, (cid, species, status, note, raw_path) in enumerate(cands, 1):
        try:
            e = embed_candidate(model, raw_path)
        except Exception as exc:
            print(f"  SKIP {cid}: {type(exc).__name__}: {exc}")
            continue
        e_n = e / (np.linalg.norm(e) + 1e-12)
        sims = E_ref @ e_n

        # Top hit per species class
        best_per_species = {}
        for j, r in enumerate(refs):
            s = float(sims[j])
            if r.species not in best_per_species or s > best_per_species[r.species][0]:
                best_per_species[r.species] = (s, r.name)

        # Decide: is the top non-target match strong AND clearly above SRKW/humpback?
        srkw_top = best_per_species.get("srkw", (0.0, "-"))[0]
        hb_top = best_per_species.get("humpback", (0.0, "-"))[0]
        target_top = max(srkw_top, hb_top)
        non_target_hits = [(sp, *best_per_species[sp])
                           for sp in best_per_species if sp in NON_TARGET]
        if not non_target_hits:
            continue
        non_target_top_species, non_target_top_sim, non_target_top_name = \
            max(non_target_hits, key=lambda x: x[1])

        margin = non_target_top_sim - target_top
        propose_promote = (
            non_target_top_sim >= PROMOTE_MIN_SIMILARITY and
            margin >= PROMOTE_MARGIN and
            status != "curious"
        )
        results.append({
            "clip_id": cid, "current_species": species, "current_status": status,
            "note": note,
            "srkw_top_sim": srkw_top, "humpback_top_sim": hb_top,
            "non_target_top_species": non_target_top_species,
            "non_target_top_sim": non_target_top_sim,
            "non_target_top_name": non_target_top_name,
            "margin_over_target": margin,
            "propose_promote": propose_promote,
        })
        if i % 25 == 0 or i == len(cands):
            print(f"  [{i}/{len(cands)}]  ({time.time()-t0:.0f}s)")

    # Sort + report
    results.sort(key=lambda r: -r["non_target_top_sim"])
    print(f"\n=== top 25 by non-target-species similarity ===")
    print(f"  {'clip':22s} {'status':>9s}  {'non-target':>16s}  {'sim':>6s}  {'margin':>7s}  {'note':30s}")
    for r in results[:25]:
        marker = " ★" if r["propose_promote"] else "  "
        n = r["note"] or ""
        if len(n) > 28: n = n[:28] + ".."
        print(f"{marker}{r['clip_id'][-22:]:22s} {r['current_status']:>9s}  "
              f"{r['non_target_top_species']:>16s}  {r['non_target_top_sim']:>6.3f}  "
              f"{r['margin_over_target']:>+7.3f}  {n:30s}")

    promote = [r for r in results if r["propose_promote"]]
    print(f"\n  ★ proposes promoting {len(promote)} clip(s) to 'curious' "
          f"(sim>={PROMOTE_MIN_SIMILARITY}, margin>={PROMOTE_MARGIN})")
    (ARTIFACTS / "candidates.json").write_text(json.dumps(results, indent=2))
    print(f"  full results -> {ARTIFACTS/'candidates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
