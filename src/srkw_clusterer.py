#!/usr/bin/env python3
"""Cluster the 350 confirmed SRKW keeps in Perch embedding space.

Step 1 of the Plan-C/D pipeline: surface ~15 acoustic groups so the user
can label clusters (instead of individual clips), then train a Perch
classifier on those labels.

  pipeline:
    1. for each `keep` SRKW clip, embed the loudest 5 s window with Perch
    2. UMAP reduce to 15 dimensions  (preserves cluster structure)
    3. HDBSCAN cluster              (auto-picks K, marks unclusterable as noise)
    4. for each cluster, pick the representative clip nearest the centroid
       — that's what the user will listen to / label

  outputs:
    models/srkw_clusterer_v0/keep_embeddings.npz        (raw embeddings)
    models/srkw_clusterer_v0/clusters.json              (per-clip cluster id + 2D coords)
    models/srkw_clusterer_v0/cluster_summary.json       (per-cluster: size, representative, top members)
    site/data/clusters.json                             (compact version for the labeling UI)
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
DB = Path("/media/y/hlabflash/whale_library/db/library.sqlite")
ART = REPO / "models/srkw_clusterer_v0"
ART.mkdir(parents=True, exist_ok=True)

PERCH_SR = 32000
PERCH_WIN_S = 5.0
INBAND_LO_HZ = 500.0
INBAND_HI_HZ = 12000.0

# HDBSCAN: min_cluster_size of 8 lands ~15-20 clusters for ~350 points and
# flags genuinely-isolated calls as noise. Larger gives fewer/coarser clusters.
HDBSCAN_MIN_CLUSTER_SIZE = 8
HDBSCAN_MIN_SAMPLES = 3

# UMAP: 15 dims preserves structure enough for HDBSCAN to find groups while
# escaping curse-of-dimensionality. 2D coords also stored for the labeling UI.
UMAP_N_COMPONENTS = 15
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0    # tight clusters; HDBSCAN cares about local density


def _resample_to_perch(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != PERCH_SR:
        n_out = int(round(len(audio) * PERCH_SR / sr))
        audio = scipy.signal.resample(audio, n_out).astype(np.float32)
    return audio.astype(np.float32)


def embed_loudest(model, audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Return (embedding, window_start_s) for the loudest in-band 5 s window."""
    audio = _resample_to_perch(audio, sr)
    win = int(PERCH_WIN_S * PERCH_SR)
    if len(audio) <= win:
        out = model.embed(audio)
        e = np.squeeze(out.embeddings, axis=1)
        return (e[0] if e.shape[0] else np.zeros(1536, np.float32), 0.0)
    hop = int(1.0 * PERCH_SR)
    starts = list(range(0, len(audio) - win + 1, hop))
    energies = []
    for s in starts:
        seg = audio[s:s + win]
        f, _, Z = scipy.signal.stft(seg, fs=PERCH_SR, nperseg=1024, noverlap=512)
        band = (f >= INBAND_LO_HZ) & (f <= INBAND_HI_HZ)
        energies.append(float(np.sum(np.abs(Z[band, :]) ** 2)))
    best = int(np.argmax(energies))
    seg = audio[starts[best]:starts[best] + win]
    out = model.embed(seg)
    e = np.squeeze(out.embeddings, axis=1)
    return (e[0] if e.shape[0] else np.zeros(1536, np.float32),
            starts[best] / PERCH_SR)


def build_embeddings(model, rows: list[tuple]) -> tuple[np.ndarray, list[dict]]:
    cache = ART / "keep_embeddings.npz"
    if cache.exists():
        print(f"  loading cached embeddings: {cache.name}")
        npz = np.load(cache, allow_pickle=True)
        return npz["embeddings"], list(npz["meta"])

    print(f"  embedding {len(rows)} clips (loudest-window pooling)...")
    t0 = time.time()
    E = np.zeros((len(rows), 1536), dtype=np.float32)
    meta = []
    for i, (cid, t, rp, conf, segs) in enumerate(rows):
        try:
            audio, sr = sf.read(rp, dtype="float32", always_2d=False)
            e, win_s = embed_loudest(model, audio, sr)
            E[i] = e
            meta.append({"clip_id": cid, "start_utc": t, "raw_path": rp,
                         "peak_confidence": conf, "n_segments": segs,
                         "window_start_s": win_s})
        except Exception as exc:
            print(f"    SKIP {cid}: {type(exc).__name__}: {exc}")
            meta.append({"clip_id": cid, "start_utc": t, "raw_path": rp,
                         "peak_confidence": conf, "n_segments": segs,
                         "window_start_s": None, "error": str(exc)})
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"    [{i+1}/{len(rows)}] {time.time()-t0:.0f}s")
    np.savez_compressed(cache, embeddings=E, meta=np.array(meta, dtype=object))
    return E, meta


def cluster_with_umap_hdbscan(E: np.ndarray):
    """Reduce + cluster. Returns (cluster_labels, coords_2d, coords_15d)."""
    import umap
    import hdbscan

    print(f"  UMAP -> {UMAP_N_COMPONENTS}D (high-dim for clustering)...")
    reducer_hi = umap.UMAP(
        n_components=UMAP_N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST, metric="cosine", random_state=42, verbose=False)
    coords_hi = reducer_hi.fit_transform(E)

    print(f"  UMAP -> 2D (for visualisation)...")
    reducer_2d = umap.UMAP(
        n_components=2, n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=0.1, metric="cosine", random_state=42, verbose=False)
    coords_2d = reducer_2d.fit_transform(E)

    print(f"  HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom")
    labels = clusterer.fit_predict(coords_hi)
    return labels, coords_2d, coords_hi


def pick_representative(cluster_idxs: np.ndarray, E_hi: np.ndarray,
                        meta: list[dict]) -> tuple[int, list[int]]:
    """Pick the clip nearest the cluster centroid, plus 4 next-nearest as
    'siblings' for the labeler to optionally play."""
    centroid = E_hi[cluster_idxs].mean(axis=0)
    dists = np.linalg.norm(E_hi[cluster_idxs] - centroid, axis=1)
    order = np.argsort(dists)
    rep = int(cluster_idxs[order[0]])
    siblings = [int(cluster_idxs[i]) for i in order[1:5]]
    return rep, siblings


def main() -> int:
    print("=== loading Perch 2.0 ===")
    from perch_hoplite.zoo import model_configs
    model = model_configs.load_model_by_name("perch_v2_cpu")

    print("\n=== loading 'keep' SRKW clips ===")
    conn = sqlite3.connect(DB)
    rows = list(conn.execute("""
        SELECT clip_id, start_utc_iso, raw_wav_path, peak_confidence, n_segments
        FROM clips
        WHERE species='SRKW' AND review_status='keep'
        ORDER BY start_unix"""))
    conn.close()
    print(f"  {len(rows)} clips to cluster")

    print("\n=== building embeddings ===")
    E, meta = build_embeddings(model, rows)

    print("\n=== clustering ===")
    labels, coords_2d, coords_hi = cluster_with_umap_hdbscan(E)
    unique = sorted(set(labels))
    n_noise = int((labels == -1).sum())
    n_clusters = sum(1 for u in unique if u != -1)
    print(f"  {n_clusters} clusters + {n_noise} noise points (out of {len(labels)})")

    # Per-cluster summary
    clusters_summary = []
    for cid in unique:
        if cid == -1:
            continue
        idxs = np.where(labels == cid)[0]
        rep, siblings = pick_representative(idxs, coords_hi, meta)
        clusters_summary.append({
            "cluster_id": int(cid),
            "size": int(len(idxs)),
            "representative_idx": rep,
            "representative_clip_id": meta[rep]["clip_id"],
            "sibling_clip_ids": [meta[i]["clip_id"] for i in siblings],
            "member_clip_ids": [meta[i]["clip_id"] for i in idxs.tolist()],
            "member_count": int(len(idxs)),
        })
    clusters_summary.sort(key=lambda c: -c["size"])

    print(f"\n=== cluster sizes ===")
    for c in clusters_summary:
        print(f"  cluster {c['cluster_id']}: {c['size']:>3d} members  "
              f"-> rep {c['representative_clip_id']}")

    # Persist all artifacts
    full = {"meta": meta,
            "cluster_labels": labels.tolist(),
            "coords_2d": coords_2d.tolist(),
            "params": {
                "umap_n_components": UMAP_N_COMPONENTS,
                "umap_n_neighbors": UMAP_N_NEIGHBORS,
                "umap_min_dist": UMAP_MIN_DIST,
                "hdbscan_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
                "hdbscan_min_samples": HDBSCAN_MIN_SAMPLES,
            }}
    (ART / "clusters.json").write_text(json.dumps(full, indent=2, default=str))
    (ART / "cluster_summary.json").write_text(json.dumps(clusters_summary, indent=2))

    # Compact version for the labeling UI under site/
    site_data = REPO / "site/data"
    site_data.mkdir(parents=True, exist_ok=True)
    ui = {"n_clusters": n_clusters, "n_noise": n_noise,
          "n_keeps": len(meta), "clusters": clusters_summary,
          "coords_2d": coords_2d.tolist(),
          "cluster_labels": labels.tolist(),
          "clip_ids": [m["clip_id"] for m in meta]}
    (site_data / "clusters.json").write_text(json.dumps(ui, default=str))
    print(f"\n  artifacts -> {ART}")
    print(f"  ui-data   -> {site_data/'clusters.json'} ({(site_data/'clusters.json').stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
