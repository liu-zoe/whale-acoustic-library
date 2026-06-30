#!/usr/bin/env python3
"""Train and apply a Perch 2.0 humpback-vs-vessel linear classifier.

Phase 2 of the project (D-020, task #18). The Multispecies model fires on
vessel noise at this hydrophone with high confidence (Mn ≈ 1.0 on confirmed
non-whale clips). Perch 2.0 gives us a 1536-dim embedding per 5 s window
trained on ~14,600 species — and the literature shows it transfers well to
cetacean tasks via linear probes on few-shot data. We have:

  - **20 user-rejected humpback clips** (negatives: confirmed not-humpback,
    mostly vessel/echo-sounder noise)
  - **1 user-kept humpback clip** (positive: confirmed humpback)
  - **64 Watkins Marine Mammal Sound Database humpback recordings**
    (positives: clean reference humpback)
  - **19 pending humpback clips** to evaluate against

Pipeline: load Perch 2.0 -> embed every 5 s window across the labeled set
-> train a regularized logistic regression on the frozen embeddings ->
score the unlabeled clips. Saves embeddings + classifier so re-running is
cheap.

    conda activate whales
    python src/perch_classifier.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
DB_PATH = Path("/media/y/hlabflash/whale_library/db/library.sqlite")
RAW_DIR = Path("/media/y/hlabflash/whale_library/audio_raw")
WATKINS_DIR = REPO / "testdata/humpback_whatkins/humpback_whale/sound"
ARTIFACTS = REPO / "models/perch_humpback_v0"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Perch 2.0 spec (from the model_configs source we inspected)
PERCH_SR = 32000
PERCH_WINDOW_S = 5.0


@dataclass
class Sample:
    """One source of audio that gets embedded into N (window) feature vectors."""
    name: str
    path: Path
    label: int   # 1 = humpback, 0 = not-humpback
    source: str  # 'watkins', 'review', or 'pending'


def load_samples() -> list[Sample]:
    """Collect labeled training samples + unlabeled pending clips to evaluate."""
    samples: list[Sample] = []
    c = sqlite3.connect(DB_PATH)
    # Positives from the user's review (just the 1 keep)
    for cid, in c.execute(
        "SELECT clip_id FROM clips WHERE species='humpback' AND review_status='keep'"):
        samples.append(Sample(cid, RAW_DIR / f"{cid}.wav", 1, "review"))
    # Negatives from the user's review (the 20 rejects)
    for cid, in c.execute(
        "SELECT clip_id FROM clips WHERE species='humpback' AND review_status='reject'"):
        samples.append(Sample(cid, RAW_DIR / f"{cid}.wav", 0, "review"))
    # Pending humpback clips — held out, scored after training
    pending = [r[0] for r in c.execute(
        "SELECT clip_id FROM clips WHERE species='humpback' "
        "AND review_status IN ('pending','uncertain')")]
    c.close()
    # Watkins positives (clean reference humpback recordings)
    for w in sorted(WATKINS_DIR.glob("*.wav")):
        samples.append(Sample(w.stem, w, 1, "watkins"))
    return samples, pending


def embed_audio(model, audio: np.ndarray, sr: int) -> np.ndarray:
    """Return (n_windows, embedding_dim) for one audio sample."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != PERCH_SR:
        n_out = int(round(len(audio) * PERCH_SR / sr))
        audio = scipy.signal.resample(audio, n_out).astype(np.float32)
    # Perch frames internally; .embed handles short and long clips.
    out = model.embed(audio.astype(np.float32))
    return np.squeeze(out.embeddings, axis=1)  # (windows, dim)


def build_embeddings(model, samples: list[Sample]) -> dict:
    """Embed every sample. Returns dict ready to write as .npz."""
    rows = []
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        try:
            audio, sr = sf.read(s.path, dtype="float32", always_2d=False)
        except Exception as exc:
            print(f"  [{i}/{len(samples)}] SKIP {s.name}: {type(exc).__name__}")
            continue
        try:
            emb = embed_audio(model, audio, sr)
        except Exception as exc:
            print(f"  [{i}/{len(samples)}] FAIL {s.name}: {type(exc).__name__}: {exc}")
            continue
        rows.append(dict(name=s.name, label=s.label, source=s.source, embeddings=emb))
        if i % 10 == 0 or i == len(samples):
            print(f"  [{i}/{len(samples)}] embedded "
                  f"({time.time()-t0:.0f}s elapsed)")
    return rows


def to_xy(rows: Iterable[dict], *, pooling: str = "mean") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aggregate per-clip window embeddings into one feature vector per clip.

    For training a clip-level keep/reject classifier, mean-pooling the windows
    is the simplest baseline (concentrates whale-vs-not signal that spans the
    whole 30 s). We could also explode to per-window samples — that would
    triple-count Watkins clips but is more honest about within-clip variance.
    Sticking with mean-pool for v0 to match clip-level review labels.
    """
    X, y, names = [], [], []
    for r in rows:
        e = r["embeddings"]
        if e.shape[0] == 0:
            continue
        if pooling == "mean":
            X.append(e.mean(axis=0))
        elif pooling == "max":
            X.append(e.max(axis=0))
        else:
            raise ValueError(pooling)
        y.append(r["label"])
        names.append(r["name"])
    return np.asarray(X), np.asarray(y), names


def main() -> int:
    print("=== loading Perch 2.0 (CPU) ===")
    from perch_hoplite.zoo import model_configs
    model = model_configs.load_model_by_name("perch_v2_cpu")

    print("\n=== inventorying samples ===")
    samples, pending_ids = load_samples()
    pos = sum(s.label == 1 for s in samples)
    neg = sum(s.label == 0 for s in samples)
    print(f"  positives (humpback): {pos}  ({sum(1 for s in samples if s.label==1 and s.source=='watkins')} Watkins + "
          f"{sum(1 for s in samples if s.label==1 and s.source=='review')} user-keep)")
    print(f"  negatives (not humpback / vessel etc.): {neg}  (user-reject)")
    print(f"  unlabeled humpback clips to score: {len(pending_ids)}")

    cache = ARTIFACTS / "embeddings.npz"
    if cache.exists():
        print(f"\n=== loading cached embeddings: {cache.name} ===")
        npz = np.load(cache, allow_pickle=True)
        labeled_rows = list(npz["labeled_rows"])
        pending_rows = list(npz["pending_rows"])
    else:
        print("\n=== embedding labeled training set ===")
        labeled_rows = build_embeddings(model, samples)
        # Also embed the pending clips (held-out evaluation)
        print("\n=== embedding pending humpback clips (held out) ===")
        pending_samples = [
            Sample(cid, RAW_DIR / f"{cid}.wav", -1, "pending") for cid in pending_ids
        ]
        pending_rows = build_embeddings(model, pending_samples)
        np.savez_compressed(
            cache,
            labeled_rows=np.array(labeled_rows, dtype=object),
            pending_rows=np.array(pending_rows, dtype=object),
        )
        print(f"  cached -> {cache}")

    X_train, y_train, names_train = to_xy(labeled_rows)
    print(f"\n=== training set: {X_train.shape[0]} clips, "
          f"{X_train.shape[1]}-dim embeddings, "
          f"{(y_train==1).sum()} pos / {(y_train==0).sum()} neg ===")

    # --- Linear probe: regularized logistic regression ---
    # Strong L2 prior — tiny training set, high-dim features. Class weights
    # balance the 65:20 positive:negative skew so the negatives aren't drowned.
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=2000, random_state=0
    )
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    print(f"  training accuracy: {train_acc:.3f}")

    # --- Leave-one-out cross-validation on the user-reviewed clips ---
    # Watkins is an easy positive class (clean recordings); the *real* test is
    # whether the classifier can tell the user's 1 keep + 20 rejects apart.
    print("\n=== LOO CV on user-reviewed clips only (the real test) ===")
    review_idx = [i for i, r in enumerate(labeled_rows) if r["source"] == "review"]
    review_X = np.stack([X_train[i] for i in review_idx])
    review_y = np.array([y_train[i] for i in review_idx])
    review_names = [names_train[i] for i in review_idx]
    correct = 0
    for hold in range(len(review_idx)):
        mask = np.ones(len(review_idx), dtype=bool); mask[hold] = False
        # combine: all-but-one review + all watkins
        watkins_idx = [i for i, r in enumerate(labeled_rows) if r["source"] == "watkins"]
        Xtr = np.vstack([review_X[mask], X_train[watkins_idx]])
        ytr = np.concatenate([review_y[mask], np.ones(len(watkins_idx))])
        cv = LogisticRegression(C=0.1, class_weight="balanced",
                                max_iter=2000, random_state=0)
        cv.fit(Xtr, ytr)
        pred = cv.predict(review_X[hold:hold+1])[0]
        prob = cv.predict_proba(review_X[hold:hold+1])[0, 1]
        ok = "OK " if pred == review_y[hold] else "MISS"
        if pred == review_y[hold]:
            correct += 1
        truth = "humpback" if review_y[hold] == 1 else "reject"
        print(f"  {ok}  {review_names[hold][-22:]:22s}  truth={truth:8s}  P(humpback)={prob:.3f}")
    print(f"  LOO accuracy: {correct}/{len(review_idx)} = {correct/len(review_idx)*100:.0f}%")

    # --- Score the 19 unlabeled / pending humpback clips ---
    if pending_rows:
        print("\n=== scoring pending humpback clips with the full-trained classifier ===")
        X_pend, _, names_pend = to_xy(pending_rows)
        prob_pend = clf.predict_proba(X_pend)[:, 1]
        order = np.argsort(-prob_pend)
        # Also pull Multispecies score for comparison
        c = sqlite3.connect(DB_PATH); ms_score = {}
        for cid, ms in c.execute(
            "SELECT clip_id, multispecies_scores FROM clips WHERE species='humpback'"):
            if ms: ms_score[cid] = json.loads(ms).get("Mn", 0.0)
        c.close()
        print(f"  {'clip':25s} {'Perch P(hb)':>12s} {'MS Mn':>8s}  Perch verdict")
        verdicts = []
        for i in order:
            cid = names_pend[i]
            mn = ms_score.get(cid, float("nan"))
            ph = prob_pend[i]
            v = "humpback" if ph >= 0.5 else "not humpback"
            print(f"  {cid[-22:]:25s}  {ph:>10.3f}  {mn:>7.2f}   {v}")
            verdicts.append({"clip_id": cid, "perch_p_humpback": float(ph),
                             "ms_mn": float(mn), "verdict": v})
        (ARTIFACTS / "pending_predictions.json").write_text(
            json.dumps(verdicts, indent=2))
        print(f"\n  wrote predictions -> {ARTIFACTS/'pending_predictions.json'}")

    # --- Persist classifier ---
    import joblib
    joblib.dump(clf, ARTIFACTS / "humpback_classifier.joblib")
    print(f"\n  classifier saved -> {ARTIFACTS/'humpback_classifier.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
