#!/usr/bin/env python3
"""Pilot SRKW call-type classifier over Perch 2.0 focused-window embeddings.

Move 1 of the D-043 follow-up plan. D-040 showed Perch cannot separate the
30 Ford-Osborne SRKW call types by unsupervised similarity (SFU reference
silhouette 0.027). Path C v2 produced 50 human-labeled clips at ~60 s/clip
of active labeling time. Full-corpus labeling (350 clips) is calendar-
bottlenecked, not hour-bottlenecked, so before committing to more labeling
we test whether the 50 labels are already enough to bootstrap a call-type
classifier for at least the head codes.

Pipeline (mirrors src/perch_classifier.py for humpback-vs-vessel, D-027):
  1. Embed the 50 focused 5s FLACs (site/labeling_test/audio_v2/) with
     Perch 2.0 CPU. One 5s window per clip -> one 1536-dim vector.
  2. Parse labels into three progressively more ambitious class schemes:
       (a) S01 vs not-S01                 -- the safest binary
       (b) S01 vs S44 vs S17 vs other     -- head codes vs aggregate rest
       (c) S01 vs S44 vs S17 vs known-tail vs unk
                                          -- unk as a first-class label
  3. Regularized logistic regression + LOO CV under each scheme.
  4. Report per-scheme accuracy so the strategic decision (label more vs
     extend coverage) can be made from evidence, not speculation.

    conda activate whales
    python src/perch_calltype_classifier.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
LABELS_PATH = REPO / "data/manual_labels/path_c_v2_50.json"
AUDIO_DIR = REPO / "site/labeling_test/audio_v2"
ARTIFACTS = REPO / "models/perch_calltype_v0"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

PERCH_SR = 32000


# ----------------------------- label parsing ------------------------------

def parse_label(raw: str) -> dict:
    """Turn a free-text human label into a structured record.

    Returns keys:
      certain    -- bool, False if label ended in "?" or was blank
      code       -- primary Ford code (S01, S44, ...) with sub-type stripped
                    or None if unmapped/blank
      subtype    -- e.g. 'i', 'ii' if present, else None
      raw        -- original string
    """
    s = raw.strip()
    if not s:
        return {"certain": False, "code": None, "subtype": None, "raw": raw}
    # Strip trailing "?" for uncertainty flag
    certain = not s.endswith("?")
    s2 = s.rstrip("?").strip()
    # Strip "SFU " prefix
    s2 = re.sub(r"^SFU\s+", "", s2, flags=re.I)
    # First S-code wins (handles "S01 around 2s and S08i around 4s" ->
    # primary code S01, sub-type None). This is an oversimplification for
    # multi-call clips but only 1 clip in the training set has multiple
    # codes, so the impact is small.
    m = re.search(r"S(\d+)(i+|v)?", s2, re.I)
    if not m:
        return {"certain": False, "code": None, "subtype": None, "raw": raw}
    code = f"S{int(m.group(1)):02d}"
    subtype = m.group(2).lower() if m.group(2) else None
    return {"certain": certain, "code": code, "subtype": subtype, "raw": raw}


# ------------------------------ embedding --------------------------------

def embed_focused_clips(model, clip_ids: list[str]) -> np.ndarray:
    """One 5s window per clip -> (n_clips, 1536)."""
    rows = []
    t0 = time.time()
    for i, cid in enumerate(clip_ids, 1):
        p = AUDIO_DIR / f"{cid}_focus.flac"
        audio, sr = sf.read(p, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        assert sr == PERCH_SR, f"{p}: expected {PERCH_SR} Hz, got {sr}"
        out = model.embed(audio.astype(np.float32))
        emb = np.squeeze(out.embeddings, axis=1)  # (windows, dim)
        assert emb.shape[0] == 1, (
            f"{cid}: expected 1 window for 5s @ 32kHz, got {emb.shape[0]}")
        rows.append(emb[0])
        if i % 10 == 0 or i == len(clip_ids):
            print(f"  [{i}/{len(clip_ids)}] embedded  "
                  f"({time.time()-t0:.0f}s)")
    return np.stack(rows)


# ------------------------------ classifier ------------------------------

def loo_cv(X: np.ndarray, y: np.ndarray, names: list[str],
           C: float = 0.1) -> tuple[float, list[dict]]:
    """Leave-one-out cross-validation. Returns (accuracy, per-clip records)."""
    from sklearn.linear_model import LogisticRegression
    n = len(y)
    correct = 0
    records = []
    for hold in range(n):
        mask = np.ones(n, dtype=bool); mask[hold] = False
        clf = LogisticRegression(C=C, class_weight="balanced",
                                 max_iter=2000, random_state=0)
        # Skip if training set doesn't have all classes (rare with 4-way
        # split and a class of size 3)
        classes_in_train = np.unique(y[mask])
        if len(classes_in_train) < 2:
            records.append({"name": names[hold], "truth": str(y[hold]),
                            "pred": "SKIP-single-class", "correct": False,
                            "probs": {}})
            continue
        clf.fit(X[mask], y[mask])
        pred = clf.predict(X[hold:hold+1])[0]
        probs = clf.predict_proba(X[hold:hold+1])[0]
        prob_map = {str(c): float(p) for c, p in zip(clf.classes_, probs)}
        ok = (pred == y[hold])
        if ok: correct += 1
        records.append({"name": names[hold], "truth": str(y[hold]),
                        "pred": str(pred), "correct": bool(ok),
                        "probs": prob_map})
    return correct / n, records


def report(scheme_name: str, y: np.ndarray, records: list[dict]) -> None:
    print(f"\n  Class distribution: {dict(Counter(y.tolist()))}")
    acc = sum(r["correct"] for r in records) / len(records)
    print(f"  LOO accuracy: {sum(r['correct'] for r in records)}/{len(records)} "
          f"= {acc*100:.0f}%")
    # Per-class breakdown
    per = Counter()
    per_correct = Counter()
    for r in records:
        per[r["truth"]] += 1
        if r["correct"]: per_correct[r["truth"]] += 1
    print(f"  Per-class recall:")
    for cls in sorted(per):
        n = per[cls]; c = per_correct[cls]
        bar = "█" * c + "░" * (n - c)
        print(f"    {cls:15s}  {c}/{n} = {c/n*100:3.0f}%  {bar}")
    # Confusion listing (misses only, short-form)
    misses = [r for r in records if not r["correct"] and r["pred"] != "SKIP-single-class"]
    if misses:
        print(f"  Misses ({len(misses)}):")
        for r in misses:
            top2 = sorted(r["probs"].items(), key=lambda kv: -kv[1])[:2]
            top2s = ", ".join(f"{k}={v:.2f}" for k, v in top2)
            print(f"    {r['name'][-22:]:22s}  truth={r['truth']:12s} "
                  f"pred={r['pred']:12s}  top-2: {top2s}")


def main() -> int:
    print("=== loading labels ===")
    raw = json.loads(LABELS_PATH.read_text())
    parsed = {cid: parse_label(v["label"]) for cid, v in raw.items()}
    n = len(parsed)
    n_certain = sum(1 for p in parsed.values() if p["certain"] and p["code"])
    n_uncertain = sum(1 for p in parsed.values() if not p["certain"] and p["code"])
    n_unk = sum(1 for p in parsed.values() if p["code"] is None)
    print(f"  {n} clips: {n_certain} certain-code, "
          f"{n_uncertain} uncertain-code, {n_unk} unmapped")

    # ----------------------- Perch embeddings (cached) ---------------------
    clip_ids = list(parsed.keys())
    cache = ARTIFACTS / "focused_embeddings.npz"
    if cache.exists():
        print(f"\n=== loading cached embeddings: {cache.name} ===")
        npz = np.load(cache, allow_pickle=True)
        cache_ids = list(npz["clip_ids"])
        E = npz["embeddings"]
        assert cache_ids == clip_ids, "cache mismatch — delete to rebuild"
    else:
        print("\n=== loading Perch 2.0 (CPU) ===")
        from perch_hoplite.zoo import model_configs
        model = model_configs.load_model_by_name("perch_v2_cpu")
        print("\n=== embedding 50 focused 5s clips ===")
        E = embed_focused_clips(model, clip_ids)
        np.savez_compressed(cache, clip_ids=np.array(clip_ids),
                            embeddings=E.astype(np.float32))
        print(f"  cached -> {cache.relative_to(REPO)}")
    print(f"  embeddings shape: {E.shape}")

    # ---------------- three progressively more ambitious schemes ------------
    # In each scheme we exclude uncertain-labeled clips from TRAINING (they
    # are noise in the target label) but evaluate them separately below.
    idx_certain_code = [i for i, cid in enumerate(clip_ids)
                        if parsed[cid]["certain"] and parsed[cid]["code"]]
    idx_unk = [i for i, cid in enumerate(clip_ids) if parsed[cid]["code"] is None]
    idx_uncertain = [i for i, cid in enumerate(clip_ids)
                     if not parsed[cid]["certain"] and parsed[cid]["code"]]

    # ---- scheme A: S01 vs not-S01 (binary, all certain-code clips + unk) --
    print("\n\n=== SCHEME A: S01 vs not-S01 ===")
    print("    (unmapped clips count as not-S01; uncertain clips excluded)")
    idx_a = idx_certain_code + idx_unk
    y_a = np.array(["S01" if parsed[clip_ids[i]]["code"] == "S01" else "not-S01"
                    for i in idx_a])
    X_a = E[idx_a]
    names_a = [clip_ids[i] for i in idx_a]
    acc_a, rec_a = loo_cv(X_a, y_a, names_a)
    report("A", y_a, rec_a)

    # ---- scheme B: S01 vs S44 vs S17 vs other-known -----------------------
    print("\n\n=== SCHEME B: S01 vs S44 vs S17 vs other-known ===")
    print("    (unmapped clips EXCLUDED — this scheme only classifies clips")
    print("     that DO map to a Ford code; use scheme C for full behaviour)")
    def bucket_b(cid):
        c = parsed[cid]["code"]
        if c in ("S01", "S44", "S17"): return c
        return "other-known"
    idx_b = idx_certain_code
    y_b = np.array([bucket_b(clip_ids[i]) for i in idx_b])
    X_b = E[idx_b]
    names_b = [clip_ids[i] for i in idx_b]
    acc_b, rec_b = loo_cv(X_b, y_b, names_b)
    report("B", y_b, rec_b)

    # ---- scheme C: S01 vs S44 vs S17 vs known-tail vs unk -----------------
    print("\n\n=== SCHEME C: S01 vs S44 vs S17 vs known-tail vs unk ===")
    print("    (unk = the 22% that don't map to Ford codes, kept as a class")
    print("     because in production the classifier must handle them)")
    def bucket_c(cid):
        c = parsed[cid]["code"]
        if c is None: return "unk"
        if c in ("S01", "S44", "S17"): return c
        return "known-tail"
    idx_c = idx_certain_code + idx_unk
    y_c = np.array([bucket_c(clip_ids[i]) for i in idx_c])
    X_c = E[idx_c]
    names_c = [clip_ids[i] for i in idx_c]
    acc_c, rec_c = loo_cv(X_c, y_c, names_c)
    report("C", y_c, rec_c)

    # ---------------- uncertain-label sanity check -------------------------
    # Train the scheme-C classifier on all certain+unk data, then look at
    # what it predicts for the ?-tagged clips. Does the classifier agree
    # with the labeler's tentative guess?
    if idx_uncertain:
        from sklearn.linear_model import LogisticRegression
        print("\n\n=== BONUS: how does the scheme-C classifier predict the "
              "?-tagged clips? ===")
        print("    (tentative human guess vs classifier top-1 -- indirect")
        print("     agreement check; if it matches often, that's another")
        print("     signal that the model has learned something real)")
        clf = LogisticRegression(C=0.1, class_weight="balanced",
                                 max_iter=2000, random_state=0)
        clf.fit(X_c, y_c)
        X_u = E[idx_uncertain]
        proba = clf.predict_proba(X_u)
        preds = clf.predict(X_u)
        agree = 0; countable = 0
        for j, i in enumerate(idx_uncertain):
            cid = clip_ids[i]
            true_bucket = bucket_c(cid)   # bucket the ?-code goes into
            pred = preds[j]
            top2 = sorted(zip(clf.classes_, proba[j]), key=lambda kv: -kv[1])[:2]
            top2s = ", ".join(f"{k}={v:.2f}" for k, v in top2)
            match = "✓" if pred == true_bucket else " "
            if true_bucket != "unk":  # only count where guess is a real code
                countable += 1
                if pred == true_bucket: agree += 1
            print(f"    {match} {cid[-22:]:22s}  guessed={parsed[cid]['raw'][:12]:12s} "
                  f"-> bucket={true_bucket:11s}  pred={pred:11s}  top-2: {top2s}")
        if countable:
            print(f"  Agreement on ?-tagged clips (excl. unk buckets): "
                  f"{agree}/{countable} = {agree/countable*100:.0f}%")

    # ---------------- persist artifacts ------------------------------------
    import joblib
    summary = {
        "n_clips": n, "n_certain": n_certain, "n_uncertain": n_uncertain,
        "n_unk": n_unk,
        "schemes": {
            "A_S01_vs_not": {"acc": acc_a, "n": len(y_a),
                             "class_counts": dict(Counter(y_a.tolist()))},
            "B_head_vs_other": {"acc": acc_b, "n": len(y_b),
                                "class_counts": dict(Counter(y_b.tolist()))},
            "C_head_vs_tail_vs_unk": {"acc": acc_c, "n": len(y_c),
                                      "class_counts": dict(Counter(y_c.tolist()))},
        },
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  summary -> {(ARTIFACTS/'results.json').relative_to(REPO)}")

    # Save both schemes:
    #   - Scheme A (S01 vs not-S01) is the DEPLOYED classifier — highest LOO
    #     accuracy (79%) and low-confidence-on-misses property that makes a
    #     P>=0.70 confidence gate sensible (see D-044 disposition).
    #   - Scheme C (adds unk as class) kept as a research/inspection artifact
    #     — same training data, richer output, worse point accuracy.
    from sklearn.linear_model import LogisticRegression
    clf_a = LogisticRegression(C=0.1, class_weight="balanced",
                               max_iter=2000, random_state=0)
    clf_a.fit(X_a, y_a)
    joblib.dump(clf_a, ARTIFACTS / "calltype_classifier_scheme_a.joblib")
    print(f"  classifier A (deployed) -> "
          f"models/perch_calltype_v0/calltype_classifier_scheme_a.joblib")

    clf_c = LogisticRegression(C=0.1, class_weight="balanced",
                               max_iter=2000, random_state=0)
    clf_c.fit(X_c, y_c)
    joblib.dump(clf_c, ARTIFACTS / "calltype_classifier_scheme_c.joblib")
    print(f"  classifier C (research)  -> "
          f"models/perch_calltype_v0/calltype_classifier_scheme_c.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
