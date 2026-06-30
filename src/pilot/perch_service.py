"""Pipeline-side Perch 2.0 helpers — one load, two operations.

Used by `run_batch.py` to annotate every clip the pipeline produces with:
  1. **`perch_p_humpback`** — humpback-vs-vessel probability from the trained
     classifier in models/perch_humpback_v0/ (D-027). Applied to every clip
     species='humpback' produced by Phase 4b Multispecies detection. Low P
     flags clips the user is likely to reject (vessel/echo-sounder noise).
  2. **`nearest_ref_call`/`pod`/`similarity`** — tentative Ford-Osborne call-
     type and pod via Perch nearest-neighbor against the catalogue in
     models/srkw_call_labeler_v0/ (D-031). Applied to every SRKW clip.

Designed so that if Perch / its artifacts are missing the pipeline degrades
gracefully — `run_batch.py` logs a warning and continues without the Perch
columns, same pattern as the Multispecies model.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
import soundfile as sf

log = logging.getLogger(__name__)

PERCH_SR = 32000
HUMPBACK_ART = Path(__file__).resolve().parents[2] / "models/perch_humpback_v0"
SRKW_ART = Path(__file__).resolve().parents[2] / "models/srkw_call_labeler_v0"


@dataclass
class PerchAnnotations:
    """Per-clip Perch-derived annotations. None where not applicable / available."""
    perch_p_humpback: Optional[float] = None
    nearest_ref_call: Optional[str] = None
    nearest_ref_pod: Optional[str] = None
    nearest_ref_similarity: Optional[float] = None


class PerchService:
    """Loads Perch 2.0 once + the humpback classifier + the SRKW reference set.

    Lazy on the heavy bits — Perch itself takes ~10 s and ~400 MB; loading is
    deferred until first use so a `--skip-perch` invocation pays no cost.
    """

    def __init__(self):
        self._model = None
        self._humpback_clf = None
        self._ref_dicts = None       # list of dicts: {call_code, pods, source}
        self._ref_embeddings = None  # (n_refs, 1536), L2-normalized

    def _ensure_model(self):
        if self._model is not None:
            return
        log.info("loading Perch 2.0 (CPU)")
        from perch_hoplite.zoo import model_configs
        self._model = model_configs.load_model_by_name("perch_v2_cpu")

    def _ensure_humpback_classifier(self):
        if self._humpback_clf is not None:
            return
        import joblib
        path = HUMPBACK_ART / "humpback_classifier.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Perch humpback classifier missing at {path}; "
                "run src/perch_classifier.py first."
            )
        self._humpback_clf = joblib.load(path)
        log.info("loaded Perch humpback classifier")

    def _ensure_srkw_refs(self):
        if self._ref_embeddings is not None:
            return
        path = SRKW_ART / "reference_embeddings.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"SRKW reference embeddings missing at {path}; "
                "run src/srkw_call_labeler.py first."
            )
        npz = np.load(path, allow_pickle=True)
        self._ref_dicts = list(npz["refs"])
        E = npz["embeddings"]
        self._ref_embeddings = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        log.info("loaded %d SRKW Ford-Osborne reference embeddings", len(self._ref_dicts))

    def _embed_wav(self, wav_path) -> np.ndarray:
        """Mean-pooled Perch embedding for one clip WAV (1536-dim)."""
        self._ensure_model()
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != PERCH_SR:
            n_out = int(round(len(audio) * PERCH_SR / sr))
            audio = scipy.signal.resample(audio, n_out).astype(np.float32)
        out = self._model.embed(audio.astype(np.float32))
        e = np.squeeze(out.embeddings, axis=1)
        if e.shape[0] == 0:
            return np.zeros(1536, dtype=np.float32)
        return e.mean(axis=0).astype(np.float32)

    def score_humpback(self, wav_path) -> float:
        """P(humpback) on a clip via the trained humpback-vs-vessel classifier."""
        self._ensure_humpback_classifier()
        e = self._embed_wav(wav_path).reshape(1, -1)
        return float(self._humpback_clf.predict_proba(e)[0, 1])

    def nearest_srkw_ref(self, wav_path, *, top_k: int = 5) -> tuple[str, str, float]:
        """Nearest Ford-Osborne reference for a clip. Returns
        (call_code, pod, cosine_similarity). Pod assigned by top-K voting
        over single-pod-labeled refs ('J'/'K'/'L'), or '?' if none in top-K."""
        self._ensure_srkw_refs()
        e = self._embed_wav(wav_path)
        e_n = e / (np.linalg.norm(e) + 1e-12)
        sim = self._ref_embeddings @ e_n
        order = np.argsort(-sim)
        top = int(order[0])
        best_call = self._ref_dicts[top]["call_code"]
        best_sim = float(sim[top])
        # Pod vote among top-K single-pod-labeled refs, weighted by similarity
        votes: dict[str, float] = {}
        for idx in order[:top_k]:
            pods = self._ref_dicts[idx]["pods"]
            if pods in ("J", "K", "L"):
                votes[pods] = votes.get(pods, 0.0) + float(sim[idx])
        pod = max(votes, key=votes.get) if votes else "?"
        return (best_call, pod, best_sim)

    def annotate(self, clip_id: str, wav_path, species: str) -> PerchAnnotations:
        """One-shot per-clip annotation. Catches per-clip failures and returns
        partial results rather than aborting the whole batch."""
        ann = PerchAnnotations()
        try:
            if species == "humpback":
                ann.perch_p_humpback = self.score_humpback(wav_path)
            elif species == "SRKW":
                call, pod, sim = self.nearest_srkw_ref(wav_path)
                ann.nearest_ref_call = call
                ann.nearest_ref_pod = pod
                ann.nearest_ref_similarity = sim
        except Exception as exc:
            log.warning("Perch annotation failed for %s: %s: %s",
                        clip_id, type(exc).__name__, exc)
        return ann
