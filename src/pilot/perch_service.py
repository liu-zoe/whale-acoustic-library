"""Pipeline-side Perch 2.0 helpers — one load, three operations.

Used by `run_batch.py` to annotate every clip the pipeline produces with:
  1. **`perch_p_humpback`** — humpback-vs-vessel probability from the trained
     classifier in models/perch_humpback_v0/ (D-027). Applied to every clip
     species='humpback' produced by Phase 4b Multispecies detection. Low P
     flags clips the user is likely to reject (vessel/echo-sounder noise).
  2. **`nearest_ref_call`/`pod`/`similarity`** — tentative Ford-Osborne call-
     type and pod via Perch nearest-neighbor against the catalogue in
     models/srkw_call_labeler_v0/ (D-031). Applied to every SRKW clip.
  3. **`perch_predicted_calltype`/`confidence`** — supervised Ford-code
     bucket ('S01' / 'not-S01' / 'unknown-calltype') from the Scheme-A
     classifier in models/perch_calltype_v0/ (D-044). Applied to every
     SRKW clip. Uses OrcaHello per-segment confidence to locate the call
     inside the 30 s clip (path_c_picker.py's centering approach) and
     embeds the focused 5 s window instead of mean-pooling the whole clip.

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
FOCUS_WIN_S = 5.0
CALLTYPE_CONF_GATE = 0.70  # D-044: below this, classifier abstains
HUMPBACK_ART = Path(__file__).resolve().parents[2] / "models/perch_humpback_v0"
SRKW_ART = Path(__file__).resolve().parents[2] / "models/srkw_call_labeler_v0"
CALLTYPE_ART = Path(__file__).resolve().parents[2] / "models/perch_calltype_v0"


@dataclass
class PerchAnnotations:
    """Per-clip Perch-derived annotations. None where not applicable / available."""
    perch_p_humpback: Optional[float] = None
    nearest_ref_call: Optional[str] = None
    nearest_ref_pod: Optional[str] = None
    nearest_ref_similarity: Optional[float] = None
    # D-044 call-type prediction. `perch_predicted_calltype` is one of
    # 'S01', 'not-S01', or 'unknown-calltype' (the abstain bucket when
    # max class prob < CALLTYPE_CONF_GATE). `confidence` is the max class
    # probability regardless of gate outcome.
    perch_predicted_calltype: Optional[str] = None
    perch_calltype_confidence: Optional[float] = None


def _find_call_center(orcahello, clip_wav_path) -> tuple[float, float]:
    """Use OrcaHello to locate the highest-confidence 3s segment in the clip.
    Returns (center_time_s, peak_segment_confidence). Same logic as
    src/path_c_picker.py — kept here so PerchService can be self-contained."""
    result = orcahello.detect_srkw_from_file(str(clip_wav_path))
    if not result.segment_predictions:
        return (15.0, 0.0)  # defensive fallback: middle of a 30 s clip
    best = max(result.segment_predictions, key=lambda s: float(s.confidence))
    center = float(best.start_time_s) + float(best.duration_s) / 2.0
    return (center, float(best.confidence))


def _extract_focused_window(audio_full: np.ndarray, sr: int,
                            center_s: float,
                            win_s: float = FOCUS_WIN_S) -> np.ndarray:
    """Pull a `win_s` window centered on `center_s`, clamped to clip bounds
    (shifting the window rather than truncating when we hit an edge)."""
    half = win_s / 2.0
    start = max(0.0, center_s - half)
    end = min(len(audio_full) / sr, center_s + half)
    if end - start < win_s:
        if start <= 0:
            end = min(len(audio_full) / sr, start + win_s)
        else:
            start = max(0.0, end - win_s)
    i0 = int(round(start * sr))
    i1 = int(round(end * sr))
    return audio_full[i0:i1]


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
        self._calltype_clf = None    # Scheme-A LogisticRegression (D-044)
        self._orcahello = None       # OrcaHello for focused-window centering

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

    def _ensure_calltype_classifier(self):
        if self._calltype_clf is not None:
            return
        import joblib
        path = CALLTYPE_ART / "calltype_classifier_scheme_a.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Perch call-type classifier missing at {path}; "
                "run src/perch_calltype_classifier.py first."
            )
        self._calltype_clf = joblib.load(path)
        log.info("loaded Perch call-type classifier (Scheme A, D-044)")

    def _ensure_orcahello(self):
        if self._orcahello is not None:
            return
        # Local import so that PerchService itself has no hard dependency
        # on OrcaHello — call-type prediction fails soft if OrcaHello is
        # unavailable, but nearest-ref and humpback-scoring still work.
        from pilot import detect as det
        log.info("loading OrcaHello (for call-type focused-window centering)")
        self._orcahello = det.load_model()

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

    def _embed_focused(self, wav_path) -> tuple[np.ndarray, float, float]:
        """Extract the OrcaHello-centered 5 s window from a clip and Perch-
        embed it as a single 1536-dim vector. Returns
        (embedding, call_center_s, peak_segment_confidence). Used by
        `predict_calltype` — call-type classification wants the FOCUSED
        window, not the whole-clip mean-pool, because a 30 s clip can hold
        several distinct calls plus quiet stretches."""
        self._ensure_orcahello()
        self._ensure_model()
        center_s, peak_seg_conf = _find_call_center(self._orcahello, wav_path)
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        seg = _extract_focused_window(audio, sr, center_s)
        if sr != PERCH_SR:
            n_out = int(round(len(seg) * PERCH_SR / sr))
            seg = scipy.signal.resample(seg, n_out).astype(np.float32)
        out = self._model.embed(seg.astype(np.float32))
        e = np.squeeze(out.embeddings, axis=1)
        if e.shape[0] == 0:
            return (np.zeros(1536, dtype=np.float32), center_s, peak_seg_conf)
        # 5 s @ 32 kHz is exactly one Perch window; if OrcaHello centered
        # too close to a clip edge we may get a shorter window and Perch
        # produces 0 windows above — fall back to the first row when > 1.
        return (e[0].astype(np.float32), center_s, peak_seg_conf)

    def predict_calltype(self, wav_path) -> tuple[str, float]:
        """Deployed D-044 Scheme-A classifier: 'S01' vs 'not-S01' with an
        abstain bucket. Returns (label, confidence). Label is one of:
          - 'S01'               (max class prob >= gate, argmax is S01)
          - 'not-S01'           (max class prob >= gate, argmax is not-S01)
          - 'unknown-calltype'  (max class prob < gate — model abstains)
        Confidence is always the max class probability, regardless of gate."""
        self._ensure_calltype_classifier()
        e, _, _ = self._embed_focused(wav_path)
        probs = self._calltype_clf.predict_proba(e.reshape(1, -1))[0]
        classes = list(self._calltype_clf.classes_)
        conf = float(probs.max())
        if conf < CALLTYPE_CONF_GATE:
            return ("unknown-calltype", conf)
        return (str(classes[int(probs.argmax())]), conf)

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
        partial results rather than aborting the whole batch. For SRKW clips
        we now run TWO independent per-annotation try/excepts so a call-type
        failure doesn't wipe out the nearest-ref annotation (or vice versa)."""
        ann = PerchAnnotations()
        if species == "humpback":
            try:
                ann.perch_p_humpback = self.score_humpback(wav_path)
            except Exception as exc:
                log.warning("Perch humpback score failed for %s: %s: %s",
                            clip_id, type(exc).__name__, exc)
        elif species == "SRKW":
            try:
                call, pod, sim = self.nearest_srkw_ref(wav_path)
                ann.nearest_ref_call = call
                ann.nearest_ref_pod = pod
                ann.nearest_ref_similarity = sim
            except Exception as exc:
                log.warning("Perch nearest-ref failed for %s: %s: %s",
                            clip_id, type(exc).__name__, exc)
            try:
                label, conf = self.predict_calltype(wav_path)
                ann.perch_predicted_calltype = label
                ann.perch_calltype_confidence = conf
            except Exception as exc:
                log.warning("Perch call-type predict failed for %s: %s: %s",
                            clip_id, type(exc).__name__, exc)
        return ann
