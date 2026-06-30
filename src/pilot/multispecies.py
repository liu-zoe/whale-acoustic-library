"""Google Multispecies Whale Model — secondary species / call-type detector.

Wraps the TensorFlow SavedModel (Kaggle: google/multispecies-whale, TF2 v2),
the project's planned secondary detection layer. The model takes 24 kHz mono
audio, scores 5 s windows, and emits 12 multi-label scores (7 whale species +
5 call/vocalization-type classes). Scores are independent, not a softmax.

Relevant to this project: `Oo` cross-checks OrcaHello's SRKW detection, `Mn`
adds the humpback catch the plan wanted, and `Echolocation` / `Whistle` give
killer-whale call-type hints. Note: this model has **no gray whale** class.

The model lives at models/multispecies-whale/ (downloaded manually from
Kaggle). TensorFlow is required: `pip install tensorflow-cpu` in the env.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.signal
import soundfile as sf

# models/multispecies-whale/<variant>/ — relative to the repo root so this
# works regardless of where the repo is checked out (laptop or flash drive).
_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "models/multispecies-whale/multispecies-whale-tensorflow2-default-v2"
)

# Human-readable names for the model's terse class codes (genus/species
# abbreviations and call-type names).
CLASS_LABELS: Dict[str, str] = {
    "Oo": "killer whale",
    "Mn": "humpback whale",
    "Eg": "N. Atlantic right whale",
    "Be": "Bryde's whale",
    "Bp": "fin whale",
    "Bm": "blue whale",
    "Ba": "minke whale",
    "Upcall": "right whale upcall (call type)",
    "Gunshot": "right whale gunshot (call type)",
    "Call": "right whale call (call type)",
    "Echolocation": "killer whale echolocation (call type)",
    "Whistle": "killer whale whistle (call type)",
}

# Maps the model's call-type classes to the catalog `call_type` vocabulary
# (see catalog.py schema: discrete | click | whistle | unknown).
CALL_TYPE_MAP = {"Call": "discrete", "Echolocation": "click", "Whistle": "whistle"}


def call_type_from_multispecies(
    scores: Optional[Dict[str, float]], threshold: float = 0.5
) -> str:
    """Pick the dominant SRKW call type from a Multispecies score dict.

    Returns the catalog value ('discrete' | 'click' | 'whistle') for the
    highest-scoring call-type class at or above ``threshold``, else
    'unknown'. The full multi-label picture is preserved separately in the
    `multispecies_scores` JSON column — this is just a convenient single label.
    """
    if not scores:
        return "unknown"
    top = max(CALL_TYPE_MAP, key=lambda k: scores.get(k, 0.0))
    if scores.get(top, 0.0) >= threshold:
        return CALL_TYPE_MAP[top]
    return "unknown"


class MultispeciesClassifier:
    """Loads the SavedModel once; scores clips of arbitrary length."""

    def __init__(self, model_dir: Path = _MODEL_DIR):
        import tensorflow as tf  # lazy: TF import is slow and optional

        self._tf = tf
        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"Multispecies model not found at {model_dir}. "
                "Download it from Kaggle (see models/multispecies-whale/)."
            )
        self.model = tf.saved_model.load(str(model_dir))
        self._score_fn = self.model.signatures["score"]
        md = self.model.signatures["metadata"]()
        self.sample_rate = int(md["input_sample_rate"])         # 24000
        self.context_width = int(md["context_width_samples"])   # 120000 (5 s)
        self.class_names = [s.decode() for s in md["class_names"].numpy()]

    def score_clip(self, audio: np.ndarray, sr: int, *, hop_s: float = 1.0) -> Dict[str, float]:
        """Score one clip; return {class_name: max score across all windows}.

        The model slides a 5 s window across the clip every ``hop_s`` seconds.
        Because the classes are multi-label, the per-class max answers "did
        this species/call appear anywhere in the clip?".
        """
        tf = self._tf
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if sr != self.sample_rate:
            n_out = int(round(len(audio) * self.sample_rate / sr))
            audio = scipy.signal.resample(audio, n_out).astype(np.float32)

        # The model needs at least one full 5 s window.
        if len(audio) < self.context_width:
            audio = np.concatenate(
                [audio, np.zeros(self.context_width - len(audio), dtype=np.float32)]
            )

        step = max(1, int(round(hop_s * self.sample_rate)))
        scores = self._score_fn(
            waveform=tf.constant(audio.reshape(1, -1, 1)),
            context_step_samples=tf.constant(step, dtype=tf.int64),
        )["score"].numpy()              # (1, n_windows, 12)

        per_class_max = scores[0].max(axis=0)   # (12,)
        return {c: float(s) for c, s in zip(self.class_names, per_class_max)}

    def score_wav(self, wav_path, *, hop_s: float = 1.0) -> Dict[str, float]:
        """Score a clip WAV file on disk; returns {class_name: max score}."""
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        return self.score_clip(audio, sr, hop_s=hop_s)

    def score_windows(self, audio: np.ndarray, sr: int, *, hop_s: float = 2.5):
        """Score every 5 s window across the audio; for *detection*, not just
        a clip-level summary.

        Returns a list of ``(window_start_s, {class_name: score})`` — window i
        covers ``[i*hop_s, i*hop_s + 5 s]``. (score_clip collapses this to the
        per-class max; score_windows keeps the per-window detail so a caller
        can localize where a species vocalizes.)
        """
        tf = self._tf
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            n_out = int(round(len(audio) * self.sample_rate / sr))
            audio = scipy.signal.resample(audio, n_out).astype(np.float32)
        if len(audio) < self.context_width:
            audio = np.concatenate(
                [audio, np.zeros(self.context_width - len(audio), dtype=np.float32)]
            )
        step = max(1, int(round(hop_s * self.sample_rate)))
        scores = self._score_fn(
            waveform=tf.constant(audio.reshape(1, -1, 1)),
            context_step_samples=tf.constant(step, dtype=tf.int64),
        )["score"].numpy()[0]            # (n_windows, 12)
        hop_actual = step / self.sample_rate
        return [
            (i * hop_actual, {c: float(s) for c, s in zip(self.class_names, row)})
            for i, row in enumerate(scores)
        ]
