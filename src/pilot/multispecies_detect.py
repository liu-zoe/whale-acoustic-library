"""Multispecies primary detection — scan chunk WAVs for a target species.

Phase 4.5 only *scores* clips OrcaHello already cut, so non-SRKW species are
invisible unless they vocalize inside an orca detection. This module runs the
Google Multispecies model directly over the raw chunk audio, so a species
OrcaHello does not detect (humpback) can be found independently.

It emits ``SegmentDetection`` records — the same type OrcaHello produces — so
the existing clustering (`cluster.cluster_detections`) and clip machinery
(`clip.make_all_clips`) are reused unchanged. ``confidence`` carries the
target class's per-window score.
"""
from __future__ import annotations

import logging
from typing import List

import soundfile as sf

from . import config as C
from .detect import SegmentDetection

log = logging.getLogger(__name__)

# The model analyses fixed 5 s windows (context_width / sample_rate).
_WINDOW_S = 5.0


def detect_chunk(
    clf,
    chunk,
    *,
    target_class: str = C.MULTISPECIES_DETECT_CLASS,
    hop_s: float = C.MULTISPECIES_SCAN_HOP_S,
) -> List[SegmentDetection]:
    """Score one chunk WAV; emit a SegmentDetection per 5 s window, with
    ``confidence`` set to the target class's score."""
    audio, sr = sf.read(chunk.wav_path, dtype="float32", always_2d=False)
    out: List[SegmentDetection] = []
    for start_s, scores in clf.score_windows(audio, sr, hop_s=hop_s):
        out.append(
            SegmentDetection(
                chunk_index=chunk.index,
                chunk_wav_path=chunk.wav_path,
                chunk_start_unix=chunk.start_unix,
                seg_start_in_chunk_s=start_s,
                seg_duration_s=_WINDOW_S,
                confidence=scores.get(target_class, 0.0),
            )
        )
    return out


def detect_all(
    clf,
    chunks,
    *,
    target_class: str = C.MULTISPECIES_DETECT_CLASS,
    hop_s: float = C.MULTISPECIES_SCAN_HOP_S,
    progress_every: int = 100,
) -> List[SegmentDetection]:
    """Scan every chunk for the target class. Per-chunk failures are skipped."""
    dets: List[SegmentDetection] = []
    for i, chunk in enumerate(chunks):
        try:
            dets.extend(
                detect_chunk(clf, chunk, target_class=target_class, hop_s=hop_s)
            )
        except Exception as exc:
            log.warning(
                "multispecies inference failed for chunk %d (%s): %s",
                chunk.index, getattr(chunk, "name", "?"), exc,
            )
        if (i + 1) % progress_every == 0:
            log.info("multispecies scan %d / %d chunks done", i + 1, len(chunks))
    if dets:
        log.info(
            "multispecies scan complete: %d windows, max %s score %.3f",
            len(dets), target_class, max(d.confidence for d in dets),
        )
    return dets
