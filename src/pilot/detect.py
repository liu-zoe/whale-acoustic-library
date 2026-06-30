"""OrcaHello detection on chunk WAVs.

Returns absolute UTC-time-anchored per-segment confidence records.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from . import config as C

sys.path.insert(0, str(C.INFERENCE_SRC))
from model.inference import OrcaHelloSRKWDetectorV1  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class SegmentDetection:
    """One model output segment, anchored in absolute UTC time."""

    chunk_index: int
    chunk_wav_path: str
    chunk_start_unix: float
    seg_start_in_chunk_s: float
    seg_duration_s: float
    confidence: float

    @property
    def seg_start_unix(self) -> float:
        return self.chunk_start_unix + self.seg_start_in_chunk_s

    @property
    def seg_end_unix(self) -> float:
        return self.seg_start_unix + self.seg_duration_s


def load_model() -> OrcaHelloSRKWDetectorV1:
    log.info("loading %s from HuggingFace...", C.HF_REPO_ID)
    model = OrcaHelloSRKWDetectorV1.from_pretrained(C.HF_REPO_ID)
    model.eval()
    return model


def detect_chunk(model, chunk) -> List[SegmentDetection]:
    """Run inference on a single chunk WAV. Returns one record per segment."""
    result = model.detect_srkw_from_file(chunk.wav_path)
    out: List[SegmentDetection] = []
    for seg in result.segment_predictions:
        out.append(
            SegmentDetection(
                chunk_index=chunk.index,
                chunk_wav_path=chunk.wav_path,
                chunk_start_unix=chunk.start_unix,
                seg_start_in_chunk_s=float(seg.start_time_s),
                seg_duration_s=float(seg.duration_s),
                confidence=float(seg.confidence),
            )
        )
    return out


def detect_all(model, chunks, *, progress_every: int = 25) -> List[SegmentDetection]:
    detections: List[SegmentDetection] = []
    for i, chunk in enumerate(chunks):
        try:
            detections.extend(detect_chunk(model, chunk))
        except Exception as exc:
            log.warning("inference failed for chunk %d (%s): %s", chunk.index, chunk.name, exc)
            continue
        if (i + 1) % progress_every == 0:
            n_pos = sum(1 for d in detections if d.confidence > C.LOCAL_DETECTION_THRESHOLD)
            log.info(
                "inference %d / %d chunks done; %d positive segments so far",
                i + 1, len(chunks), n_pos,
            )
    return detections
