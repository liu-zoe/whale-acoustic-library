"""Cluster contiguous positive segments into detection events."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .detect import SegmentDetection
from . import config as C


@dataclass
class DetectionEvent:
    """A single detection event spanning one or more contiguous positive segments."""

    start_unix: float
    end_unix: float
    peak_confidence: float
    mean_confidence: float
    n_segments: int
    member_segments: List[SegmentDetection] = field(default_factory=list)

    @property
    def midpoint_unix(self) -> float:
        return 0.5 * (self.start_unix + self.end_unix)

    @property
    def duration_s(self) -> float:
        return self.end_unix - self.start_unix


def cluster_detections(
    detections: List[SegmentDetection],
    *,
    threshold: float = C.LOCAL_DETECTION_THRESHOLD,
    gap_s: float = C.CLUSTER_GAP_S,
) -> List[DetectionEvent]:
    """Merge contiguous positive segments separated by ≤ gap_s into events.

    detections must be the full list; we filter to positives and sort by time
    here to be safe.
    """
    pos = sorted(
        [d for d in detections if d.confidence > threshold],
        key=lambda d: d.seg_start_unix,
    )
    if not pos:
        return []

    events: List[DetectionEvent] = []
    cur: List[SegmentDetection] = [pos[0]]
    for d in pos[1:]:
        prev_end = cur[-1].seg_end_unix
        if d.seg_start_unix - prev_end <= gap_s:
            cur.append(d)
        else:
            events.append(_finalize(cur))
            cur = [d]
    events.append(_finalize(cur))
    return events


def _finalize(group: List[SegmentDetection]) -> DetectionEvent:
    confs = [g.confidence for g in group]
    return DetectionEvent(
        start_unix=group[0].seg_start_unix,
        end_unix=group[-1].seg_end_unix,
        peak_confidence=max(confs),
        mean_confidence=sum(confs) / len(confs),
        n_segments=len(group),
        member_segments=group,
    )
