"""Download a 1-day HLS session from Orcasound S3 in 60 s WAV chunks.

Uses the OrcasoundHLSClient from aifororcas-livesystem to handle the
m3u8 walking + ffmpeg conversion. We just orchestrate the time range
and persist the chunk metadata for downstream stages.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from . import config as C

# Make the InferenceSystem source importable.
sys.path.insert(0, str(C.INFERENCE_SRC))
from orcasound_hls import OrcasoundHLSClient  # noqa: E402
from orcasound_hls.types import OrcasoundHLSSegment  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """Persistent record of a downloaded chunk."""

    index: int
    wav_path: str
    start_unix: float
    end_unix: float
    duration_s: float
    name: str

    @property
    def start_iso(self) -> str:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(self.start_unix, tz=timezone.utc).isoformat()


def list_segments(start_unix: float, end_unix: float) -> List[OrcasoundHLSSegment]:
    """Return the list of HLS segment chunks covering the requested window."""
    client = OrcasoundHLSClient(bucket=C.BUCKET, hydrophone_id=C.HYDROPHONE_ID)
    return client.get_segments(
        start_unix=start_unix,
        end_unix=end_unix,
        segment_size=C.DOWNLOAD_CHUNK_S,
    )


def download_chunks(
    segments: List[OrcasoundHLSSegment],
    dest_dir: Path,
    *,
    limit: int | None = None,
    progress_every: int = 25,
) -> List[ChunkRecord]:
    """Materialize each chunk as a WAV. Skip chunks whose WAV already exists.

    Parameters
    ----------
    segments
        Output of :func:`list_segments`.
    dest_dir
        Directory where chunk WAVs land.
    limit
        Optional upper bound on number of chunks (for smoke tests).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    records: List[ChunkRecord] = []
    if limit is not None:
        segments = segments[:limit]
    for i, seg in enumerate(segments):
        wav_path = dest_dir / f"{seg.name}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 0:
            log.debug("skip existing %s", wav_path.name)
        else:
            try:
                seg.download_as_wav(str(dest_dir))
            except Exception as exc:
                log.warning("download failed for chunk %d (%s): %s", i, seg.name, exc)
                continue
        records.append(
            ChunkRecord(
                index=i,
                wav_path=str(wav_path),
                start_unix=seg.start_unix,
                end_unix=seg.end_unix,
                duration_s=seg.duration_s,
                name=seg.name,
            )
        )
        if (i + 1) % progress_every == 0:
            log.info("downloaded %d / %d chunks", i + 1, len(segments))
    return records


def save_manifest(records: List[ChunkRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records], indent=2))


def load_manifest(path: Path) -> List[ChunkRecord]:
    raw = json.loads(path.read_text())
    return [ChunkRecord(**r) for r in raw]
