"""Cut 30 s clips around each detection event, denoise, and render spectrograms."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import soundfile as sf
import scipy.signal

from . import config as C
from .cluster import DetectionEvent
from .download import ChunkRecord
from .dsp import (
    bandpass_for_species,
    spectral_denoise,
    inband_snr_db,
    save_wav,
    render_spectrogram_for_species,
    SPECIES_DENOISE_PROP_DECREASE,
    DENOISE_PROP_DECREASE,
)

log = logging.getLogger(__name__)


@dataclass
class ClipRecord:
    """Materialised detection clip with paths and metadata."""

    clip_id: str
    start_unix: float
    end_unix: float
    raw_wav_path: str
    clean_wav_path: str
    spectrogram_path: str
    sample_rate: int
    duration_s: float
    snr_db: float
    peak_confidence: float
    mean_confidence: float
    n_segments: int


def _slug_from_unix(unix_ts: float) -> str:
    dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _stitch_audio_at(
    chunks: List[ChunkRecord], target_start_unix: float, target_end_unix: float
) -> tuple[np.ndarray, int]:
    """Read audio across one or more contiguous chunk WAVs to cover [target_start, target_end].

    Returns (samples, sample_rate). Pads with zeros if data is missing at edges.
    """
    # Find overlapping chunks (in time order)
    overlapping = [
        c for c in chunks
        if c.end_unix > target_start_unix and c.start_unix < target_end_unix
    ]
    if not overlapping:
        raise RuntimeError(
            f"No chunks cover [{target_start_unix}, {target_end_unix}]"
        )
    overlapping.sort(key=lambda c: c.start_unix)

    pieces: List[np.ndarray] = []
    sr_seen: Optional[int] = None
    cursor_unix = target_start_unix
    for chunk in overlapping:
        if cursor_unix >= target_end_unix:
            break
        with sf.SoundFile(chunk.wav_path) as f:
            sr = f.samplerate
            if sr_seen is None:
                sr_seen = sr
            elif sr != sr_seen:
                raise RuntimeError(
                    f"Sample-rate mismatch across chunks: {sr_seen} vs {sr}"
                )
            chunk_audio_start_unix = chunk.start_unix
            chunk_audio_dur = f.frames / float(sr)
            chunk_audio_end_unix = chunk_audio_start_unix + chunk_audio_dur

            # Map the requested window onto frame indices within this chunk.
            start_in_chunk = max(0.0, cursor_unix - chunk_audio_start_unix)
            end_in_chunk = min(chunk_audio_dur, target_end_unix - chunk_audio_start_unix)
            if end_in_chunk <= start_in_chunk:
                # Move cursor to chunk end and continue
                cursor_unix = chunk_audio_end_unix
                continue
            f.seek(int(start_in_chunk * sr))
            n_frames = int((end_in_chunk - start_in_chunk) * sr)
            data = f.read(n_frames, dtype="float32", always_2d=False)
            if data.ndim == 2:
                # Mix down to mono
                data = data.mean(axis=1)
            pieces.append(data.astype(np.float32))
            cursor_unix = chunk_audio_start_unix + end_in_chunk

    if not pieces:
        raise RuntimeError(
            f"All overlapping chunks were empty for [{target_start_unix}, {target_end_unix}]"
        )

    audio = np.concatenate(pieces)
    sr = sr_seen
    # Pad with zeros if we fell short of the requested duration
    target_n = int(round((target_end_unix - target_start_unix) * sr))
    if len(audio) < target_n:
        pad = np.zeros(target_n - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad])
    elif len(audio) > target_n:
        audio = audio[:target_n]
    return audio, sr


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    n_out = int(round(len(audio) * sr_out / sr_in))
    return scipy.signal.resample(audio, n_out).astype(np.float32)


def make_clip(
    event: DetectionEvent,
    chunks: List[ChunkRecord],
    *,
    duration_s: float = C.CLIP_DURATION_S,
    out_sr: int = C.CLIP_SAMPLE_RATE,
    tag: str = "",
    species: str = "SRKW",
) -> Optional[ClipRecord]:
    """Cut a single 30 s clip around the event midpoint, denoise, and render spectrogram.

    ``tag`` is inserted into the clip_id (e.g. "hb_" for humpback clips) so
    clips from different detectors at overlapping times stay distinct.
    ``species`` selects the denoise bandpass and spectrogram y-range —
    humpback uses 30 Hz-5 kHz (vs SRKW's 300 Hz-15 kHz) so the low-frequency
    song energy survives and is visible.
    """
    half = duration_s / 2.0
    start_u = event.midpoint_unix - half
    end_u = event.midpoint_unix + half

    try:
        raw, sr_in = _stitch_audio_at(chunks, start_u, end_u)
    except RuntimeError as exc:
        log.warning("clip extraction failed for event %.3f: %s", event.midpoint_unix, exc)
        return None

    # Resample to canonical 48 kHz for the library.
    raw48 = _resample(raw, sr_in, out_sr)

    # Cleaned: species-appropriate bandpass to the call band, then a
    # species-appropriate spectral-gate denoise. Humpback needs a lower
    # low-cut than SRKW (song energy sits below 300 Hz) AND a gentler gate
    # (the faint song at this hydrophone is easily over-attenuated).
    band = bandpass_for_species(raw48, out_sr, species)
    prop = SPECIES_DENOISE_PROP_DECREASE.get(species, DENOISE_PROP_DECREASE)
    clean = spectral_denoise(band, out_sr, prop_decrease=prop)

    # SNR is an in-band signal-quality estimate of the raw detection, not a
    # measure of how much the denoiser removed.
    snr = inband_snr_db(raw48, out_sr)

    slug = _slug_from_unix(event.start_unix)
    clip_id = f"orcasound_lab_{tag}{slug}_n{event.n_segments}"
    raw_path = C.AUDIO_RAW_DIR / f"{clip_id}.wav"
    clean_path = C.AUDIO_CLEAN_DIR / f"{clip_id}.wav"
    spec_path = C.SPECTROGRAMS_DIR / f"{clip_id}.png"

    save_wav(raw_path, raw48, out_sr)
    save_wav(clean_path, clean, out_sr)
    render_spectrogram_for_species(
        spec_path, clean, out_sr,
        f"{clip_id}  conf={event.peak_confidence:.2f}  segs={event.n_segments}",
        species,
    )

    return ClipRecord(
        clip_id=clip_id,
        start_unix=start_u,
        end_unix=end_u,
        raw_wav_path=str(raw_path),
        clean_wav_path=str(clean_path),
        spectrogram_path=str(spec_path),
        sample_rate=out_sr,
        duration_s=duration_s,
        snr_db=snr,
        peak_confidence=event.peak_confidence,
        mean_confidence=event.mean_confidence,
        n_segments=event.n_segments,
    )


def make_all_clips(
    events: List[DetectionEvent],
    chunks: List[ChunkRecord],
    *,
    tag: str = "",
    species: str = "SRKW",
) -> List[ClipRecord]:
    out: List[ClipRecord] = []
    for ev in events:
        rec = make_clip(ev, chunks, tag=tag, species=species)
        if rec is not None:
            out.append(rec)
    return out
