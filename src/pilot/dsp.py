"""Pure-DSP helpers: bandpass, spectral-gate denoise, SNR, spectrogram render.

Deliberately free of project imports (no config / cluster / download) so this
module can be reused both by the pilot pipeline (clip.py) and by standalone
tools like redenoise.py without pulling in boto3, the HLS client, etc.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- SRKW call band ---------------------------------------------------------
# Discrete calls, whistles and the bulk of click energy live in ~300 Hz-15 kHz.
# A gentle bandpass removes sub-300 Hz vessel rumble and out-of-band hiss
# without touching call energy.
BANDPASS_LOW_HZ = 300.0
BANDPASS_HIGH_HZ = 15000.0

# --- Humpback song band -----------------------------------------------------
# Humpback song units (moans, grunts, social calls) sit mostly between
# ~30 Hz and ~5 kHz, with substantial energy below 300 Hz. The SRKW bandpass
# silently removes that — so humpback clips need a wider low end, and their
# spectrograms should plot only the humpback band so the structure is visible.
HUMPBACK_BANDPASS_LOW_HZ = 30.0
HUMPBACK_BANDPASS_HIGH_HZ = 6000.0           # matches the Orcasound humpback catalogue
HUMPBACK_SPECTROGRAM_MAX_HZ = 6000.0         # ditto — show the harmonic stacks
# Humpback song at this hydrophone is faint compared to the catalogue's clean
# samples, so the spectral gate is dialled back from the SRKW default (0.6) so
# weak song units are not attenuated into the noise floor.
HUMPBACK_DENOISE_PROP_DECREASE = 0.3

# Per-species pipeline parameters. Add new species here as needed.
SPECIES_BANDPASS = {
    "SRKW":     (BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ),
    "humpback": (HUMPBACK_BANDPASS_LOW_HZ, HUMPBACK_BANDPASS_HIGH_HZ),
}
SPECIES_SPECTROGRAM_MAX_HZ = {
    "SRKW":     12000.0,
    "humpback": HUMPBACK_SPECTROGRAM_MAX_HZ,
}
SPECIES_DENOISE_PROP_DECREASE = {
    # SRKW value matches DENOISE_PROP_DECREASE below; inlined to avoid
    # a module-level forward reference.
    "SRKW":     0.6,
    "humpback": HUMPBACK_DENOISE_PROP_DECREASE,
}

# --- Spectral-gate denoise --------------------------------------------------
# prop_decrease is the headline strength knob:
#   1.0 = aggressive  (noise-only bins fully gated; calls can sound thin)
#   0.0 = passthrough (no denoising at all)
# 0.6 is a moderate setting: ambient noise drops ~8 dB while faint calls and
# the natural character of the recording survive. Tune this first if the
# cleaned audio still sounds off.
DENOISE_PROP_DECREASE = 0.6
DENOISE_N_STD = 1.5
DENOISE_NOISE_PERCENTILE = 25.0


def bandpass(
    audio: np.ndarray,
    sr: int,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
) -> np.ndarray:
    """Zero-phase 4th-order Butterworth bandpass."""
    nyq = sr / 2.0
    high = min(high_hz, nyq * 0.99)
    sos = scipy.signal.butter(
        4, [low_hz / nyq, high / nyq], btype="bandpass", output="sos"
    )
    return scipy.signal.sosfiltfilt(sos, audio).astype(np.float32)


def spectral_denoise(
    audio: np.ndarray,
    sr: int,
    *,
    n_fft: int = 2048,
    hop: int = 512,
    noise_percentile: float = DENOISE_NOISE_PERCENTILE,
    n_std: float = DENOISE_N_STD,
    prop_decrease: float = DENOISE_PROP_DECREASE,
) -> np.ndarray:
    """Gentle spectral-gating denoise.

    Ocean ambient noise is broadband and overlaps whale calls in time and
    frequency, so wavelet soft-thresholding (the old approach) could not
    separate the two -- it just attenuated everything, signal included.

    Instead we estimate a per-frequency-bin noise floor from the quietest
    frames of the clip (calls are intermittent bursts, so most STFT frames are
    ambient-only), then apply a *soft* gain. ``prop_decrease`` caps how far
    noise-only bins are pushed down, so the result stays natural and faint
    calls survive rather than being muted.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < n_fft:
        return audio

    f, t, Z = scipy.signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Z)
    phase = np.angle(Z)

    # Per-bin noise floor: a low percentile across time captures the ambient
    # level without needing a dedicated noise-only reference segment.
    noise_floor = np.percentile(mag, noise_percentile, axis=1, keepdims=True)
    noise_std = mag.std(axis=1, keepdims=True)
    thresh = noise_floor + n_std * noise_std

    # Soft mask in [0, 1]: ~1 where signal clearly exceeds the floor, ~0 in
    # noise-only bins.
    mask = np.maximum(mag - thresh, 0.0) / (mag + 1e-12)

    # Smooth the mask over frequency and time to suppress "musical noise"
    # (isolated surviving bins that warble).
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel /= kernel.sum()
    mask = scipy.signal.fftconvolve(mask, kernel, mode="same")
    mask = np.clip(mask, 0.0, 1.0)

    # prop_decrease blends the mask with passthrough so noise-only bins are
    # attenuated, not annihilated: the gain floor is (1 - prop_decrease).
    gain = (1.0 - prop_decrease) + prop_decrease * mask

    Z_clean = gain * mag * np.exp(1j * phase)
    _, out = scipy.signal.istft(
        Z_clean, fs=sr, nperseg=n_fft, noverlap=n_fft - hop
    )
    out = np.asarray(out, dtype=np.float32)
    # istft can return a slightly different length; match the input exactly.
    if len(out) < len(audio):
        out = np.concatenate(
            [out, np.zeros(len(audio) - len(out), dtype=np.float32)]
        )
    return out[: len(audio)]


def inband_snr_db(
    audio: np.ndarray,
    sr: int,
    *,
    low_hz: float = 500.0,
    high_hz: float = 12000.0,
) -> float:
    """Estimate in-band SNR: loud-frame vs. quiet-frame energy in the call band.

    Computed in the 500 Hz-12 kHz SRKW call band. The 90th-percentile frame
    energy stands in for 'call present', the 25th for 'ambient only'. Positive
    means the call sits clearly above the noise floor. Unlike the old metric
    (which compared the denoised clip to its own residual and so just measured
    how much got removed), this is a real, interpretable signal-quality score.
    """
    audio = np.asarray(audio, dtype=np.float32)
    f, _, Z = scipy.signal.stft(audio, fs=sr, nperseg=2048, noverlap=1536)
    band = (f >= low_hz) & (f <= high_hz)
    if not np.any(band):
        return 0.0
    frame_energy = np.sum(np.abs(Z[band, :]) ** 2, axis=0)
    if frame_energy.size == 0:
        return 0.0
    signal_e = float(np.percentile(frame_energy, 90.0))
    noise_e = float(np.percentile(frame_energy, 25.0))
    return 10.0 * math.log10((signal_e + 1e-12) / (noise_e + 1e-12))


def save_wav(path, audio: np.ndarray, sr: int) -> None:
    """Write mono 16-bit PCM, clipped to [-1, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


def bandpass_for_species(
    audio: np.ndarray, sr: int, species: str = "SRKW"
) -> np.ndarray:
    """Apply the species-appropriate bandpass (SRKW: 300 Hz-15 kHz, humpback:
    30 Hz-5 kHz). Falls back to SRKW parameters for unknown species."""
    lo, hi = SPECIES_BANDPASS.get(species, (BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ))
    return bandpass(audio, sr, lo, hi)


def render_spectrogram_for_species(
    path, audio: np.ndarray, sr: int, title: str, species: str = "SRKW"
) -> None:
    """Render a spectrogram with the species-appropriate y-axis cap so
    low-frequency calls (humpback) are not crushed into the bottom strip
    of a wide-band plot."""
    max_hz = SPECIES_SPECTROGRAM_MAX_HZ.get(species, 12000.0)
    render_spectrogram(path, audio, sr, title, max_hz=max_hz)


def render_spectrogram(
    path, audio: np.ndarray, sr: int, title: str, *, max_hz: float = 12000.0
) -> None:
    """Render a dB-scaled spectrogram PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    f, t, Sxx = scipy.signal.spectrogram(
        audio, fs=sr, nperseg=1024, noverlap=768, scaling="spectrum"
    )
    db = 10.0 * np.log10(Sxx + 1e-12)
    db -= db.max()
    ax.pcolormesh(t, f, db, shading="auto", cmap="magma", vmin=-80.0, vmax=0.0)
    ax.set_ylim(0, min(sr / 2.0, max_hz))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
