#!/usr/bin/env python3
"""Re-denoise existing library clips with the improved spectral-gate pipeline.

Reads each clip in audio_raw/, regenerates audio_clean/ and spectrograms/, and
updates snr_db in the SQLite catalog. The raw clips are never modified, so this
is fully repeatable: tune the knobs in pilot/dsp.py and run again. It does NOT
need the download scratch buffer, ffmpeg, or the OrcaHello model.

The denoise + spectrogram are species-aware (humpback uses a much lower low-cut
than SRKW, and a 0-2 kHz spectrogram); the species is read per-clip from the
catalog. Use ``--species humpback`` to re-process only humpback clips, etc.

    conda activate whales
    python src/redenoise.py                       # all clips
    python src/redenoise.py --species humpback    # humpback only
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot.dsp import (  # noqa: E402
    bandpass_for_species,
    spectral_denoise,
    inband_snr_db,
    save_wav,
    render_spectrogram_for_species,
    SPECIES_DENOISE_PROP_DECREASE,
    DENOISE_PROP_DECREASE,
)

LIBRARY = Path("/media/y/hlabflash/whale_library")
RAW_DIR = LIBRARY / "audio_raw"
CLEAN_DIR = LIBRARY / "audio_clean"
SPEC_DIR = LIBRARY / "spectrograms"
DB_PATH = LIBRARY / "db" / "library.sqlite"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default=None,
                    help="re-denoise only clips of this species (e.g. humpback). "
                         "Default: all clips.")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"no catalog at {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    # clip_id -> species map from the catalog (so we apply the right pipeline)
    species_by_id = {
        r[0]: (r[1] or "SRKW")
        for r in conn.execute("SELECT clip_id, species FROM clips")
    }

    raws = sorted(RAW_DIR.glob("*.wav"))
    if args.species is not None:
        raws = [p for p in raws if species_by_id.get(p.stem) == args.species]
    if not raws:
        print(f"no matching clips found", file=sys.stderr)
        return 1

    print(f"re-denoising {len(raws)} clips from {RAW_DIR}"
          + (f" (species={args.species})" if args.species else "") + "\n")
    print(f"{'clip_id':40s} {'species':>9s} {'old snr':>9s} {'new snr':>9s}")
    print("-" * 70)

    for raw_path in raws:
        clip_id = raw_path.stem
        species = species_by_id.get(clip_id, "SRKW")
        audio, sr = sf.read(raw_path, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) == 2:
            audio = audio.mean(axis=1)

        band = bandpass_for_species(audio, sr, species)
        prop = SPECIES_DENOISE_PROP_DECREASE.get(species, DENOISE_PROP_DECREASE)
        clean = spectral_denoise(band, sr, prop_decrease=prop)

        save_wav(CLEAN_DIR / f"{clip_id}.wav", clean, sr)
        render_spectrogram_for_species(
            SPEC_DIR / f"{clip_id}.png", clean, sr, clip_id, species
        )

        new_snr = inband_snr_db(audio, sr)
        old_snr = conn.execute(
            "SELECT snr_db FROM clips WHERE clip_id=?", (clip_id,)
        ).fetchone()
        old_snr = old_snr[0] if old_snr else None
        conn.execute(
            "UPDATE clips SET snr_db=? WHERE clip_id=?", (new_snr, clip_id)
        )
        old_str = f"{old_snr:9.1f}" if old_snr is not None else f"{'n/a':>9s}"
        print(f"{clip_id:40s} {species:>9s} {old_str} {new_snr:9.1f}")

    conn.commit()
    conn.close()
    print("-" * 70)
    print(f"done: {len(raws)} clean clips + spectrograms regenerated; snr_db updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
