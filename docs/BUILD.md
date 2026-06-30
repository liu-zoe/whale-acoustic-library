# Building the Whale Acoustic Library

How this app is built — architecture, components, and how to reproduce it.
This is a living document; it grows as the project does. The terse,
chronological decision log lives in [`../DECISIONS.md`](../DECISIONS.md).

---

## 1. What it is

A searchable acoustic library of whale calls — denoised, labeled, playable
audio clips with matching spectrograms — built from public hydrophone data.
The first focus is **Southern Resident Killer Whales (SRKW)** recorded by the
[Orcasound](https://www.orcasound.net) network.

The app takes raw underwater audio, finds the moments a whale is calling,
cuts and cleans short clips around them, identifies the species/call type,
cross-references known sightings, and catalogs everything into a database
with a browser-based review UI.

---

## 2. Data source

- **Network:** Orcasound community hydrophones, streaming live to the public
  AWS S3 bucket `audio-orcasound-net` (anonymous/unsigned access).
- **Format:** HLS streaming — short `.ts` audio segments under Unix-timestamp
  folders, indexed by `.m3u8` playlists.
- **Scope so far:** the `rpi_orcasound_lab` hydrophone (Orcasound Lab).
- **Sightings:** the [Acartia](https://acartia.io) dataset of confirmed whale
  sightings, used to corroborate detections by time and location.

---

## 3. Pipeline architecture

`src/run_pilot.py` orchestrates the run end to end. The stages:

```
 S3 (audio-orcasound-net)
   │
 1 Download ......... pull a day's .ts segments, assemble 60 s WAV chunks
   │
 2 Detect ........... OrcaHello scores every 3 s window for SRKW calls
   │
 3 Cluster .......... merge contiguous positive windows into detection events
   │
 4 Clip ............. cut a 30 s clip per event; denoise; render spectrogram
   │
 4.5 Multispecies ... Google model scores species + call type per clip
   │
 5 Catalog .......... write clip metadata to SQLite; cross-reference Acartia
   │
 6 Review page ...... render the static HTML review UI
   │
 7 Cleanup .......... delete intermediate chunk WAVs
```

The design is **iterative batch processing**: download a slice of time,
process it, keep the detection clips, discard the bulk raw audio, move on.
That keeps disk use bounded no matter how much calendar time is covered.

---

## 4. The detection model stack

Two models, layered from specific to general:

| Layer | Model | Role |
|---|---|---|
| Primary | **OrcaHello** (`orcasound/orcahello-srkw-detector-v1`, HuggingFace) | SRKW yes/no on 3 s windows. Decides what becomes a clip. |
| Secondary | **Google Multispecies Whale Model** (Kaggle `google/multispecies-whale`) | 24 kHz, 5 s windows, 12 multi-label classes — 7 species + 5 call types. Adds species ID + call-type opinion. |

OrcaHello is the gatekeeper; Multispecies is a second, independent opinion
that both cross-checks SRKW detections and would catch humpback/blue/fin/etc.

**Known gap:** neither model covers **gray whales** — the Multispecies model's
12 classes do not include them. Gray-whale coverage is currently unmet.

---

## 5. Signal processing

- **Clips:** 30 s, centered on the detection, 48 kHz / 16-bit mono WAV. Both a
  raw clip and a cleaned clip are kept, so nothing is lost.
- **Denoising:** a gentle **spectral gate** — bandpass to the call band
  (300 Hz–15 kHz), then attenuate frequency bins that sit near a
  per-clip noise floor, leaving call energy intact. The strength knob is
  `DENOISE_PROP_DECREASE` in `src/pilot/dsp.py`. (An earlier wavelet
  approach over-smoothed and was replaced — see `DECISIONS.md` D-008.)
- **Spectrograms:** one PNG per clip, dB-scaled, rendered from the cleaned audio.

---

## 6. Data model

A single **SQLite** database, `whale_library/db/library.sqlite`, table
`clips` — one row per detection clip. Key fields: timestamps, file paths
(raw / cleaned / spectrogram), OrcaHello confidence, in-band SNR, Acartia
sighting count, `species` / `call_type`, the full Multispecies score JSON,
and a `review_status` (pending / keep / reject / uncertain). The schema is
designed to migrate to PostgreSQL later if needed.

---

## 7. Review UI

`src/review_server.py` is a small Flask app. It serves a responsive,
multi-column grid (`src/pilot/ui.py`) where every clip is a card with its
spectrogram, raw + cleaned audio, metrics, and keep/reject/uncertain buttons
that write straight back to SQLite. Sort and filter controls make it a
side-by-side comparison view. Run it and open `http://127.0.0.1:5000`.

---

## 8. Repository layout

```
whale_acoustic_library/
  src/
    run_pilot.py         orchestrator — the whole pipeline
    pilot/
      config.py          paths + constants (location-independent)
      download.py        S3 / HLS download + WAV assembly
      detect.py          OrcaHello inference
      cluster.py         group positive windows into events
      clip.py            cut 30 s clips
      dsp.py             bandpass, spectral-gate denoise, spectrograms, SNR
      multispecies.py    Google Multispecies model wrapper
      catalog.py         SQLite schema + writes
      crossref.py        Acartia sighting cross-reference
      ui.py              renders the review HTML
    redenoise.py         re-denoise existing clips (standalone tool)
    classify_species.py  run Multispecies over existing clips (standalone)
    review_server.py     Flask review UI server
    pick_pilot_day.py / probe_s3.py / test_orcahello.py   setup + probes
  models/                model weights (OrcaHello InferenceSystem, Multispecies)
  docs/                  this document
  DECISIONS.md           append-only decision log — the build journal
  SESSION_RESUME.md      crash-recovery / how-to-resume notes
  REVIEW_GUIDE.md        guidance for the human review step

# pipeline output (written to the project's data drive)
whale_library/
  audio_raw/  audio_clean/  spectrograms/  db/library.sqlite  review/
```

---

## 9. Environment & running it

- **Environment:** a Conda env named `whales`, Python 3.11.
- **Core dependencies:** the OrcaHello InferenceSystem stack
  (`torch`/`torchaudio`/`torchvision`, `librosa`, `huggingface-hub`,
  `safetensors`), TensorFlow for the Multispecies model, plus `numpy`,
  `scipy`, `soundfile`, `matplotlib`, `boto3`, `m3u8`, `ffmpeg-python`,
  `opencv-python-headless`, `flask`. The exact OrcaHello pins are in
  `models/aifororcas-livesystem/InferenceSystem/pyproject.toml`.
- **System dependency:** `ffmpeg` (for HLS `.ts` → WAV conversion) —
  `sudo apt-get install -y ffmpeg`.
- **No GPU required** — all models run on CPU.

Run a pilot:

```bash
conda activate whales
python src/pick_pilot_day.py      # choose a day to process
python src/run_pilot.py           # run the full pipeline
python src/review_server.py       # then open http://127.0.0.1:5000
```

Useful flags on `run_pilot.py`: `--smoke-chunks N` (process only N chunks),
`--skip-download`, `--skip-multispecies`, `--keep-chunks`.

---

## 10. How it was built

The project is built **incrementally and validated at each step**, rather
than all at once:

1. **Plan** the scope, data source, and model choices.
2. **One-day pilot** — run the whole pipeline over a single high-activity
   day to shake out the design cheaply.
3. **Human review** of the pilot's clips to confirm the detections are real.
4. **Refine** based on what the pilot showed — e.g. the denoising algorithm
   was rewritten, and the Multispecies model was added as a second opinion.
5. **Scale** to longer time spans once the pipeline is trusted.

Every non-trivial choice — and the reason behind it — is recorded in
`DECISIONS.md`, which doubles as the project's build journal.

---

## 11. Status & limitations

- The one-day pilot (2025-07-14, Orcasound Lab) is complete: 36 detection
  clips, all human-confirmed as real killer-whale calls.
- **Next:** a one-week batch — the first true unattended multi-day run.
- **Limitations:** gray whales are not covered by either model; the
  detection threshold (0.5) has only been tested on a whale-rich day; the
  SNR figure is an in-band quality estimate, not a calibrated dB SNR.
