# Whale Acoustic Library

> *A pilot project building a curated acoustic library of Southern Resident
> Killer Whale (SRKW) calls from the [Orcasound](https://www.orcasound.net/)
> hydrophone network — combining off-the-shelf detection models, a custom
> processing pipeline, and human-in-the-loop review.*

**Status:** Pilot complete. 92 days of audio (2025 Q3) from the Orcasound Lab
hydrophone have been processed, reviewed, and catalogued. Planned next-phase
scope: additional hydrophone nodes, additional years.

## Headlines from the pilot

- **350 confirmed SRKW vocalisations** — out of 548 model detections, fully
  human-reviewed against published reference catalogues.
- **96% of confirmed activity came from 10 days out of 92** — SRKW vocal
  activity is extremely bursty at this site.
- **71% of activity occurs in the 17:00–01:00 PDT band** (evening / early
  night) — a clear nocturnal foraging signal.
- **Zero confirmed humpback whale vocalisations** despite 42 model
  detections — every one was rejected on review, and an independently-trained
  classifier agreed on all 36 firm rejects.
- **The largest single source of false positives is vessel noise** — two
  days (08-07, 08-25) generated 75% of August's rejected detections.

**Full results, charts, and discussion: [`docs/FINDINGS.md`](docs/FINDINGS.md).**

**Browse the data interactively** — companion site under
[`site/`](site/index.html): activity timeline (filterable by review status),
diurnal pattern, 15-clip showcase gallery with audio + spectrograms,
catalog browser of all 548 detections. Designed for GitHub Pages — once
the repo is pushed, enable Pages on the `main` branch from the `/site`
folder and the URL will be `https://<your-user>.github.io/<repo-name>/`.

Locally: `cd site && python -m http.server 8000`, then open
[http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## How it works (one-glance)

```
Orcasound S3 (public) → HLS chunks
  → OrcaHello SRKW detector (HuggingFace)
  → Google Multispecies whale model (Kaggle)        ← parallel humpback path
  → cluster positive windows into events
  → 30 s clip extraction + species-aware denoise + spectrogram
  → Google Perch 2.0 annotation (humpback P + Ford-Osborne call NN)
  → SQLite catalog + Acartia sighting cross-reference
  → static-HTML review dashboard
```

**Full architecture, components, and how to run it:
[`docs/BUILD.md`](docs/BUILD.md).**

## Repository layout

```
.
├── README.md                       ← you are here
├── DECISIONS.md                    ← chronological build journal (D-001 … D-034)
├── docs/
│   ├── FINDINGS.md                 ← what 92 days of audio showed
│   ├── BUILD.md                    ← architecture + how to reproduce
│   ├── bioacoustics-notes.md       ← interpretive notes from review
│   ├── srkw-reference-library.md   ← Ford-Osborne + favorites inventory
│   ├── perch-2.0-notes.md          ← Perch evaluation + role
│   ├── perch-v0-classifier.md      ← humpback-vs-vessel classifier
│   ├── gray-whale-research.md      ← out-of-scope investigation
│   ├── investigation/
│   │   └── august-false-positives.md
│   └── figures/                    ← chart PNGs for FINDINGS.md
├── REVIEW_GUIDE.md                 ← guidance for the human reviewer
├── src/
│   ├── run_batch.py                ← multi-day batch driver
│   ├── pilot/                      ← pipeline modules
│   ├── perch_classifier.py         ← humpback-vs-vessel agile-modeling tool
│   ├── srkw_call_labeler.py        ← Ford-Osborne nearest-neighbour labeller
│   ├── other_mammal_search.py      ← non-target similarity search
│   ├── review_server.py            ← Flask review dashboard
│   ├── redenoise.py                ← standalone denoise rerun
│   └── classify_species.py         ← standalone Multispecies scorer
├── models/
│   ├── perch_humpback_v0/          ← trained humpback classifier
│   └── srkw_call_labeler_v0/       ← Ford-Osborne reference embeddings
├── data/
│   └── library.sqlite              ← 548-row catalog snapshot (sanitised, relative paths)
├── scripts/
│   └── export_for_site.py          ← SQLite → JSON for the companion site
└── site/                           ← static companion site (GitHub Pages target)
    ├── index.html
    ├── audio/         (15 clips × raw + clean FLAC = 30 files, ~30 MB)
    ├── spectrograms/  (15 PNG, ~10 MB)
    └── data/          (catalog, timeline, diurnal, showcase JSON)
```

The audio data itself is **not** included in the repo — see "Reproducing"
below for how to fetch and process it. The SQLite catalog with all 548
clip records and review labels *is* included, as is the small `models/`
output set.

## Reproducing

You need:

- **Python 3.11** in a conda env (the project uses Miniconda):
  ```bash
  conda create -n whales -c conda-forge -c defaults python=3.11 \
      numpy scipy matplotlib pysoundfile pandas pip
  conda activate whales
  pip install -r requirements.txt   # generated from observed deps
  ```
- **System ffmpeg** for HLS conversion: `sudo apt-get install ffmpeg`
- **Storage:** ~30 GB free for a full 92-day reprocess (rotating scratch);
  the produced library is ~3.5 GB.
- **Models:** the Multispecies model is one manual download from Kaggle
  (anonymous); Perch 2.0 and OrcaHello are auto-fetched on first use.

Then:

```bash
# Pick a day to process (optional; defaults exist):
python src/pick_pilot_day.py

# Run a multi-day batch — humpback detection on by default,
# Perch annotation on by default:
python src/run_batch.py --start 2025-07-01 --days 92

# Launch the review dashboard:
python src/review_server.py    # → http://127.0.0.1:5000
```

See [`docs/BUILD.md`](docs/BUILD.md) for details.

## How this was built

Built collaboratively with [Claude Code](https://claude.com/claude-code) — an
AI coding assistant. Pipeline design, code, and analyses were developed
iteratively across many sessions; all detection labels were assigned by a
human reviewer (a non-bioacoustician using public reference catalogues such
as [Orcasound's Ford-Osborne catalogue](https://www.orcasound.net/data/product/SRKW/call-catalog/srkw-orca-call-catalog.html)
and the [SFU HALLO call library](https://orca.research.sfu.ca/call-library/)).
See [`DECISIONS.md`](DECISIONS.md) for the chronological build journal
(D-001 through D-041) — every non-trivial choice is recorded with its
rationale.

## What this is, what this isn't

**Is:** a working end-to-end pipeline from raw Orcasound HLS audio to a
human-reviewed, catalogued, query-able SRKW library; a benchmark of three
off-the-shelf cetacean models against 2,200 hours of one site's audio; and
a record of what worked, what didn't, and what we learned along the way.

**Isn't:** an authoritative bioacoustic catalogue (the reviewer is not a
trained bioacoustician — call-type / pod labels are best-effort hypotheses,
not identifications); a generalisable detector (the false-positive patterns
and detection biases described here are specific to this hydrophone, this
quarter, and these models); a comprehensive census of marine mammal
activity (the hydrophone's 24 kHz Nyquist limits species coverage — see
"What didn't work" in `FINDINGS.md`).

## Acknowledgements

- **[Orcasound](https://www.orcasound.net/)** — for operating the hydrophone
  network, publishing the audio under CC BY-NC-SA, and curating the
  Ford-Osborne SRKW call catalogue and humpback reference samples used
  throughout this work.
- **OrcaHello** (`orcasound/orcahello-srkw-detector-v1`) — the primary
  SRKW detector used in Phase 2.
- **Google Research** — Multispecies Whale Model (Kaggle) and Perch 2.0
  bioacoustics foundation model.
- **Watkins Marine Mammal Sound Database** (WHOI / Internet Archive) —
  humpback and other-whale reference recordings used for Perch-classifier
  training and validation.
- **Acartia** (`acartia.io`) — visual sightings data for cross-reference.
- **Ford 1987**, *"A catalogue of underwater calls produced by killer whales
  (Orcinus orca) in British Columbia"* — the canonical taxonomy this work
  relies on.

## Licence

- **Code** in `src/` is released under the MIT License — see
  [`LICENSE`](LICENSE).
- **Catalogue metadata** (`docs/`, `DECISIONS.md`, the SQLite schema
  + labels): CC BY-NC-SA 4.0, derived from Orcasound under the same licence.
- **Audio data** is not redistributed by this repository — the original
  Orcasound recordings are CC BY-NC-SA 4.0 and remain hosted at
  `orcasound.net`. The Watkins recordings are freely available for non-
  commercial research from WHOI.

This is non-commercial research; for any commercial use, contact Orcasound
directly at `info@orcasound.net` for the audio licensing.
