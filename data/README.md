# `data/` — published catalog artifacts

This directory contains derived data products from the pilot run that are
small enough to ship with the repository. The audio data itself is not
included; see the project's top-level [`README.md`](../README.md) for the
licensing rationale.

## `library.sqlite`

The SQLite catalog of all 548 catalogued clips from the 2025 Q3 Orcasound
Lab pilot — schema, per-clip metadata, human review labels, model scores,
Perch annotations, the `is_curious` tag.

**File paths are relative.** The `raw_wav_path`, `clean_wav_path`, and
`spectrogram_path` columns are stored relative to a library root
(e.g. `audio_raw/orcasound_lab_20250714T225009Z_n1.wav`), not absolute.
Combine with your local library root to resolve. In Python:

```python
import sqlite3
from pathlib import Path

LIBRARY_ROOT = Path("~/whale_library").expanduser()  # adjust to your setup
c = sqlite3.connect("data/library.sqlite")
for clip_id, rel_path in c.execute(
    "SELECT clip_id, raw_wav_path FROM clips WHERE review_status='keep' LIMIT 5"):
    print(clip_id, LIBRARY_ROOT / rel_path)
```

If you only want metadata (labels, scores, notes, timestamps), no audio
root is needed — just query the catalog.

### Schema (29 columns)

| Column | Type | Notes |
|---|---|---|
| `clip_id` | TEXT PK | e.g. `orcasound_lab_20250714T225411Z_n5` |
| `hydrophone_location` | TEXT | currently always `orcasound_lab` |
| `start_unix`, `end_unix` | REAL | clip time window |
| `start_utc_iso`, `end_utc_iso` | TEXT | ISO 8601 |
| `species` | TEXT | `SRKW` or `humpback` |
| `call_type` | TEXT | from Multispecies — `discrete` / `click` / `whistle` / `unknown` |
| `raw_wav_path`, `clean_wav_path`, `spectrogram_path` | TEXT | absolute paths (see caveat above) |
| `sample_rate`, `duration_s` | INTEGER, REAL | 48000, 30.0 |
| `snr_db` | REAL | in-band SNR estimate (call band, raw clip) |
| `peak_confidence`, `mean_confidence`, `n_segments` | REAL, REAL, INTEGER | from the detector |
| `detection_model` | TEXT | `orcahello-srkw-detector-v1` or `multispecies-whale` |
| `detection_threshold` | REAL | 0.5 throughout |
| `acartia_sightings_within_24h_50km` | INTEGER | SRKW visual sightings near hydrophone |
| `multispecies_scores` | TEXT (JSON) | all 12 Multispecies class scores |
| `multispecies_top`, `multispecies_top_score` | TEXT, REAL | highest-scoring class + score |
| `perch_p_humpback` | REAL | humpback-vs-vessel classifier (humpback clips only) |
| `nearest_ref_call`, `nearest_ref_pod`, `nearest_ref_similarity` | TEXT, TEXT, REAL | Perch Ford-Osborne nearest neighbor (SRKW clips only) |
| `review_status` | TEXT | `keep` / `reject` / `uncertain` / `pending` |
| `review_note` | TEXT | free text from reviewer |
| `is_curious` | INTEGER | `0` / `1` — independent tag, can coexist with any review_status |

## `../models/perch_humpback_v0/humpback_classifier.joblib`

The trained scikit-learn logistic-regression classifier used for the
humpback-vs-vessel filter. 13 KB. Operates on 1536-dim Perch 2.0
embeddings. See [`../docs/perch-v0-classifier.md`](../docs/perch-v0-classifier.md)
for training details and validation (95% LOO accuracy on user-reviewed clips).

## JSON prediction snapshots

- `../models/perch_humpback_v0/pending_predictions.json` — per-clip Perch
  humpback probabilities for the 21 unreviewed humpback clips at the time
  the classifier was first trained (D-027).
- `../models/srkw_call_labeler_v0/predictions.json` — Ford-Osborne
  nearest-reference predictions for all 506 SRKW clips (D-031).
- `../models/other_mammal_search_v0/candidates.json` — non-target similarity
  search results for reject + curious clips (D-033).

These are also written into the SQLite catalog's columns; the JSON files
preserve the exact computation step for reproducibility.
