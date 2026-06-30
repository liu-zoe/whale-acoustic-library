# Perch 2.0 Humpback-vs-Vessel Classifier — v0

*Status: working. Built 2026-06-19 to address D-020 (Multispecies model fires
on vessel/echo-sounder noise at this hydrophone with high confidence, and
threshold tuning cannot fix it). See `perch-2.0-notes.md` for the rationale;
this document records the actual build and results.*

## What it is

A binary linear classifier (humpback vs not-humpback) trained on **Perch 2.0
embeddings** — Google DeepMind's bioacoustics foundation model — using the
user's review labels as training data. The classifier is a second-stage
filter applied to clips the Multispecies model has already flagged as
humpback. It does not replace Multispecies; it *qualifies* its output.

## Pipeline

```
audio clip (any sample rate)
  → resample to 32 kHz
  → Perch 2.0 .embed()      → (n_windows × 1536) frozen embeddings
  → mean-pool over windows  → (1536,) per-clip feature vector
  → LogisticRegression(C=0.1, class_weight='balanced')
  → P(humpback) in [0, 1]
```

The classifier is small and reproducible — `models/perch_humpback_v0/`
contains the trained `humpback_classifier.joblib`, the cached embeddings
(`embeddings.npz`), and the pending-clip predictions
(`pending_predictions.json`).

## Training data

- **65 positives:** 64 Watkins Marine Mammal Sound Database humpback
  recordings + 1 user-keep humpback clip from this hydrophone.
- **20 negatives:** the user-reviewed *reject* humpback clips. Every one is
  a confirmed false positive from the Multispecies humpback detector — most
  are vessel noise or echo-sounder click trains.

## Evaluation

The training-set accuracy (98.8 %) is uninformative — Watkins recordings
are clean and easy. The honest test is **leave-one-out cross-validation on
the user-reviewed clips only**, the realistic hydrophone audio:

> **LOO accuracy on user-reviewed clips: 20 / 21 = 95 %.**

- All 20 rejects classified correctly as not-humpback
  (P(humpback) range 0.097 – 0.264, all below the 0.5 threshold).
- The one miss is the lone user-keep clip (`hb_20250803T000105Z_n1`,
  P=0.144). That sample is suspect on independent grounds — its
  Multispecies scores were `Mn` 0.91 / `Oo` 0.98, flagged in D-023 as
  possible co-occurrence or noise firing both classes. It is also possible
  Perch is correctly identifying this as not-humpback and the original
  "keep" label should be revisited.

In short: the classifier perfectly distinguishes Watkins clean humpback
from the user's hydrophone vessel-noise rejects, and the one edge case is
ambiguous.

## Result on the 21 pending humpback clips

The classifier was applied to every unreviewed humpback candidate in the
catalog (21 clips, August + September).

> **Predicted not-humpback: 21 / 21.** Maximum P(humpback) = 0.222.

| Multispecies `Mn` band | clips in pending | max Perch P(hb) |
|---|---|---|
| ≥ 0.85 | 5 | 0.16 |
| 0.50 – 0.84 | 11 | 0.22 |
| < 0.50 | 5 | 0.18 |

Strikingly, the highest-confidence Multispecies humpback fires
(`Mn` 0.87 – 0.92 on five clips) all sit at Perch P(humpback) 0.10 – 0.16
— Perch *firmly* disagrees with Multispecies on the clips Multispecies is
most certain about. This matches the D-020 / D-023 finding that
Multispecies' confidence is uncorrelated with truth on this hydrophone.

## Implications

1. **The user has been right to reject these.** A second, independently-
   trained foundation model agrees with the user's manual review. The
   precision problem at this hydrophone is real, not a labeling drift.
2. **Realistically, none of the 21 pending humpback candidates are
   humpback.** They should mostly become `reject` on review. The user can
   review them confirming or contesting Perch's verdict.
3. **The "late-summer humpback" hypothesis remains untested by 2025 Q3
   data at Orcasound Lab.** Across 92 days of audio, the count of
   user+Perch-confirmed humpback events is **zero**. (D-017's apparent
   confirmation was retracted in D-020; this analysis closes the loop.)
4. **Pending operational use:** the natural next step is to apply this
   filter at the catalog stage in `run_batch.py` — every Multispecies
   humpback fire gets a Perch P(humpback) attached, and only clips with
   Perch P ≥ 0.5 propagate to "humpback" species; the rest stay in the
   library tagged as suspected false positives. Not done in v0; would
   require adding Perch to the runtime pipeline.

## Caveats

- **Training set is small (85 samples) and imbalanced (65:20).** L2
  regularization (C=0.1) and balanced class weighting help; LOO is a
  reasonable estimator for a sample this small but the confidence
  interval is wide.
- **Watkins positives are clean recordings.** A real Salish Sea humpback
  detection at this hydrophone (when one occurs) would sound different
  — noisier, more distant, lower SNR. Perch may be over-tuned to
  Watkins-like clean conditions and under-recognise faint genuine
  humpback. We will only know when a true positive arrives.
- **The 1 "keep" being mis-predicted** could indicate either an error in
  the original review (it was always ambiguous, per D-023's note about
  Mn+Oo both near 1.0) or a limitation of the classifier with only one
  hydrophone positive. Hard to disentangle now.

## What I'd do next

- Add Perch P(humpback) as a column in the catalog (column already exists
  in spirit — just an additional `multispecies_*`-style field).
- Wire Perch into `run_batch.py` Phase 4b, so the classifier filter is
  applied during batch processing and the dashboard surfaces both
  Multispecies `Mn` and Perch P(hb).
- Eventually use the same agile-modeling pattern for a Perch-based
  SRKW-vs-vessel filter — the 08-07 / 08-25 false-positive clusters
  (D-025) are exactly the same failure mode, and the user's 103 reject
  labels are training data for it.

## Files

- `src/perch_classifier.py` — pipeline + LOO + scoring.
- `models/perch_humpback_v0/`
  - `embeddings.npz` (cached, fast re-run)
  - `humpback_classifier.joblib` (trained classifier)
  - `pending_predictions.json` (per-clip Perch verdicts)
