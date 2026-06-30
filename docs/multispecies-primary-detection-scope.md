# Scope — Multispecies Primary-Detection Pass

*Design scope for letting the Google Multispecies model detect non-SRKW
species (realistically humpback, maybe minke) directly, rather than only
annotating OrcaHello's clips. Not yet implemented — this is the plan.*

## 1. Goal

Add a second, independent detection path so the pipeline can surface
**humpback** (and possibly **minke**) whales, not just SRKW.

Out of scope: blue / fin (rare in these inland waters), N. Atlantic right and
Bryde's (wrong ocean / wrong habitat) — see `gray-whale-research.md` reasoning.
The Multispecies model emits all 12 classes; we act only on the realistic ones.

## 2. Why it doesn't work today

The Multispecies model currently runs only in **Phase 4.5**, scoring clips
**OrcaHello already cut**. OrcaHello fires on SRKW only, so non-orca species
are invisible unless they happen to vocalize inside an orca detection. Across
the current 111 clips, 0 have any non-orca species ≥ 0.5 — because every clip
exists *because OrcaHello found an orca there*.

## 3. Proposed design — a new "Phase 2b"

Run the Multispecies model as a **primary detector** over the same chunk WAVs
OrcaHello scans, in parallel with it:

```
 Phase 2   OrcaHello scan ........ 3 s windows -> SRKW detections
 Phase 2b  Multispecies scan ..... 5 s windows -> humpback/minke detections   [NEW]
 Phase 3   cluster ............... both detection streams -> events
 Phase 4   clip + denoise + spec . one 30 s clip per event (any species)
 Phase 4.5 Multispecies scoring ... unchanged — annotates every clip
 Phase 5   catalog ............... rows tagged with detecting model + species
```

The Multispecies model is already loaded for Phase 4.5 — Phase 2b reuses it,
no extra model load.

## 4. Component changes

| Component | Change | Size |
|---|---|---|
| `pilot/multispecies.py` | New `score_windows(audio, sr, hop_s)` returning **per-window** scores + window start times (today's `score_clip` only returns the per-class max). | small |
| `pilot/multispecies_detect.py` *(new)* | Run the model over chunk WAVs, threshold the target classes, emit detection records (mirrors `detect.py`'s role for OrcaHello). | medium |
| `pilot/cluster.py` | Generalize so it can cluster Multispecies detections too (currently tied to OrcaHello `SegmentDetection`). | small–med |
| `pilot/clip.py` (`ClipRecord`) + `DetectionEvent` | Carry `detection_model` and `species` per event/clip (today `species` is hardcoded `"SRKW"`, `detection_model` is a run-level argument). | medium |
| `pilot/catalog.py` (`insert_clips`) | Read `detection_model` / `species` **per clip** instead of per run. | small |
| `run_batch.py` / `run_pilot.py` | Add Phase 2b; merge the two event lists before Phase 4. | small–med |
| clip-id scheme | Include a species/model tag so a humpback clip and an orca clip at overlapping times get distinct IDs. | small |

This is a **moderate** change — ~7 files, one new module, mostly small edits.
The real refactor is making `species` / `detection_model` per-clip.

## 5. Data-model changes

The `clips` table already has `species` and `detection_model` columns — they
just need to vary per row:

- OrcaHello clips: `detection_model = orcahello-srkw-detector-v1`, `species = SRKW`.
- Multispecies clips: `detection_model = multispecies-whale`, `species =
  humpback` / `minke`.
- `peak_confidence` / `mean_confidence` hold the **detecting model's** score
  for that clip (OrcaHello's, or the Multispecies target-class score).
- Phase 4.5 still runs on every clip, so `multispecies_scores` is populated
  for all clips regardless of which model detected them.

No schema migration needed — only how the columns are filled.

## 6. Cost

A primary scan processes **all** the audio, not just clips. Rough estimate
(scaling from OrcaHello's ~66 min/day for ~42k windows):

- hop **1.0 s** → ~86k windows/day → ~1.5–2 h/day
- hop **2.5 s** → ~35k windows/day → ~45–60 min/day  ← recommended

Humpback calls/song units last seconds, so a 2.5 s hop catches them with
margin. At 2.5 s hop, Phase 2b adds roughly **+1 h/day** — about +50 % to a
batch's runtime (7-day batch: ~18 h → ~25 h). Exact speed should be measured
on one day before committing.

## 7. Open decisions (need your input)

1. **Target species:** humpback only, or humpback + minke? (Minke is a
   plausible-but-rarer Salish Sea visitor; including it adds some false-
   positive risk.)
2. **Per-class thresholds:** start at 0.5 (same as OrcaHello), or higher for
   caution? Multispecies scores are independent per class, not a softmax.
3. **Scan hop:** 2.5 s (recommended, ~1 h/day) vs 1.0 s (finer, ~2 h/day).
4. **Back-processing:** apply to future batches only, or also re-download &
   re-scan the already-processed days (2025-07-14…07-21)?

## 8. Risks & unknowns

- **Yield is uncertain.** We don't yet know how much humpback is on this
  hydrophone in the July window — could be a handful of clips or many.
  Worth a one-day validation run before a full batch.
- **False positives.** The model's fin-whale class already reached 0.26 on
  pure-orca clips; a primary scan over noisy audio will produce some false
  hits. A manual review pass (same UI) is the safeguard.
- **Denoising mismatch.** The clip bandpass (300 Hz–15 kHz) is SRKW-tuned and
  would trim some low humpback-song energy. Acceptable initially; a
  per-species bandpass is a possible later refinement.
- **Orca double-counting.** The Multispecies scan also lights up on orcas
  (`Oo`); Phase 2b must exclude `Oo` from its target classes so OrcaHello
  stays the single source of truth for SRKW.

## 9. Suggested rollout

1. Implement Phase 2b behind a flag (`--multispecies-detect`), default off.
2. **Validation run:** one day, humpback only, measure runtime + eyeball the
   hits in the review UI.
3. Tune threshold/hop from what the validation shows.
4. Enable for the next multi-day batch.

## 10. Effort estimate

Implementation + the one-day validation run: roughly **one focused working
session**. The full multi-day batch with Phase 2b enabled is then just
runtime (overnight+).
