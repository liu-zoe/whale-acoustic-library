# Whale Acoustic Library — Decisions Log

This is an append-only log of decisions made during autonomous execution.
Reversible decisions are recorded but not gated. Irreversible decisions
(data deletion, schema breaks, model selection that affects accumulated
labels) require user confirmation before being acted on.

---

## 2026-04-29

### D-001 — Pilot will run OrcaHello only (skip Multispecies)
- **Reason:** Google Multispecies Whale Model is hosted on Kaggle. User has
  no `~/.kaggle/kaggle.json`. Adding a Kaggle account is out of scope for
  the pilot.
- **Impact:** Pilot will detect SRKW (the primary target) and miss
  humpback/gray. Catalog rows from the pilot will all have
  `detection_model_used = 'OrcaHello'`; the `species` column will be
  `'SRKW'` or `'unknown'`. Re-running with Multispecies later is additive
  (new rows or augmented labels) — no data loss.
- **Reversible.**

### D-002 — Project layout
- Code + venv + scratch buffer + model weights → `~/whale_acoustic_library/`
  (193 GB free on `/`, plenty)
- Final library (clips, spectrograms, SQLite) → `/media/y/hlabflash/whale_library/`
- `/media/y/hlabflash/` always mounted during the project (per user)
- **Reversible.**

### D-003 — Conda over venv
- Installing Miniconda per user preference (familiar with it; portable
  across environments). Project env will be Python 3.11 (broadest model
  support, e.g. some TF SavedModels still lag on 3.12).
- **Reversible.**

### D-004 — Hardware-driven concurrency
- 4c/8t CPU, 12 GB free RAM, no GPU.
- ffmpeg conversion: 4 parallel workers.
- OrcaHello inference: single-process, batch=8 (VGGish-class model is
  small; CPU-only).
- **Reversible.**

### D-007 — OrcaHello segment duration is 3 s, not 2.45 s
- The plan said 2.45 s windows (from the older `aifororcas-orcaml` repo).
- The current HuggingFace V1 model uses **3 s windows**. Catalog and
  metadata will reflect 3 s.
- Verified with `model.detect_srkw_from_file()` on the bundled test WAV —
  29 segments × 3 s = 87 s ≈ length of the 65 s test file (with padding).
- **Reversible** — purely a documentation update.

### D-006 — Use orcasound/aifororcas-livesystem InferenceSystem as a library
- Discovered the deployed inference system is Python 3.11+, uses `uv`, and
  loads weights from HuggingFace at `orcasound/orcahello-srkw-detector-v1`
  (public, no auth). Replaces both the older `aifororcas-orcaml` repo and
  the Google Drive weights link.
- Strategy: clone alongside our project, use its `OrcaHelloSRKWDetectorV1`,
  `AudioPreprocessor`, and HLS/ffmpeg helpers as imported modules. Our
  added value is the clip-extraction + denoise + SQLite + Acartia-crossref
  + review UI layers built on top.
- **Reversible** — if the InferenceSystem APIs prove brittle, we can fall
  back to direct HuggingFace `from_pretrained` + custom orchestration.

### D-005 — Pilot data: 2025-07-14, Orcasound Lab, single HLS session
- S3 prefix: `rpi_orcasound_lab/hls/1752476425/`
- 8,638 `.ts` segments, ~1.7 GB download, ~24 h audio (single continuous
  recording session — no restart-gap stitching required).
- Day chosen because Acartia shows 6 SRKW sightings within 50 km of the
  lab on that UTC day — highest-overlap candidate in 2025 Q3 of the 9
  days with both S3 coverage and Acartia hits.
- Selection criteria + report saved at
  `~/whale_acoustic_library/logs/pilot_day.json`.
- **Reversible** — re-run with another candidate (e.g. 2025-09-04 or
  2025-09-22) costs only download + inference time.

---

## 2026-05-16

### D-008 — Denoising switched from wavelet soft-threshold to spectral gating
- **Problem:** the pilot's cleaned clips sounded worse than the raw clips
  (muted, weakened signal). Cause: `_wavelet_denoise` used the universal
  VisuShrink threshold (~5.3x noise sigma for a 30 s clip), which over-
  smooths, and soft-thresholding shrinks signal coefficients along with
  noise. Wavelet denoising also assumes a sparse signal in white noise,
  which broadband ocean ambient is not — it added musical-noise streak
  artifacts (visible as vertical streaks in the old spectrograms).
- **Change:** new dependency-free DSP module `src/pilot/dsp.py` implements
  `spectral_denoise()` — a gentle spectral gate: per-frequency noise floor
  estimated from the quietest STFT frames, soft mask, smoothed over
  time/frequency, with `prop_decrease=0.6` so noise-only bins drop ~8 dB
  rather than being annihilated. `clip.py` now imports from `dsp.py`; the
  wavelet path and `pywt` dependency are removed.
- **SNR metric replaced:** the old `snr_db` compared the denoised clip to
  its own residual, so it just measured how much was removed (always
  negative). New `inband_snr_db` = loud-frame vs quiet-frame energy in the
  500 Hz-12 kHz call band — a real, interpretable signal-quality score.
- **Re-applied to the 36 pilot clips** via new tool `src/redenoise.py`,
  which reprocesses from `audio_raw/` (no re-download or re-inference) and
  updates `snr_db` in the catalog. Old clean clips backed up to
  `whale_acoustic_library/_backup_audio_clean_v1_wavelet/`.
- **Reversible** — raw clips are untouched; tune `DENOISE_PROP_DECREASE` in
  `dsp.py` and re-run `redenoise.py`. Review labels are unaffected: the
  underlying detection is unchanged, only audibility improved. (Supersedes
  the wavelet choice in project plan §3.5.)

### D-009 — Google Multispecies Whale Model integrated as secondary detector
- Model downloaded from Kaggle (`google/multispecies-whale`, TF2 v2) to
  `models/multispecies-whale/`. `tensorflow-cpu` added to the `whales` env
  (`pip` had to be installed into the env first).
- Model spec (read from its own `metadata` signature): 24 kHz mono input,
  5 s windows, 12 multi-label classes — 7 species (`Oo` orca, `Mn`
  humpback, `Bm` blue, `Bp` fin, `Ba` minke, `Be` Bryde's, `Eg` N. Atlantic
  right) plus 5 call-type classes (`Call`, `Echolocation`, `Whistle`,
  `Upcall`, `Gunshot`). **No gray whale class** — the plan's gray-whale
  goal (§3.2) is not served by this model.
- New code: `src/pilot/multispecies.py` (model wrapper) and
  `src/classify_species.py` (standalone scoring tool, mirrors redenoise.py).
- Catalog: three additive columns — `multispecies_scores` (JSON of all 12
  class scores), `multispecies_top`, `multispecies_top_score`. The
  OrcaHello-derived `species` / `call_type` columns were left untouched so
  the two detectors' opinions stay separable. DB backed up to
  `library.sqlite.bak-pre-multispecies` first.
- Result on the 36 pilot clips: every clip scores `Oo` top (0.78-1.00, mean
  0.985) — the Multispecies model independently confirms all 36 OrcaHello
  SRKW detections. Call types present: `Call` in 27/36 clips,
  `Echolocation` in 16/36, `Whistle` in 2/36. No humpback or other species
  (all < 0.16) — a pure-SRKW day, as expected.
- **Reversible** — raw clips untouched; columns are additive and
  recomputable by re-running `classify_species.py`.
- Pipeline wiring: `run_pilot.py` now runs Multispecies as Phase 4.5 (after
  clip materialization, before catalog insert). `catalog.py` carries three
  `multispecies_*` columns in its schema + `insert_clips`, with an
  ALTER-based migration for pre-existing catalogs. New `--skip-multispecies`
  flag; if the model weights or TensorFlow are absent the pipeline logs a
  warning and continues OrcaHello-only. Verified with unit tests of
  `catalog` + the `multispecies` helper; a full end-to-end pilot was not
  re-run here (this env lacks the OrcaHello InferenceSystem deps).

### D-010 — OrcaHello detection threshold stays at 0.5 for the 1-week batch
- The pilot review confirmed all 36 detections as real killer-whale clips
  (0 rejects) at the 0.5 local threshold, and the Multispecies model
  independently scored every clip `Oo` >= 0.78. So 0.5 holds for the next
  (1-week) batch. Caveat: the pilot day was whale-rich — the false-positive
  rate is not truly tested until the batch hits quiet days. Revisit then.

### D-011 — call_type populated from the Multispecies model
- The catalog `call_type` column is now derived from the Multispecies
  call-type classes: dominant of Call->discrete, Echolocation->click,
  Whistle->whistle at score >= 0.5, else 'unknown'. Logic in
  `multispecies.call_type_from_multispecies()`, used by both
  `catalog.insert_clips` (pipeline) and `classify_species.py`. Backfilled the
  36 pilot clips: 20 discrete, 10 click, 5 unknown, 1 whistle. The full
  multi-label scores remain in `multispecies_scores`.

### D-012 — Repo made location-independent; whales env rebuilt for the pilot
- `config.REPO_ROOT` now resolves from the file location instead of a
  hardcoded `/home/y/whale_acoustic_library`, so the project runs from the
  flash drive (or anywhere) without edits.
- The `whales` env was rebuilt with the full OrcaHello InferenceSystem deps
  (torch/torchaudio/torchvision 2.9.x CPU, librosa, huggingface-hub,
  safetensors, pydub, m3u8, ffmpeg-python, opencv-python-headless, pytz).
  Verified: OrcaHello detector and Multispecies model both load. Remaining:
  system `ffmpeg` needs `sudo apt-get install -y ffmpeg` (sudo password).

### D-013 — 1-week detection batch (2025-07-15..07-21) complete
- New `src/run_batch.py` drives the pipeline day-by-day (run_pilot.py is
  single-day): each day is fully processed, then its chunk WAVs are deleted
  before the next — peak scratch stays ~1 day. Ran 17.9 h wall time, 0 day
  failures. Batch started at 07-15 (not the pilot's 07-14) so the 36
  reviewed pilot clips and their labels were not touched by INSERT OR REPLACE.
- 75 new detection clips; catalog now 111. Per day: 07-15 (2), 07-16 (66),
  07-17 (5), 07-18 (2), 07-19/20/21 (0).
- Coverage gaps in the Orcasound stream: 07-16 had ~15.7 h of audio, 07-20
  ~7 h; the other days near-full 24 h.
- Threshold signal: many new detections sit right at the 0.5 threshold
  (conf 0.50-0.55), and 8 of the 75 are NOT corroborated by Multispecies
  (`Oo` < 0.5) — likely false positives the whale-rich pilot day never
  surfaced. This is the data to revisit the threshold with (see D-010).
- `pandas` was added to the `whales` env (needed by `crossref.py`).

### D-014 — Multispecies primary detection built; 07-02 validation inconclusive
- Phase 2b added: `pilot/multispecies_detect.py` scans chunk audio with the
  Multispecies model (humpback / `Mn`, threshold 0.5, 2.5 s hop) and reuses
  the OrcaHello clustering + clip path. New `run_batch.py --multispecies-detect`
  flag; humpback clips get a `hb_` clip-id tag and `species=humpback`.
- `catalog.insert_clips` now (a) preserves `review_status`/`review_note`
  across re-runs — `INSERT OR REPLACE` no longer wipes review labels — and
  (b) takes a per-call `species` argument.
- Validation run: 2025-07-02 (chosen for full 24 h audio + humpback
  sightings near the lab; 07-23 was dropped — only ~7 h of audio). Result:
  **0 humpback, 0 SRKW** — an acoustically quiet day. The Multispecies scan
  ran clean over 1031 chunks in ~15 min (cheap; total run 2.8 h, dominated
  by download + OrcaHello).
- **Inconclusive:** 0 detections neither confirms nor refutes the detector.
  Acartia *visual* sightings are a weak proxy — summer feeding-ground
  humpbacks vocalize sparsely. Next steps proposed: a positive-control test
  on known humpback audio, and adding max-score logging so 0-detection days
  still report the highest humpback score seen.

### D-015 — Positive-control test confirms the humpback detection path works
- Ran the exact Phase 2b chain (`score_windows` -> `multispecies_detect` ->
  `cluster`) on known humpback recordings from the Watkins Marine Mammal
  Sound Database (`testdata/`, via `src/positive_control.py`).
- Result: humpback detected in 6/8 recordings, peak `Mn` scores 0.56-0.93
  (one 10-min recording produced 18 events). The 2 misses are a 2.8 s
  sub-window snippet and one faint clip (max 0.245) — not wiring failures.
  Orca clips (negative control) score `Mn` <= 0.015 — no false positives.
- **Conclusion:** the detector and the Phase 2b wiring are sound. The 07-02
  zero result (D-014) therefore means no humpback vocalized detectably at
  the hydrophone that day — not a broken detector. Humpback detection is
  validated and ready to enable on real batches.

### D-016 — Back-processed 07-14..07-21 for humpback: 1 borderline detection
- Re-ran `run_batch --multispecies-detect` over the 8 already-processed days
  (21.7 h, 0 errors). Review labels survived — the 36 pilot 'keep' clips are
  intact, confirming the review-preservation fix through a real re-run.
  Catalog: 111 SRKW + 1 humpback = 112 clips.
- Humpback: **1 clip**, on 2025-07-17 (detection-scan score 0.756). Per-day
  max humpback scores (the new logging): 07-14 0.41, 07-15 0.42, 07-16 0.48
  (near-miss), 07-17 0.76, 07-18 0.32, 07-19 0.26, 07-20 0.00 (~7 h audio),
  07-21 0.39.
- The one humpback clip is **borderline**: its clip-level Multispecies
  re-score is `Oo` 0.54 > `Mn` 0.41 — on the full 30 s the model leans
  killer whale (07-17 also had 5 SRKW clips; orcas were around). Needs human
  review. Note: the Acartia cross-ref column is SRKW-only, so it does not
  corroborate humpback.
- Verdict: humpback is genuinely sparse at Orcasound Lab in this July week —
  one borderline candidate in 8 days. The detector is sound (D-015); it is
  simply a quiet hydrophone/season for humpback.

### D-017 — September week (09-04..09-10) confirms late-summer humpback
- Ran `run_batch --multispecies-detect` over 2025-09-04..09-10 (25.8 h, 0
  errors). Catalog: 209 SRKW + 6 humpback = 215 clips (103 new pending).
- **5 humpback clips, all on 09-04**, clustered between 11:22–11:26 UTC — a
  single ~4-minute humpback vocalizing bout. 2 are unambiguous (clip-level
  `Mn` 0.984 / 0.998 with `Oo` < 0.05), 3 are borderline but firmly not
  orca (`Oo` < 0.03 on all three). Max scan score 0.99.
- The other 6 days: 0 humpback, max scan scores 0.06–0.34 — comfortably
  under threshold. Humpback was confined to that one ~4-minute window.
- Acartia corroborates: **4 humpback sightings within 50 km of the lab on
  09-04** — independent visual + acoustic confirmation on the same day.
- SRKW that week: 98 new clips (24 on 09-04, 14 on 09-05, **58 on 09-10** —
  a big orca day rivaling 07-16). 09-04 was a multi-species day.
- **The late-summer hypothesis is empirically supported**: September
  yielded confident humpback (5 clips, peak 0.99); the July week barely
  did (1 borderline). Recommendation: enable `--multispecies-detect` by
  default on future batches; consider extending coverage into late
  September / October.
- The review-preservation fix held through this run: 78 `keep` + 15
  `reject` + 19 `uncertain` reviews stayed intact while 103 new pending
  rows were added.

### D-018 — Per-species denoise + spectrogram (the humpback was being silenced)
- User's review of the 5 Sep-04 humpback clips revealed all 6 humpback
  clips going to `uncertain`, with a note on the strongest one
  (`Mn` 0.998): *"I cannot quite detect any humpback sound but curious
  why the conf is so high?"* The model was right; the pipeline was wrong.
- Root cause: the denoise bandpass was SRKW-tuned (300 Hz-15 kHz), but
  humpback song lives mostly **20-300 Hz**. The clean clips had **99.99%**
  of the humpback energy removed (20-100 Hz dropped from 1.79e-5 to
  2.77e-14; 100-300 Hz from 1.45e-5 to 6.14e-8). The user was listening to
  audio that no longer contained humpback.
- Fix: per-species pipeline in `dsp.py` (`bandpass_for_species`,
  `render_spectrogram_for_species`) and `clip.py` (`make_clip` takes
  `species`). Humpback uses **30 Hz-5 kHz** bandpass and a **0-2 kHz**
  spectrogram y-range; SRKW unchanged. Lookup in `SPECIES_BANDPASS` and
  `SPECIES_SPECTROGRAM_MAX_HZ` — easy to extend.
- Re-denoised the 6 humpback clips via `redenoise.py --species humpback`
  (now species-aware). Old v1 versions kept in
  `_backup_humpback_v1_srkw_bandpass/` for before/after comparison —
  100M+x more energy preserved at 20-100 Hz, 50x at 100-300 Hz, higher
  bands unchanged.
- The fact that two months of confident humpback detections were
  effectively un-auditable until a non-expert reviewer flagged it is a
  good case for *humans-in-the-loop* in detection pipelines.

### D-019 — `--multispecies-detect` is now the default in run_batch.py
- Switched from opt-in `action="store_true"` to default-on via
  `argparse.BooleanOptionalAction`. Pass `--no-multispecies-detect` to
  skip. The Sep week (D-017) showed the cost (+~15 min/day at 2.5 s hop)
  is negligible and the humpback yield can be real when present.

### D-020 — Humpback "detections" on this hydrophone are mostly false positives
- After re-review of the v3 humpback clips (D-018) against the Watkins
  reference *and* the Orcasound humpback catalogue, the user concluded:
  **0 confirmed humpback**, 1 uncertain (07-17 — note: *"not like Watkins
  or Orcasound"*), 5 reject (all 09-04, including the strongest at scan
  0.99 / clip-level `Mn` **1.00**). User reviewer notes flagged *"engine
  sound or some kind of machine sound."*
- Visual confirmation: the 5 09-04 clips' spectrograms show uniform
  horizontal bands at ~200-500 Hz across the full 30 s — characteristic
  of vessel engine noise (low-frequency fundamentals + harmonics), not
  humpback song. The Watkins reference shows clear rapid pulse trains;
  ours do not. The 5 detections cluster in a 4-minute window — almost
  certainly a single vessel passing near the hydrophone.
- **This qualifies D-017.** The "late-summer hypothesis empirically
  supported" finding was based on the 5 09-04 clips, which we now know
  to be false positives. The hypothesis is NOT supported by our data; it
  remains untested. **There is no confirmed humpback in the processed
  Jul/Sep audio.**
- **Threshold tuning cannot fix this.** The worst false positive scored
  scan 0.99 / Mn 1.00 — at the ceiling. No simple cutoff would have
  rejected it while preserving any chance of catching real humpback.
- The Multispecies model's positive-control success on Watkins (D-015,
  6/8 fired) is unchanged: the model can detect humpback when present.
  It just isn't *precise* on this hydrophone's audio — vessel engine
  harmonics fool it. Different acoustic environment from training data,
  different false-positive rate.
- **Disposition: Option A — status quo, no code changes.**
  `--multispecies-detect` stays default-on (D-019). Humpback detections
  go to `pending` and require review, same as today. The 5 reject + 1
  uncertain are now labeled training data for Option C below.
- **Deferred: Option C — Perch 2.0 agile modeling.** Train a custom
  humpback-vs-vessel classifier on top of Perch 2.0 embeddings, using
  our 5 reviewed rejects as negatives and Watkins/Orcasound humpback
  samples as positives. Tracked separately. See `docs/perch-2.0-notes.md`.
- This whole arc is a textbook case for humans-in-the-loop in detection
  pipelines: a confident-looking model output (Mn 1.00, scan 0.99) was a
  false positive that a careful non-expert reviewer caught because they
  compared the audio + spectrograms against baseline reference samples.

### D-021 — Rest-of-July batch (07-22..07-31) — SRKW activity drops; 4 humpback candidates
- Ran `run_batch.py --start 2025-07-22 --days 10` (29.6 h, 0 errors).
  Catalog: 220 SRKW + 10 humpback = 230 clips. Reviews preserved (162 keep
  + 30 uncertain + 23 reject + 15 new pending).
- **SRKW activity dropped sharply.** Only **11 SRKW clips across 10 days**
  (07-24: 7, 07-30: 1, 07-31: 3; six days produced none). Contrast with
  07-15..07-21's 75 SRKW. The first two weeks of July were the peak; by
  late July, the orcas had largely moved on.
- **4 humpback candidates** on 4 *different* days (07-26, 07-27, 07-28,
  07-31) — a different shape from D-020's 4-minute cluster of 5 (which
  was a single vessel). Detail:
  - 07-26 13:33  scan 0.81  Mn 0.70  Oo 0.60
  - 07-27 23:31  scan 0.90  Mn 0.84  Oo 0.86
  - 07-28 00:36  scan 0.65  Mn 0.17  Oo 0.80  (orca-leaning at clip level)
  - 07-31 21:25  scan 0.59  Mn 0.32  Oo 0.38  (borderline both)
- **Acartia corroboration is partial:** 07-26 had 2 humpback sightings
  within 50 km, 07-27 had 1 — matches the model fires on those two days.
  07-28 and 07-31 have no nearby sightings; clip-level scores are also
  weaker for those two.
- D-020 stands: these still require review. But the *shape* (spread
  across 4 days; not a uniform vessel-like band; partial Acartia match)
  is genuinely different from the Sep-04 cluster. The 07-26 and 07-27
  detections are the most credible humpback candidates the project has
  produced so far.
- Coverage gaps: 07-23 (~7 h), 07-24 (~6.7 h), 07-29 (~9.5 h), 07-31
  (~15.5 h); other days full 24 h.

### D-022 — Early-July batch (07-01..07-13) — July complete; humpback hits don't line up with Acartia
- Ran `run_batch.py --start 2025-07-01 --days 13` (43.9 h, 0 errors).
  Catalog: 232 SRKW + 14 humpback = 246 clips. Reviews preserved.
- **July is now fully processed (31 days).** Clean SRKW rise-and-fall:
  12 clips early (07-01..07-13), **111 mid-month peak** (07-14..07-21),
  11 late (07-22..07-31). The first two weeks of July were the orca peak.
- 4 new humpback candidates:
  - 07-06 08:34  scan 0.88  Mn 0.78  Oo 0.07  (high-confidence single hit)
  - 07-09 04:49 / 05:11 / 05:14  three clips in ~25 min  scan 0.51-0.57
    Mn 0.36-0.57  **Oo 0.00 on all three** (tight cluster)
- **Acartia *inverse* correlation:** humpback was sighted within 50 km on
  07-02, 07-05, 07-10, 07-11, 07-13 (6 sightings). The model fired on
  07-06 and 07-09 — **days with NO Acartia sightings**. The 5 days with
  sightings produced **0 model fires.** Both halves of this pattern point
  away from real-humpback and toward the D-020 vessel-false-positive read.
- 07-09's cluster of 3 in 25 min is structurally identical to D-020's
  Sep-04 vessel pattern. The 07-06 single hit looks high-confidence but
  has the same Oo-near-zero / no-Acartia signature.
- The 07-26 / 07-27 hits from D-021 still stand out as the only humpback
  candidates with same-day Acartia corroboration in the whole dataset.

### D-023 — August batch (08-01..08-31) — biggest run yet; Q3 nearly complete
- Ran `run_batch.py --start 2025-08-01 --days 31` (94.2 h ≈ 3.9 days,
  0 errors — the biggest single batch the project has done). Catalog:
  347 SRKW + 29 humpback = 376 clips. Reviews preserved.
- SRKW: **115 new clips across 31 days, very bursty.** ~84% on just 4
  days: 08-01 (41), 08-07 (20), 08-25 (27), 08-27 (9). The other 27 days
  averaged <1 clip each. Different rhythm from mid-July's sustained 8-day
  peak.
- Humpback: 15 candidates across 11 days, with three sub-patterns:
  - **Classic D-020 vessel signature** (high `Mn`, near-zero `Oo`):
    08-30 (Mn 0.93, Oo 0.00), 08-09 06:39 (Mn 0.77, Oo 0.00), and the
    borderline-low pair 08-16 / 08-17 05:22.
  - **Both `Mn` and `Oo` near 1.0**: 08-03 (Mn 0.91, Oo 0.98), 08-11
    (Mn 0.92, Oo 0.99), 08-09 02:36 (Mn 0.81, Oo 0.82), 08-27 (Mn 0.80,
    Oo 0.83). Either co-occurrence of orca + humpback, or noise that
    fires both classes. Distinct from the D-020 single-class signature.
  - **Borderline/mixed**: the remaining 7, mostly Mn 0.3-0.6, Oo varies.
- **Acartia in August: 12 sightings across 10 days** (08-01, 08-02 × 3,
  08-03, 08-05, 08-14, 08-17, 08-18, 08-22, 08-28, 08-29).
  Cross-reference: **2 same-day matches** — 08-03 and 08-17 — vs 9 model
  fires with no Acartia and 8 Acartia days with no fires. Inverse
  correlation still dominant but not absolute; better than D-022 (0
  matches) but worse than D-021's 2/4.
- Coverage gaps: 08-12 (~12.8 h), 08-14 (~19.4 h), 08-19 (~7 h), 08-20
  (~17 h), 08-22 (~7 h), 08-23 (~17 h), 08-24 (~7 h), 08-28 (~7 h),
  08-29 (~17 h). Otherwise near-full 24 h.
- **Q3 progress: 53 of ~92 days now processed** (all of July + all of
  August + 09-04..09-10). Remaining: 09-01..09-03 + 09-11..09-30 =
  **23 days** to fully complete the original plan-§3.1 Q3 scope.

### D-024 — Dashboard fix: re-render `index.html` when the DB is newer
- **Symptom:** user came back after a few hours of reviewing and saw 161
  pending again — appearing to have lost ~118 reviews.
- **Diagnosis:** the reviews were never lost. `POST /api/review/` updates
  SQLite immediately; the live DB still had 186 keep + 103 reject + 44
  uncertain + 43 pending. But `review_server.py`'s `/` route served a
  static `index.html` snapshot rendered at the end of the August batch
  (2026-06-07), only re-rendering if the file did not exist. On browser
  reload the user saw the stale post-batch snapshot, not the live state.
- **Fix:** `/` now re-renders whenever `library.sqlite.st_mtime` is newer
  than `index.html.st_mtime`. After any `/api/review/` write, the next
  page load picks up the new state automatically. Cheap (mtime check) and
  avoids re-rendering on every page load.
- This was a UX bug with no data loss, but it could easily have made a
  reviewer abandon the work — worth a permanent fix.

### D-025 — August SRKW false positives traced to two specific days
- User reviewing flagged a high August reject rate (64%, vs ~16% in July).
  Investigation in `docs/investigation/august-false-positives.md`.
- **75% of August's 61 SRKW rejects came from just 08-07 and 08-25.** Event-
  driven, not systemic device drift.
- **08-07** — single 20-min event 06:06-06:26 UTC. All 20 clips visually
  identical broadband haze (no tonal/click structure). Consistent with one
  slow vessel transit or sustained mechanical noise.
- **08-25** — multi-event day across 17 h (06:48-23:52 UTC), multiple
  bursts including a heavy 11-clip 23h cluster. Spectrograms vary; several
  show regular **click-train patterns characteristic of vessel echo-sounders
  / fish-finders** (the periodic sonar pulses look like SRKW echolocation
  clicks to OrcaHello).
- **Not threshold-tunable** — several 08-25 false positives crossed scan
  0.99. The "August SRKW story" in D-023 should be re-read: 08-07 and
  08-25 contributed essentially zero real orca; the actual August SRKW
  haul is ~21 clips spread across the month, not the raw 115.
- This is the same failure mode that motivated deferred task #18: a
  Perch-based agile classifier could distinguish real SRKW from this
  vessel-noise pattern. Putting the labeled rejects to work later.

### D-026 — Rest-of-September batch + Q3 complete
- Ran `run_batch.py --start 2025-09-01 --days 30` (105.6 h ≈ 4.4 days, 0
  errors). The Sep week (09-04..09-10, originally D-017) was re-run;
  review labels were preserved (D-018 fix verified end-to-end at scale).
- **All of 2025 Q3 is now processed: 92 days, 548 clips.**
  - July: 134 SRKW + 9 humpback
  - August: 115 SRKW + 15 humpback  (per D-025, ~75% of August SRKW were
    false positives concentrated on 08-07 + 08-25)
  - **September: 257 SRKW + 18 humpback**  &mdash; the dominant month
- Big SRKW days in September: **09-22 (85 clips), 09-10 (58, re-run),
  09-24 (46), 09-04 (24, re-run), 09-05 (14, re-run), 09-15 (7), 09-25
  (6).** 09-22 is the biggest single day in the entire dataset and
  belongs on the review priority list when the user returns &mdash; with
  D-025 in mind (08-25 had similar headline size but turned out to be
  vessel noise).
- 13 new humpback hits in September (in addition to the 5 D-020/Sep-week
  reproduced). Score signatures range from `Mn` 0.51 / `Oo` 0.01 (clean
  humpback signature) to `Mn` 0.87 / `Oo` 0.97 (both high &mdash; co-
  occurrence or noise firing both classes).
- **Acartia cross-reference for September: 3 of 8 humpback-fire days had
  same-day sightings within 50 km** (09-03, 09-24, 09-27). That is the
  best correlation seen so far &mdash; better than August (2/11) and far
  better than July (0/2 in D-022). Pattern is still mostly inverse, but
  improving in late summer / early fall.
- Coverage: only minor gaps (09-06 ~17 h, 09-17 ~21.6 h, 09-19 ~15.6 h,
  09-30 ~23.2 h). All other 26 days near-full 24 h.
- **215 clips pending review** when user returns.
- The original §3.1 plan scope (2025 Q3, Orcasound Lab) is now complete.

### D-027 — Perch 2.0 humpback-vs-vessel classifier (task #18) — works
- Built per `docs/perch-v0-classifier.md`. `src/perch_classifier.py` loads
  Perch 2.0 (anonymous `kagglehub` download, ~388 MB, no auth needed),
  mean-pools per-clip embeddings, trains an L2-regularized logistic
  regression on 65 positives (64 Watkins + 1 user-keep) and 20 negatives
  (user-reject humpback clips, mostly vessel noise per D-020 / D-025).
- **LOO accuracy on user-reviewed clips: 20/21 = 95%.** All 20 rejects
  classified as not-humpback (max P=0.264). The 1 miss is the lone user
  "keep" (P=0.144) — its Multispecies score signature was always suspect
  (Mn 0.91 / Oo 0.98), so we cannot tell whether Perch is wrong or the
  original keep label was.
- **Verdict on the 21 unreviewed humpback candidates: 21/21 predicted
  not-humpback** (max P=0.222). Perch firmly disagrees with Multispecies'
  highest-confidence fires (5 clips at Mn ≥ 0.87 all land at Perch
  P ≤ 0.16). An independently-trained foundation model corroborates the
  user's manual rejection pattern.
- **Practical implication:** across 92 days of 2025 Q3 audio, the count of
  user-and-Perch-corroborated humpback events at Orcasound Lab is **zero**.
  D-017's apparent confirmation is fully retracted; D-020's pessimistic
  read is supported.
- Caveats: small training set (85 samples), imbalanced (65:20), Watkins
  positives are clean recordings (the classifier may be over-tuned to
  clean conditions and under-recognise faint real humpback when one
  arrives). Won't know until a true positive appears.
- Artifacts in `models/perch_humpback_v0/` (embeddings cache, joblib
  classifier, per-clip predictions). Not yet wired into `run_batch.py` —
  current usage is a one-shot tool to score the existing pending queue.
  Wiring into the live pipeline + applying the same pattern to SRKW
  vessel-noise filtering (D-025) are the natural next moves.

### D-028 — Bioacoustics notes promoted to `docs/bioacoustics-notes.md`
- User review surfaced two repeated interpretive questions (echolocation
  spectrogram appearance; what overlapping calls + clicks mean). Both
  answered and persisted to a living `docs/bioacoustics-notes.md` doc, so
  the knowledge is captured outside the chat transcript.
- Also documented: SRKW slow clicks (>400 ms ICI from the Orcasound click
  catalogue) — visually spaced vertical broadband lines rather than the
  buzz of fast echolocation. Relevant to several "is this SRKW?"
  judgment calls in the queue.

### D-029 — `curious` review status added
- Added a fifth review label between `uncertain` and `reject`:
  **`curious`** — "interesting / possibly real but unfamiliar, save for
  further investigation." Distinct from `uncertain` ("I cannot tell
  keep-or-reject"). Examples of curious clips so far: spaced-pulse sounds
  the reviewer thinks are mammal but not standard SRKW; "very slow
  click?" candidates that may be the slow-click SRKW category.
- Changes: `review_server.ALLOWED_STATUS` adds `curious`; `pilot/ui.py`
  gains a purple-bordered card class, a `curious` filter button, and a
  per-card `curious` action button. SQLite schema is unchanged
  (`review_status TEXT`, no constraint).

### D-030 — SRKW reference library acquired (task #25)
- 57 audio files / 56 MB persisted under `testdata/srkw_reference/`.
  Bulk: the Orcasound no-narration Ford-Osborne bundle (45 FLAC / MP3 /
  OGG mirrors covering 30 distinct discrete-call types S01-S46) plus the
  3 pod-favorite calls (J-S01, K-S16, L-S19) plus 8 click samples, 1
  whistle sample, and 4 paired SRKW-call-vs-vessel-noise samples.
- Pod labeling is sparse in the source: only 16/45 FO samples have
  explicit pod tags (J:5, K:1, L:2 single-pod, rest "R" mixed-resident or
  unlabeled). Enough for tentative pod hints, not a confident classifier.
- Source: Orcasound public data products
  (https://orcasound.net/data/product/SRKW/). Licence: CC BY-NC-SA 4.0;
  this project is non-commercial research, so compliant. Attribution to
  Orcasound required in any downstream publication.
- Inventory + caveats documented in `docs/srkw-reference-library.md`.
- The SFU Ford-style catalogue (orca.research.sfu.ca/call-library) loads
  via JavaScript and exposed no API to the static fetch — deferred for a
  later devtools-driven re-attempt if richer data is needed.

### D-031 — Tentative call-type / pod labels via Perch nearest neighbor (task #26)
- `src/srkw_call_labeler.py`: embeds all 48 reference clips with
  Perch 2.0 (cached to `models/srkw_call_labeler_v0/reference_embeddings.npz`),
  then for each of the 506 catalog SRKW clips computes mean-pooled Perch
  embedding and finds the most similar reference. Writes three new catalog
  columns: `nearest_ref_call`, `nearest_ref_pod`, `nearest_ref_similarity`.
  Pod assigned by top-5 similarity-weighted vote among single-pod-labeled
  refs (or '?' if no pod-labeled ref in top-5).
- Wall time: 48 min for 506 clips on CPU. Re-runnable cheaply if cached.
- Result: **similarity is mostly low** (median 0.230, max 0.658, only 13
  clips ≥0.5). Call distribution heavily skewed: S17 (163) + S01 (128) =
  57% of all assignments. Pod skew: J=333, ?=146, L=18, K=9 — driven by
  the relative density of pod-labeled refs (4 S33-J samples and 1 J-pod
  favorite vs only 1 K and 2 L references).
- **Important signal:** despite low absolute similarities, `keep` clips
  have higher median similarity (0.273) than `reject` clips (0.186).
  Perch nearest-neighbor *is* detecting something useful even at low
  absolute values — the labels are weak hints with real signal-to-noise.
- The labels are **hypotheses for human verification, not identifications**.
  Use `nearest_ref_similarity` as a confidence band; only sim ≥ 0.5
  should be trusted as a strong call-type assignment.
- Dashboard updated: `pilot/ui.py` shows a colored `ref` tag per SRKW clip
  (green ≥0.5, olive 0.35-0.5, grey <0.35), plus a new "nearest-ref
  similarity ↓" sort option. Restart server for new code to load.

### D-032 — Perch wired into `run_batch.py` for operational annotation
- New `pilot/perch_service.py` wraps Perch 2.0 model load + the two
  operations (humpback-vs-vessel classifier from D-027; SRKW Ford-Osborne
  nearest-neighbor from D-031). Lazy model load — `PerchService()`
  constructs without loading; first `annotate()` triggers the load. Per-
  clip exceptions are caught inside `annotate()` so one bad clip never
  aborts a day.
- `pilot/catalog.py`: added `perch_p_humpback` column to the schema and a
  `_perch_fields()` formatter; `insert_clips` accepts new
  `perch_by_clip_id` kwarg. The other three Perch columns (`nearest_ref_*`)
  were already added by D-031's labeler — migration is a no-op for
  catalogs that have run it. Schema now has 29 columns total.
- `src/run_batch.py`: new Phase 4.6 between Multispecies scoring and
  catalog insert. Each SRKW clip gets `nearest_ref_call/pod/similarity`;
  each humpback clip gets `perch_p_humpback`. Daily summary line logs
  Perch SRKW-similarity median/max and Perch humpback-pass count vs
  total. New `--perch / --no-perch` flag (default: on).
- Lazy load means `--no-perch` runs pay no cost. With `--perch` on, expect
  +~10s for model load per batch + ~5-10s per clip for embedding+lookup.
  For a 30-day batch with ~150 clips that's ~25 minutes extra — small
  relative to OrcaHello inference.
- Operational implication: from the next batch onward, every new SRKW
  clip lands with a tentative call-type label and similarity score; every
  new humpback clip lands with a Perch verdict. The 376 existing clips
  already have these columns from the one-shot tools (D-027, D-031).

### D-033 — Other-mammal search: weak findings, with one genuine candidate (task #24)
- Hardware reality check first: harbour porpoise / Dall's porpoise
  echolocate at 110-150 kHz, well above our 48 kHz hydrophone's 24 kHz
  Nyquist. Porpoise is **physically not recordable** at this site; an
  embedding search can't find what the signal chain cannot capture.
- In-band non-SRKW non-humpback marine mammals we can capture: pinnipeds
  (seal / sea lion, <5 kHz) and the other baleen whales (minke / fin /
  blue / right). Of those, blue / fin / minke / Bryde's / N. Atlantic
  right are already scored per-clip by the Multispecies model (Phase 4.5,
  D-009); pinnipeds are not. The Watkins "best of whales" Internet
  Archive package is whale-only — no pinniped audio.
- Acquired 121 Watkins recordings for minke / finback / N. Atlantic right
  whale (32 MB) as additional non-target references for embedding search.
- Built `src/other_mammal_search.py`: combines 234 references (48 SRKW
  Ford-Osborne + 65 humpback positives + 121 Watkins extras) and computes
  Perch nearest neighbor for every reject + curious clip (138 candidates).
  Surfaces clips whose nearest neighbor is a non-target species AND whose
  similarity to non-target exceeds similarity to SRKW/humpback by a margin.
- Result: 19 candidates with positive margin (closer to non-target than
  to SRKW/humpback). **14 of the 19 are from the 2025-08-07 vessel-noise
  cluster** (per D-025): Perch is fooled by the cluster's low-frequency
  broadband haze matching Watkins finback-whale moans. The user has
  already labeled those clips as rejects, so we revert their auto-
  promotion and treat this as a known limitation of the method.
- Only 2 candidates met the strict threshold (sim ≥ 0.30, margin ≥ +0.05):
  one (08-07) was reverted as D-025 known vessel; one is genuinely new:
  **hb_20250817T052311Z_n1** (humpback reject, finback whale match
  sim 0.461). Promoted to `curious` with a "possibly finback" hypothesis
  for human verification.
- **Honest takeaway:** the agile-modeling pattern (Perch + small reference
  pool) gives weak findings on this hydrophone for non-target species —
  similarities are mostly < 0.5 and the low-frequency vessel-noise pattern
  contaminates the baleen-whale references. A confident non-target
  classifier here would need (a) hydrophone-recorded reference data for
  the target species, and (b) hard-negative training against the D-025
  vessel-noise patterns. Both are larger projects than the agile-modeling
  starter pass intended here.
- Artifacts: `testdata/watkins_extras/` (121 WAVs), `src/other_mammal_search.py`,
  `models/other_mammal_search_v0/candidates.json` (full per-clip scores).

### D-034 — `curious` promoted from review status to independent tag
- User feedback (after working with the labels): "I want the curious tag
  to be not mutually exclusive to keep, uncertain, and reject but rather
  a second layer on top of it." Specifically: a clip can have SRKW
  vocalization AND an unfamiliar mammal sound — should be both `keep` AND
  `curious`. Example: `orcasound_lab_20250716T032404Z_n3`.
- Change:
  - **Schema**: new `is_curious INTEGER DEFAULT 0` column. `curious` is no
    longer a valid `review_status` value (removed from
    `review_server.ALLOWED_STATUS`).
  - **API**: new endpoint `POST /api/curious/<clip_id>` with body
    `{is_curious: bool, note: optional_str}`. Note is *appended* to
    `review_note` (not replaced) so any per-clip context the user adds
    when flagging accumulates over time.
  - **UI**: per-card curious button is now a toggle (`★ curious` on/off,
    purple when on, grey when off); a separate `★ curious only` filter
    button in the top bar (orthogonal to the keep/reject/uncertain
    filters). A purple star badge appears in the top-right corner of
    every card with `is_curious=1`. Border colour still reflects
    `review_status` only.
- Migration: the 9 clips that were previously `review_status='curious'`
  were each set to `is_curious=1` plus a `review_status` reassignment per
  context. The user's explicit example
  (`orcasound_lab_20250716T032404Z_n3`) became `keep` + `is_curious=1`;
  the other 8 became `uncertain` + `is_curious=1` as the safe default,
  preserving the "needs further look" semantics. The user can re-promote
  any of those to `keep` directly via the dashboard.
- After this migration the full review queue is **0 pending** — every
  catalog clip has a review_status assigned. The next batch run will
  re-introduce pending clips for the new detections.

### D-035 — Pilot wrap-up: findings document + GitHub-ready repo
- Wrote `docs/FINDINGS.md` (~2.5k words, 3 figures, showcase appendix
  of 9 clips). Document re-frames the pilot in context: the work is a
  pilot for a planned multi-site, multi-year project; the human reviewer
  is explicitly a non-bioacoustician; the first denoise algorithm was
  retrofit-corrected (D-008 story now told as a methodology lesson).
- Generated 3 figures (`docs/figures/`): SRKW daily timeline, diurnal
  pattern, humpback model disagreement chart.
- Rewrote `README.md` as the GitHub front door — pilot status, headline
  findings, architecture sketch, repo layout, reproducing instructions,
  acknowledgements, licence.
- Added `LICENSE` (MIT for code; explicit CC BY-NC-SA inheritance for
  catalogue metadata; explicit non-redistribution note for audio).
- Added `requirements.txt` with pinned versions of all runtime deps.
- Added `.gitignore` excluding ~1.2 GB of large/non-ours material:
  third-party model checkouts (Orcasound InferenceSystem, Multispecies),
  Watkins + Orcasound reference datasets, embedding caches, logs,
  scratch buffer, historical backups.
- Added `data/library.sqlite` — snapshot of the catalog so repo readers
  can inspect schema + review labels without needing the audio. Plus
  `data/README.md` documenting the 29-column schema and a path-rewrite
  recipe for users who want to wire up their own audio later.
- **Final repo size: ~3.2 MB** of code + docs + catalog + small classifier
  artifacts. Ready to `git init && git add . && git commit && gh repo create`.

### D-036 — Catalog paths sanitised + Tier-B companion site built
- Sanitised the shipped `data/library.sqlite` to use relative paths
  (e.g. `audio_raw/orcasound_lab_...wav` instead of
  `/media/y/hlabflash/whale_library/audio_raw/...`). All 1644 path
  entries (548 clips × 3 columns) rewritten. The live runtime DB at
  `/media/y/hlabflash/whale_library/db/library.sqlite` is intentionally
  untouched — the pipeline code expects absolute paths there. Documented
  the convention + Python rewrite recipe in `data/README.md`.
- Built the companion site under `site/`:
  - 15-clip Tier-B showcase (one clip per day with confirmed SRKW
    activity, picked deterministically by
    `peak_confidence × log(n_segments+1)` from the keeps that are not
    flagged `is_curious`). Original 30-target shrank to 15 because the
    dataset is more bursty than the headline number implied (only 15
    days have confirmed keeps).
  - Audio transcoded raw + clean to FLAC (15 × 2 = 30 files, ~30 MB).
    All 15 PNG spectrograms copied alongside (~10 MB). Total site
    media: ~45 MB — under the 100 MB GitHub repo warning, so no Git
    LFS dependency.
  - `scripts/export_for_site.py` reads the shipped catalog and produces
    4 JSON files (catalog 296 KB, timeline 6 KB, diurnal <1 KB,
    showcase 4 KB) for the JS frontend.
  - `site/index.html`: vanilla HTML + Vega-Lite charts + plain JS. No
    build step. Sections: At-a-glance stats, activity-over-time chart
    (stacked-bar, status-filterable), diurnal-pattern chart, showcase
    audio gallery, full-catalog browser (searchable + filterable).
    Local smoke test: all routes 200, all JSON loads, audio + PNG
    serve correctly.
- README and DECISIONS updated. Site is ready for GitHub Pages
  deployment from `main` branch, `/site` folder.

### D-037 — Companion-site charts switched to inline SVG; v2 SRKW labeler
- Site charts (timeline + diurnal) initially used Vega-Lite from a CDN.
  User reported they didn't render — root cause likely the timeline
  spec referencing a non-existent `status_order` field. Rather than chase
  Vega-Lite bugs, rewrote both charts as plain inline SVG generated by
  vanilla JS. Zero CDN dependency, works offline. User confirmed they
  render now.
- Pulled call_type / pod tags from the showcase cards and the catalog
  browser per user feedback that the v1 labels were too unreliable to
  publish ("S01 mislabeled some as S44 / S37ii landed on S38").
- Harvested 3 new reference data sources via `scripts/harvest_refs.py`:
  - **BR2011Nora**: 22 AIFF files of Nora (a NRKW) doing FO calls S01-S19+
  - **CWRAudiosamples**: 28 WAV calls from the Center for Whale Research
    (S01-S44 including sub-types S2i, S2iii)
  - **Scalls_Snipped_by_BR**: 294 individually-snipped per-call-type
    recordings from 3 dates × multiple individuals, with pod inferable
    from individual ID prefix (C* = K pod, A* = J pod, L* = L pod).
  Total reference pool: **48 → 390 (8× growth).** 159 MB on disk.
  All material is CC BY-NC-SA from Orcasound's public data products.
- Built **`src/srkw_call_labeler_v2.py`** with two structural changes:
  1. **Loudest-window pooling.** v1 mean-pooled all 6 Perch windows from
     a 30 s clip; v2 selects the window with highest in-band energy and
     uses only that window's embedding. Also stores the selected window
     start time as `nearest_ref_window_start_s`.
  2. **Margin-threshold suppression.** If the top reference's similarity
     beats the second-nearest by less than 0.03, the label is suppressed
     (set to NULL). Captures honest uncertainty.
  New catalog columns: `nearest_ref_window_start_s`, `nearest_ref_margin`,
  `nearest_ref_labeler_version`.
- v2 results vs v1 on all 506 SRKW clips:
  - Median similarity 0.230 -> **0.308**, max 0.66 -> **0.74**.
  - **S17 over-attractor eliminated** (was 123 in v1, gone from v2's top 12).
  - 86 clips at sim ≥ 0.5 (vs 13 in v1) — better confident-match yield.
  - **382 / 506 clips (75%) now suppressed** as below the margin
    threshold — v1 was confidently wrong on many of these; v2 is honestly
    uncertain. Different failure mode, more useful for a publishable
    dataset.
  - **New imbalance: K-pod over-attractor** — BR Scalls is dominated by
    C-series individuals (K pod). New ref imbalance: most pod tags
    default to K. Pod labels are still tentative; user's earlier decision
    to omit pod from the showcase remains correct.
  - Specific user-flagged clips: `20250724T174455Z_n1` (user said
    didn't sound like S01) is now correctly suppressed; `20250716T050930Z_n7`
    (user said sounded like S44, not S01) still labels S01 — S44 has only
    one reference in our pool.
- Suppression rate is high (75%) — even with 8× more refs, the
  domain gap between archival/CWR recordings and our hydrophone audio
  is still large enough that nearest-neighbour matching is fragile.
  This is the natural cue to move to the user's Plan C: cluster the 506
  keeps and train a classifier on the user's labels of cluster
  representatives. Tracked as task #41's natural follow-up.

### D-038 — Cluster the 350 keep SRKW clips into 7 acoustic groups
- `src/srkw_clusterer.py`: embeds each keep clip with loudest-window
  Perch pooling, reduces 1536-D embeddings via UMAP (n_components=15,
  n_neighbors=15, min_dist=0, cosine metric), clusters with HDBSCAN
  (min_cluster_size=8). Also computes 2D UMAP coords for visualisation.
  installed deps: `umap-learn`, `hdbscan`.
- Result: **7 clusters + 4 noise = 100% of the 350 keeps assigned.**
  Sizes: 129, 78, 56, 29, 25, 15, 14. The biggest 3 = 75% of all keeps.
  This is much more labelable than 350 individual clips.
- For each cluster, picked the clip nearest the centroid as the
  representative + 4 next-nearest as siblings (for the labeler to
  cross-check). Cached:
    `models/srkw_clusterer_v0/keep_embeddings.npz` (raw 1536-D embeddings)
    `models/srkw_clusterer_v0/clusters.json`       (per-clip cluster id + 2D)
    `models/srkw_clusterer_v0/cluster_summary.json`(per-cluster summary)
    `site/data/clusters.json`                      (compact UI feed, 42 KB)
- v2-labeler best-guesses per cluster representative for reference
  (these aren't trusted, but useful starting context):
    cluster 3 (129 members): S03 sim 0.58
    cluster 4  (78 members): S04 sim 0.38
    cluster 6  (56 members): suppressed (sim 0.26)
    cluster 1  (29 members): suppressed (sim 0.42)
    cluster 2  (25 members): S01 sim 0.52
    cluster 5  (15 members): suppressed (sim 0.27)
    cluster 0  (14 members): S14 sim 0.39
- Built `site/clusters.html` — a static page with the 7 cluster cards,
  representative audio + spectrogram + 4 siblings each, label input
  (quick-pick dropdown + free-text + note). Labels save to browser
  localStorage and can be exported as JSON. Page references audio
  served by the existing review_server at 127.0.0.1:5000.
- Next step (after user labels): train a Perch logistic-regression
  classifier on cluster_id → label mapping, propagate to all 350 keeps,
  update catalog, restore call labels to the companion site if accuracy
  meets a self-reported bar.

### D-039 — SFU call library harvested + cluster UI fixes
- User audited the reference data ("I see at least 44 call types at
  SFU with multiple samples each") and was right to ask: I had not
  successfully harvested the SFU library before (the page is JS-rendered;
  earlier static fetches found nothing). Re-investigated by reading
  `js/GridPanel.js` source and found the data path:
  `https://orca.research.sfu.ca/call-library/catalogs/srkw.json`.
- `scripts/harvest_sfu.py`: pulls all 123 entries (audio MP3 + WebP
  spectrogram + metadata) into `testdata/srkw_reference/sfu/`.
  Filenames normalised as `S{NN}{subtype}__pods-{J,K,L}__seq{NN}.mp3`
  so the v2 labeler can parse them deterministically. ~12 MB total.
- The crucial advantage of SFU vs our existing sources:
  - **Proper sub-type breakouts** — S02i, S02ii, S02iii, S08i, S08ii,
    S37i, S37ii are separate entries (none of our other sources
    distinguish these). The user explicitly noted that v1 mislabeled
    an S37ii as S38; that's now addressable.
  - **Better pod balance** — J:65, L:74, K:24 (vs BR Scalls' heavy
    K-pod skew).
  - **2-9 takes per call code** (vs 1 take for most FO entries).
- v2 labeler updated: `collect_references()` now includes SFU; cache
  invalidated so next run picks them up. **513 total references / 43
  distinct call types** (was 390 / 30 before SFU). 11× the original 48.
- Cluster UI fixes per user feedback:
  - **Sibling spectrograms** now shown alongside audio in a grid (user
    noted spectrograms help locate the call segment).
  - **Multi-label labelling explained** in the help panel: comma-
    separated codes (`S01,S17`) for clusters spanning call types; or
    `multiple` when undisentanglable.
  - **"OK to label multiple clusters the same"** explicit — HDBSCAN
    groups by acoustic similarity, which can split S01 into recording-
    condition / individual / intensity variants. Same-label clusters
    help the classifier learn the variation.
- v2 re-run is *not* needed for the in-progress clustering workflow —
  cluster_id → user label is the training signal, references just
  provided initial hints. v2 will be re-run later if we want refreshed
  nearest-ref columns for non-keep clips.

### D-040 — Falsifying test: Perch can't separate SRKW call types (Path A)
- User reported that listening to cluster representatives didn't reveal
  coherent call-type groups. To diagnose "is it the clusters or my ear?"
  ran a falsifying test: embed all 123 SFU references (where call type
  is known to expert standard) and quantify separation by call_code in
  the same UMAP space as the 350 keeps.
- Result: **silhouette score 0.027 across 30 SFU call codes** (0 = random
  separation, 1 = perfect). Most codes have larger intra-class than
  inter-class distance — two recordings of the same call type are
  typically further apart in Perch embedding space than they are from
  the nearest other call type. All 123 SFU refs project into just 2 of
  the 7 keep-clusters, with mixed call codes within.
- Interpretation: Perch 2.0 (bird-trained) transfers to *ecotype* tasks
  per the Perch-2.0-whale paper, but not to *within-species call-type*
  discrimination at Ford catalogue granularity. The acoustic scale
  doesn't match. User's "I don't think the data is good" reading was
  diagnostically correct — it was the data approach, not their ear.
- Disposition (Path A): document this as a real publishable negative
  finding in `docs/FINDINGS.md` under "What didn't work". Drop call-type
  labels from the v1 publication. Project ships as-is.

### D-041 — Path C test setup: 7 diverse focused segments for manual labeling
- User asked to test manual labeling at small scale (5-7 clips) before
  committing to it for the full 350. Built a one-off picker:
  - Quality gate: peak_confidence ≥ 0.95 AND SNR in top quartile (≥ 5.0 dB)
  - 11 candidates passed → farthest-point sampling in Perch cosine space
    picks the 7 most-different clips.
  - For each: trim to the loudest 5 s window (just the call moment, no
    ambient padding), save as FLAC + render focused spectrogram.
- Built `site/labeling_test.html`: minimal labeling page showing focused
  segment (audio + spectrogram) alongside the full 30 s clip (audio +
  spectrogram), text input for Ford code, optional note, auto-save to
  localStorage, export-to-JSON button. Pre-linked to SFU + Orcasound
  catalogues for reference.
- Self-reflection prompts in each card (per-clip: confidence rating,
  time taken, multiple listens?) so the user can self-assess at the end
  whether scaling to 350 is feasible.
- Goal of the test: empirical answer to "is manual labeling at scale
  the right path?". If 5+ of the 7 land clearly with high confidence,
  Path C scales. If most are guesses, we accept call-type labels are
  out of scope for v1.

### D-042 — Published on GitHub: liu-zoe/whale-acoustic-library
- Repo: <https://github.com/liu-zoe/whale-acoustic-library> (public,
  MIT for code / CC BY-NC-SA for data).
- Description: "Pilot SRKW acoustic library — 92 days at Orcasound Lab,
  2025 Q3. Built with Claude Code."
- Topics: bioacoustics, orca, srkw, killer-whale,
  passive-acoustic-monitoring, orcasound, claude-code, ai-assisted.
- Initial commit: 130 files, 49.6 MB. Commit author scoped per-repo to
  `liu-zoe <liuzoe@proton.me>`; global git config unchanged.
- README updated with a "How this was built" paragraph crediting Claude
  Code + the human reviewer process + pointing at DECISIONS.md.
- GitHub Pages: built-in branch source only supports `/` or `/docs`, but
  the companion site lives at `/site`. Added
  `.github/workflows/pages.yml` (Actions workflow that uploads `site/`
  as the Pages artifact on every push to main). Pages live at
  <https://liu-zoe.github.io/whale-acoustic-library/>.
- Smoke-tested live URL: site root, JSON data, and showcase audio all
  served 200.

### D-043 — Path C viability confirmed: manual labeling scales (50-clip test)
- Follow-up to D-041's 7-clip test. Selector rewritten (`src/path_c_picker.py`)
  to use OrcaHello per-segment confidence for window centering — v1's raw
  in-band energy missed the real call on ~28% of clips because vessel
  noise had higher energy than SRKW calls. Two rounds: 30 clips
  (2026-07-06) then 20 more (2026-07-12), one labeler, using SFU HALLO +
  Orcasound Ford-Osborne libraries as references.
- Timing held across independent batches. Batch 1 (first-time labeler):
  34.3 min active, 69 s/clip. Batch 2 (returning, one week later):
  16.0 min active, 48 s/clip, zero breaks. **Combined pace: 60 s/clip
  active** — 30% faster in batch 2 with essentially identical pool quality
  (OH peak-seg 0.88 vs 0.91; SNR 6.0 vs 5.5 dB), i.e. real trained-labeler
  speedup, not easier clips.
- Label distribution stable across batches:

  | bucket | batch 1 | batch 2 | combined |
  |---|---:|---:|---:|
  | Confident (clean Ford code) | 60% | 65% | **62%** |
  | Uncertain (?-tagged) | 20% | 10% | **16%** |
  | Unmapped (blank) | 20% | 25% | **22%** |

- **The 22% unmapped is a floor, not a labeling failure.** SRKW vocalizations
  outside Ford's discrete-call catalogue — novel variants, whistles,
  aberrant calls, click-trains. In any future call-type classifier, "unk"
  should be a first-class label rather than a bucket of rejects.
- 16 distinct Ford codes across 50 clips. **S01 dominates at 46%** of
  non-empty labels — matches Ford & Ellis 1999's report of S01 as one of
  the highest-frequency J-pod calls; Orcasound Lab sits in J-pod's summer
  core habitat. Emergent labeler behaviour in batch 2 (unprompted): first
  multi-call annotation (`S01 around 2s and S08i around 4s`), first
  sub-type annotations (`S02i or S02ii`).
- 350-clip extrapolation: **5.8 h active** at combined pace, 4.7 h at
  batch-2 trained pace. Well below D-041's 20-26 h projection — the
  OrcaHello centering + labeler training curve halved the per-clip cost.
- Caveat that dictated the follow-up (D-044): 50 clips is Zipfian.
  S01 has n=19; S44 has 5; S17 has 3; six other codes have 1 each.
  A supervised classifier will be strong on S01, marginal on the mid-tier,
  untrainable on singletons. All 50 clips are Orcasound Lab (J-pod
  dominant); K and L pods have different call vocabularies (S16 for K;
  S02/S10/S37 prominent for L) and a classifier trained here needs
  cross-node validation before general use.
- Disposition: Path C is viable. Manual labeling of the full 350 was
  planned as a one-time ~6 h push, but before committing we test whether
  the 50 labels are already enough for a pilot classifier (see D-044).
- Artifacts: `data/manual_labels/path_c_v2_50.json`,
  `src/path_c_picker.py`, `site/labeling_test.html` + `data_v2.json`.

### D-044 — Pilot call-type classifier viable on head codes; targeted labeling replaces exhaustive labeling
- Move 1 of the D-043 follow-up plan. Labeling of the remaining ~300 clips
  turned out to be calendar-bottlenecked (two weeks passed with zero
  additional labeling), not hour-bottlenecked. Before committing more
  labeling time, tested whether the 50 labels already in hand were
  sufficient to bootstrap a supervised call-type classifier for at least
  the head codes.
- Perch 2.0 CPU embeddings on the 50 focused 5s FLACs from Path C v2 —
  one 1536-dim vector per clip. Regularized logistic regression
  (C=0.1, class_weight=balanced) with leave-one-out cross-validation
  under three progressively more ambitious schemes:

  | Scheme | Classes | LOO acc | Majority baseline | Lift |
  |---|---|---:|---:|---:|
  | A — S01 vs not-S01 | 2 | **79%** | 57% | +22 pp |
  | B — S01 / S44 / S17 / other-known | 4 | 71% | 58% | +13 pp |
  | C — head codes / known-tail / unk | 5 | 57% | 43% | +14 pp |

- Scheme A: S01 recall 83%, not-S01 recall 75%. All 9 misses have
  P(wrong-class) in [0.53, 0.68] — no confident errors, meaning a
  confidence-thresholded classifier (P ≥ 0.70) trades coverage for
  precision on borderline calls.
- Scheme B: S01 recall 89%; S44 (n=3) and S17 (n=2) not trainable at this
  sample size, as expected.
- Scheme C: unk recall 64% — Perch embeddings do distinguish "canonical
  Ford discrete call" from "SRKW vocalization outside catalogue," which
  validates unk-as-first-class-label from D-043.
- Uncertain-tag agreement check: on the 8 `?`-tagged clips (labels
  the human wasn't sure about), classifier agreed on 2/8 = 25%. Weak —
  but ?-tagged means labeler-of-last-resort, so tells us more about the
  labels than the classifier.
- Disposition (ship): wire the Scheme-A S01-vs-not-S01 classifier into
  `src/run_batch.py` as a new `perch_predicted_calltype` column with a
  P ≥ 0.70 confidence gate (below → `unknown-calltype`). Analogous to
  the D-027/D-032 humpback-classifier deployment. Head codes S44 and
  S17 remain unreliable and are reported as `not-S01`.
- Disposition (backfill): apply the classifier to all 506 existing SRKW
  clips in the Q3 corpus (in addition to going-forward ingestion), so
  the companion site can show S01 vs not-S01 breakdown across all of
  2025 Q3 immediately.
- Disposition (cross-node policy, forward-looking): when applied to
  non-Lab nodes (Bush Point etc.), predictions run in **shadow mode** —
  stored in DB with `cross_node_unvalidated=true`, hidden from the
  public site until we've labeled ≥20 K-pod / ≥20 L-pod clips at the
  new node and either confirmed transfer or retrained.
- Disposition (labeling plan): replace "label all 350" with **targeted
  labeling**. Future sessions surface only clips where the classifier
  is most uncertain (P near 0.5), in 20-clip chunks matching the
  batch-2 proven pace (~16 min/session). ~5 such sessions would add the
  ~100 labels needed to make S44/S17 trainable, but the S01 detector
  already auto-labels ~40% of the corpus, so forward progress is no
  longer blocked on the labeler's calendar.
- Caveats: all training data is Orcasound Lab / J-pod-dominant; 50-label
  base means all accuracy numbers have wide CIs; confidence gating
  trades coverage for precision by design.
- Artifacts: `src/perch_calltype_classifier.py`,
  `models/perch_calltype_v0/calltype_classifier_scheme_c.joblib`,
  `models/perch_calltype_v0/results.json`.

### D-045 — Full 2025 coverage at three Salish Sea nodes (expansion phase)
- Follow-through on D-044's forward plan. Once the Scheme-A S01 classifier
  was deployed (b), extending coverage without waiting for more labelling
  became the next-most-valuable move. Two dimensions to extend: **time**
  (Q3 → Q3 + Q4) and **space** (single-node → multi-node).
- Scope: **7 batches, ~184 days each × 3 additional nodes + 1 extra
  quarter at the home node.** Concretely:

  | Batch | Node | Period | Rationale |
  |---|---|---|---|
  | c) | rpi_orcasound_lab | 2025-10-01 → 2025-12-31 (92 d) | complete 2025 H2 at the home node |
  | 1 | rpi_bush_point | 2025-07-01 → 2025-09-30 (92 d) | K/L pod passage — highest cross-pod value |
  | 2 | rpi_bush_point | 2025-10-01 → 2025-12-31 (92 d) | continuity |
  | 3 | rpi_port_townsend | 2025-07-01 → 2025-09-30 (92 d) | K/L transit corridor |
  | 4 | rpi_port_townsend | 2025-10-01 → 2025-12-31 (92 d) | continuity |
  | 5 | rpi_sunset_bay | 2025-07-01 → 2025-09-30 (92 d) | southern Salish Sea |
  | 6 | rpi_sunset_bay | 2025-10-01 → 2025-12-31 (92 d) | continuity |

- **Pipeline parameterisation.** `run_batch.py` gains a `--hydrophone-id`
  argument (default `rpi_orcasound_lab`). `pilot/config.py NODES` dict
  registers each node's lat/lon + dominant pods. Downstream:
  `pilot/download.py.list_segments` accepts the id;
  `pilot/crossref.py.load_acartia_near_node(hid)` cross-references
  sightings against that node's coordinates; `pilot/catalog.py.insert_clips`
  writes `hydrophone_location` = the id with `rpi_` stripped and derives
  `cross_node_unvalidated = (hydrophone_location != 'orcasound_lab')`.
- **Cross-node classifier policy (per D-044 disposition).** The Scheme-A
  S01 classifier still runs on non-Lab clips (predictions stored in DB
  as normal), but `cross_node_unvalidated=1` on those rows. The
  companion site's call-type breakdown chart and showcase-card badges
  explicitly filter those out — they exist in the data for research
  access, but are not published as authoritative until we have ≥20
  K-pod / ≥20 L-pod validation labels per node.
- **Execution.** Batches run **sequentially, not in parallel** —
  concurrent runs share OrcaHello + Perch + Multispecies (~5-8 GB) and
  the machine's memory shows "allocation exceeds 10% of free system
  memory" warnings under a single load. Chained via
  `bin/run_expansion.sh` under `nohup` so the sequence survives Claude
  Code session end. Per-day failures already skip forward in
  `run_batch.py`; the shell script's per-batch loop doesn't `set -e`
  so a batch-level failure doesn't stop the chain.
- Estimated wall time: ~10-15 days total (prior Q3 Lab runs suggest
  1.5-2 days per 92-day batch). Runs quietly in background.
- Disposition:
  - **Ship code changes and the runner script now** as part of the
    b) + c) commit sequence (v0.2 tag).
  - **Kick off `bin/run_expansion.sh` immediately after publishing v0.2.**
  - **Do NOT auto-tag v0.3 on completion** — each node adds new data
    that should get a real look before publishing; expansion completion
    should trigger a per-node review + a manual v0.3 tag.
- Deferred: (a) per-node call-type retraining once K/L pod labels are
  collected; (b) node-specific SNR calibration (different ambient
  profiles at Bush Point / Port Townsend vs Haro Strait); (c) inter-node
  event correlation (was there a J-pod pass detected at Lab AND at
  another node within the same window?).
- Artifacts: `bin/run_expansion.sh`, `src/pilot/config.py::NODES`,
  new logs at `logs/batch_*.log`.
