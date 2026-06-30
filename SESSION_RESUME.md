# Session Resume / Crash Recovery

This file lives **on the flash drive** so the project can be picked up on any
machine, even if the working laptop dies. It is the single place to look first
when resuming. Keep it current.

## How Claude Code sessions persist (and the catch)

- Every `claude` CLI session has a **UUID** and is saved as a transcript at
  `~/.claude/projects/<sanitized-cwd>/<uuid>.jsonl` on the machine it ran on.
- Resume a session with:
  - `claude --resume` — interactive picker of past sessions
  - `claude --resume <uuid>` — resume a specific one
  - `claude --continue` — resume the most recent
- **The catch:** that transcript is on the *local laptop disk only*. If the
  laptop dies, the transcript (and the in-session context) is gone — this is
  exactly why the original project thread was unrecoverable.

So: **do not rely on the transcript.** Rely on this drive instead.

## Current session

- Session UUID: `95e37320-86a6-434c-90d5-face5697b7a8`
- To resume this exact thread (only if the same laptop survives):
  `claude --resume 95e37320-86a6-434c-90d5-face5697b7a8`

## What survives a crash (the durable record)

1. **This flash drive** — code, logs, models, `DECISIONS.md`, this file.
2. `DECISIONS.md` — append-only decision log; read it to see all choices.
3. Claude's memory at `~/.claude/projects/-home-y/memory/` — but that is also
   laptop-local, so the flash drive remains the source of truth.

## Project status — as of 2026-05-16

- 1-day pilot complete (2025-07-14, Orcasound Lab): 36 detection clips,
  all reviewed and labeled `keep` (0 rejects).
- Denoising rewritten (wavelet -> spectral gating); 36 clips re-denoised.
  See `DECISIONS.md` D-008.
- Google Multispecies model integrated as secondary detector; all 36 clips
  scored (every clip confirmed `Oo`/killer whale). See `DECISIONS.md` D-009.
- `call_type` populated from Multispecies; detection threshold confirmed at
  0.5; `whales` env rebuilt with full OrcaHello deps; repo paths made
  location-independent. See `DECISIONS.md` D-010..D-012.
- 1-week batch (2025-07-15..07-21) complete via `run_batch.py`: 75 new
  clips, catalog now 111. See `DECISIONS.md` D-013.
- Multispecies primary detection (humpback) built — `run_batch.py
  --multispecies-detect`. 07-02 validation found 0 humpback (quiet day);
  a positive-control test on known humpback recordings then confirmed the
  detector works (6/8, scores 0.56-0.93). See `DECISIONS.md` D-014, D-015.
- Back-processed 07-14..07-21 for humpback: 1 borderline humpback clip
  (07-17); catalog now 112 (111 SRKW + 1 humpback). See `DECISIONS.md` D-016.
- September week 09-04..09-10 with humpback detection: 5 humpback clips
  on 09-04 (peak Mn 0.998), late-summer hypothesis confirmed; catalog 215
  (209 SRKW + 6 humpback). See `DECISIONS.md` D-017.
- User review of 09-04 humpback clips revealed a pipeline bug — the SRKW
  bandpass was deleting 99.99% of humpback energy. Per-species denoise +
  spectrogram added; 6 humpback clips re-denoised (v1 preserved).
  `--multispecies-detect` now default-on. See `DECISIONS.md` D-018, D-019.
- After v3 re-render with the wider band + gentler gate, user re-reviewed
  against Watkins + Orcasound catalogue refs: **0 confirmed humpback**,
  1 uncertain, 5 reject (vessel noise mistaken for humpback). D-017's
  "late-summer hypothesis confirmed" is now qualified. Disposition:
  status quo, no code changes; humpback detections always need review.
  Perch 2.0 agile modeling deferred as the real fix. See `DECISIONS.md` D-020.
- Q3-complete-followup round (2026-06-21): added `curious` review status
  (D-029); 14 clips now tagged curious. Built and trained Perch 2.0
  humpback-vs-vessel classifier (D-027), 95% LOO accuracy, deployed.
  Acquired the Orcasound SRKW reference library — 30 Ford-Osborne call
  types + 3 favorites + clicks/whistles/vessel-pair samples (D-030).
  Tentatively labeled all 506 SRKW clips with `nearest_ref_call/pod/sim`
  via Perch nearest neighbor (D-031); weak labels (median sim 0.23) but
  keep/reject discrimination is real (keeps have higher median similarity
  than rejects). Wired Perch into `run_batch.py` (D-032). Other-mammal
  similarity search produced 1 genuine candidate (hb_20250817 -> "possibly
  finback") plus a clear demonstration that the method is fooled by the
  D-025 vessel-noise pattern (D-033). Catalog now has 4 new Perch-derived
  columns: `perch_p_humpback`, `nearest_ref_call`, `nearest_ref_pod`,
  `nearest_ref_similarity`.
- Rest-of-July batch (07-22..07-31): 11 new SRKW (late-July SRKW activity
  dropped sharply) + 4 humpback candidates on 4 *different* days — a
  different shape from D-020's vessel-cluster. 07-26 and 07-27 partly
  corroborated by Acartia sightings. Catalog now 230 (220 SRKW + 10
  humpback). 15 clips pending review. See `DECISIONS.md` D-021.
- Early-July batch (07-01..07-13): **July fully covered** (31 days). 12
  new SRKW (pre-peak buildup) + 4 humpback (1 high-conf single on 07-06,
  3 clustered in 25 min on 07-09 — vessel-like). **Inverse Acartia
  correlation** strengthens the D-020 read: model fires on days WITHOUT
  sightings, silent on days WITH them. Catalog 246 (232 SRKW + 14
  humpback). 31 clips pending. See `DECISIONS.md` D-022.
- August batch (08-01..08-31, 31 days, biggest run yet at 94.2 h):
  115 new SRKW (bursty — 84% on 4 days: 08-01/08-07/08-25/08-27) +
  15 humpback candidates in mixed patterns (some classic D-020 vessel,
  some `Mn`+`Oo`-both-high suggesting co-occurrence/noise, some
  borderline). 2 same-day Acartia matches (08-03, 08-17); rest still
  inverse correlation. Catalog 376 (347 SRKW + 29 humpback). 161
  pending. See `DECISIONS.md` D-023. **Q3 53 of ~92 days processed;
  23 days remain (rest of Sep)** to complete the original plan.
- Dashboard cache bug: `/` was serving stale HTML, making reviews look
  lost. Fixed (re-render when DB is newer). User's 118 interim reviews
  were always in the DB. See `DECISIONS.md` D-024.
- Investigation of August false positives:
  46/61 (75%) come from just 08-07 (single 20-min vessel/noise event)
  and 08-25 (multi-event day with click-train patterns from vessel
  echo-sounders). Real August SRKW ~21 clips, not 115. Writeup at
  `docs/investigation/august-false-positives.md`. See `DECISIONS.md` D-025.
- Rest-of-September batch (09-01..09-30): 105.6 h, 0 errors. **All of
  2025 Q3 now processed (92 days)** — the original §3.1 plan scope is
  complete. Catalog **548 clips (506 SRKW + 42 humpback)**, 215 pending.
  09-22 is the biggest single day on record (85 SRKW); needs priority
  review with D-025 in mind. September has the best Acartia/humpback
  correlation seen so far (3/8 vs August's 2/11). See `DECISIONS.md` D-026.
- Perch 2.0 humpback-vs-vessel classifier built (task #18 done):
  LOO accuracy 20/21 = 95% on user-reviewed clips; **all 21 pending
  humpback clips predicted not-humpback** (Perch independently corroborates
  the user's reject pattern). Across all 92 Q3 days, user+Perch-confirmed
  humpback count = **0**. Artifacts in `models/perch_humpback_v0/`. See
  `DECISIONS.md` D-027, `docs/perch-v0-classifier.md`.

## Environment

- Conda env `whales` (Python 3.11) at `/home/y/miniconda3/envs/whales`.
  Recreate with: `conda create -n whales -c conda-forge -c defaults
  python=3.11 numpy scipy matplotlib pysoundfile`
- Project code expects to live at `~/whale_acoustic_library/` (a copy is on
  this drive under `whale_acoustic_library/`).

## Open tasks

- [ ] Review the call-type/pod tentative labels on the dashboard
      (sort by "nearest-ref similarity ↓"). See `docs/srkw-reference-library.md`
      and D-031 for caveats.
- [ ] Review the 14 `curious` clips, including the new Perch-flagged
      finback candidate (`hb_20250817T052311Z_n1`).
- [ ] Apply the same agile-modeling pattern to SRKW vessel-noise filtering
      (D-025 vessel/echo-sounder false positives). 103 reject labels
      already available as training data.
- [ ] Optional: fine-tune `DENOISE_PROP_DECREASE` on the larger library.
- [ ] (Future iteration) v2 SRKW labeler with loudest-window pooling +
      larger reference pool if call-type labeling needs better discrimination.
