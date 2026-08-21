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

## Current state (updated 2026-07-29) — v0.2 shipped, expansion chain running autonomously

### v0.2 (D-043, D-044, D-045) — S01 detector deployed + expansion started
- 50 hand-labeled clips from Path C v2 → Perch 2.0 supervised Scheme-A
  classifier (`S01 vs not-S01`, 79% LOO accuracy).
  Confidence-gated at P ≥ 0.70; abstains as `unknown-calltype` on borderline.
- Pipeline changes (all in `pilot/perch_service.py` + `pilot/catalog.py`):
  - `PerchService.predict_calltype()` uses OrcaHello per-segment confidence
    to locate the call inside a 30 s clip, then Perch-embeds the focused
    5 s window, then predicts.
  - New DB columns: `perch_predicted_calltype`, `perch_calltype_confidence`,
    `cross_node_unvalidated`.
  - `run_batch.py` gains `--hydrophone-id`; downloads, Acartia lookup,
    and catalog inserts all thread the node through.
  - `pilot/config.py::NODES` registers coordinates for the 5 known
    Orcasound public nodes.
- Backfilled onto all 503 pre-existing SRKW Q3 clips via
  `src/backfill_calltype.py`.
- Companion site: new "Call-type breakdown" section with SVG stacked bar
  (only home-node keep+uncertain clips shown; shadow-mode predictions
  from other nodes exist in DB but hidden). Showcase cards get an S01
  badge when the classifier didn't abstain.
- FINDINGS.md updated with a "Follow-up" paragraph on the Perch
  call-type limitation section, framing the pilot classifier as the
  positive counter-result.

### Expansion chain (D-045) — LAUNCHED at 2026-07-30T18:27 -05:00

**PID at kickoff**: 3280342 (bash bin/run_expansion.sh) + 3280348 (first
child, Q4 Lab batch). Launched via `nohup ... & disown` so the chain
survives Claude Code session end.

### Expansion chain (D-045) — 7 batches, sequential
- Runner: `bin/run_expansion.sh` under `nohup` — runs 7 batches
  sequentially, each ~1.5-2 days of compute:
  1. `q4_lab`  — Orcasound Lab, 2025 Q4 (completes 2025 H2 at home node)
  2. `bp_q3`   — Bush Point, Q3     (K/L pod passage)
  3. `bp_q4`   — Bush Point, Q4
  4. `pt_q3`   — Port Townsend, Q3  (K/L transit corridor)
  5. `pt_q4`   — Port Townsend, Q4
  6. `sb_q3`   — Sunset Bay, Q3     (southern Salish Sea)
  7. `sb_q4`   — Sunset Bay, Q4
- Total estimated wall time: **~3-4 months** (revised 2026-07-31 after
  observing Q4 Lab's actual ~5-hour-per-audio-day pace; my initial
  "10-15 days" estimate was wrong by ~10×).
- Non-Lab clips run in shadow mode (`cross_node_unvalidated=1` in DB) —
  S01 classifier still fires and predictions are stored, but hidden from
  the site until per-node validation labels exist.
- Serial because concurrent runs would OOM (each pipeline loads
  OrcaHello + Perch + Multispecies ~5-8 GB).

### How to check on the expansion chain
```
bash bin/expansion_status.sh          # snapshot of all 7 batches + DB counts
tail -f logs/batch_q4_lab.log         # follow the current batch
tail -f logs/expansion.out            # follow the chain's own log
pgrep -af "run_batch|run_expansion"   # confirm the chain is still alive
```

### KNOWN INCIDENTS: two BP fill-ins owed (2026-08-17 and 2026-08-18)
Both Bush Point batches OOM-crashed on the first high-volume day they
hit. See D-045 for full analysis. **Total 151 days need reprocessing.**

- **BP Q3** killed on 2025-07-22 (481 events). **71 days lost** (07-22..09-30).
- **BP Q4** killed on 2025-10-12 (257 events). **80 days lost** (10-12..12-31).

Fill-in scripts written; launch in sequence after main chain (memory
constraint — one pipeline at a time):
```
# Verify no run_batch.py is running:
pgrep -af run_batch.py || echo "safe to launch"

# BP Q3 fill-in first (~15 days of compute at BP pace):
nohup bash bin/bp_q3_fillin.sh > logs/bp_q3_fillin.out 2>&1 &
disown
# WAIT for it to finish before launching next:
wait

# Then BP Q4 fill-in (~17 days at BP pace):
nohup bash bin/bp_q4_fillin.sh > logs/bp_q4_fillin.out 2>&1 &
disown
```

**These fill-ins can OOM AGAIN** on the same big-activity days. If
exit=137: check last "day complete" line in the fill-in log, then
relaunch with `--start` bumped to the DAY AFTER the crash.

Watch for **similar OOM on PT Q3, PT Q4, SB Q3, SB Q4** — Port
Townsend and Sunset Bay sit on comparable K/L pod corridors and may
hit the same failure mode. If they do, fill-in scripts follow the
same pattern (write more via `bin/*_fillin.sh` templates).

### If the chain died (machine reboot, OOM kill, etc.) — how to resume
```
# 1. See which batches have already completed (look for BATCH COMPLETE)
grep "BATCH COMPLETE" logs/batch_*.log

# 2. Edit bin/run_expansion.sh: comment out the run_batch calls for the
#    completed batches, keep only the remaining ones.

# 3. Relaunch with nohup:
nohup bash bin/run_expansion.sh > logs/expansion_resumed.out 2>&1 &
disown
```

### Publication cadence
Tag `v0.2` was cut immediately after b) + expansion-code shipped
(commit `fd1b493`), including the S01 classifier + all multi-node
plumbing. Per D-045: **do not auto-tag v0.3 on expansion completion** —
each new node's data deserves a review pass before publishing. Cut
v0.3 manually after per-node review.

### Prior state (pre-v0.2, retained for context)

Path A published, Path C awaiting review

### Publication (Path A, complete)
- Repo live and public: <https://github.com/liu-zoe/whale-acoustic-library>
  - Owner: `liu-zoe` (per-repo git config; NOT the default account on this
    machine — see `~/.claude/projects/-home-y/memory/whale-acoustic-library-github.md`)
  - Tag `v0.1-pilot` with GitHub Release
  - Topics: bioacoustics, orca, srkw, killer-whale, passive-acoustic-
    monitoring, orcasound, claude-code, ai-assisted
- Companion site live: <https://liu-zoe.github.io/whale-acoustic-library/>
  - Deployed via `.github/workflows/pages.yml` on push to `main` when
    `site/**` changes (branch source can't target `/site`).
  - Charts are inline SVG (Vega-Lite was pulled after render bugs).
- Local git in sync with `origin/main`, last commit `ab40e4c`
  ("Path C v2: OrcaHello-centered focused-window selector + 30 clips").
  See `DECISIONS.md` D-034 → D-042 for the publication decision trail.

### Path C (test in progress)
Perch could NOT separate the 30 SRKW call types (silhouette 0.027 on SFU
references — falsifying test, D-040). Decision: publish v1 without
call-type labels and try Path C — a small manual-labeling test — to see
if labeling 350 clips by hand is even feasible.

- **v1 (7 clips)**: user labeled — ~20.8 min → projects to 20-26 h for
  350 clips. Also revealed that the loudest-window selector missed the
  real SRKW call on ~28% of clips because vessel noise had higher
  in-band energy. See D-041.
- **v2 (30 clips)**: selector rewritten to use OrcaHello per-segment
  confidence as the "where is the call" signal (`src/path_c_picker.py`).
  30 diverse focused 5 s clips generated at `site/labeling_test/audio_v2/`
  + `spectrograms_v2/`, page at `site/labeling_test.html` uses
  `data_v2.json` and localStorage key `srkw-labeling-test-v2`. **User is
  reviewing these now.**
- After user exports `manual_labels_test_v2.json`: analyze timing +
  label distribution, decide whether to grow to 50 with
  `python src/path_c_picker.py --target-total 50` (script is idempotent
  and appends to existing picks) or move on.

## How to pick up after a network drop / re-login

1. **This session (if same laptop):**
   `claude --resume 95e37320-86a6-434c-90d5-face5697b7a8`
   Transcript is at `/home/y/.claude/projects/-home-y/95e37320-…jsonl`
   (~12 MB) and persists across reboots.
2. **Labeling test — offline safe.** The page at
   `http://127.0.0.1:8000/labeling_test.html` needs a local static server
   (see below); audio + spectrograms are FLAC/PNG files on the flash
   drive. Only the "full 30 s context clip" audio comes from
   review_server on :5000. Neither needs the internet. Progress
   auto-saves to browser localStorage under key `srkw-labeling-test-v2`
   — closing the tab/browser is safe as long as the SAME browser is
   reopened. When done, click **Export labels** to download
   `manual_labels_test_v2.json`.
3. **Servers to restart if they died:**
   ```
   cd /media/y/hlabflash/whale_acoustic_library
   /home/y/miniconda3/envs/whales/bin/python src/review_server.py &
   /home/y/miniconda3/envs/whales/bin/python -m http.server 8000 -d site &
   ```
   `review_server.py` at :5000 serves the full 30 s clips + spectrograms
   from the SQLite catalog; `http.server` at :8000 serves the static
   labeling page. As of the last audit, `review_server.py` (PID 2437533)
   has been running for 10+ days.
4. **Git recovery.** Everything through `ab40e4c` is pushed to
   `origin/main`. If the flash drive is lost:
   `gh repo clone liu-zoe/whale-acoustic-library` gets code + docs +
   site back. Catalog SQLite and models are NOT in the repo — those
   only live on this drive.

## Open tasks

- [ ] User reviews 30 Path C v2 clips, exports `manual_labels_test_v2.json`,
      shares back. Then: timing analysis + go/no-go on 350-clip manual
      labeling.
- [ ] Optional: grow Path C to 50 clips (`--target-total 50`) after v2
      results are in.
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
