# Bioacoustics Learning Notes

Living document of domain knowledge accumulated during review and analysis.
Separate from `DECISIONS.md` (which logs build decisions) — these are
*interpretive* facts about whale sounds and how they appear in our pipeline.

---

## Reading echolocation clicks on a spectrogram

When a SRKW click appears as a **vertical bar that starts strong at the top
of the display and fades downward — not reaching the very bottom**, that is
the expected appearance, not an artifact. Why:

- A click is a very brief (~µs), **broadband** impulse — energy across many
  frequencies simultaneously. A single click therefore renders as a vertical
  line on a spectrogram (all frequencies at one moment in time).
- **SRKW click energy actually peaks at ~30-80 kHz**, well above our
  display's top (the model itself works up to ~24 kHz; our spectrograms cap
  at 12 kHz for SRKW). What we are looking at is only the lower-frequency
  "tail" of each click.
- **Why nothing at the very bottom:** the SRKW denoise pipeline applies a
  300 Hz **high-pass filter** (`dsp.SPECIES_BANDPASS['SRKW'] = (300, 15000)`).
  Anything below 300 Hz is removed by design — that is mostly vessel rumble,
  electrical hum, and low-frequency ambient noise.

So the "top-heavy, missing the bottom" appearance is the combination of
(a) clicks being naturally high-frequency-dominant and
(b) the design choice to filter out sub-300 Hz noise. Both intentional.

## Multiple animals when echolocation and discrete calls overlap

When you see **echolocation clicks and a discrete (tonal) call overlapping
in the same 30 s window**, the most parsimonious interpretation is **two or
more individual orcas vocalizing concurrently**, not a single animal doing
both at once. Why:

- Both signal types are produced via the killer whale's nasal / phonic-lip
  apparatus. The physical mechanism makes simultaneous production by a
  single individual difficult — calls and clicks from one animal are
  generally sequential, not concurrent.
- Salish Sea SRKW travel as social pods (J, K, L), commonly multiple
  animals together. Concurrent vocalisations across individuals are
  expected behaviour.

**Practical note:** these overlap clips are ecologically rich — they
capture pod activity, not just a lone animal. Worth flagging in review
notes; they would be valuable for any future multi-animal source-separation
work.

## SRKW slow clicks vs. continuous tones

The Orcasound click catalogue formally documents **seven ICI (inter-click
interval) categories**, including a "**very slow (>400 ms ICI)**" class.
At >400 ms between clicks you can perceive each click as a distinct tick
rather than the buzz of fast echolocation. On a spectrogram, slow clicks
appear as **spaced vertical broadband lines** rather than continuous
horizontal bands.

This matters because: when a clip "doesn't sound like typical SRKW
echolocation" but has spaced ticks, it may still be SRKW — just in the slow
echolocation regime. (Discovered while investigating the user's question
about `2025-09-22T03:09:14`.)

Reference:
- https://www.orcasound.net/portfolio/srkw-click-catalog/
- The seven categories: very fast (<50 ms), fast (50-100 ms),
  medium (100-250 ms), slow (250-400 ms), very slow (>400 ms),
  buzz (continuous), sweep (amplitude-modulated).

## SRKW discrete call sub-types (e.g. S37i vs S37ii)

The Ford-Osborne SRKW catalogue defines discrete call types (S01, S02,
…). Several of these have documented *sub-types* — variants that share
the same call code but differ in fine structure (e.g. **S37i** and
**S37ii**, **S18i** / **S18ii**, **S40R** / **S40L**). Sub-types are
ecologically meaningful: they track lineage and behavioural context.

The no-narration FO bundle we have (`testdata/srkw_reference/ford_osborne/`)
provides 30 distinct call types but **does not separate sub-types** in
its filenames — `FO-S37.flac` is a single take, not split as
`FO-S37i.flac` / `FO-S37ii.flac`. Consequence: our Perch nearest-neighbor
labeler (D-031) cannot distinguish S37i from S37ii; it will label both
as "S37" at best, or sometimes pick a similar-family call like S38.
A clip the user identifies as S37ii landing on "S38" is a labeler
limitation, not a bug.

To distinguish sub-types we would need either (a) sub-type-labelled
reference takes (the SFU library — `orca.research.sfu.ca/call-library/` —
might have this if we figure out the API), or (b) train a finer
discriminator with the user's own labeled clips once enough sub-typed
positives accumulate.
