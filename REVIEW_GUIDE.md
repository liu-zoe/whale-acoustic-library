# Manual Review Guide — for a non-expert reviewer

You do **not** need to be a bioacoustics expert to do this well. Here is how to
think about the review step.

## What your job actually is

You are **not** identifying *which* call it is (S1 vs. S16 vs. a whistle) —
that is expert work, and the catalog has ~25+ SRKW discrete call types. Your
job is the simpler triage question:

> "Is there a real whale sound in this clip, or did the model fire on noise?"

Three labels, and "uncertain" is a perfectly good answer:

- **keep** — there is clearly a whale sound in here.
- **reject** — clearly noise, a boat, or empty water; the model was wrong.
- **uncertain** — you genuinely can't tell. Use this freely. A pile of
  "uncertain" labels is honest data, not a failure.

Leave `call_type` and `species` alone — OrcaHello only says "SRKW-ish or not".

## The spectrogram is your main tool (more than the audio)

Look at the picture first, then listen to confirm.

- **Whale calls** = *structured tonal contours* — wavy lines that curve and
  sweep in frequency, usually in the ~0.5–10 kHz band, often with
  **harmonic stacks** (several parallel curved bands at once). They look
  deliberate and organic.
- **Boat / vessel noise** = broadband vertical smears, or steady flat
  horizontal bands that don't change — machine-like, monotonous.
- **Rain / static / hiss** = uniform speckled haze with no structure.
- **Echolocation clicks** = a row of evenly spaced vertical ticks.

If you see curved, sweeping, harmonically-stacked lines → almost certainly a
real call → **keep**. If you only see flat bands, smears, or haze →
**reject**.

## Use the authoritative reference, not random YouTube

Orcasound publishes a labeled SRKW call catalog with **spectrograms + audio
for each call type** — match what you see/hear against these:

- SRKW call catalog (spectrograms + recordings of each discrete call):
  https://www.orcasound.net/data/product/SRKW/call-catalog/srkw-orca-call-catalog.html
- "Learn the favorite calls of J, K, and L pods":
  https://www.orcasound.net/learn/learn-the-favorite-calls-of-the-southern-resident-killer-whales-j-k-and-l-pods/
- Orcasound learn hub: https://www.orcasound.net/learn/

This is far better than YouTube because the spectrograms are shown the same
way ours are, so you can visually pattern-match.

## Corroborating signals already in each clip card

- **n_segments** — how many 3 s windows OrcaHello flagged in a row. A clip
  with many clustered positive segments is more trustworthy than a lone
  1-segment hit.
- **peak_confidence** — the model's own probability. Higher = more confident,
  but it is not proof; trust your eyes/ears over the number.
- **Acartia sightings** — confirmed visual orca sightings near the hydrophone
  around that time. Corroboration, not proof.

## Practical tips

- It's fine to be conservative: when torn between keep and reject, choose
  **uncertain**. We can revisit those as a batch.
- Re-review is cheap — the clips are now denoised better (D-008), so calls
  are easier to hear. Consider a second pass on anything you marked
  uncertain the first time.
- Don't agonize per-clip. First impression from the spectrogram is usually
  right.
