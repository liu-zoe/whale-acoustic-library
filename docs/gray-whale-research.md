# Gray-Whale Sound Detection — Research Notes

*Exploratory investigation (low priority). Question: can we cover gray whales,
given that neither OrcaHello (SRKW-only) nor the Google Multispecies model
includes a gray-whale class?*

## Summary

There is **no off-the-shelf gray-whale acoustic detector** to drop in, and
gray whales are largely a **geographic/seasonal mismatch** for this project's
current setup. Covering them would mean building a custom detector. It is a
real gap, but a low-stakes one for the project as currently scoped.

## 1. Is there an existing model?

No. Bioacoustic ML for cetaceans is mature — CNNs on spectrograms, multi-
species models, NOAA's passive-acoustic archives and species-specific
detectors (right whale, Antarctic blue/fin, humpback, sperm whale). But:

- The **Google Multispecies model** — the broadest one available, already in
  our stack — deliberately omits gray whales (its 7 species are orca,
  humpback, blue, fin, minke, Bryde's, N. Atlantic right).
- No published ready-to-use gray-whale detector or large annotated
  gray-whale call dataset surfaced. Gray whales are an acoustic-monitoring
  gap relative to humpback/orca/right-whale.

## 2. What gray-whale calls look like

Gray whales vocalize **low and pulsed**, roughly **20 Hz – 2 kHz**:

- **M3** — low-frequency moans, ~48 Hz mean, ~1.8 s; the dominant call on
  migration (~47% of the repertoire).
- **M1** — second most common migration call (~37%).
- **S1 "knock" calls** — pulsed bursts, 3–18 pulses each, individual pulses
  100–1600 Hz; the dominant sound in the breeding lagoons.

Most call energy sits **below a few hundred Hz** — much lower than SRKW
discrete calls, whistles, and clicks.

## 3. Relevance to this project (the important part)

- **Geography:** the Salish Sea gray whales — the "Sounders," ~a dozen
  individuals — feed in **North Puget Sound** (Whidbey/Camano Island,
  Saratoga Passage, Port Susan, Snohomish Delta). The Orcasound **Lab**
  hydrophone is on the west side of San Juan Island (Haro Strait), ~50–100 km
  away in different waters. Gray whales are rarely near this hydrophone.
- **Season:** the Sounders are present mainly **March–May** (spring), with
  occasional Dec–Jan visits. This project's focus is the **SRKW summer peak,
  July–September** — almost no overlap with gray-whale presence.

So for the current hydrophone + season, gray whales would contribute very
little even with a perfect detector.

## 4. Technical implications if we did pursue it

- Our clip denoising bandpass is **300 Hz – 15 kHz** — tuned for SRKW. It
  would **cut most gray-whale call energy** (M3 at ~48 Hz is well below
  300 Hz). The 48 kHz hydrophone *captures* low frequencies fine; the
  pipeline's processing just discards them.
- A gray-whale path would need its own low-frequency front-end and detector.

## 5. Options, if it becomes a priority

1. **Low-frequency pulse-train / energy template detector** for the S1
   "knock" calls — lightweight, no ML, no training data. Detects the
   characteristic 3–18-pulse bursts. Cheapest to prototype.
2. **Custom-trained CNN** — most accurate, but needs an annotated gray-whale
   call dataset, which does not readily exist. Significant effort.
3. **Re-scope the data** — process a Sounder-habitat hydrophone (if one
   exists in the Orcasound network nearer North Puget Sound) over the
   **spring** months. Without nearby spring data, no detector helps.

## Recommendation

Leave gray whales out of scope for now. The geography and season mismatch
mean the payoff is low for this hydrophone and time window. Revisit only if
the project later expands to spring data or a North Puget Sound hydrophone —
at which point a simple low-frequency knock-pulse detector (option 1) is the
sensible first try. Document this as a known limitation rather than a TODO.

## Sources

- [Gray whale audio — Discovery of Sound in the Sea](https://dosits.org/galleries/audio-gallery/marine-mammals/baleen-whales/gray-whale/)
- [Gray Whale Call Types Recorded During Migration — Frontiers in Marine Science](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2018.00329/full)
- [Migrating gray whale call and blow rates — Scientific Reports](https://www.nature.com/articles/s41598-019-49115-y)
- [Gray whales of the Salish Sea — Encyclopedia of Puget Sound](https://www.eopugetsound.org/article/gray-whales-salish-sea)
- [The Sounders — Cascadia Research](https://cascadiaresearch.org/project_page/sounders_names/)
- [Deep learning benchmark for baleen whale detection — Schall et al. 2024](https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.392)
- [NOAA Passive Acoustic Data archive](https://www.ncei.noaa.gov/products/passive-acoustic-data)
