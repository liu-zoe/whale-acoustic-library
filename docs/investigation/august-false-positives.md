# August SRKW false-positive clusters (08-07, 08-25)

*Diagnostic written while the user is away. Source: review feedback that
"a large number of cases, mostly in August, are artifacts of the recording
devices but got labeled SRKW instead."*

## Headline

August's elevated SRKW reject rate (64% vs ~16% in July) is **not** uniform
device drift across the month. It is **driven by two specific days**:

| day | SRKW clips | reject rate |
|---|---|---|
| **2025-08-07** | 20 | **100 %** (20 / 20) |
| **2025-08-25** | 27 | **96 %** (26 / 27) |
| all other August days combined | 48 | 31 % |

**46 of August's 61 SRKW rejects (75 %) came from just those two days.** This
is event-driven, not systemic.

## 2025-08-07 — single ~20-minute noise event

- All 20 clips fall in a **single 20-minute window**, **06:06 → 06:26 UTC**
  (= 23:06 → 23:26 PDT, i.e. very late evening local time).
- Median gap between adjacent clips: **0.6 min**. Max gap: 4.7 min. Tight.
- Confidence range 0.50–0.93 (mean 0.69). `n_segments` mostly 1, max 7.
- Spectrogram montage (`montage_08-07.png`): **all 20 clips look visually
  identical** — uniform broadband purple-pink haze, no tonal structure, no
  click trains, no visible call onset/offset. Just sustained low-level
  noise with the same character across every clip.

**Best interpretation:** a single persistent noise source was present for
~20 minutes — most likely a vessel slowly transiting near the hydrophone
or a sustained mechanical/environmental noise (e.g., generator, distant
pile-driving, anchor handling). OrcaHello's 3-s windows kept scoring above
0.5 the whole time, producing 20 quasi-identical "clips" of the same thing.

## 2025-08-25 — multiple events through the day

Very different pattern from 08-07:

- 27 clips spread **06:48 → 23:52 UTC** — full 17-hour spread, not a single
  burst.
- Median gap 7.5 min, max gap **495 min (~8 h)** — there are clear quiet
  stretches between activity bursts.
- Hourly distribution (UTC):
  - 06h: 1, 07h: 2  &larr; pre-dawn activity
  - 15h: 1, 16h: 3, 17h: 4, 18h: 4  &larr; afternoon cluster (12 clips)
  - 20h: 1
  - **23h: 11**  &larr; heavy late-evening cluster
- Confidence range 0.57–0.99 (mean **0.81** — much higher than 08-07).
- Spectrogram montage (`montage_08-25.png`): clips **do not all look alike.**
  Several show **regular vertical click-train patterns** characteristic of
  vessel **echo-sounders / depth-finders / fish-finders** (periodic sonar
  pulses, very regular spacing). Others show broadband noise or short tonal
  fragments. Distinct events, distinct sources.

**Best interpretation:** **multiple vessel passages** through the day, at
least one (perhaps several) using active echo-sounding equipment whose
periodic pulses OrcaHello mistakes for SRKW echolocation clicks. The 23h
cluster of 11 clips probably represents a single boat lingering or
transiting slowly near the hydrophone for ~30 minutes at night.

## Why OrcaHello fires on these

OrcaHello is a binary SRKW-vs-not classifier trained on Orcasound's labelled
data, originally tuned for vocalisation detection. It is **not robustness-
hardened against**:

- **Vessel echo-sounder click trains** — regular broadband ticks superficially
  resemble SRKW echolocation clicks.
- **Sustained broadband ambient with vessel character** — slow-moving boats,
  generators, or anchor-handling noise can produce energy in the call band
  that scores marginally above 0.5 for tens of minutes at a time.

The 0.5 threshold is fine for *real* calls (which usually score 0.8+); the
problem is that consistent low-level noise can hover just over 0.5 long
enough to produce many windows. `n_segments` is low (mean 1.8 on 08-07,
1.5 on 08-25) — these are single 3-second pops, not the multi-second
sustained calls real SRKW produce.

## Implications

1. **Not a hydrophone defect.** The other 27 August days had reject rate
   ~31 %, consistent with July's 16 % background; the deployment is fine.
2. **Not threshold-tunable.** Several 08-25 false positives crossed 0.99 —
   raising the threshold to 0.9 would only filter the cleanest real calls
   first.
3. **This is exactly the failure mode** that drove deferred-task #18: Perch
   2.0 agile modeling with these labeled rejects as negatives would let us
   build a *secondary* "real-SRKW-vs-vessel-noise" classifier — same lever
   as the planned humpback-vs-vessel classifier, but for SRKW.
4. **The 4 high-reject days (08-01, 08-07, 08-25, plus 08-27)** that drove
   August's apparent SRKW activity are now correctly characterised:
   - 08-01: real orca day (29 % reject — 10 keeps + uncertain)
   - **08-07: not a real orca day** (100 % reject — single 20-min vessel)
   - **08-25: not a real orca day** (96 % reject — multi-vessel day)
   - 08-27: mixed (needs the remaining reviews)

The "August SRKW story" in D-023 — "bursty across 4 high days" — should be
read in this light: 08-01 is the only one of those four that meaningfully
contributed orca clips. 08-07 and 08-25 contributed essentially zero. The
*real* August SRKW haul is closer to **~21 clips spread across the month**,
not the raw 115 figure.

## Files

- `montage_08-07.png` — all 20 rejected clips on 08-07.
- `montage_08-25.png` — all 26 rejected clips on 08-25.
