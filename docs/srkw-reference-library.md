# SRKW Reference Library

Acquired 2026-06-21 from the Orcasound public data products under
`/data/product/SRKW/`. Persisted locally under
`testdata/srkw_reference/` (analogous to `testdata/humpback_whatkins/` for
the humpback positive-control set). Used by the Perch nearest-neighbor
labeler (`src/srkw_call_labeler.py`) to assign tentative call-type and pod
labels to user-confirmed SRKW clips.

## What we have

| collection | files | content | role |
|---|---|---|---|
| `ford_osborne/` (no-narration bundle, 5.6 MB zip) | **45 FLAC + 45 MP3 + 45 OGG + 45 PNG spectrograms** | Ford-Osborne discrete-call catalogue: **30 distinct call types** (S01-S46), 1-5 takes each | call-type nearest-neighbor labeling |
| `favorites/` | 3 MP3 | J-pod-S01, K-pod-S16, L-pod-S19 — one canonical "favorite call" per pod | one-shot pod positives |
| `clicks/` | 8 MP3 + 4 OGG | Various SRKW echolocation samples: fast / slow / slowed10× / "varied JK", click trains, calls+clicks together | reference for the 0928-style slow-click hypothesis; could later support a click-vs-call sub-classifier |
| `whistles/` | 1 MP3 | "whistle-examples" — multiple SRKW whistle samples in one file | reference for the rarer whistle category (D-018: only 4 of 111 SRKW clips had Multispecies Whistle >= 0.5) |
| `calls_vs_vessel/` | 4 MP3 | L-pod S19 calls recorded in/after vessel noise, plus an isolated vessel-noise sample | ground-truth pairs for the planned SRKW-vs-vessel filter (D-025 / task #18-style) |

Total: **57 audio files, 56 MB.** Source organization preserved one level
down so provenance is obvious.

## Pod labeling — sparser than ideal

Of the 45 Ford-Osborne samples, only 16 have explicit pod tags in their
filenames; the rest are unlabeled or labeled "R" (mixed-resident, not pod-
specific). Plus the 3 favorites.

Adding everything up, single-pod-labeled positives by pod:

| pod | n |
|---|---|
| J | 5  (4 FO S33 variants + 1 favorite S01) |
| K | 1  (1 favorite S16) |
| L | 2  (1 FO S06-L + 1 favorite S19) |

**That is far below what would make a confident pod classifier** — but it's
enough to populate a tentative `nearest_ref_pod` column the human reviewer
can validate or correct. We treat pod assignment as a hypothesis, not a
verdict. The "R"-labeled samples are excluded from pod voting (resident
killer whale = unspecified J/K/L); a clip whose top-5 Perch neighbours are
all R gets `pod = "?"`.

## Call-type labeling — actually usable

30 distinct call types is most of the Ford catalogue. For each user-
confirmed SRKW clip, the nearest reference in Perch embedding space
provides a tentative `nearest_ref_call` label (S01, S33, etc.) plus a
`nearest_ref_similarity` cosine score. High similarity (≥ 0.6, say) is a
reasonable label; lower scores indicate the clip doesn't closely match any
of our 45 references and should be considered unclassified.

Two important caveats:
- Some Ford codes have only one reference take (e.g., S01, S05). Nearest-
  neighbor labels for those are inherently fragile.
- The catalogue spans the published "stereotyped calls" — calls that don't
  match the Ford catalogue (variants, novel calls, unstereotyped utterances)
  will still get labeled with their closest match, even when there isn't a
  good fit. This is why the `similarity` score matters as much as the
  label itself.

## Pipeline integration (`src/srkw_call_labeler.py`)

One-shot script. Loads Perch 2.0, embeds all 48 references (cached as
`models/srkw_call_labeler_v0/reference_embeddings.npz`), embeds every SRKW
clip in the catalog, finds nearest neighbor, writes three new catalog
columns:

```
ALTER TABLE clips ADD COLUMN nearest_ref_call TEXT;
ALTER TABLE clips ADD COLUMN nearest_ref_pod TEXT;
ALTER TABLE clips ADD COLUMN nearest_ref_similarity REAL;
```

Re-runnable cheaply once embeddings are cached.

## Licence

All Orcasound material is **Creative Commons Attribution-NonCommercial-
ShareAlike 4.0 International**. This project is non-commercial research, so
compliant; attribution to "Orcasound (orcasound.net)" required in any
downstream publication. Ford 1987 "A catalogue of underwater calls produced
by killer whales (Orcinus orca) in British Columbia" is the canonical
academic citation for the call type taxonomy.

## What's NOT here (and where to look later)

- **SFU call library** (`orca.research.sfu.ca/call-library/`) — the page
  appears to render data via JavaScript; no API or audio URLs were
  visible in the static HTML. Worth a deeper look (browser devtools network
  tab) — likely a richer Ford-style catalogue with more takes per call.
- **Non-target marine mammals** (porpoise, seal, sea lion) — task #24
  needs its own acquisition (likely Watkins or DOSITS).
- **Per-pod-specific recordings** — beyond the 3 favorites and the few
  pod-tagged FO entries, we don't have isolated J / K / L pod recordings.
  Could be extracted manually from pod-known historical recordings if the
  user wants better pod-classifier training data later.
