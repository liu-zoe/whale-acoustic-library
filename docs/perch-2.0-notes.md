# Perch 2.0 — Evaluation Notes

*Can we leverage Google's Perch 2.0 for this project? Short answer: not for
the immediate humpback-detection goal, but yes later, for things the current
model stack genuinely cannot do.*

## What Perch 2.0 is

A bioacoustics **foundation / embedding model** from Google DeepMind, trained
on ~14,600 species (mostly birds, also some mammals/amphibians/insects) — with
**almost no marine-mammal training data**. It is *not* a whale classifier; it
turns 5 s audio windows into embeddings ("fingerprints").

The Dec-2025 paper *"Perch 2.0 transfers 'whale' to underwater tasks"* shows
that, despite the lack of marine training data, Perch 2.0 embeddings transfer
**very well** to cetacean tasks via few-shot **linear probing** — beating
other embedding models on baleen-whale species and killer-whale ecotype
benchmarks.

## How you'd use it — agile modeling

You don't get a ready classifier. The workflow (`perch-hoplite`, pip):
embed your audio → search by an example clip → label a handful of hits →
`train_linear_classifier()` on the frozen embeddings → active-learning loop.
A usable custom classifier in ~an hour — *if you have labeled examples*.

## Why it is NOT the tool for the humpback validation run

The **Google Multispecies model** (already in our stack) is a ready-made
12-class classifier that **already includes humpback** — zero labeled data,
zero training. Perch 2.0 would be *more* work for the same immediate goal:
we'd first need labeled humpback examples and a training step. For "detect
humpback now," Multispecies primary detection is the direct path.

## Where Perch 2.0 IS worth adopting later

1. **Fine-grained SRKW call-type classification** — the project plan's
   intended "reserve" use. Multispecies only gives coarse Call / Echolocation
   / Whistle; Perch + agile modeling could classify specific discrete call
   types (S1, S16, …) from the library's own clips.
2. **Custom detectors where no model exists** — e.g. gray whales (see
   `gray-whale-research.md`). Agile modeling on a few labeled examples is the
   realistic path when there is no off-the-shelf model.
3. **Similarity search / library curation** — embed the whole library, find
   clips similar to a reference call, surface odd or mislabeled clips; could
   also assist the non-expert review step.

## Recommendation

- **Now:** stick with the Multispecies model for the humpback work. Don't add
  Perch to the validation run.
- **Later (phase 2):** adopt Perch 2.0 for call-type classification and
  custom detectors. Natural sequencing — the clip library we are building now
  becomes the labeled training data Perch's agile modeling needs.

## Sources

- [Perch 2.0 transfers 'whale' to underwater tasks (arXiv 2512.03219)](https://arxiv.org/abs/2512.03219)
- [Perch 2.0: The Bittern Lesson for Bioacoustics (arXiv 2508.04665)](https://arxiv.org/html/2508.04665v1)
- [How AI trained on birds is surfacing underwater mysteries — Google Research](https://research.google/blog/how-ai-trained-on-birds-is-surfacing-underwater-mysteries/)
- [perch-hoplite — GitHub](https://github.com/google-research/perch-hoplite)
