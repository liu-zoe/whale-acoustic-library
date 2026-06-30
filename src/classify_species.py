#!/usr/bin/env python3
"""Run the Google Multispecies Whale Model over the existing library clips.

Adds a secondary species / call-type opinion to each clip without re-running
download or OrcaHello inference. Reads audio_raw/, scores with
pilot/multispecies.py, and writes results into new columns of the SQLite
catalog. Raw clips are never modified, so this is safe to re-run.

    conda activate whales
    python src/classify_species.py
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot.multispecies import (  # noqa: E402
    MultispeciesClassifier,
    CLASS_LABELS,
    call_type_from_multispecies,
)

LIBRARY = Path("/media/y/hlabflash/whale_library")
RAW_DIR = LIBRARY / "audio_raw"
DB_PATH = LIBRARY / "db" / "library.sqlite"

# New, additive columns — the OrcaHello-derived `species` / `call_type`
# columns are left untouched so the two detectors' opinions stay separable.
NEW_COLUMNS = {
    "multispecies_scores": "TEXT",      # JSON {class_code: score} for all 12
    "multispecies_top": "TEXT",         # highest-scoring class code
    "multispecies_top_score": "REAL",   # its score
}


def ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(clips)")}
    for col, col_type in NEW_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE clips ADD COLUMN {col} {col_type}")


def main() -> int:
    raws = sorted(RAW_DIR.glob("*.wav"))
    if not raws:
        print(f"no clips in {RAW_DIR}", file=sys.stderr)
        return 1
    if not DB_PATH.exists():
        print(f"no catalog at {DB_PATH}", file=sys.stderr)
        return 1

    backup = DB_PATH.parent / (DB_PATH.name + ".bak-pre-multispecies")
    if not backup.exists():
        shutil.copy2(DB_PATH, backup)
        print(f"catalog backed up -> {backup.name}")

    print("loading Multispecies model (TensorFlow)...")
    clf = MultispeciesClassifier()

    conn = sqlite3.connect(DB_PATH)
    ensure_columns(conn)

    print(f"\nscoring {len(raws)} clips (24 kHz, 5 s windows)\n")
    print(f"{'clip_id':40s} {'top class':>34s} {'Oo':>6s} {'Mn':>6s}")
    print("-" * 90)
    for raw in raws:
        clip_id = raw.stem
        audio, sr = sf.read(raw, dtype="float32", always_2d=False)
        scores = clf.score_clip(audio, sr)
        top = max(scores, key=scores.get)
        conn.execute(
            "UPDATE clips SET multispecies_scores=?, multispecies_top=?, "
            "multispecies_top_score=?, call_type=? WHERE clip_id=?",
            (
                json.dumps({k: round(v, 4) for k, v in scores.items()}),
                top,
                scores[top],
                call_type_from_multispecies(scores),
                clip_id,
            ),
        )
        top_label = f"{top} ({CLASS_LABELS.get(top, top)})"
        print(f"{clip_id:40s} {top_label:>34s} {scores['Oo']:6.3f} {scores['Mn']:6.3f}")
    conn.commit()
    conn.close()

    print("-" * 90)
    print(
        f"done: {len(raws)} clips scored; results in the clips.multispecies_* "
        f"columns (full 12-class scores as JSON in multispecies_scores)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
