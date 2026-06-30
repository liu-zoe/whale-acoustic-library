"""SQLite catalog for the whale acoustic library."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .clip import ClipRecord
from .multispecies import call_type_from_multispecies
from . import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    clip_id TEXT PRIMARY KEY,
    hydrophone_location TEXT NOT NULL,
    start_unix REAL NOT NULL,
    end_unix REAL NOT NULL,
    start_utc_iso TEXT NOT NULL,
    end_utc_iso TEXT NOT NULL,
    species TEXT,                       -- SRKW | unknown | (later: humpback, gray)
    call_type TEXT,                     -- discrete | click | whistle | unknown
    raw_wav_path TEXT NOT NULL,
    clean_wav_path TEXT NOT NULL,
    spectrogram_path TEXT NOT NULL,
    sample_rate INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    snr_db REAL NOT NULL,
    peak_confidence REAL NOT NULL,
    mean_confidence REAL NOT NULL,
    n_segments INTEGER NOT NULL,
    detection_model TEXT NOT NULL,      -- e.g. orcahello-srkw-detector-v1
    detection_threshold REAL NOT NULL,
    acartia_sightings_within_24h_50km INTEGER NOT NULL DEFAULT 0,
    review_status TEXT NOT NULL DEFAULT 'pending',  -- pending | keep | reject | uncertain
    review_note TEXT,
    multispecies_scores TEXT,           -- JSON: Google Multispecies 12-class scores
    multispecies_top TEXT,              -- highest-scoring Multispecies class code
    multispecies_top_score REAL         -- score of multispecies_top
);
CREATE INDEX IF NOT EXISTS idx_clips_start_unix ON clips(start_unix);
CREATE INDEX IF NOT EXISTS idx_clips_review ON clips(review_status);
"""


def get_conn(path: Optional[Path] = None) -> sqlite3.Connection:
    p = path or C.DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn


# Columns added after the original schema; migrated into pre-existing catalogs.
_LATER_COLUMNS = {
    "multispecies_scores": "TEXT",
    "multispecies_top": "TEXT",
    "multispecies_top_score": "REAL",
    # Perch-derived columns (D-027, D-031, D-032). Populated either by
    # run_batch.py on new clips or by the one-shot tools (perch_classifier.py,
    # srkw_call_labeler.py) for retro-fitting.
    "perch_p_humpback": "REAL",
    "nearest_ref_call": "TEXT",
    "nearest_ref_pod": "TEXT",
    "nearest_ref_similarity": "REAL",
    # `is_curious` is independent of review_status (D-034): a clip can be
    # `keep` AND curious (e.g. SRKW present alongside an unfamiliar sound).
    # 0 / 1 stored as INTEGER (SQLite). NULL on legacy rows = "not flagged".
    "is_curious": "INTEGER DEFAULT 0",
}


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # Migrate catalogs created before these columns existed.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(clips)")}
    for col, col_type in _LATER_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE clips ADD COLUMN {col} {col_type}")
    conn.commit()


def _multispecies_fields(scores: Optional[dict]) -> dict:
    """Format a Multispecies score dict into the three catalog columns."""
    if not scores:
        return {
            "multispecies_scores": None,
            "multispecies_top": None,
            "multispecies_top_score": None,
        }
    top = max(scores, key=scores.get)
    return {
        "multispecies_scores": json.dumps(
            {k: round(v, 4) for k, v in scores.items()}
        ),
        "multispecies_top": top,
        "multispecies_top_score": scores[top],
    }


def _perch_fields(ann: Optional[dict]) -> dict:
    """Format a PerchAnnotations-ish dict into the four catalog columns.

    Accepts either a PerchAnnotations dataclass or a plain dict with the
    same field names; missing fields become NULL.
    """
    if not ann:
        return {
            "perch_p_humpback": None,
            "nearest_ref_call": None,
            "nearest_ref_pod": None,
            "nearest_ref_similarity": None,
        }
    get = (lambda k: getattr(ann, k, None)) if hasattr(ann, "perch_p_humpback") \
          else (lambda k: ann.get(k))
    return {
        "perch_p_humpback": get("perch_p_humpback"),
        "nearest_ref_call": get("nearest_ref_call"),
        "nearest_ref_pod": get("nearest_ref_pod"),
        "nearest_ref_similarity": get("nearest_ref_similarity"),
    }


def insert_clips(
    conn: sqlite3.Connection,
    clips: List[ClipRecord],
    *,
    detection_model: str,
    detection_threshold: float,
    species: str = "SRKW",
    sightings_by_clip_id: dict[str, int] | None = None,
    multispecies_by_clip_id: dict[str, dict[str, float]] | None = None,
    perch_by_clip_id: dict[str, dict] | None = None,
) -> None:
    sightings_by_clip_id = sightings_by_clip_id or {}
    multispecies_by_clip_id = multispecies_by_clip_id or {}
    perch_by_clip_id = perch_by_clip_id or {}
    # Preserve human review state across re-runs: INSERT OR REPLACE rewrites
    # the whole row, so carry forward any existing review_status / review_note
    # rather than letting them reset to the schema default ('pending').
    existing_review = {
        r[0]: (r[1], r[2])
        for r in conn.execute(
            "SELECT clip_id, review_status, review_note FROM clips"
        )
    }
    rows = []
    from datetime import datetime, timezone
    for c in clips:
        start_iso = datetime.fromtimestamp(c.start_unix, tz=timezone.utc).isoformat()
        end_iso = datetime.fromtimestamp(c.end_unix, tz=timezone.utc).isoformat()
        rows.append(
            {
                "clip_id": c.clip_id,
                "hydrophone_location": "orcasound_lab",
                "start_unix": c.start_unix,
                "end_unix": c.end_unix,
                "start_utc_iso": start_iso,
                "end_utc_iso": end_iso,
                "species": species,
                "call_type": call_type_from_multispecies(
                    multispecies_by_clip_id.get(c.clip_id)
                ),
                "raw_wav_path": c.raw_wav_path,
                "clean_wav_path": c.clean_wav_path,
                "spectrogram_path": c.spectrogram_path,
                "sample_rate": c.sample_rate,
                "duration_s": c.duration_s,
                "snr_db": c.snr_db,
                "peak_confidence": c.peak_confidence,
                "mean_confidence": c.mean_confidence,
                "n_segments": c.n_segments,
                "detection_model": detection_model,
                "detection_threshold": detection_threshold,
                "acartia_sightings_within_24h_50km": int(
                    sightings_by_clip_id.get(c.clip_id, 0)
                ),
                **_multispecies_fields(multispecies_by_clip_id.get(c.clip_id)),
                **_perch_fields(perch_by_clip_id.get(c.clip_id)),
                "review_status": existing_review.get(
                    c.clip_id, ("pending", None))[0],
                "review_note": existing_review.get(
                    c.clip_id, ("pending", None))[1],
            }
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO clips (
            clip_id, hydrophone_location, start_unix, end_unix,
            start_utc_iso, end_utc_iso,
            species, call_type,
            raw_wav_path, clean_wav_path, spectrogram_path,
            sample_rate, duration_s, snr_db,
            peak_confidence, mean_confidence, n_segments,
            detection_model, detection_threshold,
            acartia_sightings_within_24h_50km,
            multispecies_scores, multispecies_top, multispecies_top_score,
            perch_p_humpback,
            nearest_ref_call, nearest_ref_pod, nearest_ref_similarity,
            review_status, review_note
        ) VALUES (
            :clip_id, :hydrophone_location, :start_unix, :end_unix,
            :start_utc_iso, :end_utc_iso,
            :species, :call_type,
            :raw_wav_path, :clean_wav_path, :spectrogram_path,
            :sample_rate, :duration_s, :snr_db,
            :peak_confidence, :mean_confidence, :n_segments,
            :detection_model, :detection_threshold,
            :acartia_sightings_within_24h_50km,
            :multispecies_scores, :multispecies_top, :multispecies_top_score,
            :perch_p_humpback,
            :nearest_ref_call, :nearest_ref_pod, :nearest_ref_similarity,
            :review_status, :review_note
        )
        """,
        rows,
    )
    conn.commit()
