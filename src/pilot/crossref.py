"""Cross-reference detection clips with Acartia sightings."""
from __future__ import annotations

import math
from datetime import timedelta
from typing import List

import pandas as pd

from . import config as C
from .clip import ClipRecord


SRKW_TYPES = {"Southern Resident Orca", "Killer Whale (Orca)"}


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_acartia_near_lab() -> pd.DataFrame:
    """Load Acartia, filter to SRKW types within radius of the lab.

    Returned df has columns: created (UTC), latitude, longitude, dist_km.
    """
    df = pd.read_csv(C.ACARTIA_CSV)
    df["created"] = pd.to_datetime(df["created"], errors="coerce", utc=True)
    df = df.dropna(subset=["created"])
    df = df[df["type"].isin(SRKW_TYPES)].copy()
    df["dist_km"] = df.apply(
        lambda r: _haversine_km(r["latitude"], r["longitude"], C.LAB_LAT, C.LAB_LON),
        axis=1,
    )
    return df[df["dist_km"] <= C.ACARTIA_RADIUS_KM].copy()


def count_sightings_for_clips(
    clips: List[ClipRecord], near: pd.DataFrame
) -> dict[str, int]:
    """For each clip, count Acartia sightings within ±ACARTIA_TIME_WINDOW_HOURS."""
    out: dict[str, int] = {}
    if near.empty:
        return {c.clip_id: 0 for c in clips}
    times = near["created"].values  # numpy datetime64
    delta = pd.Timedelta(hours=C.ACARTIA_TIME_WINDOW_HOURS)
    for c in clips:
        clip_time = pd.Timestamp(c.start_unix, unit="s", tz="UTC")
        mask = (times >= (clip_time - delta).to_datetime64()) & (
            times <= (clip_time + delta).to_datetime64()
        )
        out[c.clip_id] = int(mask.sum())
    return out
