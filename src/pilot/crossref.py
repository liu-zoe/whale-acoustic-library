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


def load_acartia_near_node(hydrophone_id: str | None = None) -> pd.DataFrame:
    """Load Acartia, filter to SRKW types within radius of the given node.

    Uses `C.NODES[hydrophone_id]` for lat/lon. Falls back to Orcasound Lab
    when `hydrophone_id` is None (preserves the pre-multi-node behaviour).
    Returned df has columns: created (UTC), latitude, longitude, dist_km.
    """
    if hydrophone_id and hydrophone_id in C.NODES:
        lat, lon = C.NODES[hydrophone_id]["lat"], C.NODES[hydrophone_id]["lon"]
    else:
        lat, lon = C.LAB_LAT, C.LAB_LON
    df = pd.read_csv(C.ACARTIA_CSV)
    df["created"] = pd.to_datetime(df["created"], errors="coerce", utc=True)
    df = df.dropna(subset=["created"])
    df = df[df["type"].isin(SRKW_TYPES)].copy()
    df["dist_km"] = df.apply(
        lambda r: _haversine_km(r["latitude"], r["longitude"], lat, lon),
        axis=1,
    )
    return df[df["dist_km"] <= C.ACARTIA_RADIUS_KM].copy()


# Back-compat alias for callers that predate the multi-node parameter.
def load_acartia_near_lab() -> pd.DataFrame:
    return load_acartia_near_node(None)


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
