"""Combine S3 coverage with Acartia sightings to pick a pilot day.

Selects a day in 2025 Q3 with:
  (a) at least one full HLS session directory at Orcasound Lab, AND
  (b) one or more Acartia killer-whale sightings within ~50 km of the lab.

Writes a small report to stdout and saves the selected day + dir to
/home/y/whale_acoustic_library/logs/pilot_day.json.
"""
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "audio-orcasound-net"
PREFIX = "rpi_orcasound_lab/hls/"
LAB_LAT, LAB_LON = 48.5583362, -123.1735774
ACARTIA_CSV = "/media/y/hlabflash/acartia_2026-02-28.csv"
RADIUS_KM = 50.0

# Acartia "type" values relevant to SRKW
SRKW_TYPES = {"Southern Resident Orca", "Killer Whale (Orca)"}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def main():
    # 1) Get S3 timestamp dirs for 2025 Q3
    print("Listing S3 timestamps for 2025 Q3...")
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    q3_start = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp())
    q3_end = int(datetime(2025, 10, 1, tzinfo=timezone.utc).timestamp())

    by_day = defaultdict(list)
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []) or []:
            sub = cp["Prefix"].split("/")[-2]
            try:
                ts = int(sub)
            except ValueError:
                continue
            if not (q3_start <= ts < q3_end):
                continue
            d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            by_day[d].append(ts)

    s3_days = sorted(by_day)
    print(f"  {len(s3_days)} unique UTC days with coverage in 2025 Q3")

    # 2) Read Acartia, filter SRKW within radius and within Q3
    print(f"Loading Acartia from {ACARTIA_CSV}...")
    df = pd.read_csv(ACARTIA_CSV)
    print(f"  {len(df)} total sightings")
    df["created"] = pd.to_datetime(df["created"], errors="coerce", utc=True)
    df = df.dropna(subset=["created"])
    df = df[df["type"].isin(SRKW_TYPES)]
    df = df[
        (df["created"] >= "2025-07-01") &
        (df["created"] < "2025-10-01")
    ]
    df["dist_km"] = df.apply(
        lambda r: haversine_km(r["latitude"], r["longitude"], LAB_LAT, LAB_LON),
        axis=1,
    )
    near = df[df["dist_km"] <= RADIUS_KM].copy()
    near["day"] = near["created"].dt.strftime("%Y-%m-%d")
    print(f"  SRKW sightings in 2025 Q3 within {RADIUS_KM} km of Orcasound Lab: {len(near)}")

    sightings_by_day = near.groupby("day").size().to_dict()

    # 3) Find days with both S3 coverage AND Acartia hits
    overlap = [(d, sightings_by_day[d]) for d in s3_days if d in sightings_by_day]
    overlap.sort(key=lambda x: x[1], reverse=True)

    print()
    print(f"Days with both S3 coverage and SRKW sightings near the lab: {len(overlap)}")
    print("Top 10 candidates:")
    for d, n in overlap[:10]:
        print(f"  {d}: {n} sightings, S3 dirs={len(by_day[d])}")

    if not overlap:
        print("No overlap found — consider widening RADIUS_KM or different time window.")
        return 1

    # 4) Pick the topmost candidate, get its first S3 dir, count segments
    best_day, best_n = overlap[0]
    best_dirs = sorted(by_day[best_day])
    pilot_dir_epoch = best_dirs[0]
    pilot_prefix = f"{PREFIX}{pilot_dir_epoch}/"
    print(f"\nSelected pilot day: {best_day} ({best_n} SRKW sightings nearby)")
    print(f"Selected pilot HLS session: {pilot_prefix}")

    # Count .ts segments and m3u8 size
    seg_count = 0
    total_bytes = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=pilot_prefix):
        for obj in page.get("Contents", []) or []:
            if obj["Key"].endswith(".ts"):
                seg_count += 1
                total_bytes += obj["Size"]
    print(f"Segments in pilot session: {seg_count}")
    print(f"Approx download size: {total_bytes / 1e6:.1f} MB")
    # Each .ts is ~10s, so estimate audio duration
    est_audio_s = seg_count * 10
    print(f"Approx audio duration: {est_audio_s/3600:.1f} hours")

    # 5) Save selection
    out = {
        "selected_day_utc": best_day,
        "selected_session_epoch": pilot_dir_epoch,
        "selected_session_s3_prefix": pilot_prefix,
        "approx_segments": seg_count,
        "approx_download_mb": round(total_bytes / 1e6, 1),
        "approx_audio_hours": round(est_audio_s / 3600, 1),
        "acartia_sightings_within_50km_that_day": int(best_n),
        "all_s3_session_epochs_that_day": best_dirs,
    }
    out_path = Path("/home/y/whale_acoustic_library/logs/pilot_day.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved selection to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
