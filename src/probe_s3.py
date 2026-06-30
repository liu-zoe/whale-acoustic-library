"""Probe Orcasound S3 bucket to characterize available data for the pilot.

Lists timestamp-named directories under rpi_orcasound_lab and prints summary
statistics. We use unsigned access since the bucket is public.
"""
import sys
from datetime import datetime, timezone
from collections import defaultdict

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "audio-orcasound-net"
PREFIX = "rpi_orcasound_lab/hls/"


def main():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    # First, list the timestamp directory names (CommonPrefixes from delimited list)
    print(f"Bucket: {BUCKET}")
    print(f"Prefix: {PREFIX}")
    print()

    pager = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX, Delimiter="/")
    timestamps = []
    for page in pager:
        for cp in page.get("CommonPrefixes", []) or []:
            # cp["Prefix"] like "rpi_orcasound_lab/hls/1693939200/"
            sub = cp["Prefix"].split("/")[-2]
            try:
                ts = int(sub)
                timestamps.append(ts)
            except ValueError:
                continue

    print(f"Found {len(timestamps)} timestamp directories")
    if not timestamps:
        return 1

    timestamps.sort()
    earliest = datetime.fromtimestamp(timestamps[0], tz=timezone.utc)
    latest = datetime.fromtimestamp(timestamps[-1], tz=timezone.utc)
    print(f"Earliest: {earliest.isoformat()} (epoch {timestamps[0]})")
    print(f"Latest:   {latest.isoformat()} (epoch {timestamps[-1]})")

    # Bucket timestamps that fall in 2025 Q3 (Jul–Sep)
    q3_start = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp())
    q3_end = int(datetime(2025, 10, 1, tzinfo=timezone.utc).timestamp())
    q3 = [t for t in timestamps if q3_start <= t < q3_end]
    print(f"\n2025 Q3 directories: {len(q3)}")
    if q3:
        # Group by day
        by_day = defaultdict(list)
        for t in q3:
            d = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
            by_day[d].append(t)
        days = sorted(by_day)
        print(f"  spanning {len(days)} unique UTC days, {days[0]} to {days[-1]}")
        # Pick a likely candidate: a day with the most directories (longest coverage)
        best_day = max(days, key=lambda d: len(by_day[d]))
        print(f"  busiest day: {best_day} with {len(by_day[best_day])} directories")
        # Print a small sample
        for d in days[:5]:
            print(f"    {d}: {len(by_day[d])} dirs, first epoch {by_day[d][0]}")
        print("    ...")
        for d in days[-5:]:
            print(f"    {d}: {len(by_day[d])} dirs, first epoch {by_day[d][0]}")

    # Pick the first 2025 Q3 dir and inspect its contents
    if q3:
        sample = q3[0]
        sample_prefix = f"{PREFIX}{sample}/"
        print(f"\nSample dir contents: {sample_prefix}")
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=sample_prefix, MaxKeys=10)
        for obj in resp.get("Contents", []):
            print(f"  {obj['Key']}  ({obj['Size']} B)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
