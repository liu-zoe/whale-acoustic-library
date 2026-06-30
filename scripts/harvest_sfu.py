#!/usr/bin/env python3
"""Harvest the SFU SRKW call library — 123 entries / 30 call types.

This is the catalogue at https://orca.research.sfu.ca/call-library/ . Its
defining advantage over the Orcasound Ford-Osborne bundle: proper sub-type
breakouts (S02i / S02ii / S02iii, S08i / S08ii, S37i / S37ii) and balanced
pod coverage (J=65, L=74, K=24 takes), plus 2-9 takes per call code.

Downloads each MP3 + its WebP spectrogram into
testdata/srkw_reference/sfu/ . Filename normalisation:

    sfu/S{NN}{subtype}__pods-{J,K,L}__seq.{mp3,webp}

so the v2 labeler can parse (call_type, sub_type, pod) deterministically.
A JSON inventory is dropped at sfu/inventory.json with the original
catalogue entries for provenance.

Licence: SFU's HALLO project is DFO-funded research; the call library is
publicly served. Treating as research-use under fair-use until the
catalogue's own LICENCE link is verified for redistribution. For our
purposes (local model training, no redistribution), this is fine.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "testdata/srkw_reference/sfu"
DST.mkdir(parents=True, exist_ok=True)

BASE = "https://orca.research.sfu.ca/call-library"
CATALOG_URL = f"{BASE}/catalogs/srkw.json"
MEDIA_PREFIX = "catalogs/"  # entries' audio_file paths are relative to /call-library/catalogs/

UA = "whale-acoustic-library/0.1 (research; SFU HALLO call library)"


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def normalize_name(call_type: str, pods: list[str], seq: int) -> str:
    """Build a stable filename that the v2 labeler can parse."""
    pods_part = ",".join(sorted(pods)) if pods else "none"
    return f"S{call_type[1:]}__pods-{pods_part}__seq{seq:02d}"


def main() -> int:
    print(f"=== fetching SFU catalog index ===")
    catalog = json.loads(fetch(CATALOG_URL).decode("utf-8"))
    calls = catalog["calls"]
    print(f"  {len(calls)} entries in catalogue")

    # Group entries by call_type so we can number takes consistently
    by_call: dict[str, list] = {}
    for c in calls:
        by_call.setdefault(c["call_type"], []).append(c)

    print(f"  {len(by_call)} distinct call types")
    print(f"\n=== downloading audio + spectrograms ===")
    t0 = time.time()
    inventory = []
    audio_count = 0
    spec_count = 0
    for code in sorted(by_call):
        takes = by_call[code]
        for seq, entry in enumerate(takes):
            pods = entry.get("pod") or []
            name_stem = normalize_name(code, pods, seq)

            # Audio
            audio_rel = entry["audio_file"]
            audio_url = f"{BASE}/{MEDIA_PREFIX}{urllib.parse.quote(audio_rel)}"
            audio_dst = DST / f"{name_stem}.mp3"
            if not audio_dst.exists():
                try:
                    audio_dst.write_bytes(fetch(audio_url))
                    audio_count += 1
                except Exception as exc:
                    print(f"  AUDIO FAIL {audio_rel}: {exc}")
                    continue

            # Spectrogram
            image_rel = entry.get("image_file")
            spec_size = 0
            if image_rel:
                image_url = f"{BASE}/{MEDIA_PREFIX}{urllib.parse.quote(image_rel)}"
                # Most are .webp; preserve original extension
                ext = Path(image_rel).suffix
                spec_dst = DST / f"{name_stem}{ext}"
                if not spec_dst.exists():
                    try:
                        spec_dst.write_bytes(fetch(image_url))
                        spec_count += 1
                    except Exception:
                        pass
                spec_size = spec_dst.stat().st_size if spec_dst.exists() else 0

            inventory.append({
                "filename_stem": name_stem,
                "call_type": code,
                "pods": pods,
                "clan": entry.get("clan"),
                "matrilines": entry.get("matrilines"),
                "subclan": entry.get("subclan"),
                "audio_file": str(audio_dst.relative_to(REPO)),
                "audio_size": audio_dst.stat().st_size,
                "spectrogram_file": (
                    str((DST / f"{name_stem}{Path(image_rel).suffix}").relative_to(REPO))
                    if image_rel else None),
                "spectrogram_size": spec_size,
                "orig_audio_path": audio_rel,
                "orig_image_path": image_rel,
            })

        elapsed = time.time() - t0
        print(f"  {code}: {len(takes)} takes  ({elapsed:.0f}s)")

    print(f"\n=== SUMMARY ===")
    print(f"  audio downloaded: {audio_count} new (+ existing)")
    print(f"  spectrograms downloaded: {spec_count} new (+ existing)")
    print(f"  inventory entries: {len(inventory)}")
    audio_total = sum(r["audio_size"] for r in inventory)
    spec_total = sum(r["spectrogram_size"] for r in inventory)
    print(f"  audio total size: {audio_total/1024/1024:.1f} MB")
    print(f"  spectrogram total: {spec_total/1024/1024:.1f} MB")

    inv_path = DST / "inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2))
    print(f"\n  inventory -> {inv_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
