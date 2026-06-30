#!/usr/bin/env python3
"""Harvest extra SRKW reference recordings into testdata/srkw_reference/.

Sources (all CC BY-NC-SA via Orcasound):
  - BR2011Nora: 22 AIFF files (Nora S##) — saved as testdata/srkw_reference/nora/
  - CWR samples: 26 WAVs (S{##}cwr.wav)         → testdata/srkw_reference/cwr/
  - BR Scalls tree: date / individual / call_type / clip.wav  →
                                                     testdata/srkw_reference/br_scalls/

The BR Scalls tree is harvested recursively — three dates × multiple
individuals × multiple call types, ~33 WAVs per leaf in the sampled case.
File names are normalised so we can parse them back later:

    br_scalls/{date}_{individual}_{call}_{originalstem}.wav

For all sources, we record the parsed (date, individual, call_type, pod)
in testdata/srkw_reference/inventory_extra.json so the labeler can build
a richer reference pool without re-parsing filenames.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF_ROOT = REPO / "testdata/srkw_reference"

BASE = "https://orcasound.net/data/product/SRKW/call-catalog"
NORA_DIR = f"{BASE}/BR2011Nora/"
CWR_DIR  = f"{BASE}/BR8TB-tosort/OrcaCalls/CWRAudiosamples/"
SCALLS_DIR = f"{BASE}/BR8TB-tosort/OrcaCalls/Scalls_Snipped_by_BR_S10s/"

UA = "whale-acoustic-library/0.1 (research; orcasound CC-BY-NC-SA)"
LINK_PAT = re.compile(r'<a href="([^"?][^"]*)"', re.IGNORECASE)


def list_dir(url: str) -> list[str]:
    """Parse Apache mod_autoindex HTML into a sorted list of names (relative)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    names = []
    for m in LINK_PAT.findall(html):
        if m.startswith("/") or m.startswith("?") or m == "../":
            continue
        if m == ".":
            continue
        names.append(urllib.parse.unquote(m))
    return sorted(set(names))


def download(url: str, dst: Path) -> int:
    if dst.exists() and dst.stat().st_size > 0:
        return dst.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    dst.write_bytes(data)
    return len(data)


def harvest_simple(label: str, url_dir: str, suffixes: tuple[str, ...],
                   dst_dir: Path) -> list[dict]:
    """Download every file ending in any of `suffixes` from a flat directory."""
    print(f"\n=== {label}: listing {url_dir}")
    names = list_dir(url_dir)
    targets = [n for n in names if n.lower().endswith(suffixes)]
    print(f"  found {len(targets)} files")
    out = []
    t0 = time.time()
    for i, n in enumerate(targets, 1):
        size = download(url_dir + urllib.parse.quote(n), dst_dir / n)
        out.append({"file": n, "size": size, "source": label})
        if i % 10 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] {time.time()-t0:.0f}s")
    return out


def harvest_scalls(label: str, root_url: str, dst_dir: Path) -> list[dict]:
    """Recursively walk Scalls_Snipped_by_BR_S10s/{date}/{individual}/{call_type}/*.wav."""
    print(f"\n=== {label}: scraping tree at {root_url}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    dates = [d.rstrip("/") for d in list_dir(root_url) if d.endswith("/")]
    print(f"  {len(dates)} date(s) found: {dates}")
    t0 = time.time()
    for date in dates:
        date_url = f"{root_url}{date}/"
        individuals = [d.rstrip("/") for d in list_dir(date_url) if d.endswith("/")]
        print(f"  {date}: {len(individuals)} individual(s) — {individuals}")
        for ind in individuals:
            ind_url = f"{date_url}{ind}/"
            call_types = [d.rstrip("/") for d in list_dir(ind_url) if d.endswith("/")]
            for ct in call_types:
                ct_url = f"{ind_url}{ct}/"
                try:
                    wavs = [w for w in list_dir(ct_url) if w.lower().endswith(".wav")]
                except Exception as exc:
                    print(f"    SKIP {ct_url}: {exc}")
                    continue
                for w in wavs:
                    # Normalise: br_scalls/{date}_{individual}_{call}_{origstem}.wav
                    # so a single flat folder is easy to glob and filenames parse.
                    safe = re.sub(r"[^A-Za-z0-9._-]", "_", w)
                    out_name = f"{date}_{ind}_{ct}_{safe}"
                    size = download(ct_url + urllib.parse.quote(w),
                                    dst_dir / out_name)
                    out.append({"file": out_name, "date": date,
                                "individual": ind, "call_type": ct,
                                "orig_stem": w, "size": size,
                                "source": label})
        elapsed = time.time() - t0
        print(f"  {date} done — {sum(1 for r in out if r['date']==date)} files "
              f"in {elapsed:.0f}s")
    return out


def parse_inventory(harvested: dict) -> dict:
    """Add call_type and pod metadata for each downloaded record."""
    # Nora filename: "Nora S01.aiff" / "Nora S33.aiff"
    NORA_PAT = re.compile(r"Nora S(\d+)(?:i+)?\.aiff", re.IGNORECASE)
    # CWR filename: "S10cwr.wav" / "S2icwr.wav" / "S2iiicwr.wav"
    CWR_PAT = re.compile(r"S(\d+)(i*)cwr\.wav", re.IGNORECASE)
    # Individual prefix maps to pod:
    #   A** = J pod ; C** = K pod ; L** would be L pod
    POD_MAP = lambda ind: {"A": "J", "C": "K", "L": "L"}.get(ind[:1].upper(), "?")

    for rec in harvested["nora"]:
        m = NORA_PAT.match(rec["file"])
        rec["call_type"] = f"S{int(m.group(1)):02d}" if m else None
        rec["pod"] = None  # Nora is a NRKW; pod not directly applicable
    for rec in harvested["cwr"]:
        m = CWR_PAT.match(rec["file"])
        if m:
            base = f"S{int(m.group(1)):02d}"
            subtype = m.group(2).lower()  # "" / "i" / "iii"
            rec["call_type"] = base + subtype
        else:
            rec["call_type"] = None
        rec["pod"] = None  # CWR samples not pod-tagged in filenames
    for rec in harvested["br_scalls"]:
        rec["pod"] = POD_MAP(rec["individual"])
    return harvested


def main() -> int:
    harvested = {"nora": [], "cwr": [], "br_scalls": []}

    harvested["nora"] = harvest_simple(
        "BR2011Nora", NORA_DIR, (".aiff", ".aif"),
        REF_ROOT / "nora")
    harvested["cwr"] = harvest_simple(
        "CWRAudiosamples", CWR_DIR, (".wav",),
        REF_ROOT / "cwr")
    harvested["br_scalls"] = harvest_scalls(
        "BR_Scalls", SCALLS_DIR, REF_ROOT / "br_scalls")

    harvested = parse_inventory(harvested)

    # Persist inventory
    inv_path = REF_ROOT / "inventory_extra.json"
    inv_path.write_text(json.dumps(harvested, indent=2))

    total_files = sum(len(v) for v in harvested.values())
    total_size = sum(r["size"] for v in harvested.values() for r in v)
    print(f"\n=== SUMMARY ===")
    for k, v in harvested.items():
        sz = sum(r["size"] for r in v)
        print(f"  {k:>10s}: {len(v):>4d} files, {sz/1024/1024:>5.1f} MB")
    print(f"  {'total':>10s}: {total_files:>4d} files, {total_size/1024/1024:>5.1f} MB")
    print(f"  inventory: {inv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
