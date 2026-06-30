"""Generate the static HTML review / comparison page for the library.

A responsive multi-column grid of clip cards — spectrogram, raw + cleaned
audio, OrcaHello + Multispecies metrics — with client-side sort and filter so
the clips can be compared side by side. Each card has keep/reject/uncertain
controls that POST to review_server.py, which writes back to SQLite.
"""
from __future__ import annotations

import json
import sqlite3
import urllib.parse
from html import escape
from pathlib import Path

from . import config as C


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Whale Acoustic Library — Compare & Review</title>
  <style>
    body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 16px; background: #111; color: #eee; }
    h1 { margin: 0 0 8px; font-size: 20px; }
    .topbar { position: sticky; top: 0; background: #111; padding: 8px 0 12px; z-index: 10; border-bottom: 1px solid #333; }
    .controls { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; font-size: 13px; color: #aaa; }
    .controls select { background: #2a2a2a; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 3px 6px; }
    button { font-size: 13px; padding: 5px 10px; border-radius: 4px; border: 1px solid #555; background: #2a2a2a; color: #eee; cursor: pointer; }
    button:hover { filter: brightness(1.25); }
    button.f.active { background: #34506e; border-color: #5a8cc0; }
    .summary { color: #aaa; font-size: 12px; margin-top: 8px; }
    .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fill, minmax(440px, 1fr)); margin-top: 14px; }
    .card { background: #1c1c1c; border-radius: 8px; padding: 10px; border-left: 4px solid #555; }
    .card.keep { border-left-color: #2e8b57; }
    .card.reject { border-left-color: #b22222; }
    .card.uncertain { border-left-color: #d2b04c; }
    .card.pending { border-left-color: #555; }
    /* is_curious is independent of review_status; star badge in the top-right */
    .card { position: relative; }
    .curious-badge { position: absolute; top: 6px; right: 8px; color: #b08bff;
                     font-size: 18px; line-height: 1; }
    .card:not(.is-curious) .curious-badge { display: none; }
    button.curious.on { background: #3a2660; border-color: #7a4ed2; color: #fff; }
    button.curious.off { background: #2a2a2a; border-color: #555; color: #888; }
    .meta { font-size: 11px; color: #aaa; margin-bottom: 6px; line-height: 1.5; }
    .meta b { color: #fff; }
    .tag { background: #333; padding: 2px 6px; border-radius: 4px; margin-right: 4px; white-space: nowrap; }
    .tag.ms { background: #2a3a4a; }
    .tag.ref { background: #2a4030; }
    .tag.ref.high { background: #2e7a4a; color: #fff; }
    .tag.ref.mid { background: #4a6b3a; color: #eee; }
    .tag.ref.low { background: #3a3a30; color: #999; }
    .tag.sp-humpback { background: #6a4a7a; color: #fff; }
    .tag.sp-SRKW { background: #2f5d7a; color: #fff; }
    .status-badge { font-size: 11px; padding: 2px 6px; border-radius: 4px; background: #444; }
    .spec { display: block; width: 100%; height: auto; border-radius: 4px; }
    .audio-row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin: 6px 0; }
    .audio-row label { color: #888; font-size: 11px; margin-right: 2px; }
    audio { height: 32px; vertical-align: middle; }
    .actions { margin-top: 6px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
    button.keep { background: #1f5e3a; border-color: #2e8b57; }
    button.reject { background: #5e1f1f; border-color: #b22222; }
    button.uncertain { background: #5a4a18; border-color: #d2b04c; }
    .note { flex: 1; min-width: 120px; padding: 4px; background: #2a2a2a; color: #eee; border: 1px solid #555; border-radius: 4px; font-size: 12px; }
  </style>
</head>
<body>
  <div class="topbar">
    <h1>Whale Acoustic Library — Compare &amp; Review</h1>
    <div class="controls">
      <label>sort:&nbsp;
        <select id="sort" onchange="applySort()">
          <option value="conf">OrcaHello confidence &darr;</option>
          <option value="time">time &uarr;</option>
          <option value="snr">in-band SNR &darr;</option>
          <option value="msoo">Multispecies Oo &darr;</option>
          <option value="refsim">nearest-ref similarity &darr;</option>
          <option value="species">species</option>
          <option value="status">review status</option>
        </select>
      </label>
      <span>filter:
        <button class="f active" data-f="all" onclick="applyFilter(this)">all</button>
        <button class="f" data-f="keep" onclick="applyFilter(this)">keep</button>
        <button class="f" data-f="uncertain" onclick="applyFilter(this)">uncertain</button>
        <button class="f" data-f="reject" onclick="applyFilter(this)">reject</button>
        <button class="f" data-f="pending" onclick="applyFilter(this)">pending</button>
      </span>
      <span>&nbsp;&nbsp;tag:
        <button class="f" data-f="curious" onclick="applyFilter(this)">★ curious only</button>
      </span>
    </div>
    <div class="summary" id="summary">__SUMMARY__</div>
  </div>
  <div class="grid" id="grid">
__CARDS__
  </div>
<script>
const grid = document.getElementById("grid");
let curFilter = "all";

function applySort() {
  const k = document.getElementById("sort").value;
  const cards = [...grid.children];
  cards.sort((a, b) => {
    if (k === "conf")   return b.dataset.conf - a.dataset.conf;
    if (k === "snr")    return b.dataset.snr  - a.dataset.snr;
    if (k === "msoo")   return b.dataset.msoo - a.dataset.msoo;
    if (k === "time")   return a.dataset.time - b.dataset.time;
    if (k === "refsim")  return b.dataset.refsim - a.dataset.refsim;
    if (k === "species") return (a.dataset.species || "").localeCompare(b.dataset.species || "");
    if (k === "status") return a.dataset.status.localeCompare(b.dataset.status);
    return 0;
  });
  cards.forEach(c => grid.appendChild(c));
}

function applyFilter(btn) {
  if (btn) {
    curFilter = btn.dataset.f;
    document.querySelectorAll("button.f").forEach(b => b.classList.toggle("active", b === btn));
  }
  for (const c of grid.children) {
    let show;
    if (curFilter === "all")           show = true;
    else if (curFilter === "curious")  show = c.dataset.curious === "1";
    else                                show = c.dataset.status === curFilter;
    c.style.display = show ? "" : "none";
  }
  updateCounts();
}

function updateCounts() {
  const all = [...grid.children];
  const by = {keep: 0, reject: 0, uncertain: 0, pending: 0};
  all.forEach(c => { by[c.dataset.status] = (by[c.dataset.status] || 0) + 1; });
  const shown = all.filter(c => c.style.display !== "none").length;
  document.getElementById("summary").textContent =
    `${all.length} clips · keep ${by.keep} · uncertain ${by.uncertain} `
    + `· reject ${by.reject} · pending ${by.pending} · showing ${shown}`;
}

async function setStatus(clipId, status, btn) {
  const noteInput = document.getElementById("note-" + clipId);
  const note = noteInput ? noteInput.value : "";
  const r = await fetch("/api/review/" + encodeURIComponent(clipId), {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({status, note}),
  });
  if (!r.ok) { alert("Failed to save: " + r.status); return; }
  const card = btn.closest(".card");
  card.classList.remove("keep", "reject", "uncertain", "pending");
  card.classList.add(status);
  card.dataset.status = status;
  const badge = card.querySelector(".status-badge");
  if (badge) badge.textContent = status;
  applyFilter();
}

async function toggleCurious(clipId, btn) {
  const card = btn.closest(".card");
  const newFlag = card.dataset.curious !== "1";
  const r = await fetch("/api/curious/" + encodeURIComponent(clipId), {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({is_curious: newFlag}),
  });
  if (!r.ok) { alert("Failed to save: " + r.status); return; }
  card.classList.toggle("is-curious", newFlag);
  card.dataset.curious = newFlag ? "1" : "0";
  btn.classList.toggle("on", newFlag);
  btn.classList.toggle("off", !newFlag);
  applyFilter();
}

applySort();
updateCounts();
</script>
</body>
</html>
"""


def _files_url(path: str) -> str:
    """Convert a local library file path into a review_server.py-served URL."""
    rel = Path(path).relative_to(C.LIBRARY_ROOT)
    return "/files/" + urllib.parse.quote(str(rel))


def _multispecies_tags(row: sqlite3.Row) -> str:
    """Render Multispecies killer-whale + top call-type tags, if scored."""
    raw = row["multispecies_scores"] if "multispecies_scores" in row.keys() else None
    if not raw:
        return ""
    try:
        scores = json.loads(raw)
    except (ValueError, TypeError):
        return ""
    tags = f"<span class='tag ms'>Oo {scores.get('Oo', 0.0):.2f}</span>"
    call_types = {k: scores.get(k, 0.0) for k in ("Call", "Echolocation", "Whistle")}
    top_call = max(call_types, key=call_types.get)
    if call_types[top_call] >= 0.5:
        tags += f"<span class='tag ms'>{top_call} {call_types[top_call]:.2f}</span>"
    return tags


def _ref_call_tag(row: sqlite3.Row) -> str:
    """Render the nearest-Ford-Osborne-call + pod + similarity (SRKW clips only).

    Color-codes by similarity band so a quick scroll surfaces high-confidence
    matches: green (>=0.5), olive (0.35-0.5), grey (<0.35 = noise / weak match).
    """
    keys = row.keys()
    if "nearest_ref_call" not in keys or not row["nearest_ref_call"]:
        return ""
    sim = row["nearest_ref_similarity"] or 0.0
    band = "high" if sim >= 0.5 else "mid" if sim >= 0.35 else "low"
    pod = row["nearest_ref_pod"] or "?"
    call = row["nearest_ref_call"]
    return f"<span class='tag ref {band}'>{escape(call)}-{escape(pod)} ({sim:.2f})</span>"


def _ref_similarity(row: sqlite3.Row) -> float:
    keys = row.keys()
    if "nearest_ref_similarity" not in keys or row["nearest_ref_similarity"] is None:
        return 0.0
    return float(row["nearest_ref_similarity"])


def _multispecies_oo(row: sqlite3.Row) -> float:
    raw = row["multispecies_scores"] if "multispecies_scores" in row.keys() else None
    if not raw:
        return 0.0
    try:
        return float(json.loads(raw).get("Oo", 0.0))
    except (ValueError, TypeError):
        return 0.0


def render(out_path: Path) -> int:
    """Render the static review/comparison HTML. Returns clip count."""
    conn = sqlite3.connect(str(C.DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = list(
        conn.execute("SELECT * FROM clips ORDER BY peak_confidence DESC, start_unix ASC")
    )
    conn.close()

    cards = []
    for row in rows:
        clip_id = row["clip_id"]
        status = row["review_status"] or "pending"
        # is_curious column may not exist on very old catalogs
        is_curious = bool(row["is_curious"]) if "is_curious" in row.keys() and row["is_curious"] else False
        curious_classes = " is-curious" if is_curious else ""
        spec_url = _files_url(row["spectrogram_path"])
        species = row["species"] or "?"
        meta_html = (
            f"<span class='tag sp-{escape(species)}'>{escape(species)}</span>"
            f"<span class='tag'>conf {row['peak_confidence']:.2f}</span>"
            f"<span class='tag'>SNR {row['snr_db']:.1f} dB</span>"
            f"<span class='tag'>segs {row['n_segments']}</span>"
            f"<span class='tag'>Acartia &plusmn;24h: {row['acartia_sightings_within_24h_50km']}</span>"
            f"{_multispecies_tags(row)}"
            f"{_ref_call_tag(row)}"
            f"<span class='status-badge'>{escape(status)}</span>"
        )
        card = f"""
    <div class="card {escape(status)}{curious_classes}" id="card-{escape(clip_id)}"
         data-status="{escape(status)}" data-conf="{row['peak_confidence']:.4f}"
         data-snr="{row['snr_db']:.4f}" data-time="{row['start_unix']:.0f}"
         data-msoo="{_multispecies_oo(row):.4f}" data-species="{escape(species)}"
         data-refsim="{_ref_similarity(row):.4f}"
         data-curious="{'1' if is_curious else '0'}"><span class="curious-badge">★</span>
      <div class="meta">
        <b>{escape(row['start_utc_iso'])}</b><br>
        <code>{escape(clip_id)}</code><br>{meta_html}
      </div>
      <a href="{spec_url}" target="_blank" title="open full-size spectrogram">
        <img class="spec" loading="lazy" src="{spec_url}" alt="spectrogram"/>
      </a>
      <div class="audio-row">
        <label>raw</label>
        <audio controls preload="none" src="{_files_url(row['raw_wav_path'])}"></audio>
        <label>cleaned</label>
        <audio controls preload="none" src="{_files_url(row['clean_wav_path'])}"></audio>
      </div>
      <div class="actions">
        <button class="keep" onclick="setStatus('{escape(clip_id)}','keep',this)">keep</button>
        <button class="reject" onclick="setStatus('{escape(clip_id)}','reject',this)">reject</button>
        <button class="uncertain" onclick="setStatus('{escape(clip_id)}','uncertain',this)">uncertain</button>
        <button class="curious {'on' if is_curious else 'off'}" onclick="toggleCurious('{escape(clip_id)}',this)">★ curious</button>
        <input id="note-{escape(clip_id)}" class="note" placeholder="optional note"
               value="{escape(row['review_note'] or '')}"/>
      </div>
    </div>"""
        cards.append(card)

    summary = f"{len(rows)} clips"
    html = HTML_TEMPLATE.replace("__CARDS__", "\n".join(cards)).replace(
        "__SUMMARY__", summary
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return len(rows)
