"""Tiny Flask app to serve the review UI and record keep/reject decisions.

Run with:
    ~/miniconda3/envs/whales/bin/python src/review_server.py

Then visit http://127.0.0.1:5000/
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

import sys

# Allow running this file directly without `pip install -e`.
sys.path.insert(0, str(Path(__file__).parent))
from pilot import config as C
from pilot import ui as ui_mod


REVIEW_HTML = C.REVIEW_DIR / "index.html"
ALLOWED_STATUS = {"keep", "reject", "uncertain", "pending"}


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        # Re-render whenever the SQLite catalog is newer than the static
        # HTML, so review labels written via /api/review survive a page
        # reload. Without this, the dashboard kept serving a snapshot
        # frozen at the end of the last batch run — making the user's
        # interim reviews look lost on browser refresh (D-024).
        needs_render = (
            not REVIEW_HTML.exists()
            or C.DB_PATH.stat().st_mtime > REVIEW_HTML.stat().st_mtime
        )
        if needs_render:
            n = ui_mod.render(REVIEW_HTML)
            app.logger.info("re-rendered review page with %d clips", n)
        return REVIEW_HTML.read_text(), 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.route("/refresh")
    def refresh():
        n = ui_mod.render(REVIEW_HTML)
        return f"refreshed: {n} clips", 200

    @app.route("/files/<path:rel>")
    def files(rel):
        # serve files from inside the library tree only
        return send_from_directory(str(C.LIBRARY_ROOT), rel)

    @app.route("/api/review/<clip_id>", methods=["POST"])
    def review(clip_id):
        payload = request.get_json(force=True) or {}
        status = payload.get("status")
        note = payload.get("note") or None
        if status not in ALLOWED_STATUS:
            abort(400, f"bad status: {status}")
        conn = sqlite3.connect(str(C.DB_PATH))
        cur = conn.execute(
            "UPDATE clips SET review_status = ?, review_note = ? WHERE clip_id = ?",
            (status, note, clip_id),
        )
        conn.commit()
        conn.close()
        if cur.rowcount == 0:
            abort(404, "clip not found")
        return jsonify({"ok": True, "clip_id": clip_id, "status": status})

    @app.route("/api/curious/<clip_id>", methods=["POST"])
    def curious(clip_id):
        """Toggle the `is_curious` flag independently of review_status (D-034).
        Body: {"is_curious": true|false, "note": "optional appended note"}."""
        payload = request.get_json(force=True) or {}
        flag = bool(payload.get("is_curious"))
        note = payload.get("note") or None
        conn = sqlite3.connect(str(C.DB_PATH))
        if note:
            # Append-only: preserve existing note, append the new one with separator
            row = conn.execute("SELECT review_note FROM clips WHERE clip_id=?",
                               (clip_id,)).fetchone()
            existing = row[0] if row else None
            combined = f"{existing} | {note}" if existing else note
            cur = conn.execute(
                "UPDATE clips SET is_curious = ?, review_note = ? WHERE clip_id = ?",
                (1 if flag else 0, combined, clip_id))
        else:
            cur = conn.execute(
                "UPDATE clips SET is_curious = ? WHERE clip_id = ?",
                (1 if flag else 0, clip_id))
        conn.commit()
        conn.close()
        if cur.rowcount == 0:
            abort(404, "clip not found")
        return jsonify({"ok": True, "clip_id": clip_id, "is_curious": flag})

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
