"""Per-point inspection history for the tomato pipeline (SQLite, stdlib only).

The robot returns to the same plant/point repeatedly, so "is it getting worse?" is often more
useful than any single photo. Each analysed inspection is stored with its full result JSON, the
sensor readings and a small thumbnail; trend() compares a new result with the previous inspection of
the same point.

A fresh connection is opened per call: the web server is multi-threaded and sqlite3 connections
must not be shared across threads.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Healthy leaves already vary by up to ~6 points of "affected" share between photos (lesion.py,
# measured on PlantVillage), so smaller changes than 10 points can't be told from noise.
STABLE_BAND = 0.10
THUMB_SIDE = 200


class TomatoHistory:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS tomato_inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    point_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    condition_confidence REAL,
                    severity TEXT NOT NULL,
                    affected_fraction REAL,
                    status_label TEXT NOT NULL,
                    sensors TEXT,
                    result TEXT NOT NULL,
                    thumb BLOB
                )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_point_ts ON tomato_inspections(point_id, ts)")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, point_id: str, result: dict, bgr: np.ndarray, sensors: dict | None = None) -> int:
        h, w = bgr.shape[:2]
        s = THUMB_SIDE / max(h, w)
        small = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO tomato_inspections(point_id, ts, condition, condition_confidence, severity, "
                "affected_fraction, status_label, sensors, result, thumb) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (point_id, ts, result["condition"], result["conditionConfidence"], result["severity"],
                 result["affectedFraction"], result["statusLabel"], json.dumps(sensors or {}),
                 json.dumps(result, ensure_ascii=False), buf.tobytes() if ok else None))
            return cur.lastrowid

    def previous(self, point_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT ts, condition, affected_fraction FROM tomato_inspections WHERE point_id=? "
                "ORDER BY ts DESC, id DESC LIMIT 1", (point_id,)).fetchone()
            count = c.execute("SELECT COUNT(*) FROM tomato_inspections WHERE point_id=?", (point_id,)).fetchone()[0]
        if row is None:
            return None
        return {"ts": row["ts"], "condition": row["condition"], "affected_fraction": row["affected_fraction"],
                "count": count}

    def trend(self, point_id: str, condition: str, affected: float | None) -> dict | None:
        """Change versus the previous inspection of this point, or None on a first visit.
        Call BEFORE add() for the current result."""
        prev = self.previous(point_id)
        if prev is None:
            return None
        out = {"previousTs": prev["ts"], "previousCondition": prev["condition"],
               "conditionChanged": prev["condition"] != condition, "inspectionsSoFar": prev["count"],
               "previousAffectedFraction": prev["affected_fraction"], "deltaAffected": None,
               "direction": "unknown"}
        if affected is not None and prev["affected_fraction"] is not None:
            delta = affected - prev["affected_fraction"]
            out["deltaAffected"] = round(delta, 4)
            out["direction"] = "worsening" if delta >= STABLE_BAND else "improving" if delta <= -STABLE_BAND else "stable"
        return out

    def series(self, point_id: str, limit: int = 200) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, ts, condition, severity, affected_fraction, status_label, sensors FROM "
                "tomato_inspections WHERE point_id=? ORDER BY ts ASC, id ASC LIMIT ?", (point_id, limit)).fetchall()
        return [{**dict(r), "sensors": json.loads(r["sensors"] or "{}")} for r in rows]

    def points(self) -> list[dict]:
        """Latest inspection per point, most recently visited first, with the damage series for a sparkline."""
        with self._conn() as c:
            ids = [r[0] for r in c.execute(
                "SELECT point_id FROM tomato_inspections GROUP BY point_id ORDER BY MAX(ts) DESC")]
        out = []
        for pid in ids:
            s = self.series(pid)
            out.append({"point_id": pid, "latest": s[-1], "count": len(s),
                        "damage": [x["affected_fraction"] for x in s]})
        return out

    def thumb(self, inspection_id: int) -> bytes | None:
        with self._conn() as c:
            row = c.execute("SELECT thumb FROM tomato_inspections WHERE id=?", (inspection_id,)).fetchone()
        return bytes(row["thumb"]) if row and row["thumb"] else None

    def result(self, inspection_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT result FROM tomato_inspections WHERE id=?", (inspection_id,)).fetchone()
        return json.loads(row["result"]) if row else None
