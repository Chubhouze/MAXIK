"""Per inspection-point history. The robot visits the same points repeatedly, so change over time
is a strong signal (shrinking plant = wilting/weak growth, rising yellow share = yellowing)."""
import json
import sqlite3
from pathlib import Path


class InspectionHistory:
    def __init__(self, db_path: str | Path = "inspections.db"):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                point_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                image_path TEXT,
                label TEXT,
                confidence REAL,
                stats TEXT,
                sensors TEXT
            )"""
        )
        self.conn.commit()

    def last(self, point_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT stats FROM inspections WHERE point_id=? ORDER BY ts DESC LIMIT 1", (point_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def add(self, point_id: str, ts: str, image_path: str, report: dict):
        self.conn.execute(
            "INSERT INTO inspections(point_id, ts, image_path, label, confidence, stats, sensors) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                point_id, ts, image_path, report["label"], report["confidence"],
                json.dumps(report["color_stats"]), json.dumps(report.get("sensors", {})),
            ),
        )
        self.conn.commit()
