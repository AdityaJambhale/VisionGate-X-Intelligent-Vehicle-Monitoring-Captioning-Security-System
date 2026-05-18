"""
VisionGate X — Database module
Handles all SQLite interactions: schema creation, inserts, queries.
"""

import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Schema ─────────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    source          TEXT,                   -- video filename or 'live'
    frame_number    INTEGER,
    vehicle_type    TEXT,
    plate_number    TEXT,
    helmet_status   TEXT,                   -- 'helmet' | 'no_helmet' | 'unknown'
    confidence      REAL,
    caption         TEXT,
    bbox_json       TEXT,                   -- JSON string of bounding boxes
    snapshot_path   TEXT
);

CREATE TABLE IF NOT EXISTS violations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id    INTEGER REFERENCES detections(id),
    timestamp       TEXT    NOT NULL,
    plate_number    TEXT,
    violation_type  TEXT,
    description     TEXT,
    fine_inr        REAL,
    challan_id      TEXT,
    challan_path    TEXT,
    notified        INTEGER DEFAULT 0       -- 0=no, 1=yes
);

CREATE TABLE IF NOT EXISTS challans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    challan_id      TEXT    UNIQUE NOT NULL,
    plate_number    TEXT,
    violation_type  TEXT,
    fine_inr        REAL,
    issued_at       TEXT,
    status          TEXT DEFAULT 'pending', -- 'pending' | 'paid' | 'cancelled'
    pdf_path        TEXT,
    owner_name      TEXT,
    owner_contact   TEXT
);

CREATE INDEX IF NOT EXISTS idx_detections_plate    ON detections(plate_number);
CREATE INDEX IF NOT EXISTS idx_detections_ts       ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_violations_plate    ON violations(plate_number);
CREATE INDEX IF NOT EXISTS idx_challans_challan_id ON challans(challan_id);
"""


class Database:
    """Thread-safe SQLite wrapper for VisionGate X."""

    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self._init_schema()

    # ── Internal helpers ───────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)

    # ── Detections ─────────────────────────────────────────────────────────────

    def insert_detection(
        self,
        source: str,
        frame_number: int,
        vehicle_type: str,
        plate_number: str,
        helmet_status: str,
        confidence: float,
        caption: str,
        bbox_json: dict | list | None = None,
        snapshot_path: str = "",
    ) -> int:
        ts = datetime.now().isoformat(timespec="seconds")
        bbox_str = json.dumps(bbox_json) if bbox_json else "{}"
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO detections
                   (timestamp, source, frame_number, vehicle_type, plate_number,
                    helmet_status, confidence, caption, bbox_json, snapshot_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (ts, source, frame_number, vehicle_type, plate_number,
                 helmet_status, confidence, caption, bbox_str, snapshot_path),
            )
            return cur.lastrowid

    def get_detections(
        self,
        limit: int = 200,
        plate_filter: str = "",
        helmet_filter: str = "",
    ) -> list[dict]:
        query = "SELECT * FROM detections WHERE 1=1"
        params = []
        if plate_filter:
            query += " AND plate_number LIKE ?"
            params.append(f"%{plate_filter}%")
        if helmet_filter:
            query += " AND helmet_status = ?"
            params.append(helmet_filter)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── Violations ─────────────────────────────────────────────────────────────

    def insert_violation(
        self,
        detection_id: int,
        plate_number: str,
        violation_type: str,
        description: str,
        fine_inr: float,
        challan_id: str = "",
        challan_path: str = "",
    ) -> int:
        ts = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO violations
                   (detection_id, timestamp, plate_number, violation_type,
                    description, fine_inr, challan_id, challan_path)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (detection_id, ts, plate_number, violation_type,
                 description, fine_inr, challan_id, challan_path),
            )
            return cur.lastrowid

    def get_violations(self, limit: int = 200, plate_filter: str = "") -> list[dict]:
        query = "SELECT * FROM violations WHERE 1=1"
        params = []
        if plate_filter:
            query += " AND plate_number LIKE ?"
            params.append(f"%{plate_filter}%")
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def mark_violation_notified(self, violation_id: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE violations SET notified=1 WHERE id=?", (violation_id,)
            )

    # ── Challans ───────────────────────────────────────────────────────────────

    def insert_challan(
        self,
        challan_id: str,
        plate_number: str,
        violation_type: str,
        fine_inr: float,
        pdf_path: str = "",
        owner_name: str = "",
        owner_contact: str = "",
    ) -> int:
        ts = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO challans
                   (challan_id, plate_number, violation_type, fine_inr,
                    issued_at, pdf_path, owner_name, owner_contact)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (challan_id, plate_number, violation_type, fine_inr,
                 ts, pdf_path, owner_name, owner_contact),
            )
            return cur.lastrowid

    def get_challans(self, limit: int = 200, status_filter: str = "") -> list[dict]:
        query = "SELECT * FROM challans WHERE 1=1"
        params = []
        if status_filter:
            query += " AND status = ?"
            params.append(status_filter)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def update_challan_status(self, challan_id: str, status: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE challans SET status=? WHERE challan_id=?",
                (status, challan_id),
            )

    # ── Analytics ──────────────────────────────────────────────────────────────

    def get_summary_stats(self) -> dict:
        with self._conn() as conn:
            total_detections  = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            total_violations  = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
            total_challans    = conn.execute("SELECT COUNT(*) FROM challans").fetchone()[0]
            total_fines       = conn.execute("SELECT COALESCE(SUM(fine_inr),0) FROM challans").fetchone()[0]
            no_helmet_count   = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE helmet_status='no_helmet'"
            ).fetchone()[0]
        return {
            "total_detections": total_detections,
            "total_violations": total_violations,
            "total_challans":   total_challans,
            "total_fines_inr":  total_fines,
            "no_helmet_count":  no_helmet_count,
        }
