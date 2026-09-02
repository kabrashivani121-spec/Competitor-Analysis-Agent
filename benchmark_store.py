"""SQLite persistence for benchmarking reports, notes, and portfolio entries."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

import config


class BenchmarkStore:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = str(db_path or (config.RUNTIME_DIR / "benchmarking.db"))
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                );
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                );
                CREATE TABLE IF NOT EXISTS portfolio (
                    ticker TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    shares REAL NOT NULL DEFAULT 0,
                    cost_basis REAL NOT NULL DEFAULT 0,
                    current_price REAL NOT NULL DEFAULT 0,
                    fair_value REAL NOT NULL DEFAULT 0,
                    currency TEXT NOT NULL DEFAULT '$',
                    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                );
                """
            )

    def save_report(self, mode: str, subject: str, content: str, metadata: dict[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO reports (mode, subject, content, metadata_json) VALUES (?, ?, ?, ?)",
                (mode, subject, content, json.dumps(metadata, ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def list_reports(self, mode: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        query = "SELECT id, mode, subject, created_at, metadata_json FROM reports"
        params: list[Any] = []
        if mode:
            query += " WHERE mode = ?"
            params.append(mode)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_report(self, report_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        try:
            result["metadata"] = json.loads(result.pop("metadata_json"))
        except (json.JSONDecodeError, TypeError):
            result["metadata"] = {}
        return result

    def delete_report(self, report_id: int):
        with self._connect() as connection:
            connection.execute("DELETE FROM reports WHERE id = ?", (report_id,))

    def add_note(self, subject: str, title: str, content: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO notes (subject, title, content) VALUES (?, ?, ?)",
                (subject.strip(), title.strip(), content.strip()),
            )
            return int(cursor.lastrowid)

    def list_notes(self, subject: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM notes"
        params: tuple[Any, ...] = ()
        if subject:
            query += " WHERE subject = ?"
            params = (subject,)
        query += " ORDER BY id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def delete_note(self, note_id: int):
        with self._connect() as connection:
            connection.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    def upsert_portfolio(
        self,
        ticker: str,
        company_name: str,
        shares: float,
        cost_basis: float,
        current_price: float,
        fair_value: float,
        currency: str,
    ):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO portfolio
                    (ticker, company_name, shares, cost_basis, current_price, fair_value, currency)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    company_name=excluded.company_name,
                    shares=excluded.shares,
                    cost_basis=excluded.cost_basis,
                    current_price=excluded.current_price,
                    fair_value=excluded.fair_value,
                    currency=excluded.currency,
                    updated_at=datetime('now', 'localtime')
                """,
                (
                    ticker.upper(),
                    company_name,
                    shares,
                    cost_basis,
                    current_price,
                    fair_value,
                    currency,
                ),
            )

    def portfolio_frame(self) -> pd.DataFrame:
        with self._connect() as connection:
            frame = pd.read_sql_query("SELECT * FROM portfolio ORDER BY ticker", connection)
        if frame.empty:
            return frame
        frame["Market value"] = frame["shares"] * frame["current_price"]
        frame["Cost value"] = frame["shares"] * frame["cost_basis"]
        frame["Unrealized P/L"] = frame["Market value"] - frame["Cost value"]
        frame["Fair-value upside"] = frame.apply(
            lambda row: row["fair_value"] / row["current_price"] - 1
            if row["current_price"]
            else 0,
            axis=1,
        )
        return frame

    def delete_portfolio(self, ticker: str):
        with self._connect() as connection:
            connection.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker.upper(),))
