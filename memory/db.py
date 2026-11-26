"""SQLite 跨会话记忆持久化：术语库 / 用户偏好 / 历史经验 / 运行记录。"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import ExperienceItem, PreferenceItem, TermItem

_SCHEMA = """
CREATE TABLE IF NOT EXISTS terms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term TEXT NOT NULL,
    standard_form TEXT DEFAULT '',
    locked INTEGER DEFAULT 0,
    lang TEXT DEFAULT 'zh',
    note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT DEFAULT '',
    note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT DEFAULT '',
    source_text TEXT NOT NULL,
    rewritten_text TEXT NOT NULL,
    tags TEXT DEFAULT '',
    quality REAL DEFAULT 0.8,
    note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    mode TEXT DEFAULT 'mock',
    status TEXT DEFAULT 'delivered',
    paragraphs INTEGER DEFAULT 0,
    reflection_rounds INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost REAL DEFAULT 0.0,
    latency_ms REAL DEFAULT 0.0,
    verify_pass_rate REAL DEFAULT 1.0
);
"""


class MemoryDB:
    """轻量 SQLite 封装：每次调用独立连接，线程安全且便于测试隔离。"""

    def __init__(self, db_path: Path | str, seed_dir: Optional[Path | str] = None):
        self.db_path = Path(db_path)
        self.seed_dir = Path(seed_dir) if seed_dir else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
        self._seed_if_empty()

    def _seed_if_empty(self) -> None:
        if not self.seed_dir:
            return
        with self._conn() as conn:
            if conn.execute("SELECT COUNT(*) c FROM terms").fetchone()["c"] == 0:
                self._seed_table(conn, "seed_terms.json", "terms",
                                 ["term", "standard_form", "locked", "lang", "note"])
            if conn.execute("SELECT COUNT(*) c FROM preferences").fetchone()["c"] == 0:
                self._seed_table(conn, "seed_preferences.json", "preferences",
                                 ["key", "value", "note"])
            if conn.execute("SELECT COUNT(*) c FROM experiences").fetchone()["c"] == 0:
                self._seed_table(conn, "seed_experiences.json", "experiences",
                                 ["title", "source_text", "rewritten_text", "tags", "quality", "note"])

    def _seed_table(self, conn: sqlite3.Connection, fname: str, table: str, cols: List[str]) -> None:
        fpath = self.seed_dir / fname
        if not fpath.exists():
            return
        rows = json.loads(fpath.read_text(encoding="utf-8"))
        placeholders = ",".join("?" * len(cols))
        for r in rows:
            conn.execute(
                f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
                [r.get(c) for c in cols],
            )

    # ------------------------------------------------------------------ #
    # terms
    # ------------------------------------------------------------------ #
    def add_term(self, item: TermItem) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO terms (term, standard_form, locked, lang, note) VALUES (?,?,?,?,?)",
                (item.term, item.standard_form, int(item.locked), item.lang, item.note),
            )
            return int(cur.lastrowid)

    def list_terms(self) -> List[TermItem]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM terms ORDER BY id").fetchall()
        return [TermItem(id=r["id"], term=r["term"], standard_form=r["standard_form"],
                         locked=bool(r["locked"]), lang=r["lang"], note=r["note"]) for r in rows]

    def locked_terms(self) -> List[TermItem]:
        return [t for t in self.list_terms() if t.locked]

    def delete_term(self, term_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM terms WHERE id=?", (term_id,))

    # ------------------------------------------------------------------ #
    # preferences
    # ------------------------------------------------------------------ #
    def set_preference(self, item: PreferenceItem) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO preferences (key, value, note) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, note=excluded.note",
                (item.key, item.value, item.note),
            )
            return int(cur.lastrowid or 0)

    def list_preferences(self) -> List[PreferenceItem]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM preferences ORDER BY id").fetchall()
        return [PreferenceItem(id=r["id"], key=r["key"], value=r["value"], note=r["note"]) for r in rows]

    def preferences_dict(self) -> Dict[str, str]:
        return {p.key: p.value for p in self.list_preferences()}

    def delete_preference(self, pref_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM preferences WHERE id=?", (pref_id,))

    # ------------------------------------------------------------------ #
    # experiences
    # ------------------------------------------------------------------ #
    def add_experience(self, item: ExperienceItem) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO experiences (title, source_text, rewritten_text, tags, quality, note)"
                " VALUES (?,?,?,?,?,?)",
                (item.title, item.source_text, item.rewritten_text, item.tags, item.quality, item.note),
            )
            return int(cur.lastrowid)

    def list_experiences(self) -> List[ExperienceItem]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM experiences ORDER BY quality DESC, id").fetchall()
        return [ExperienceItem(id=r["id"], title=r["title"], source_text=r["source_text"],
                               rewritten_text=r["rewritten_text"], tags=r["tags"],
                               quality=r["quality"], note=r["note"]) for r in rows]

    def delete_experience(self, exp_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM experiences WHERE id=?", (exp_id,))

    # ------------------------------------------------------------------ #
    # runs / metrics
    # ------------------------------------------------------------------ #
    def record_run(self, *, session_id: str, mode: str, status: str, paragraphs: int,
                   reflection_rounds: int, prompt_tokens: int, completion_tokens: int,
                   cost: float, latency_ms: float, verify_pass_rate: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (session_id, created_at, mode, status, paragraphs,"
                " reflection_rounds, prompt_tokens, completion_tokens, total_tokens,"
                " cost, latency_ms, verify_pass_rate) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (session_id, time.time(), mode, status, paragraphs, reflection_rounds,
                 prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
                 cost, latency_ms, verify_pass_rate),
            )

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def metrics_summary(self) -> Dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total_runs,"
                " COALESCE(AVG(reflection_rounds),0) AS avg_reflection_rounds,"
                " COALESCE(SUM(total_tokens),0) AS total_tokens,"
                " COALESCE(SUM(cost),0) AS total_cost,"
                " COALESCE(AVG(latency_ms),0) AS avg_latency_ms,"
                " COALESCE(AVG(verify_pass_rate),0) AS avg_verify_pass_rate,"
                " COALESCE(SUM(CASE WHEN status='delivered' THEN 1 ELSE 0 END),0) AS delivered_runs"
                " FROM runs"
            ).fetchone()
            by_status = conn.execute(
                "SELECT status, COUNT(*) c FROM runs GROUP BY status").fetchall()
        d = dict(row)
        d["pass_rate"] = (d["delivered_runs"] / d["total_runs"]) if d["total_runs"] else 0.0
        d["by_status"] = {r["status"]: r["c"] for r in by_status}
        return d
