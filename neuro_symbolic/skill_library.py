from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import sqlite_vec

from memory.memory_store import HashEmbedder, TextEmbedder
from neuro_symbolic.skill_dsl import BehaviorSkill


@dataclass(frozen=True)
class StoredSkill:
    skill_id: int
    skill: BehaviorSkill
    uses_anomaly: bool
    use_count: int
    success_count: int
    last_executed_tick: int | None
    distance: float | None = None


class SkillLibrary:
    """Vector-indexed store of symbolically verified reusable behavior skills."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedder: TextEmbedder | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.embedder: TextEmbedder = embedder or HashEmbedder(dim=384)
        self._lock = threading.RLock()
        self._closed = False
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    trigger_context TEXT NOT NULL,
                    skill_json TEXT NOT NULL,
                    uses_anomaly INTEGER NOT NULL,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    last_executed_tick INTEGER
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS skill_vec USING vec0(
                    embedding float[{self.embedder.dim}] distance_metric=cosine
                );
                """
            )
            self.conn.commit()

    def count(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM skills").fetchone()
            return int(row["n"]) if row is not None else 0

    def list_names(self) -> list[str]:
        with self._lock:
            rows = self.conn.execute("SELECT name FROM skills ORDER BY id ASC").fetchall()
            return [str(r["name"]) for r in rows]

    def use_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT name, use_count FROM skills ORDER BY id ASC"
            ).fetchall()
            return {str(r["name"]): int(r["use_count"]) for r in rows}

    def _skill_embedding_text(self, skill: BehaviorSkill) -> str:
        prims = " ".join(s.primitive for s in skill.steps)
        targets = " ".join(s.target_object for s in skill.steps if s.target_object)
        return f"{skill.name} {skill.trigger_context} {prims} {targets}"

    def add_skill(self, skill: BehaviorSkill, uses_anomaly: bool) -> int:
        """Insert or update a verified skill and sync its sqlite-vec embedding."""
        skill_json = skill.model_dump_json()
        vec = self.embedder.embed(self._skill_embedding_text(skill))
        serialized = sqlite_vec.serialize_float32(vec.tolist())

        with self._lock:
            existing = self.conn.execute(
                "SELECT id FROM skills WHERE name = ?",
                (skill.name,),
            ).fetchone()

            if existing is not None:
                skill_id = int(existing["id"])
                self.conn.execute(
                    """
                    UPDATE skills
                    SET trigger_context = ?, skill_json = ?, uses_anomaly = ?
                    WHERE id = ?
                    """,
                    (skill.trigger_context, skill_json, int(uses_anomaly), skill_id),
                )
                self.conn.execute("DELETE FROM skill_vec WHERE rowid = ?", (skill_id,))
                self.conn.execute(
                    "INSERT INTO skill_vec(rowid, embedding) VALUES (?, ?)",
                    (skill_id, serialized),
                )
            else:
                cur = self.conn.execute(
                    """
                    INSERT INTO skills (name, trigger_context, skill_json, uses_anomaly)
                    VALUES (?, ?, ?, ?)
                    """,
                    (skill.name, skill.trigger_context, skill_json, int(uses_anomaly)),
                )
                skill_id = int(cur.lastrowid)
                self.conn.execute(
                    "INSERT INTO skill_vec(rowid, embedding) VALUES (?, ?)",
                    (skill_id, serialized),
                )

            self.conn.commit()
            return skill_id

    def get_by_name(self, name: str) -> StoredSkill | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM skills WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        return self._row_to_stored(row, distance=None)

    def retrieve(
        self,
        context_query: str,
        k: int = 3,
        max_distance: float = 0.85,
        current_tick: int | None = None,
    ) -> list[StoredSkill]:
        """Retrieve semantically matching skills, filtering out skills on cooldown."""
        if k <= 0:
            return []
        query_vec = self.embedder.embed(context_query)
        serialized = sqlite_vec.serialize_float32(query_vec.tolist())

        with self._lock:
            total = self.count()
            if total == 0:
                return []

            fetch_k = min(total, max(k * 3, 6))
            knn_rows = self.conn.execute(
                """
                SELECT rowid, distance
                FROM skill_vec
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance ASC
                """,
                (serialized, fetch_k),
            ).fetchall()
            if not knn_rows:
                return []

            dist_map = {int(r["rowid"]): float(r["distance"]) for r in knn_rows}
            placeholders = ",".join("?" for _ in dist_map)
            rows = self.conn.execute(
                f"SELECT * FROM skills WHERE id IN ({placeholders})",
                tuple(dist_map.keys()),
            ).fetchall()

        results: list[StoredSkill] = []
        for row in rows:
            sid = int(row["id"])
            dist = dist_map[sid]
            if dist > max_distance:
                continue
            stored = self._row_to_stored(row, distance=dist)
            if (
                current_tick is not None
                and stored.last_executed_tick is not None
                and (current_tick - stored.last_executed_tick) < stored.skill.cooldown_ticks
            ):
                continue
            results.append(stored)

        results.sort(key=lambda s: s.distance if s.distance is not None else 999.0)
        return results[:k]

    def record_execution(self, name: str, tick: int, success: bool = True) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE skills
                SET use_count = use_count + 1,
                    success_count = success_count + ?,
                    last_executed_tick = ?
                WHERE name = ?
                """,
                (int(success), tick, name),
            )
            self.conn.commit()

    def total_reuses(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(CASE WHEN use_count > 1 THEN use_count - 1 ELSE 0 END), 0) AS n "
                "FROM skills"
            ).fetchone()
            return int(row["n"]) if row is not None else 0

    @staticmethod
    def _row_to_stored(row: sqlite3.Row, distance: float | None) -> StoredSkill:
        return StoredSkill(
            skill_id=int(row["id"]),
            skill=BehaviorSkill.model_validate_json(str(row["skill_json"])),
            uses_anomaly=bool(row["uses_anomaly"]),
            use_count=int(row["use_count"]),
            success_count=int(row["success_count"]),
            last_executed_tick=(
                int(row["last_executed_tick"]) if row["last_executed_tick"] is not None else None
            ),
            distance=distance,
        )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()