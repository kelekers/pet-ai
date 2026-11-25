from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import numpy as np
import sqlite_vec


class TextEmbedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray:
        """Encode text into a unit-normalized float32 vector of length self.dim."""


class HashEmbedder:
    """Deterministic, zero-download 384D char-trigram embedder for fast offline testing."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        cleaned = f" {text.lower().strip()} "
        tokens = cleaned.split()
        ngrams = [cleaned[i : i + 3] for i in range(max(0, len(cleaned) - 2))]
        for feature in (*tokens, *ngrams):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-8 else vec


class FastEmbedder:
    """CPU-only ONNX Runtime embedder using fastembed (all-MiniLM-L6-v2, 384D)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.dim = 384

    def embed(self, text: str) -> np.ndarray:
        raw = next(self._model.embed([text]))
        vec = np.asarray(raw, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-8 else vec


@dataclass(frozen=True)
class Episode:
    episode_id: int
    tick: int
    entity: str
    action: str
    event_tag: str
    summary: str
    delta_p: float
    delta_a: float
    delta_d: float
    importance: float
    x: float
    y: float
    distance: float | None = None


class EpisodicMemoryStore:
    """Associative episodic memory backed by SQLite + sqlite-vec cosine KNN."""

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
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tick INTEGER NOT NULL,
                    entity TEXT NOT NULL,
                    action TEXT NOT NULL,
                    event_tag TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    delta_p REAL NOT NULL,
                    delta_a REAL NOT NULL,
                    delta_d REAL NOT NULL,
                    importance REAL NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_entity ON episodes(entity);
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
                    embedding float[{self.embedder.dim}] distance_metric=cosine
                );
                """
            )
            self.conn.commit()

    def count(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()
            return int(row["n"]) if row is not None else 0

    def record_episode(
        self,
        tick: int,
        entity: str,
        action: str,
        event_tag: str,
        summary: str,
        delta_p: float,
        delta_a: float,
        delta_d: float,
        importance: float,
        x: float,
        y: float,
    ) -> int:
        """Persist an episode and index its vector embedding in sqlite-vec."""
        vec = self.embedder.embed(f"{entity} {event_tag} {action}: {summary}")
        serialized = sqlite_vec.serialize_float32(vec.tolist())
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO episodes (
                    tick, entity, action, event_tag, summary,
                    delta_p, delta_a, delta_d, importance, x, y
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tick,
                    entity,
                    action,
                    event_tag,
                    summary,
                    delta_p,
                    delta_a,
                    delta_d,
                    importance,
                    x,
                    y,
                ),
            )
            episode_id = int(cur.lastrowid)
            self.conn.execute(
                "INSERT INTO memory_vec(rowid, embedding) VALUES (?, ?)",
                (episode_id, serialized),
            )
            self.conn.commit()
            return episode_id

    def retrieve_similar(
        self,
        query_text: str,
        k: int = 5,
        entity: str | None = None,
    ) -> list[Episode]:
        """Run sqlite-vec cosine KNN retrieval with optional entity filtering."""
        if k <= 0:
            return []
        query_vec = self.embedder.embed(query_text)
        serialized = sqlite_vec.serialize_float32(query_vec.tolist())

        with self._lock:
            total = self.count()
            if total == 0:
                return []

            fetch_k = min(total, k * 4 if entity is not None else k)
            knn_rows = self.conn.execute(
                """
                SELECT rowid, distance
                FROM memory_vec
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
                f"SELECT * FROM episodes WHERE id IN ({placeholders})",
                tuple(dist_map.keys()),
            ).fetchall()

        episodes: list[Episode] = []
        for r in rows:
            if entity is not None and r["entity"] != entity:
                continue
            ep_id = int(r["id"])
            episodes.append(
                Episode(
                    episode_id=ep_id,
                    tick=int(r["tick"]),
                    entity=str(r["entity"]),
                    action=str(r["action"]),
                    event_tag=str(r["event_tag"]),
                    summary=str(r["summary"]),
                    delta_p=float(r["delta_p"]),
                    delta_a=float(r["delta_a"]),
                    delta_d=float(r["delta_d"]),
                    importance=float(r["importance"]),
                    x=float(r["x"]),
                    y=float(r["y"]),
                    distance=dist_map[ep_id],
                )
            )
        episodes.sort(key=lambda ep: ep.distance if ep.distance is not None else 999.0)
        return episodes[:k]

    def entity_affinity(
        self,
        entity: str,
        current_tick: int,
        recency_decay: float = 0.004,
    ) -> tuple[float, float]:
        """Compute recency- and importance-weighted (trust [-1, 1], fear [0, 1]) for an entity."""
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT tick, delta_p, delta_a, delta_d, importance
                FROM episodes
                WHERE entity = ?
                """,
                (entity,),
            ).fetchall()
        if not rows:
            return (0.0, 0.0)

        weighted_valence = 0.0
        weighted_fear = 0.0
        weight_sum = 0.0
        for r in rows:
            age = max(0, current_tick - int(r["tick"]))
            w = float(r["importance"]) * math.exp(-recency_decay * age)
            dp = float(r["delta_p"])
            da = float(r["delta_a"])
            dd = float(r["delta_d"])
            # Positive pleasure builds trust; negative pleasure + high arousal + low dominance = fear
            weighted_valence += w * (dp + 0.25 * dd)
            fear_signal = max(0.0, -dp) * (1.0 + max(0.0, da)) * (1.0 + max(0.0, -dd))
            weighted_fear += w * fear_signal
            weight_sum += w

        if weight_sum <= 1e-8:
            return (0.0, 0.0)

        # Saturate smoothly via tanh as evidence accumulates (weight_sum scales confidence)
        evidence_gain = 1.0 - math.exp(-0.55 * weight_sum)
        trust = float(np.tanh(2.4 * (weighted_valence / weight_sum)) * evidence_gain)
        fear = float(np.clip(math.tanh(2.2 * (weighted_fear / weight_sum)) * evidence_gain, 0.0, 1.0))
        return (trust, fear)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()