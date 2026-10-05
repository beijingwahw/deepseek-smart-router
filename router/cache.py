"""语义缓存 (GPTCache 路线): 相似问题直接返回缓存答案, 零成本零延迟.

设计要点 (对齐 2025 前沿实践):
  - 向量相似度 + 硬边界: 只在同一难度档内命中, 防止"简单档缓存污染困难档"
  - 软阈值: 余弦相似度 >= threshold (默认 0.90) 才算命中
  - TTL 过期 + 容量上限 (LRU 淘汰最久未命中条目)
  - 只缓存"安全请求": 非流式 / 无工具 / 无视觉 (调用方负责判断)
  - 命中计入 stats(tier="cache", cost=0), 节省额直接体现在看板
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time

import numpy as np

from .embedder import cosine

SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    last_hit REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    tier TEXT NOT NULL,
    query TEXT NOT NULL,
    vector BLOB NOT NULL,
    response TEXT NOT NULL,
    model TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_tier ON semantic_cache(tier);
"""


class SemanticCache:
    def __init__(self, db_path: str, embedder, threshold: float = 0.90,
                 ttl_seconds: int = 86400, max_entries: int = 10000):
        self.embedder = embedder
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")  # 多组件共享库防锁
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---------- 查询 ----------

    def lookup(self, query: str, tier: str) -> dict | None:
        """找相似缓存. 命中返回答案 dict (OpenAI 格式), 否则 None."""
        vec = self.embedder.embed(query)
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, vector, response, model FROM semantic_cache "
                "WHERE tier=?", (tier,)).fetchall()
        best_id, best_sim, best = None, -1.0, None
        for r in rows:
            if now - r["ts"] > self.ttl:
                continue  # 过期
            sim = cosine(vec, np.frombuffer(r["vector"], dtype=np.float32))
            if sim > best_sim:
                best_id, best_sim, best = r["id"], sim, r
        if best is None or best_sim < self.threshold:
            self.misses += 1
            return None
        with self._lock:
            self._conn.execute(
                "UPDATE semantic_cache SET hits=hits+1, last_hit=? WHERE id=?",
                (now, best_id))
            self._conn.commit()
        self.hits += 1
        body = json.loads(best["response"])
        body.setdefault("router", {})["cache"] = {
            "hit": True, "similarity": round(best_sim, 4),
            "cached_model": best["model"]}
        return body

    # ---------- 写入 ----------

    def store(self, query: str, tier: str, response: dict, model: str) -> None:
        vec = self.embedder.embed(query)
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO semantic_cache (ts, last_hit, tier, query, vector,"
                " response, model) VALUES (?,?,?,?,?,?,?)",
                (now, now, tier, query[:2000], vec.tobytes(),
                 json.dumps(response, ensure_ascii=False), model))
            # 容量控制: 淘汰最久未命中的
            self._conn.execute(
                """DELETE FROM semantic_cache WHERE id NOT IN (
                       SELECT id FROM semantic_cache
                       ORDER BY last_hit DESC LIMIT ?)""",
                (self.max_entries,))
            # 顺手清过期
            self._conn.execute("DELETE FROM semantic_cache WHERE ts < ?",
                               (now - self.ttl,))
            self._conn.commit()

    # ---------- 统计 ----------

    def stats(self) -> dict:
        total = self.hits + self.misses
        with self._lock:
            size = self._conn.execute(
                "SELECT COUNT(*) AS n FROM semantic_cache").fetchone()["n"]
        return {"enabled": True, "entries": size, "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "threshold": self.threshold, "embedder": self.embedder.name}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
