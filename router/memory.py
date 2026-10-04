"""kNN 经验回忆: 相似历史任务的成败指导当下路由 (RouteLLM 语义路由路线).

与学习引擎 (learner.py) 的区别:
  - learner 是全局统计: "ds-r1 做数学题总体怎么样"
  - memory 是局部检索: "和这次问题最相似的几次, 都是谁答的, 答得怎么样"

每次调用完成, 把 (问题向量, 命中模型, 奖励) 存入经验库;
路由时检索 top-k 最相似经验, 对候选模型加回忆奖惩.
与缓存共用向量底座, 零额外依赖.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import numpy as np

from .embedder import cosine

SCHEMA = """
CREATE TABLE IF NOT EXISTS experience (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    query TEXT NOT NULL,
    vector BLOB NOT NULL,
    model TEXT NOT NULL,
    reward REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exp_ts ON experience(ts);
"""


class ExperienceMemory:
    def __init__(self, db_path: str, embedder, k: int = 5,
                 min_sim: float = 0.7, max_entries: int = 5000):
        self.embedder = embedder
        self.k = k
        self.min_sim = min_sim
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def record(self, query: str, model: str, reward: float) -> None:
        if not query.strip():
            return
        vec = self.embedder.embed(query)
        with self._lock:
            self._conn.execute(
                "INSERT INTO experience (ts, query, vector, model, reward)"
                " VALUES (?,?,?,?,?)",
                (time.time(), query[:2000], vec.tobytes(), model,
                 max(0.0, min(1.0, reward))))
            self._conn.execute(
                """DELETE FROM experience WHERE id NOT IN (
                       SELECT id FROM experience ORDER BY ts DESC LIMIT ?)""",
                (self.max_entries,))
            self._conn.commit()

    def recall(self, query: str) -> dict[str, float]:
        """检索相似经验, 返回 {模型: 相似度加权平均奖励} (无经验则空)."""
        if not query.strip():
            return {}
        vec = self.embedder.embed(query)
        with self._lock:
            rows = self._conn.execute(
                "SELECT vector, model, reward FROM experience").fetchall()
        scored = []
        for r in rows:
            sim = cosine(vec, np.frombuffer(r["vector"], dtype=np.float32))
            if sim >= self.min_sim:
                scored.append((sim, r["model"], r["reward"]))
        scored.sort(key=lambda x: -x[0])
        agg: dict[str, dict] = {}
        for sim, model, reward in scored[: self.k]:
            v = agg.setdefault(model, {"w": 0.0, "wr": 0.0})
            v["w"] += sim
            v["wr"] += sim * reward
        return {m: round(v["wr"] / v["w"], 3) for m, v in agg.items()
                if v["w"] > 0}

    def bonus(self, model: str, recalled: dict[str, float]) -> float:
        """模型在相似经验中的奖惩值 (-1 ~ +1). 无数据为 0."""
        wr = recalled.get(model)
        if wr is None:
            return 0.0
        return round((wr - 0.5) * 2, 3)

    def size(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) AS n FROM experience").fetchone()["n"]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
