"""成本统计: SQLite 持久化 + 节省金额基线对比.

基线: 同样的 token 如果全部走最贵档 (always-hard) 或全部走标准档
(always-standard) 要花多少钱 —— 差额即路由插件省下的钱.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    tier TEXT NOT NULL,          -- 实际命中档位 trivial/standard/hard
    model TEXT NOT NULL,         -- 实际调用模型
    score INTEGER NOT NULL,      -- 难度分
    reasons TEXT NOT NULL,       -- 评分理由
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    baseline_hard_usd REAL NOT NULL DEFAULT 0,      -- 若全走 hard 档的成本
    baseline_standard_usd REAL NOT NULL DEFAULT 0,  -- 若全走 standard 档的成本
    latency_ms INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,        -- ok / fallback / error
    error TEXT,
    query TEXT                   -- 用户问题摘要 (供经验回忆/反馈引用)
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
"""


def calc_cost(price: dict, prompt_tokens: int, completion_tokens: int,
              cache_hit_tokens: int = 0) -> float:
    hit_rate = price.get("cache_hit_input", price["input"])
    in_cost = ((prompt_tokens - cache_hit_tokens) * price["input"]
               + cache_hit_tokens * hit_rate) / 1_000_000
    return in_cost + completion_tokens * price["output"] / 1_000_000


class Stats:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            # 老库迁移: v3 新增 query 列
            cols = {r[1] for r in self._conn.execute(
                "PRAGMA table_info(requests)").fetchall()}
            if "query" not in cols:
                self._conn.execute(
                    "ALTER TABLE requests ADD COLUMN query TEXT")
            self._conn.commit()

    def record(self, *, tier: str, model: str, score: int, reasons: str,
               prompt_tokens: int, completion_tokens: int, cost: float,
               baseline_hard: float, baseline_standard: float,
               latency_ms: int, status: str, error: str | None = None,
               query: str = "") -> int:
        """记录一次请求, 返回 request_id (供反馈闭环引用)."""
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO requests
                   (ts, tier, model, score, reasons, prompt_tokens,
                    completion_tokens, cost_usd, baseline_hard_usd,
                    baseline_standard_usd, latency_ms, status, error, query)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (time.time(), tier, model, score, reasons, prompt_tokens,
                 completion_tokens, cost, baseline_hard, baseline_standard,
                 latency_ms, status, error, query[:2000]),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_request(self, request_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
        return dict(row) if row else None

    def daily_spend(self) -> float:
        """今日 (本地时区零点起) 的实际花费."""
        day_start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        with self._lock:
            row = self._conn.execute(
                """SELECT COALESCE(SUM(cost_usd),0) AS s FROM requests
                   WHERE ts >= ? AND status != 'error'""",
                (day_start,)).fetchone()
        return row["s"]

    def latency_by_model(self) -> dict[str, dict]:
        """每个模型的平均延迟与调用数 (供看板展示)."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT model, AVG(latency_ms) AS avg_ms, COUNT(*) AS n
                   FROM requests WHERE status != 'error' GROUP BY model"""
            ).fetchall()
        return {r["model"]: {"avg_latency_ms": round(r["avg_ms"] or 0),
                             "calls": r["n"]} for r in rows}

    def summary(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) AS n,
                          COALESCE(SUM(cost_usd),0) AS cost,
                          COALESCE(SUM(baseline_hard_usd),0) AS bh,
                          COALESCE(SUM(baseline_standard_usd),0) AS bs,
                          COALESCE(SUM(prompt_tokens),0) AS pt,
                          COALESCE(SUM(completion_tokens),0) AS ct
                   FROM requests WHERE status != 'error'"""
            ).fetchone()
            tiers = self._conn.execute(
                "SELECT tier, COUNT(*) AS n FROM requests GROUP BY tier"
            ).fetchall()
            recent = self._conn.execute(
                """SELECT ts, tier, model, score, reasons, prompt_tokens,
                          completion_tokens, cost_usd, latency_ms, status, error
                   FROM requests ORDER BY id DESC LIMIT 50"""
            ).fetchall()
        d = dict(row)
        d["saved_vs_hard"] = d["bh"] - d["cost"]
        d["saved_vs_standard"] = d["bs"] - d["cost"]
        d["tiers"] = {r["tier"]: r["n"] for r in tiers}
        d["recent"] = [dict(r) for r in recent]
        return d

    def close(self) -> None:
        with self._lock:
            self._conn.close()
