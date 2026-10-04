"""自适应学习引擎: Thompson Sampling 反馈闭环.

这是 Genesis 版的核心 —— 路由器不再只信静态画像, 而是从真实使用结果中学习:

  - 每个 (模型, 任务标签) 维护一个 Beta 分布 (alpha=成功强度, beta=失败强度)
  - 调度时对适配度做 Thompson 采样乘性调整:
      表现好的模型被放大, 表现差的被压制, 同时保留探索空间
  - 双通道奖励:
      显式: POST /v1/feedback 用户/harness 打分 (0-1)
      隐式: 调用成功 0.75 / 调用失败 0.1 (弱信号, 防漂移)
  - 全部持久化在 SQLite, 重启不丢失学习成果

无数据时调整系数 = 1.0 (完全中性, 退化为静态 best_fit).
"""

from __future__ import annotations

import random
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS rewards (
    model TEXT NOT NULL,
    tag TEXT NOT NULL,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    trials INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (model, tag)
);
"""

# 调整系数钳制范围: 最多放大 1.5x, 最多压到 0.5x
ADJUST_MIN, ADJUST_MAX = 0.5, 1.5
# 隐式信号奖励值
REWARD_SUCCESS = 0.75
REWARD_FAILURE = 0.1


class Learner:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _sample_one(self, model: str, tag: str) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT alpha, beta, trials FROM rewards WHERE model=? AND tag=?",
                (model, tag)).fetchone()
        if row is None or row["trials"] == 0:
            return 1.0  # 无数据: 中性
        # Beta(1,1) 均值 0.5 -> 乘 2 使中性值为 1.0, 再钳制
        return max(ADJUST_MIN, min(ADJUST_MAX,
                                   2.0 * random.betavariate(row["alpha"],
                                                            row["beta"])))

    def adjustment(self, model: str, tags: list[str] | None) -> float:
        """模型在本次任务上的学习调整系数 (0.5-1.5)."""
        keys = tags or ["*"]
        samples = [self._sample_one(model, t) for t in keys]
        return round(sum(samples) / len(samples), 4)

    def record(self, model: str, tags: list[str] | None, reward: float) -> None:
        """记录一次结果. reward ∈ [0,1]: 1=完美, 0=彻底失败."""
        reward = max(0.0, min(1.0, reward))
        with self._lock:
            for tag in (tags or ["*"]):
                self._conn.execute(
                    """INSERT INTO rewards (model, tag, alpha, beta, trials)
                       VALUES (?,?,?,?,1)
                       ON CONFLICT(model, tag) DO UPDATE SET
                           alpha = alpha + excluded.alpha - 1.0,
                           beta = beta + excluded.beta - 1.0,
                           trials = trials + 1""",
                    (model, tag, 1.0 + reward, 1.0 + (1.0 - reward)))
            self._conn.commit()

    def summary(self) -> dict[str, dict]:
        """每个模型的学习状态: 平均胜率 / 反馈次数 (聚合所有标签)."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT model,
                          SUM(alpha - 1) AS wins,
                          SUM(beta - 1) AS losses,
                          SUM(trials) AS trials
                   FROM rewards GROUP BY model""").fetchall()
        out = {}
        for r in rows:
            total = r["wins"] + r["losses"]
            out[r["model"]] = {
                "win_rate": round(r["wins"] / total, 3) if total > 0 else 0.5,
                "trials": r["trials"],
            }
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()
