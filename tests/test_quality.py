"""质量提升回归测试: 修复过的真 bug 不得复现.

注意: 本文件导入 router.main (有全局副作用), 按字母序最后执行.
"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from router.classifier import estimate_tokens
from router.main import _reload_if_changed, app
from router.stats import Stats


# ---------- 热重载 bug: force 必须真的重载 ----------

def test_admin_reload_force_actually_reloads(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("thresholds: {trivial: 90, hard: 95}\n", encoding="utf-8")
    import router.main as m
    assert _reload_if_changed(force=True, path=str(cfg_file)) is True
    assert m.TH == {"trivial": 90, "hard": 95}  # 修复前: force 重载是空操作


def test_reload_skipped_when_mtime_unchanged(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("thresholds: {trivial: 88, hard: 96}\n", encoding="utf-8")
    import router.main as m
    m._config_mtime = None
    assert _reload_if_changed(path=str(cfg_file)) is False  # 首次只记录 mtime
    assert _reload_if_changed(path=str(cfg_file)) is False  # mtime 未变不重载


# ---------- feedback 校验 bug: 非数字 score 必须 422 而非 500 ----------

def test_feedback_rejects_non_numeric_score():
    client = TestClient(app)
    resp = client.post("/v1/feedback", json={"model": "ds-v3", "score": "abc"})
    assert resp.status_code == 422  # 修复前: float("abc") 抛出 500


def test_feedback_rejects_out_of_range():
    client = TestClient(app)
    resp = client.post("/v1/feedback", json={"model": "ds-v3", "score": 1.5})
    assert resp.status_code == 422


def test_feedback_accepts_valid():
    client = TestClient(app)
    resp = client.post("/v1/feedback", json={"model": "ds-v3", "score": 0.8})
    assert resp.status_code == 200 and resp.json()["ok"] is True


def test_feedback_unknown_model_422():
    client = TestClient(app)
    resp = client.post("/v1/feedback", json={"model": "ghost", "score": 0.5})
    assert resp.status_code == 422


# ---------- CJK token 估算精度 ----------

def test_estimate_tokens_cjk_not_underestimated():
    # 修复前按 4 字符/token: "你好世界" 只有 1 token, 严重低估
    assert estimate_tokens("你好世界") >= 4


def test_estimate_tokens_ascii():
    assert estimate_tokens("hello world test") == 4


def test_estimate_tokens_mixed():
    t = estimate_tokens("用 Python 写一个函数")
    assert 5 <= t <= 12  # 6 个汉字 + 少量 ASCII


# ---------- SQLite WAL 并发模式 ----------

def test_sqlite_wal_enabled(tmp_path):
    st = Stats(str(tmp_path / "w.db"))
    mode = st._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    st.close()


# ---------- requirements 声明完整性 ----------

def test_requirements_declares_numpy():
    from pathlib import Path
    reqs = (Path(__file__).parent.parent / "requirements.txt").read_text()
    assert "numpy" in reqs
