"""Genesis 功能测试: 学习引擎 / 预算守卫 / 运行时管理."""

import random

from router.budget import apply_budget
from router.config import load_config
from router.learner import Learner
from router.proxy import SmartRouter
from router.stats import Stats


# ---------- 学习引擎 ----------

def _learner(tmp_path):
    return Learner(str(tmp_path / "learn.db"))


def test_learner_neutral_without_data(tmp_path):
    ln = _learner(tmp_path)
    assert ln.adjustment("ds-r1", ["math"]) == 1.0
    assert ln.adjustment("unknown-model", []) == 1.0
    ln.close()


def test_learner_rewards_good_model(tmp_path):
    random.seed(42)
    ln = _learner(tmp_path)
    for _ in range(20):
        ln.record("ds-r1", ["math"], 1.0)   # 一直表现好
        ln.record("ds-v3", ["math"], 0.0)   # 一直表现差
    # 采样多次取均值, 好模型应显著高于差模型
    good = sum(ln.adjustment("ds-r1", ["math"]) for _ in range(50)) / 50
    bad = sum(ln.adjustment("ds-v3", ["math"]) for _ in range(50)) / 50
    assert good > 1.2 and bad < 0.8
    s = ln.summary()
    assert s["ds-r1"]["win_rate"] > 0.9
    assert s["ds-v3"]["win_rate"] < 0.1
    assert s["ds-r1"]["trials"] == 20
    ln.close()


def test_learner_persists_across_restart(tmp_path):
    ln = _learner(tmp_path)
    ln.record("ds-r1", ["code"], 1.0)
    ln.close()
    ln2 = _learner(tmp_path)  # 模拟重启
    assert ln2.summary()["ds-r1"]["trials"] == 1
    ln2.close()


def test_learner_tags_fallback_to_star(tmp_path):
    ln = _learner(tmp_path)
    ln.record("m", [], 1.0)
    assert ln.summary()["m"]["trials"] == 1  # 无标签记入 "*" 桶
    ln.close()


def test_learner_integrated_in_suitability(tmp_path):
    random.seed(7)
    ln = _learner(tmp_path)
    for _ in range(30):
        ln.record("ds-v3", ["code"], 0.0)  # ds-v3 写代码一直被差评
    cfg = load_config()
    r = SmartRouter(cfg, learner=ln)
    task = {"tags": ["code"], "est_tokens": 10,
            "needs_vision": False, "needs_tools": False}
    score, detail = r._suitability("ds-v3", task, "standard", 3.0)
    assert detail["learned_adjustment"] < 1.0  # 被学习压制
    ln.close()


def test_learner_disabled_means_static(tmp_path):
    cfg = load_config()
    r = SmartRouter(cfg, learner=None)  # 不带学习引擎
    task = {"tags": ["code"], "est_tokens": 10,
            "needs_vision": False, "needs_tools": False}
    _, detail = r._suitability("ds-v3", task, "standard", 3.0)
    assert "learned_adjustment" not in detail


# ---------- 预算守卫 ----------

def test_budget_disabled_passthrough():
    assert apply_budget("hard", 999.0, {"enabled": False}) == ("hard", None)
    assert apply_budget("hard", 999.0, {}) == ("hard", None)


def test_budget_under_cap():
    assert apply_budget("hard", 1.0, {"enabled": True, "daily_usd": 10}) == ("hard", None)


def test_budget_downgrade_on_exceed():
    tier, note = apply_budget("hard", 11.0, {"enabled": True, "daily_usd": 10})
    assert tier == "standard" and "降档" in note


def test_budget_standard_stays_on_exceed():
    tier, note = apply_budget("standard", 11.0, {"enabled": True, "daily_usd": 10})
    assert tier == "standard" and note is not None  # 已是最低开销档, 只提示


def test_budget_block_on_exceed():
    tier, action = apply_budget("hard", 11.0,
                                {"enabled": True, "daily_usd": 10,
                                 "on_exceed": "block"})
    assert action == "block"


# ---------- 运行时禁用 ----------

def test_runtime_disable_excludes_from_candidates():
    cfg = load_config()
    r = SmartRouter(cfg)
    r.disabled.add("ds-r1")
    assert "ds-r1" not in r.candidates("hard")
    # 跨档降级链里也不应出现
    assert "ds-r1" not in r.candidate_chain("standard")
    r.disabled.discard("ds-r1")
    assert "ds-r1" in r.candidates("hard")


# ---------- 统计扩展 ----------

def test_stats_request_id_and_daily_spend(tmp_path):
    st = Stats(str(tmp_path / "s.db"))
    rid = st.record(tier="hard", model="deepseek/deepseek-reasoner", score=80,
                    reasons="t", prompt_tokens=100, completion_tokens=50,
                    cost=0.5, baseline_hard=0.5, baseline_standard=0.2,
                    latency_ms=1200, status="ok")
    assert rid >= 1
    rec = st.get_request(rid)
    assert rec["model"] == "deepseek/deepseek-reasoner"
    assert abs(st.daily_spend() - 0.5) < 1e-9
    lat = st.latency_by_model()
    assert lat["deepseek/deepseek-reasoner"]["avg_latency_ms"] == 1200
    st.close()
