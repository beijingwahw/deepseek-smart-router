"""熔断器 / 候选链 / 成本计算单元测试 (不依赖真实上游)."""

import time

from router.config import load_config
from router.proxy import CircuitBreaker, SmartRouter
from router.stats import calc_cost


# ---------- 熔断器 ----------

def test_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
    assert cb.available()
    for _ in range(3):
        cb.on_failure()
    assert not cb.available()  # 已熔断


def test_breaker_half_open_after_cooldown():
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=1)
    cb.on_failure()
    assert not cb.available()
    cb._opened_at = time.time() - 2  # 模拟冷却期已过
    assert cb.available()


def test_breaker_recovers_on_success():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)
    cb.on_failure()
    cb.on_success()
    assert cb._failures == 0 and cb.available()


# ---------- 候选链 ----------

def _router(extra_models: dict | None = None, strategy: str = "priority"):
    cfg = load_config()
    cfg["routing"]["strategy"] = strategy
    if extra_models:
        cfg["models"].update(extra_models)
    return SmartRouter(cfg)


def test_candidates_by_tier():
    r = _router()
    assert r.candidates("trivial") == ["local-qwen"]
    assert r.candidates("standard") == ["ds-v3"]
    assert r.candidates("hard") == ["ds-r1"]


def test_candidate_chain_cross_tier():
    r = _router()
    # hard 档挂了降级到 standard
    assert r.candidate_chain("hard") == ["ds-r1", "ds-v3"]
    # trivial 档挂了降级到 standard
    assert r.candidate_chain("trivial") == ["local-qwen", "ds-v3"]


def test_pinned_model_chain():
    r = _router()
    assert r.candidate_chain("hard", pinned="ds-v3") == ["ds-v3"]


def test_priority_order_within_tier():
    r = _router(extra_models={
        "gpt-4o": {"provider": "deepseek", "model": "gpt-4o", "tier": "standard",
                   "priority": 1, "enabled": True,
                   "price": {"input": 5.0, "output": 15.0}},
        "ds-v3b": {"provider": "deepseek", "model": "deepseek-chat-2",
                   "tier": "standard", "priority": 3, "enabled": True,
                   "price": {"input": 0.3, "output": 1.2}},
    })
    names = r.candidates("standard")
    assert names[0] in ("ds-v3", "gpt-4o")  # priority=1 的在前
    assert names[-1] == "ds-v3b"            # priority=3 殿后


def test_cheapest_strategy():
    r = _router(strategy="cheapest", extra_models={
        "gpt-4o": {"provider": "deepseek", "model": "gpt-4o", "tier": "standard",
                   "priority": 1, "enabled": True,
                   "price": {"input": 5.0, "output": 15.0}},
    })
    assert r.candidates("standard")[0] == "ds-v3"  # 最便宜优先


def test_round_robin_strategy():
    r = _router(strategy="round_robin", extra_models={
        "gpt-4o": {"provider": "deepseek", "model": "gpt-4o", "tier": "standard",
                   "priority": 1, "enabled": True,
                   "price": {"input": 5.0, "output": 15.0}},
    })
    first = r.candidates("standard")
    second = r.candidates("standard")
    assert first[0] != second[0]  # 轮询换序


def test_disabled_model_excluded():
    r = _router(extra_models={
        "dead-model": {"provider": "deepseek", "model": "x", "tier": "hard",
                       "enabled": False, "price": {"input": 0, "output": 0}},
    })
    assert "dead-model" not in r.candidate_chain("hard")


# ---------- 成本计算 ----------

def test_calc_cost_basic():
    price = {"input": 0.27, "output": 1.10, "cache_hit_input": 0.07}
    cost = calc_cost(price, prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert abs(cost - 1.37) < 1e-9


def test_calc_cost_with_cache_hit():
    price = {"input": 0.27, "output": 1.10, "cache_hit_input": 0.07}
    cost = calc_cost(price, prompt_tokens=1_000_000,
                     completion_tokens=0, cache_hit_tokens=1_000_000)
    assert abs(cost - 0.07) < 1e-9


def test_local_model_is_free():
    assert calc_cost({"input": 0.0, "output": 0.0}, 10**9, 10**9) == 0.0
