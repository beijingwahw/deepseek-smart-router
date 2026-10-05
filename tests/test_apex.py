"""Apex 功能测试: 影子灰度 / 延迟感知 / LLM-as-Judge / MoA 聚合."""

import asyncio

import pytest

from router.config import load_config
from router.judge import ajudge_response
from router.moa import aggregate_proposals
from router.proxy import RouteResult, SmartRouter
from router.shadow import ShadowRunner


def _resp(text: str) -> dict:
    return {"choices": [{"index": 0, "message": {"role": "assistant",
                                                 "content": text}}]}


# ---------- 影子灰度 ----------

def _router_with_shadow():
    cfg = load_config()
    cfg["models"]["new-model"] = {
        "provider": "deepseek", "model": "brand-new-llm",
        "tier": "standard", "enabled": True, "shadow": True,
        "price": {"input": 0.1, "output": 0.1}}
    return SmartRouter(cfg)


def test_shadow_model_excluded_from_candidates():
    r = _router_with_shadow()
    assert "new-model" not in r.candidates("standard")
    assert "new-model" not in r.candidate_chain("standard")


def test_shadow_promote_enters_candidates():
    r = _router_with_shadow()
    r.promoted.add("new-model")
    assert "new-model" in r.candidates("standard")


def test_shadow_runner_sampling():
    sr = ShadowRunner({"enabled": True, "sample_rate": 0.0,
                       "models": ["new-model"]})
    fired = sr.maybe_launch(_router_with_shadow(), {"messages": []}, {}, "q",
                            None, None, None)
    assert fired == []  # 采样率 0 绝不发起
    sr_off = ShadowRunner({"enabled": False, "sample_rate": 1.0,
                           "models": ["new-model"]})
    assert sr_off.maybe_launch(_router_with_shadow(), {"messages": []}, {},
                               "q", None, None, None) == []  # 未启用不发起


# ---------- 延迟感知适配 ----------

def test_latency_fit_penalizes_slow_model():
    cfg = load_config()
    r = SmartRouter(cfg)
    r.latency_stats = lambda: {
        "deepseek/deepseek-chat": {"avg_latency_ms": 20000, "calls": 10},
        "deepseek/deepseek-reasoner": {"avg_latency_ms": 500, "calls": 10},
    }
    r.routing["latency_weight"] = 0.2
    task = {"tags": ["code"], "est_tokens": 10,
            "needs_vision": False, "needs_tools": False}
    s_slow, d_slow = r._suitability("ds-v3", task, "standard", 3.0)
    s_fast, d_fast = r._suitability("ds-r1", task, "standard", 3.0)
    assert d_slow["latency_fit"] < 0.3   # 20s -> 严重惩罚
    assert d_fast["latency_fit"] > 0.8   # 500ms -> 几乎无感


def test_latency_disabled_when_no_stats():
    cfg = load_config()
    r = SmartRouter(cfg)  # latency_stats = None
    task = {"tags": [], "est_tokens": 10,
            "needs_vision": False, "needs_tools": False}
    _, detail = r._suitability("ds-v3", task, "standard", 3.0)
    assert "latency_fit" not in detail


# ---------- LLM-as-Judge ----------

class _JudgeRouter:
    models = {"judge-m": {"provider": "x", "model": "judge-m"}}

    async def chat(self, chain, payload, task=None):
        return RouteResult(model_name="judge-m", model="judge-m",
                           provider="fake", tier="trivial", usage={},
                           latency_ms=10, response=_resp("0.9"))


def test_ajudge_heuristic_passthrough():
    score, _ = asyncio.run(ajudge_response(
        _JudgeRouter(),
        _resp("这是一个完整详实的回答, 涵盖了问题的各个方面, "
              "包含推理过程和最终结论。"), {},
        {"mode": "heuristic"}))
    assert score >= 0.9


def test_ajudge_model_mode_parses_score():
    score, reasons = asyncio.run(ajudge_response(
        _JudgeRouter(), _resp("任何回答"), {},
        {"mode": "model", "model": "judge-m"}, "问题"))
    assert score == 0.9
    assert "LLM评审" in reasons[0]


def test_ajudge_falls_back_on_bad_judge():
    class _BadRouter:
        models = {"judge-m": {}}

        async def chat(self, chain, payload, task=None):
            raise RuntimeError("评审模型挂了")
    score, _ = asyncio.run(ajudge_response(
        _BadRouter(), _resp("完整且详实的回答内容, 有理有据, "
                            "逐步解释了问题的原因和解决方案。"), {},
        {"mode": "model", "model": "judge-m"}, "问题"))
    assert score >= 0.9  # 回退启发式评分


# ---------- MoA 聚合模式 ----------

class _AggRouter:
    async def chat(self, chain, payload, task=None):
        prompt = payload["messages"][0]["content"]
        assert "提案 1" in prompt and "提案 2" in prompt  # 聚合器收到了全部提案
        return RouteResult(model_name="claude", model="claude", provider="f",
                           tier="hard", usage={"prompt_tokens": 100,
                                               "completion_tokens": 50},
                           latency_ms=200,
                           response=_resp("综合后的最终答案"))


def test_aggregate_proposals():
    members = [{"model_name": "a", "ok": True, "judge_score": 0.9},
               {"model_name": "b", "ok": True, "judge_score": 0.7},
               {"model_name": "c", "ok": False, "error": "挂"}]
    proposals = {"a": _resp("提案一的内容, 足够长的回答文本。"),
                 "b": _resp("提案二的内容, 另一种角度的回答。"),
                 "c": _resp("不应被包含")}
    result = asyncio.run(aggregate_proposals(
        _AggRouter(), "claude", "原始问题", members, proposals, {}))
    assert result.response["choices"][0]["message"]["content"] == "综合后的最终答案"


def test_aggregate_requires_two_proposals():
    members = [{"model_name": "a", "ok": True, "judge_score": 0.9}]
    with pytest.raises(ValueError, match="提案不足"):
        asyncio.run(aggregate_proposals(
            _AggRouter(), "claude", "q", members, {"a": _resp("只有一个")}, {}))
