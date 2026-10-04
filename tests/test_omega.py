"""Omega 功能测试: kNN 经验回忆 / MoA 竞技场 / λ 拨盘."""

import asyncio

import pytest

from router.config import load_config
from router.embedder import HashEmbedder
from router.memory import ExperienceMemory
from router.moa import run_moa
from router.proxy import RouteResult, SmartRouter


# ---------- 经验回忆 ----------

def _memory(tmp_path, **kw):
    return ExperienceMemory(str(tmp_path / "m.db"), HashEmbedder(),
                            **{"k": 5, "min_sim": 0.7, "max_entries": 100,
                               **kw})


def test_memory_recall_biases_toward_winner(tmp_path):
    m = _memory(tmp_path)
    for _ in range(5):
        m.record("怎么重置路由器管理员密码?", "ds-v3", 1.0)   # ds-v3 答得好
        m.record("怎么重置路由器管理员密码?", "ds-r1", 0.1)   # ds-r1 答得差
    recalled = m.recall("怎么重置路由器管理员密码啊?")  # 换个问法也能检索到
    assert recalled["ds-v3"] > 0.9
    assert recalled["ds-r1"] < 0.2
    assert m.bonus("ds-v3", recalled) > 0.5
    assert m.bonus("ds-r1", recalled) < -0.5
    assert m.bonus("unknown", recalled) == 0.0
    m.close()


def test_memory_ignores_dissimilar(tmp_path):
    m = _memory(tmp_path)
    m.record("怎么重置密码?", "ds-v3", 1.0)
    assert m.recall("证明哥德巴赫猜想") == {}
    m.close()


def test_memory_empty_safe(tmp_path):
    m = _memory(tmp_path)
    assert m.recall("任何问题") == {}
    m.record("", "x", 1.0)  # 空问题不入库
    assert m.size() == 0
    m.close()


def test_memory_integrated_in_candidates(tmp_path):
    m = _memory(tmp_path)
    for _ in range(6):
        m.record("写一篇产品发布新闻稿", "claude-x", 1.0)
    cfg = load_config()
    cfg["models"]["claude-x"] = {
        "provider": "deepseek", "model": "claude-sonnet-4-5",
        "tier": "standard", "priority": 9, "enabled": True,
        "price": {"input": 3.0, "output": 15.0}}
    r = SmartRouter(cfg, memory=m)
    task = {"tags": ["writing"], "est_tokens": 20,
            "needs_vision": False, "needs_tools": False}
    report = {x["name"]: x for x in
              r.suitability_report("standard", task, "写一篇产品发布新闻稿")}
    assert report["claude-x"]["recall_bonus"] > 0
    m.close()


# ---------- MoA 竞技场 ----------

def _resp(text):
    return {"choices": [{"index": 0, "message": {"role": "assistant",
                                                 "content": text}}]}


class _FakeRouter:
    """模拟上游: good 给完整答案, bad 给拒答, down 抛异常."""

    async def chat(self, chain, payload, task=None):
        name = chain[0]
        if name == "down":
            raise RuntimeError("上游超时")
        text = ("这是一个内容详实的完整回答, 包含推理过程和最终结论, "
                "覆盖了问题的各个方面。" if name == "good"
                else "抱歉, 我无法回答。")
        return RouteResult(model_name=name, model=name, provider="fake",
                           tier="hard", usage={"prompt_tokens": 10,
                                               "completion_tokens": 20},
                           latency_ms=100, response=_resp(text))


def test_moa_picks_best_answer():
    outcome = asyncio.run(run_moa(_FakeRouter(), ["bad", "good", "down"],
                                  {"messages": []}, {}, fanout=3))
    assert outcome.winner.model_name == "good"
    assert outcome.winner_judge_score >= 0.9
    assert len(outcome.members) == 3
    bad = next(b for b in outcome.members if b["model_name"] == "bad")
    down = next(b for b in outcome.members if b["model_name"] == "down")
    assert bad["judge_score"] < 0.55 and bad["winner"] is False
    assert down["ok"] is False  # 异常被捕获, 不拖垮整场竞赛


def test_moa_all_failed_raises():
    class _AllDown:
        async def chat(self, chain, payload, task=None):
            raise RuntimeError("全挂")
    with pytest.raises(RuntimeError, match="全部参赛者失败"):
        asyncio.run(run_moa(_AllDown(), ["a", "b"], {"messages": []}, {},
                            fanout=2))


def test_moa_degraded_single_valid():
    class _OneGood:
        async def chat(self, chain, payload, task=None):
            if chain[0] == "a":
                raise RuntimeError("挂")
            return RouteResult(model_name="b", model="b", provider="f",
                               tier="hard", usage={}, latency_ms=1,
                               response=_resp("认真且完整的回答内容, 足够详细。"))
    outcome = asyncio.run(run_moa(_OneGood(), ["a", "b"], {"messages": []}, {},
                                  fanout=2))
    assert outcome.degraded is True
    assert outcome.winner.model_name == "b"


# ---------- λ 质量-成本拨盘 ----------

def _lambda_router(lam):
    cfg = load_config()
    cfg["routing"]["quality_lambda"] = lam
    cfg["models"]["premium"] = {
        "provider": "deepseek", "model": "claude-sonnet-4-5",
        "tier": "standard", "enabled": True,
        "price": {"input": 3.0, "output": 15.0}}
    return SmartRouter(cfg)


def test_lambda_quality_first_prefers_premium():
    r = _lambda_router(1.0)  # 极限质量
    task = {"tags": ["writing"], "est_tokens": 10,
            "needs_vision": False, "needs_tools": False}
    names = r.candidates("standard", task)
    assert names[0] == "premium"  # claude 写作最强, 质量优先时居首


def test_lambda_cost_first_prefers_cheap():
    r = _lambda_router(0.0)  # 极限省钱
    task = {"tags": ["writing"], "est_tokens": 10,
            "needs_vision": False, "needs_tools": False}
    names = r.candidates("standard", task)
    assert names[-1] == "premium"  # 最贵的殿后
