"""Frontier 功能测试: 语义缓存 / 级联评审."""

import time

from router.cache import SemanticCache
from router.embedder import HashEmbedder, cosine
from router.judge import (cascade_start_tier, judge_response,
                          should_escalate)


def _resp(text: str, model: str = "m") -> dict:
    return {"choices": [{"index": 0, "message": {"role": "assistant",
                                                 "content": text},
                         "finish_reason": "stop"}],
            "model": model,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


# ---------- 哈希向量 ----------

def test_hash_embedder_similar_texts():
    e = HashEmbedder()
    a = e.embed("如何重置我的密码?")
    b = e.embed("如何重置我的密码啊?")
    sim = cosine(a, b)
    assert sim > 0.7


def test_hash_embedder_dissimilar_texts():
    e = HashEmbedder()
    a = e.embed("如何重置我的密码?")
    b = e.embed("证明黎曼猜想在临界带内的零点分布")
    assert cosine(a, b) < 0.5


def test_hash_embedder_empty_safe():
    e = HashEmbedder()
    assert cosine(e.embed(""), e.embed("")) == 0.0


# ---------- 语义缓存 ----------

def _cache(tmp_path, **kw):
    return SemanticCache(str(tmp_path / "c.db"), HashEmbedder(),
                         **{"threshold": 0.75, "ttl_seconds": 3600,
                            "max_entries": 100, **kw})


def test_cache_exact_hit(tmp_path):
    c = _cache(tmp_path)
    c.store("如何重置密码?", "trivial", _resp("点设置-账号-重置"), "m1")
    hit = c.lookup("如何重置密码?", "trivial")
    assert hit is not None
    assert hit["choices"][0]["message"]["content"] == "点设置-账号-重置"
    assert hit["router"]["cache"]["hit"] is True
    assert c.hits == 1
    c.close()


def test_cache_semantic_hit(tmp_path):
    c = _cache(tmp_path)
    c.store("如何重置我的密码?", "trivial", _resp("步骤如下: 第一...第二..."), "m1")
    hit = c.lookup("如何重置我的密码啊?", "trivial")
    assert hit is not None  # 语义近似也命中
    c.close()


def test_cache_miss_on_dissimilar(tmp_path):
    c = _cache(tmp_path)
    c.store("如何重置密码?", "trivial", _resp("答案"), "m1")
    assert c.lookup("证明根号二是无理数", "trivial") is None
    assert c.misses == 1
    c.close()


def test_cache_tier_isolation(tmp_path):
    c = _cache(tmp_path)
    c.store("如何重置密码?", "trivial", _resp("答案"), "m1")
    assert c.lookup("如何重置密码?", "hard") is None  # 跨档不命中
    c.close()


def test_cache_ttl_expiry(tmp_path):
    c = _cache(tmp_path, ttl_seconds=1)
    c.store("问题甲", "trivial", _resp("答案甲"), "m1")
    with c._lock:
        c._conn.execute("UPDATE semantic_cache SET ts=?", (time.time() - 10,))
        c._conn.commit()
    assert c.lookup("问题甲", "trivial") is None
    c.close()


def test_cache_max_entries_eviction(tmp_path):
    c = _cache(tmp_path, max_entries=5)
    for i in range(10):
        c.store(f"完全不同的问题编号{i}号", "trivial", _resp(f"答{i}"), "m")
    assert c.stats()["entries"] <= 5
    c.close()


def test_cache_stats_hit_rate(tmp_path):
    c = _cache(tmp_path)
    c.store("问题乙", "trivial", _resp("答案乙"), "m")
    c.lookup("问题乙", "trivial")   # hit
    c.lookup("完全无关的另一件事", "trivial")  # miss
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1 and s["hit_rate"] == 0.5
    c.close()


# ---------- 级联评审 ----------

def test_judge_good_answer():
    score, _ = judge_response(_resp(
        "这是一个完整的回答, 包含了足够的细节和推理过程, "
        "逐步解释了问题的原因和解决方案。"))
    assert score >= 0.9


def test_judge_refusal_low_score():
    score, reasons = judge_response(_resp("抱歉, 我无法回答这个问题。"))
    assert score < 0.55
    assert any("拒答" in r or "过短" in r for r in reasons)


def test_judge_empty_zero():
    score, _ = judge_response(_resp(""))
    assert score == 0.0


def test_judge_repetition_penalty():
    score, reasons = judge_response(_resp("啊啊啊啊" * 100))
    assert score < 0.6
    assert any("重复" in r for r in reasons)


def test_judge_code_task_without_code():
    score, reasons = judge_response(
        _resp("你应该先写函数然后再测试, 整体思路就是这样, 不需要别的。" * 2),
        task={"tags": ["code"]})
    assert any("无代码块" in r for r in reasons)


def test_judge_tool_calls_pass():
    body = {"choices": [{"index": 0, "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "1", "type": "function",
                        "function": {"name": "f", "arguments": "{}"}}]}}]}
    score, _ = judge_response(body)
    assert score == 1.0


def test_cascade_tier_map():
    assert cascade_start_tier("hard") == "standard"
    assert cascade_start_tier("standard") == "trivial"
    assert cascade_start_tier("trivial") is None


def test_should_escalate():
    assert should_escalate(0.3, 0.55) is True
    assert should_escalate(0.8, 0.55) is False
