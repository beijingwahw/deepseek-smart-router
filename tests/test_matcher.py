"""任务画像识别 + 能力匹配 + best_fit 调度单元测试."""

from router.classifier import detect_task_profile
from router.config import load_config
from router.matcher import caps_for, suitability
from router.proxy import SmartRouter


def profile(text=None, messages=None, **extra):
    payload = {"messages": messages or [{"role": "user", "content": text or ""}],
               **extra}
    return detect_task_profile(payload)


# ---------- 任务画像识别 ----------

def test_detect_code_task():
    p = profile("用 Python 写一个函数实现快速排序")
    assert "code" in p["tags"]


def test_detect_math_task():
    p = profile("求解微分方程 dy/dx = x^2 的通解")
    assert "math" in p["tags"]


def test_detect_translation_task():
    p = profile("把这段话翻译成英文")
    assert "translation" in p["tags"]


def test_detect_writing_task():
    p = profile("帮我写一篇关于秋天的小红书文案")
    assert "writing" in p["tags"]


def test_detect_reasoning_task():
    p = profile("分析这个架构设计为什么会产生死锁, 找出根本原因")
    assert "reasoning" in p["tags"]


def test_detect_vision_need():
    p = profile(messages=[{"role": "user", "content": [
        {"type": "text", "text": "描述这张图"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]}])
    assert p["needs_vision"] is True


def test_detect_tools_need():
    p = profile("查一下天气", tools=[{"type": "function", "function": {
        "name": "get_weather", "parameters": {}}}])
    assert p["needs_tools"] is True


def test_detect_long_context():
    p = profile("分析以下文档\n" + "很长的内容" * 15000)  # >16000 tokens
    assert "long_context" in p["tags"]
    assert p["est_tokens"] > 16000


# ---------- 能力画像 ----------

def test_builtin_profile_reasoner():
    caps = caps_for({"model": "deepseek-reasoner"})
    assert caps.source == "builtin"
    assert caps.strengths["reasoning"] == 10
    assert caps.supports_vision is False


def test_builtin_profile_specific_before_generic():
    # qwen2.5-coder 应命中 code 画像而非 qwen 通用画像
    caps = caps_for({"model": "qwen2.5-coder:7b"})
    assert caps.strengths["code"] == 7
    assert caps.context_window == 32768


def test_config_override_profile():
    caps = caps_for({"model": "my-new-model-x",
                     "strengths": {"code": 10},
                     "supports_vision": True,
                     "context_window": 1000000})
    assert caps.source == "config"
    assert caps.strengths["code"] == 10
    assert caps.supports_vision is True
    assert caps.context_window == 1000000


def test_unknown_model_default_profile():
    caps = caps_for({"model": "totally-unknown-9000"})
    assert caps.source == "default"
    assert caps.strengths["code"] == 5


# ---------- 硬过滤 ----------

def _task(**kw):
    base = {"tags": [], "est_tokens": 100,
            "needs_vision": False, "needs_tools": False}
    base.update(kw)
    return base


def test_vision_filter():
    caps = caps_for({"model": "deepseek-chat"})  # 不支持视觉
    score, detail = suitability(caps, "standard", {"input": 1, "output": 1},
                                _task(needs_vision=True), 10, "standard")
    assert score is None and "视觉" in detail["filtered"]


def test_context_window_filter():
    caps = caps_for({"model": "qwen2.5-coder:7b"})  # 32k 窗口
    score, detail = suitability(caps, "trivial", {"input": 0, "output": 0},
                                _task(est_tokens=50000), 10, "trivial")
    assert score is None and "超出窗口" in detail["filtered"]


def test_tools_filter():
    caps = caps_for({"model": "x", "supports_tools": False})
    score, detail = suitability(caps, "standard", {"input": 1, "output": 1},
                                _task(needs_tools=True), 10, "standard")
    assert score is None and "工具" in detail["filtered"]


# ---------- 适配度排序 ----------

def test_math_task_prefers_reasoner():
    task = _task(tags=["math", "reasoning"])
    r1, _ = suitability(caps_for({"model": "deepseek-reasoner"}), "hard",
                        {"input": 0.55, "output": 2.19}, task, 3, "hard")
    coder, _ = suitability(caps_for({"model": "qwen2.5-coder:7b"}), "trivial",
                           {"input": 0, "output": 0}, task, 3, "hard")
    assert r1 > coder


def test_translation_task_prefers_strong_multilingual():
    task = _task(tags=["translation"])
    qwen, _ = suitability(caps_for({"model": "qwen-plus"}), "standard",
                          {"input": 0.5, "output": 0.5}, task, 3, "standard")
    r1, _ = suitability(caps_for({"model": "deepseek-reasoner"}), "hard",
                        {"input": 0.55, "output": 2.19}, task, 3, "standard")
    assert qwen > r1  # 翻译强项 9 vs 5


def test_cheaper_wins_when_capability_equal():
    task = _task(tags=["code"])
    cheap, _ = suitability(caps_for({"model": "unknown-a", "strengths": {"code": 8}}),
                           "standard", {"input": 0.1, "output": 0.1}, task, 3,
                           "standard")
    pricey, _ = suitability(caps_for({"model": "unknown-b", "strengths": {"code": 8}}),
                            "standard", {"input": 2, "output": 2}, task, 3,
                            "standard")
    assert cheap > pricey


# ---------- best_fit 调度 ----------

def _router_with(extra):
    cfg = load_config()
    cfg["models"].update(extra)
    return SmartRouter(cfg)


def test_best_fit_picks_strongest_in_tier():
    r = _router_with({
        "claude-writer": {
            "provider": "deepseek", "model": "claude-sonnet-4-5",
            "tier": "standard", "priority": 9, "enabled": True,
            "price": {"input": 3.0, "output": 15.0}},
    })
    task = _task(tags=["writing"])
    # 写作任务: claude (writing 10) 应排在 ds-v3 (writing 8) 之前,
    # 尽管 claude 更贵且 priority 更低
    assert r.candidates("standard", task)[0] == "claude-writer"


def test_best_fit_filtered_model_goes_last():
    r = _router_with({
        "text-only": {
            "provider": "deepseek", "model": "deepseek-chat",
            "tier": "trivial", "priority": 1, "enabled": True,
            "price": {"input": 0.1, "output": 0.1}},
    })
    task = _task(needs_vision=True)
    names = r.candidates("trivial", task)
    # 两个模型都不支持视觉 -> 都被过滤, 顺序兜底但不抛异常
    assert set(names) == {"local-qwen", "text-only"}


def test_suitability_report_contains_filtered_reason():
    r = _router_with({})
    task = _task(needs_vision=True, tags=["code"])
    report = r.suitability_report("standard", task)
    by_name = {x["name"]: x for x in report}
    assert by_name["ds-v3"]["score"] is None
    assert "视觉" in by_name["ds-v3"]["filtered"]


def test_suitability_report_sorted():
    r = _router_with({})
    report = r.suitability_report("hard", _task(tags=["math", "reasoning"]))
    scores = [x["score"] for x in report if x["score"] is not None]
    assert scores == sorted(scores, reverse=True)
