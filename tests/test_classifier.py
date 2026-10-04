"""分类器与路由决策单元测试."""

from router.classifier import classify, route

TH_TRIVIAL, TH_HARD = 25, 60


def decide(text: str, **extra) -> tuple[str, int, list[str]]:
    payload = {"messages": [{"role": "user", "content": text}], **extra}
    f = classify(payload)
    return route(f.score, TH_TRIVIAL, TH_HARD), f.score, f.reasons


# ---------- 强制标签 ----------

def test_force_hard_tag():
    tier, score, _ = decide("[think] 随便聊聊")
    assert tier == "hard" and score == 100


def test_force_trivial_tag():
    tier, score, _ = decide("[quick] 证明一下费马大定理")
    assert tier == "trivial" and score == 0


# ---------- 简单任务 ----------

def test_translate_is_trivial():
    tier, _, _ = decide("帮我把这句话翻译成英文: 今天天气不错")
    assert tier == "trivial"


def test_rename_is_trivial():
    tier, _, _ = decide("rename this variable to something clearer: tmp")
    assert tier == "trivial"


def test_simple_factual_question():
    tier, _, _ = decide("什么是闭包?")
    assert tier == "trivial"


# ---------- 常规任务 ----------

def test_normal_coding_is_standard():
    tier, _, _ = decide(
        "用 Python 写一个函数, 读取 CSV 并按第二列排序, 输出前 10 行"
    )
    assert tier == "standard"


# ---------- 困难任务 ----------

def test_architecture_design_is_hard():
    tier, score, reasons = decide(
        "设计一个支持百万并发的消息推送系统架构, 分析各种方案的权衡, "
        "重点考虑死锁与性能瓶颈, 给出复杂度分析"
    )
    assert tier == "hard"
    assert score >= 65
    assert any("推理关键词" in r for r in reasons)


def test_stack_trace_debug_is_hard():
    tier, _, _ = decide(
        "帮我 debug: 为什么会产生这个错误? 找出根本原因\n"
        "Traceback (most recent call last):\n"
        '  File "app.py", line 42, in run\n    conn.execute(sql)\n'
        "OperationalError: database is locked"
    )
    assert tier == "hard"


def test_deep_conversation_boosts_score():
    msgs = [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"第{i}轮讨论: 为什么微服务拆分后性能反而下降, 根本原因"}
            for i in range(35)]
    from router.classifier import classify as _c
    f = _c({"messages": msgs})
    assert f.score >= 30
    assert any("深层对话" in r for r in f.reasons)


def test_tool_chain_boosts_score():
    msgs = [
        {"role": "user", "content":
            "debug: 分析这个仓库为什么测试会偶发失败, 找出根本原因, 是否存在并发问题"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "pytest output ... 3 failed"},
    ]
    payload = {"messages": msgs, "tools": [{"type": "function",
               "function": {"name": "run_tests", "parameters": {}}}]}
    from router.classifier import classify as _c
    f = _c(payload)
    assert f.score >= 60  # 推理词 + 工具链 + 工具定义


# ---------- 路由边界 ----------

def test_route_boundaries():
    assert route(0, 30, 65) == "trivial"
    assert route(29, 30, 65) == "trivial"
    assert route(30, 30, 65) == "standard"
    assert route(64, 30, 65) == "standard"
    assert route(65, 30, 65) == "hard"
    assert route(100, 30, 65) == "hard"


# ---------- 多模态消息不炸 ----------

def test_multimodal_content_safe():
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "翻译这张图里的文字"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]}]}
    from router.classifier import classify as _c
    f = _c(payload)
    assert 0 <= f.score <= 100
