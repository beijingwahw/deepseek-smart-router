"""任务难度分类器.

多信号启发式评分 (0-100), 无需调用任何模型, 零延迟零成本:
  - 显式标签: [think]/[hard] 强制走 R1, [quick]/[easy] 强制走本地
  - 推理关键词 (中英双语)
  - 代码/堆栈复杂度
  - 多步骤结构
  - 上下文长度与对话深度
  - 工具调用链
  - 简单任务负向信号 (翻译/润色/格式化等)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------- 信号词表 ----------------

REASONING_KEYWORDS = [
    # 中文
    "证明", "推导", "论证", "架构设计", "设计一个", "重构", "为什么",
    "根本原因", "一步步分析", "逐步分析", "深入分析", "权衡", "取舍",
    "最优解", "复杂度分析", "并发", "死锁", "内存泄漏", "性能瓶颈",
    # English
    "prove", "derive", "step by step", "reason about", "root cause",
    "architect", "refactor", "trade-off", "tradeoff", "optimize",
    "deadlock", "race condition", "memory leak", "design a",
    "why does", "explain why", "debug",
]

TRIVIAL_KEYWORDS = [
    # 中文
    "翻译", "润色", "改写这句话", "格式化", "起个名", "命名", "错别字",
    "缩写", "扩写这句话", "换个说法", "总结一下这句话",
    # English
    "translate", "rephrase", "format this", "rename", "typo",
    "fix the grammar", "paraphrase", "shorten", "prettify",
]

FORCE_HARD_TAGS = ["[think]", "[hard]", "[r1]", "[deep]"]
FORCE_TRIVIAL_TAGS = ["[quick]", "[easy]", "[fast]", "[local]"]

STACK_TRACE_RE = re.compile(
    r"(Traceback \(most recent call last\)|at [\w$.]+\([\w.]+:\d+\)|"
    r"Exception in thread|Segmentation fault|panic:|FATAL|Error: .*\n\s+at )"
)
CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
MULTI_STEP_RE = re.compile(
    r"((首先|第一步|步骤\s*1).{0,400}(然后|其次|第二步|步骤\s*2))|"
    r"((^|\n)\s*(1[.、)]|step\s*1).{0,400}(2[.、)]|step\s*2))",
    re.IGNORECASE | re.DOTALL,
)
SIMPLE_QUESTION_RE = re.compile(
    r"^(什么是|什么是|谁发明的|what is|who is|define )", re.IGNORECASE
)

# 强设计/形式化信号: 命中任意一个, 在推理词基础上额外加权
STRONG_DESIGN_KEYWORDS = [
    "架构设计", "设计一个", "证明", "推导", "architect", "prove", "derive",
    "design a",
]

# 明确的编码请求信号 (写函数/实现接口等), 不属于简单任务
CODE_REQUEST_RE = re.compile(
    r"(写|实现|编写|开发|补全|fix|implement|write|create|develop|code)"
    r".{0,25}(函数|代码|脚本|程序|接口|类|组件|测试|"
    r"function|code|script|class|api|component|test|bug)",
    re.IGNORECASE,
)


@dataclass
class Features:
    """从一次 chat 请求中提取的特征."""

    score: int = 0
    reasons: list[str] = field(default_factory=list)
    forced: str | None = None  # "hard" / "trivial" / None

    def add(self, points: int, reason: str) -> None:
        self.score += points
        self.reasons.append(f"{'+' if points >= 0 else ''}{points} {reason}")


def _last_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):  # 多模态: 拼接文本块
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return content or ""
    return ""


def _all_text(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if content:
            parts.append(content)
    return "\n".join(parts)


def classify(payload: dict) -> Features:
    """对一次 OpenAI 格式 chat/completions 请求打分.

    payload: 请求体 (messages / tools / tool_choice 等)
    返回 Features(score 0-100, reasons, forced)
    """
    messages = payload.get("messages", []) or []
    last = _last_user_text(messages)
    full = _all_text(messages)
    f = Features()
    low = last.lower()

    # 0. 显式标签 —— 最高优先级
    for tag in FORCE_HARD_TAGS:
        if tag in low:
            f.score = 100
            f.forced = "hard"
            f.reasons.append(f"显式标签 {tag} -> 强制困难档")
            return f
    for tag in FORCE_TRIVIAL_TAGS:
        if tag in low:
            f.score = 0
            f.forced = "trivial"
            f.reasons.append(f"显式标签 {tag} -> 强制简单档")
            return f

    # 1. 推理关键词: 命中越多越可能是深度推理任务
    lower_full = full.lower()
    hits = sum(1 for kw in REASONING_KEYWORDS if kw in lower_full)
    if hits:
        points = {1: 15, 2: 25, 3: 38}.get(hits, 50)
        f.add(points, f"推理关键词 x{hits}")
        if any(kw in lower_full for kw in STRONG_DESIGN_KEYWORDS):
            f.add(15, "强设计/形式化信号")

    # 2. 代码与错误信号
    if STACK_TRACE_RE.search(full):
        f.add(25, "包含堆栈/崩溃信息")
    elif CODE_REQUEST_RE.search(full):
        f.add(30, "明确编码请求")
    code_blocks = CODE_BLOCK_RE.findall(full)
    if code_blocks:
        longest = max(len(b) for b in code_blocks)
        if longest > 800:
            f.add(15, "大段代码块")
        else:
            f.add(8, "包含代码块")

    # 3. 多步骤结构
    if MULTI_STEP_RE.search(full):
        f.add(10, "多步骤任务结构")

    # 4. 上下文规模 (越长越可能复杂)
    n_chars = len(full)
    if n_chars > 8000:
        f.add(20, f"超长上下文({n_chars}字符)")
    elif n_chars > 2000:
        f.add(10, f"较长上下文({n_chars}字符)")

    # 5. 对话深度 (harness 多轮 agent 循环)
    n_msgs = len(messages)
    if n_msgs > 30:
        f.add(15, f"深层对话({n_msgs}轮)")
    elif n_msgs > 10:
        f.add(8, f"多轮对话({n_msgs}轮)")

    # 6. 工具调用 (agent 执行中)
    if payload.get("tools") or payload.get("tool_choice"):
        f.add(10, "携带工具定义")
    if any(m.get("role") == "tool" for m in messages):
        f.add(10, "工具调用链进行中")

    # 7. 简单任务负向信号
    triv_hits = sum(1 for kw in TRIVIAL_KEYWORDS if kw in full.lower())
    if triv_hits:
        f.add(-min(30, 15 + 8 * triv_hits), f"简单任务关键词 x{triv_hits}")
    if SIMPLE_QUESTION_RE.search(last.strip()) and n_chars < 500:
        f.add(-15, "简短事实型提问")

    f.score = max(0, min(100, f.score))
    return f


def route(score: int, threshold_trivial: int, threshold_hard: int) -> str:
    """分数 -> 档位: trivial / standard / hard."""
    if score < threshold_trivial:
        return "trivial"
    if score >= threshold_hard:
        return "hard"
    return "standard"
