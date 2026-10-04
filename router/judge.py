"""级联质量评审: 便宜模型的答案够不够好? 不好就升级 (FrugalGPT 路线).

无额外模型调用的启发式评审 (零成本), 信号:
  - 拒答/不确定语 (中英文)
  - 内容过短或为空
  - 包含错误/堆栈
  - 高重复度 (模型崩坏特征)
  - 任务契合: 要代码没代码, 要长文太短

判分 0-1, 低于阈值即升级到更高档. 升级事件还会反哺学习引擎:
被升级 = 差反馈, 一次通过 = 好反馈, 级联越用越少.
"""

from __future__ import annotations

import re

UNCERTAINTY_RE = re.compile(
    r"(我不确定|我无法|我不能回答|我不知道|无法确定|作为.{0,4}AI|我没有办法|"
    r"抱歉.{0,10}(无法|不能)|I'?m not sure|I cannot|I can'?t answer|"
    r"as an AI|I apologize|I don'?t know)", re.IGNORECASE)
ERROR_RE = re.compile(
    r"(Traceback|Error:|Exception|出错了|调用失败|服务不可用)", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")


def _repetition_ratio(text: str) -> float:
    """4-gram 重复率: >0.5 通常是模型崩坏循环."""
    if len(text) < 40:
        return 0.0
    grams = [text[i:i + 4] for i in range(len(text) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def judge_response(body: dict, task: dict | None = None) -> tuple[float, list[str]]:
    """评审一次 OpenAI 格式响应. 返回 (得分 0-1, 理由列表)."""
    task = task or {}
    try:
        msg = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return 0.0, ["响应结构异常"]
    text = msg.get("content") or ""
    if msg.get("tool_calls"):  # 工具调用响应视为完整回答
        return 1.0, ["包含工具调用"]

    score = 1.0
    reasons: list[str] = []

    if not text.strip():
        return 0.0, ["回答为空"]
    if len(text) < 30:
        score -= 0.45
        reasons.append(f"回答过短({len(text)}字符)")
    if UNCERTAINTY_RE.search(text):
        score -= 0.5
        reasons.append("包含拒答/不确定语")
    if ERROR_RE.search(text):
        score -= 0.4
        reasons.append("包含错误信息")
    rep = _repetition_ratio(text)
    if rep > 0.5:
        score -= 0.5
        reasons.append(f"高重复度({rep:.0%})")

    tags = task.get("tags") or []
    if "code" in tags and not CODE_FENCE_RE.search(text):
        score -= 0.2
        reasons.append("代码任务但无代码块")
    if "writing" in tags and len(text) < 300:
        score -= 0.25
        reasons.append("写作任务但篇幅过短")

    return max(0.0, round(score, 2)), reasons


LOWER_TIER = {"hard": "standard", "standard": "trivial"}


def cascade_start_tier(tier: str) -> str | None:
    """级联起点: 降一档先答; trivial 已是最低档, 不级联."""
    return LOWER_TIER.get(tier)


def should_escalate(score: float, threshold: float) -> bool:
    return score < threshold
