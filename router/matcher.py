"""能力匹配器: 任务画像 x 模型能力画像 -> 适配度评分.

三层决策:
  1. 硬过滤: 视觉/工具需求不满足、上下文超出窗口 -> 直接出局
  2. 能力匹配: 任务标签 (code/math/reasoning/translation/writing/long_context)
     对照模型强项打分
  3. 成本微调: 适配度相近时便宜的优先

模型能力画像来源: 内置常见模型画像表 (按模型 ID 子串匹配),
可被 yaml 中模型的 strengths / context_window / supports_tools /
supports_vision 字段覆盖 —— 新模型无需改代码.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 难度档邻近关系: 相邻档可互为备份
TIER_NEIGHBORS = {
    "trivial": {"standard"},
    "standard": {"trivial", "hard"},
    "hard": {"standard"},
}

# 内置画像: (模型 ID 子串, 各项强项 0-10, 上下文窗口, 支持工具, 支持视觉)
# 顺序敏感: 更具体的 ID 放前面 (如 deepseek-reasoner 先于 deepseek-chat)
BUILTIN_PROFILES: list[tuple[str, dict, int, bool, bool]] = [
    ("deepseek-reasoner",
     {"reasoning": 10, "math": 9, "code": 8, "writing": 6, "translation": 5,
      "long_context": 6}, 65536, True, False),
    ("deepseek-chat",
     {"reasoning": 6, "math": 6, "code": 8, "writing": 8, "translation": 8,
      "long_context": 6}, 65536, True, False),
    ("o1", {"reasoning": 10, "math": 10, "code": 8, "writing": 6,
            "translation": 5, "long_context": 7}, 200000, True, True),
    ("gpt-4o-mini",
     {"reasoning": 6, "math": 6, "code": 7, "writing": 7, "translation": 8,
      "long_context": 7}, 128000, True, True),
    ("gpt-4o",
     {"reasoning": 8, "math": 8, "code": 9, "writing": 8, "translation": 8,
      "long_context": 7}, 128000, True, True),
    ("claude",
     {"reasoning": 9, "math": 8, "code": 9, "writing": 10, "translation": 8,
      "long_context": 9}, 200000, True, True),
    ("gemini",
     {"reasoning": 7, "math": 7, "code": 7, "writing": 7, "translation": 8,
      "long_context": 10}, 1048576, True, True),
    ("qwen2.5-coder",
     {"reasoning": 4, "math": 4, "code": 7, "writing": 4, "translation": 5,
      "long_context": 4}, 32768, True, False),
    ("qwen",
     {"reasoning": 6, "math": 6, "code": 7, "writing": 7, "translation": 9,
      "long_context": 7}, 131072, True, False),
    ("glm",
     {"reasoning": 6, "math": 6, "code": 7, "writing": 7, "translation": 9,
      "long_context": 8}, 131072, True, False),
    ("moonshot",  # Kimi: 长文本见长
     {"reasoning": 6, "math": 6, "code": 6, "writing": 8, "translation": 8,
      "long_context": 9}, 262144, True, False),
    ("llama",
     {"reasoning": 5, "math": 5, "code": 6, "writing": 6, "translation": 6,
      "long_context": 6}, 131072, True, False),
]

DEFAULT_STRENGTHS = {"reasoning": 5, "math": 5, "code": 5, "writing": 5,
                     "translation": 5, "long_context": 5}
DEFAULT_CONTEXT_WINDOW = 32768


@dataclass
class ModelCaps:
    strengths: dict = field(default_factory=lambda: dict(DEFAULT_STRENGTHS))
    context_window: int = DEFAULT_CONTEXT_WINDOW
    supports_tools: bool = True
    supports_vision: bool = False
    source: str = "default"  # builtin / config / default


def caps_for(model_cfg: dict) -> ModelCaps:
    """取模型能力画像: 配置覆盖 > 内置画像 > 默认."""
    caps = ModelCaps()
    model_id = model_cfg.get("model", "").lower()
    for substr, strengths, ctx, tools, vision in BUILTIN_PROFILES:
        if substr in model_id:
            caps.strengths = dict(strengths)
            caps.context_window = ctx
            caps.supports_tools = tools
            caps.supports_vision = vision
            caps.source = "builtin"
            break
    # 配置级覆盖
    if model_cfg.get("strengths"):
        caps.strengths.update(model_cfg["strengths"])
        caps.source = "config"
    if model_cfg.get("context_window"):
        caps.context_window = model_cfg["context_window"]
        caps.source = "config"
    if "supports_tools" in model_cfg:
        caps.supports_tools = bool(model_cfg["supports_tools"])
        caps.source = "config"
    if "supports_vision" in model_cfg:
        caps.supports_vision = bool(model_cfg["supports_vision"])
        caps.source = "config"
    return caps


def tier_fit(model_tier: str, task_tier: str) -> float:
    if model_tier == task_tier:
        return 1.0
    if model_tier in TIER_NEIGHBORS.get(task_tier, set()):
        return 0.55
    return 0.25


def suitability(caps: ModelCaps, model_tier: str, price: dict,
                task: dict, max_price: float, task_tier: str,
                w_capability: float = 0.60, w_tier: float = 0.25,
                w_cost: float = 0.15) -> tuple[float | None, dict]:
    """计算适配度 (0-100). 被硬过滤时返回 (None, {filtered: 原因}).

    成本用对数曲线: 价格差 10 倍才拉开一档差距,
    避免贵模型被成本项一票否决 (能力相近时便宜的才占优).
    """
    if task.get("needs_vision") and not caps.supports_vision:
        return None, {"filtered": "任务需要视觉输入, 该模型不支持"}
    if task.get("needs_tools") and not caps.supports_tools:
        return None, {"filtered": "任务需要工具调用, 该模型不支持"}
    est = task.get("est_tokens", 0)
    if est > caps.context_window * 0.9:
        return None, {"filtered":
                      f"输入约 {est} tokens, 超出窗口 {caps.context_window} 的 90%"}

    tags = task.get("tags") or []
    if tags:
        capability = sum(caps.strengths.get(t, 3) for t in tags) / (10 * len(tags))
    else:
        capability = 0.5  # 无明确类型: 中性
    tf = tier_fit(model_tier, task_tier)
    price_sum = price.get("input", 0) + price.get("output", 0)
    if max_price > 0:
        import math
        cost_fit = 1.0 - math.log1p(price_sum) / math.log1p(max_price)
    else:
        cost_fit = 1.0

    score = round(100 * (w_capability * capability + w_tier * tf
                         + w_cost * cost_fit), 1)
    return score, {
        "capability": round(capability * 100, 1),
        "matched_tags": tags,
        "tier_fit": tf,
        "cost_fit": round(cost_fit, 2),
        "caps_source": caps.source,
    }
