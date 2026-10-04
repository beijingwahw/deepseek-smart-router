"""预算守卫: 日花费限额, 超限自动降档或拒绝."""

from __future__ import annotations


def apply_budget(tier: str, daily_spend: float, budget_cfg: dict) -> tuple[str, str | None]:
    """根据今日已花费决定放行策略.

    返回 (实际档位, 动作说明):
      - 未启用或未超限: (tier, None)
      - 超限 + downgrade: hard -> standard, 其他档不变 (已经够便宜了)
      - 超限 + block: (tier, "block") 调用方应返回 429
    """
    if not budget_cfg.get("enabled"):
        return tier, None
    cap = budget_cfg.get("daily_usd", 0)
    if cap <= 0 or daily_spend < cap:
        return tier, None
    action = budget_cfg.get("on_exceed", "downgrade")
    if action == "block":
        return tier, "block"
    if tier == "hard":
        return "standard", f"日花费 ${daily_spend:.2f} 超限 ${cap:.2f}, 已降档 hard->standard"
    return tier, f"日花费 ${daily_spend:.2f} 超限 ${cap:.2f}, 当前档已是最低开销"
