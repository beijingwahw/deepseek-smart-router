"""MoA 竞技场 (Mixture-of-Agents, ICLR 2025 Spotlight).

困难任务不再只派一个模型, 而是并行 fan-out 给适配度 Top-N 模型,
用内置评审选出冠军答案 —— 多模型竞赛, 赢家通吃, 胜负反哺学习引擎.

与研究的两个关键对齐:
  - "提议者质量 > 多样性" (Princeton 2025): 直接取适配度排序的 Top-N,
    而不是为了多样性刻意混入弱模型
  - 零额外评审成本模式: 默认用启发式评审 (judge.py) 选冠军,
    不引入 LLM-as-Judge 的额外调用开销
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .judge import judge_response
from .proxy import RouteResult


@dataclass
class MoaOutcome:
    winner: RouteResult
    winner_judge_score: float
    members: list[dict] = field(default_factory=list)   # 全部参赛者成绩
    responses: dict = field(default_factory=dict)        # 提案原文 (聚合用)
    degraded: bool = False  # 只有 <=1 个有效响应时退化为普通路由


_AGGREGATE_PROMPT = (
    "你是答案聚合器。下面是多个模型对同一问题的回答提案。\n"
    "请综合各提案的优点、纠正其中的错误, 输出一个最终的、最优的回答。\n"
    "直接给出最终答案, 不要点评提案, 不要提及「提案」二字。\n\n"
    "【原始问题】\n{question}\n\n{proposals}")


async def aggregate_proposals(router, aggregator: str, question: str,
                              members: list[dict], proposals: dict,
                              task: dict) -> RouteResult:
    """MoA 聚合模式: 冠军模型作为聚合器, 把所有提案合成最终答案.

    members: run_moa 的参赛榜单; proposals: {model_name: response_body}
    """
    blocks = []
    for b in members:
        if not b["ok"]:
            continue
        body = proposals.get(b["model_name"]) or {}
        try:
            text = body["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            text = ""
        if text.strip():
            blocks.append(f"【提案 {len(blocks)+1}】\n{text[:3000]}")
    if len(blocks) < 2:
        raise ValueError("有效提案不足, 无法聚合")
    prompt = _AGGREGATE_PROMPT.format(question=question[:2000],
                                      proposals="\n\n".join(blocks))
    return await router.chat(
        [aggregator],
        {"messages": [{"role": "user", "content": prompt}]}, task)


async def run_moa(router, chain: list[str], payload: dict, task: dict,
                  fanout: int = 3) -> MoaOutcome:
    """并行调用链上前 fanout 个模型, 评审选冠军."""
    members = chain[: max(2, fanout)]
    results = await asyncio.gather(
        *(router.chat([m], payload, task) for m in members),
        return_exceptions=True)

    valid: list[tuple[RouteResult, float, list[str]]] = []
    board: list[dict] = []
    responses: dict = {}
    for name, res in zip(members, results):
        if isinstance(res, Exception):
            board.append({"model_name": name, "ok": False,
                          "error": str(res)[:200]})
            continue
        score, reasons = judge_response(res.response, task)
        valid.append((res, score, reasons))
        responses[name] = res.response
        board.append({"model_name": name, "model": res.model, "ok": True,
                      "judge_score": score, "judge_reasons": reasons,
                      "latency_ms": res.latency_ms, "usage": res.usage})

    if not valid:
        raise RuntimeError("MoA 全部参赛者失败: "
                           + "; ".join(b.get("error", "?") for b in board))
    valid.sort(key=lambda x: -x[1])
    winner, wscore, _ = valid[0]
    for b in board:
        b["winner"] = b.get("model_name") == winner.model_name
    return MoaOutcome(winner=winner, winner_judge_score=wscore,
                      members=board, responses=responses,
                      degraded=len(valid) == 1)
