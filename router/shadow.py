"""影子灰度 (Shadow Canary): 新模型零风险上线评估.

生产级模型灰度方法论:
  - 影子模型不出现在正常候选链, 用户永远不受影响
  - 按 sample_rate 对真实流量后台平行调用 (fire-and-forget)
  - 评审打分写入学习引擎与经验回忆 —— 数据攒够了, 一键 promote 转正,
    此时 learner/memory 里已有它的真实表现, 调度立即合理使用它
"""

from __future__ import annotations

import asyncio
import random


class ShadowRunner:
    def __init__(self, cfg_shadow: dict):
        self.enabled = cfg_shadow.get("enabled", False)
        self.sample_rate = cfg_shadow.get("sample_rate", 0.2)
        self.models = cfg_shadow.get("models", [])
        self.launched = 0

    def maybe_launch(self, router, payload: dict, task: dict, query: str,
                     judge_fn, learn_fn, record_fn) -> list[str]:
        """按采样率后台发起影子评估. 返回本次发起的影子模型列表."""
        if not self.enabled:
            return []
        fired = []
        for name in self.models:
            if name not in router.models:
                continue
            if random.random() >= self.sample_rate:
                continue
            asyncio.create_task(
                self._eval(router, name, payload, task, query,
                           judge_fn, learn_fn, record_fn))
            fired.append(name)
            self.launched += 1
        return fired

    @staticmethod
    async def _eval(router, name, payload, task, query,
                    judge_fn, learn_fn, record_fn):
        try:
            result = await router.chat([name], payload, task)
        except Exception:
            learn_fn(name, task, 0.1, query)
            return
        jscore, _ = await judge_fn(result.response, task)
        learn_fn(name, task, jscore, query)
        record_fn(name, result, jscore, query)
