"""路由执行层 v2: 多供应商模型池 + 协议适配 + 熔断 + 降级.

模型池视角: 每个注册模型有自己的熔断器和协议适配器;
一次请求的候选链 = [同档模型(按策略排序)...] + [跨档降级模型...],
逐个尝试, 直到成功或全部失败.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx

from .config import enabled_models
from .learner import REWARD_FAILURE, REWARD_SUCCESS
from .matcher import caps_for, suitability
from .providers import build_adapter


class CircuitBreaker:
    """简单熔断器: 连续失败 N 次进入冷却期, 冷却期内直接视为不可用."""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._opened_at: float | None = None

    def available(self) -> bool:
        if self._opened_at is None:
            return True
        if time.time() - self._opened_at >= self.cooldown_seconds:
            self._failures = 0  # 半开: 放行一次尝试
            self._opened_at = None
            return True
        return False

    def on_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def on_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.time()


class UpstreamError(Exception):
    """上游调用失败 (含超时/5xx/连接错误), 触发降级."""

    def __init__(self, model_name: str, message: str):
        super().__init__(message)
        self.model_name = model_name
        self.message = message


@dataclass
class RouteResult:
    model_name: str               # 模型池内部名称
    model: str                    # 上游实际模型 ID
    provider: str
    tier: str
    status: str = "ok"            # ok / fallback
    fell_back_from: str | None = None
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0
    response: dict | None = None


class SmartRouter:
    def __init__(self, cfg: dict, client: httpx.AsyncClient | None = None,
                 learner=None):
        self.cfg = cfg
        self.providers = cfg["providers"]
        self.models = cfg["models"]
        self.routing = cfg["routing"]
        self.learner = learner          # Genesis: 自适应学习引擎 (可选)
        self.disabled: set[str] = set()  # 运行时禁用 (Admin API)
        cb_cfg = self.routing.get("circuit_breaker", {})
        self.breakers = {
            name: CircuitBreaker(cb_cfg.get("failure_threshold", 3),
                                 cb_cfg.get("cooldown_seconds", 60))
            for name in self.models
        }
        self._adapters: dict = {}
        self._rr_counters: dict[str, int] = {}
        self._client = client

    async def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=None)
        return self._client

    # ---------- 候选链 ----------

    def _max_price(self) -> float:
        return max(
            (m["price"].get("input", 0) + m["price"].get("output", 0)
             for m in enabled_models(self.cfg).values()),
            default=0.0)

    def _suitability(self, name: str, task: dict, task_tier: str,
                     max_price: float) -> tuple[float | None, dict]:
        m = self.models[name]
        score, detail = suitability(caps_for(m), m.get("tier", "standard"),
                                    m.get("price", {}), task, max_price,
                                    task_tier)
        if score is not None and self.learner is not None:
            adj = self.learner.adjustment(name, task.get("tags") or [])
            detail["learned_adjustment"] = adj
            if adj != 1.0:
                score = round(min(100.0, score * adj), 1)
        return score, detail

    def candidates(self, tier: str, task: dict | None = None) -> list[str]:
        """某难度档的启用模型, 按策略排序 (best_fit 时按任务适配度)."""
        names = [n for n, m in enabled_models(self.cfg).items()
                 if m.get("tier") == tier and n not in self.disabled]
        strategy = self.routing.get("strategy", "best_fit")
        if strategy == "best_fit" and task is not None:
            max_price = self._max_price()
            scored = []
            for n in names:
                s, _ = self._suitability(n, task, tier, max_price)
                scored.append((n, s if s is not None else -1.0))
            names = [n for n, _ in sorted(scored, key=lambda x: -x[1])]
        elif strategy == "cheapest":
            names.sort(key=lambda n: (self.models[n]["price"].get("input", 0)
                                      + self.models[n]["price"].get("output", 0)))
        elif strategy == "round_robin":
            names.sort(key=lambda n: self.models[n].get("priority", 1))
            if names:
                i = self._rr_counters.get(tier, 0) % len(names)
                self._rr_counters[tier] = i + 1
                names = names[i:] + names[:i]
        else:  # priority
            names.sort(key=lambda n: self.models[n].get("priority", 1))
        return names

    def candidate_chain(self, tier: str, pinned: str | None = None,
                        task: dict | None = None) -> list[str]:
        """完整候选链: 指定模型 / 同档模型 + 跨档降级模型."""
        if pinned:
            return [pinned]
        chain = list(self.candidates(tier, task))
        if self.routing.get("fallback_enabled", True):
            for fb_tier in self.routing.get("cross_tier_fallback", {}).get(tier, []):
                chain += [n for n in self.candidates(fb_tier, task)
                          if n not in chain]
        return chain

    def suitability_report(self, tier: str, task: dict) -> list[dict]:
        """全池适配度报告 (供 preview 展示): 含被硬过滤的模型及原因."""
        max_price = self._max_price()
        report = []
        for name, m in enabled_models(self.cfg).items():
            score, detail = self._suitability(name, task, tier, max_price)
            report.append({
                "name": name, "provider": m["provider"], "model": m["model"],
                "tier": m.get("tier"), "score": score, **detail,
                "breaker_open": not self.breakers[name].available(),
            })
        report.sort(key=lambda r: (r["score"] is None,
                                   -(r["score"] or 0)))
        return report

    # ---------- 内部 ----------

    def _adapter(self, name: str):
        if name not in self._adapters:
            mcfg = self.models[name]
            self._adapters[name] = build_adapter(self.providers[mcfg["provider"]],
                                                 mcfg)
        return self._adapters[name]

    def _result(self, name: str, chain: list[str], started: float,
                usage: dict | None = None, response: dict | None = None) -> RouteResult:
        mcfg = self.models[name]
        return RouteResult(
            model_name=name, model=mcfg["model"], provider=mcfg["provider"],
            tier=mcfg.get("tier", "standard"),
            status="ok" if name == chain[0] else "fallback",
            fell_back_from=None if name == chain[0] else chain[0],
            usage=usage or {},
            latency_ms=int((time.time() - started) * 1000),
            response=response,
        )

    def _transient(self, e: Exception) -> bool:
        """4xx 是请求本身的问题, 换模型大概率同样失败, 不值得降级."""
        return not (isinstance(e, UpstreamError) and e.message.startswith("上游 4"))

    def _learn(self, name: str, task: dict | None, reward: float) -> None:
        if self.learner is not None:
            self.learner.record(name, (task or {}).get("tags") or [], reward)

    # ---------- 非流式 ----------

    async def chat(self, chain: list[str], payload: dict,
                   task: dict | None = None) -> RouteResult:
        if not chain:
            raise UpstreamError("-", "模型池为空: 没有可用模型")
        last_err: Exception | None = None
        for name in chain:
            if not self.breakers[name].available():
                continue
            mcfg = self.models[name]
            adapter = self._adapter(name)
            body = adapter.translate_request(
                {**payload, "model": mcfg["model"], "stream": False})
            started = time.time()
            try:
                client = await self.client()
                resp = await client.post(
                    adapter.endpoint(stream=False), json=body,
                    headers=adapter.headers(),
                    timeout=mcfg.get("timeout", 120))
                if resp.status_code >= 500:
                    raise UpstreamError(name, f"上游 {resp.status_code}")
                self.breakers[name].on_success()  # 4xx 也算上游可达
                self._learn(name, task, REWARD_SUCCESS)
                data = adapter.translate_response(resp.json())
                result = self._result(name, chain, started,
                                      usage=data.get("usage", {}) or {},
                                      response=data)
                if resp.status_code >= 400:
                    result.status = "ok"  # 4xx 原样透传给 harness
                return result
            except (httpx.TimeoutException, httpx.TransportError, UpstreamError) as e:
                self.breakers[name].on_failure()
                self._learn(name, task, REWARD_FAILURE)
                last_err = e
                if not self._transient(e):
                    raise
        raise UpstreamError(chain[0], f"候选链全部不可用: {last_err}")

    # ---------- 流式 ----------

    async def chat_stream(self, chain: list[str], payload: dict,
                          task: dict | None = None):
        """返回 (RouteResult, 异步字节迭代器, usage_holder).

        流式只在连接建立阶段降级; 建立后事件流经协议翻译器转为 OpenAI 格式透传.
        """
        if not chain:
            raise UpstreamError("-", "模型池为空: 没有可用模型")
        last_err: Exception | None = None
        for name in chain:
            if not self.breakers[name].available():
                continue
            mcfg = self.models[name]
            adapter = self._adapter(name)
            body = adapter.translate_request(
                {**payload, "model": mcfg["model"], "stream": True})
            started = time.time()
            try:
                client = await self.client()
                req = client.build_request("POST", adapter.endpoint(stream=True),
                                           json=body, headers=adapter.headers())
                resp = await client.send(req, stream=True)
                if resp.status_code >= 400:
                    await resp.aread()
                    await resp.aclose()
                    raise UpstreamError(name, f"上游 {resp.status_code}")
                self.breakers[name].on_success()
                self._learn(name, task, REWARD_SUCCESS)
                usage_holder: dict = {}
                stream = self._wrap_stream(resp, adapter.make_stream_translator(),
                                           usage_holder)
                return self._result(name, chain, started), stream, usage_holder
            except (httpx.TimeoutException, httpx.TransportError, UpstreamError) as e:
                self.breakers[name].on_failure()
                self._learn(name, task, REWARD_FAILURE)
                last_err = e
                if not self._transient(e):
                    raise
        raise UpstreamError(chain[0], f"候选链全部不可用: {last_err}")

    @staticmethod
    async def _wrap_stream(resp: httpx.Response, translator, usage_holder: dict):
        """SSE 事件流经适配器翻译成 OpenAI chunk, 同时收集 usage."""
        try:
            async for raw_line in resp.aiter_lines():
                for out in translator.feed(raw_line.encode()):
                    text = out.decode(errors="ignore")
                    if '"usage"' in text:
                        try:
                            data = json.loads(text.split("data:", 1)[1].strip())
                            if isinstance(data, dict) and data.get("usage"):
                                usage_holder.update(data["usage"])
                        except (ValueError, IndexError):
                            pass
                    yield out
        finally:
            await resp.aclose()
