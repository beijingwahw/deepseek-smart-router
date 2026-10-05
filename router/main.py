"""FastAPI 入口 v1.0 Genesis: 自我进化的多模型智能路由.

新增:
  - 学习闭环: 调用结果自动反哺调度 (隐式) + /v1/feedback 显式打分
  - 预算守卫: 日花费超限自动降档/拒绝 (budget 配置)
  - 运行时管理: /v1/admin 模型启停 + 配置热重载 (文件变更自动生效)
  - 可观测: /v1/stats 聚合学习状态 / 延迟 / 预算用量

启动: uvicorn router.main:app --host 0.0.0.0 --port 8355
Harness 接入: 把 base_url 指向 http://localhost:8355/v1 即可.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .budget import apply_budget
from .cache import SemanticCache
from .classifier import classify, detect_task_profile, route
from .config import enabled_models, load_config
from .embedder import build_embedder
from .judge import (ajudge_response, cascade_start_tier, judge_response,
                    should_escalate)
from .learner import Learner
from .memory import ExperienceMemory
from .moa import aggregate_proposals, run_moa
from .proxy import SmartRouter, UpstreamError
from .shadow import ShadowRunner
from .stats import Stats, calc_cost

CONFIG_PATH = os.environ.get("ROUTER_CONFIG")

# ---------------- 全局状态 (热重载时整体刷新) ----------------

cfg = load_config()
stats = Stats(cfg["stats"]["db_path"])
learner = Learner(cfg["stats"]["db_path"])
shadow_runner = ShadowRunner(cfg.get("shadow", {}))


def _build_memory(cfg):
    m = cfg.get("memory", {})
    if not m.get("enabled"):
        return None
    return ExperienceMemory(cfg["stats"]["db_path"], build_embedder(cfg),
                            k=m.get("k", 5), min_sim=m.get("min_sim", 0.7),
                            max_entries=m.get("max_entries", 5000))


memory = _build_memory(cfg)
router = SmartRouter(cfg, learner=learner, memory=memory)
router.latency_stats = stats.latency_by_model  # 延迟感知接线


def _build_cache(cfg):
    c = cfg.get("cache", {})
    if not c.get("enabled"):
        return None
    return SemanticCache(cfg["stats"]["db_path"], build_embedder(cfg),
                         threshold=c.get("threshold", 0.75),
                         ttl_seconds=c.get("ttl_seconds", 86400),
                         max_entries=c.get("max_entries", 10000))


cache = _build_cache(cfg)
TH = cfg["thresholds"]
ALIASES = cfg.get("aliases", {})
BUDGET = cfg.get("budget", {})
CASCADE = cfg.get("cascade", {})
MOA = cfg.get("moa", {})
JUDGE = cfg.get("judge", {})
_config_mtime: float | None = None


@asynccontextmanager
async def lifespan(_app):
    yield
    # 关闭时统一释放资源
    stats.close()
    learner.close()
    if cache:
        cache.close()
    if memory:
        memory.close()
    if router._client is not None:
        await router._client.aclose()


app = FastAPI(title="DeepSeek Smart Router", version="4.0.0",
              lifespan=lifespan)


def _reload_if_changed(force: bool = False,
                       path: str | None = None) -> bool:
    """配置文件变更时热重载: 模型池/阈值/预算/缓存/级联立即生效, 学习成果保留.

    force=True 时跳过 mtime 比较直接重载 (Admin API 用).
    返回是否发生了重载.
    """
    global cfg, router, cache, memory, shadow_runner
    global TH, ALIASES, BUDGET, CASCADE, MOA, JUDGE, _config_mtime
    path = path or CONFIG_PATH
    if not path or not Path(path).exists():
        return False
    mtime = os.path.getmtime(path)
    if not force:
        if _config_mtime is None:
            _config_mtime = mtime
            return False
        if mtime <= _config_mtime:
            return False
    cfg = load_config(path)
    disabled, promoted = router.disabled, router.promoted  # 保留运行时状态
    memory = memory if memory is not None else _build_memory(cfg)
    router = SmartRouter(cfg, learner=learner, memory=memory)
    router.latency_stats = stats.latency_by_model
    router.disabled, router.promoted = disabled, promoted
    shadow_runner = ShadowRunner(cfg.get("shadow", {}))
    new_cache = _build_cache(cfg)
    if new_cache is not None or cache is None:
        cache = new_cache or cache  # 缓存对象复用, 保住命中率
    TH = cfg["thresholds"]
    ALIASES = cfg.get("aliases", {})
    BUDGET = cfg.get("budget", {})
    CASCADE = cfg.get("cascade", {})
    MOA = cfg.get("moa", {})
    JUDGE = cfg.get("judge", {})
    _config_mtime = mtime
    return True


@app.middleware("http")
async def hot_reload_middleware(request: Request, call_next):
    _reload_if_changed()
    return await call_next(request)


# ---------------- 内部 ----------------

def _estimate_tokens(payload: dict) -> int:
    from .classifier import estimate_tokens
    total = "".join(str(m.get("content", "")) for m in payload.get("messages", []))
    return estimate_tokens(total)


def _ref_price(tier: str) -> dict:
    for m in enabled_models(cfg).values():
        if m.get("tier") == tier:
            return m["price"]
    return {"input": 0.0, "output": 0.0}


def _record(result, score: int, reasons: str, usage: dict,
            status: str, error: str | None = None, query: str = "") -> int:
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    price = cfg["models"].get(result.model_name, {}).get(
        "price", {"input": 0.0, "output": 0.0})
    return stats.record(
        tier=result.tier, model=f"{result.provider}/{result.model}",
        score=score, reasons=reasons, prompt_tokens=pt, completion_tokens=ct,
        cost=calc_cost(price, pt, ct, cache_hit),
        baseline_hard=calc_cost(_ref_price("hard"), pt, ct),
        baseline_standard=calc_cost(_ref_price("standard"), pt, ct),
        latency_ms=result.latency_ms, status=status, error=error,
        query=query,
    )


def _mem_record(query: str, model_name: str, reward: float) -> None:
    """经验回忆入库 (模型池内部名称)."""
    if memory is not None and model_name:
        memory.record(query, model_name, reward)


async def _judge(body: dict, task: dict, question: str = ""):
    """统一评审入口: 按配置走启发式或 LLM-as-Judge."""
    return await ajudge_response(router, body, task, JUDGE, question)


def _shadow_learn(name: str, task: dict, reward: float, query: str) -> None:
    learner.record(name, task.get("tags") or [], reward)
    _mem_record(query, name, reward)


def _shadow_record(name: str, result, jscore: float, query: str) -> None:
    usage = result.usage or {"prompt_tokens": 0, "completion_tokens": 0}
    _record(result, 0, f"影子评估(评审{jscore})", usage, "shadow", query=query)


def _launch_shadow(payload: dict, task: dict, query: str) -> None:
    """非流式响应完成后, 后台发起影子灰度评估."""
    shadow_runner.maybe_launch(router, payload, task, query,
                               _judge, _shadow_learn, _shadow_record)


def _plan(payload: dict) -> tuple[str, str | None, object]:
    """路由计划: (tier, pinned_model, features). 含预算守卫."""
    f = classify(payload)
    requested = str(payload.get("model") or "auto")
    if requested in cfg["models"]:
        tier, pinned = cfg["models"][requested].get("tier", "standard"), requested
    else:
        alias = ALIASES.get(requested)
        if alias and alias != "auto":
            tier, pinned = alias, None
        else:
            tier, pinned = route(f.score, TH["trivial"], TH["hard"]), None
    tier, budget_note = apply_budget(tier, stats.daily_spend(), BUDGET)
    if budget_note and budget_note != "block":
        f.reasons.append(budget_note)
    return tier, pinned, f


# ---------------- 端点 ----------------

@app.get("/health")
async def health():
    return {
        "status": "ok", "version": "4.0.0-apex",
        "models": {n: {"provider": m["provider"], "model": m["model"],
                       "tier": m.get("tier"), "enabled": m.get("enabled", True),
                       "disabled_at_runtime": n in router.disabled}
                   for n, m in cfg["models"].items()},
        "aliases": ALIASES, "thresholds": TH,
        "budget": {**BUDGET, "spent_today": round(stats.daily_spend(), 4)},
        "learner": learner.summary(),
        "cache": cache.stats() if cache else {"enabled": False},
        "cascade": {"enabled": CASCADE.get("enabled", False)},
        "memory": {"enabled": memory is not None,
                   "entries": memory.size() if memory else 0},
        "moa": {"enabled": MOA.get("enabled", False),
                "tiers": MOA.get("tiers", []),
                "mode": MOA.get("mode", "select")},
        "judge": {"mode": JUDGE.get("mode", "heuristic")},
        "shadow": {"enabled": shadow_runner.enabled,
                   "sample_rate": shadow_runner.sample_rate,
                   "models": shadow_runner.models,
                   "launched": shadow_runner.launched},
        "quality_lambda": cfg["routing"].get("quality_lambda", 0.55),
        "latency_weight": cfg["routing"].get("latency_weight", 0.05),
    }


@app.get("/v1/models")
async def list_models():
    ids = list(ALIASES.keys()) + list(enabled_models(cfg).keys())
    return {"object": "list", "data": [
        {"id": i, "object": "model", "created": 0, "owned_by": "smart-router"}
        for i in ids]}


@app.post("/v1/route/preview")
async def route_preview(request: Request):
    """干跑: 任务画像 + 全池适配度 (含学习调整系数) + 出局原因."""
    payload = await request.json()
    tier, pinned, f = _plan(payload)
    task = detect_task_profile(payload)
    query = _cache_query(payload)
    chain = router.candidate_chain(tier, pinned, task, query)
    return {
        "score": f.score, "tier": tier, "pinned": pinned,
        "task_profile": task,
        "candidates": [r for r in router.suitability_report(tier, task, query)
                       if r["name"] in chain],
        "forced": f.forced, "reasons": f.reasons,
    }


def _cache_query(payload: dict) -> str:
    """缓存键文本: 最后一条用户消息."""
    from .classifier import _last_user_text
    return _last_user_text(payload.get("messages", []) or [])


def _cacheable(payload: dict, task: dict) -> bool:
    return (cache is not None and not payload.get("stream")
            and not task["needs_tools"] and not task["needs_vision"])


class _CacheResult:
    """缓存命中的记账占位."""

    def __init__(self, cached_model: str):
        self.model_name = ""
        self.model = cached_model
        self.provider = "cache"
        self.tier = "cache"
        self.latency_ms = 0
        self.status = "cache_hit"
        self.fell_back_from = None


def _router_headers(result, score: int, request_id: int | None = None,
                    extra: dict | None = None) -> dict:
    h = {"X-Router-Tier": result.tier, "X-Router-Model": result.model,
         "X-Router-Score": str(score)}
    if request_id:
        h["X-Router-Request-Id"] = str(request_id)
    return {**h, **(extra or {})}


def _handle_cache_hit(payload, task, tier, query, f, reasons):
    """语义缓存命中则返回响应, 否则 None."""
    if not _cacheable(payload, task):
        return None
    cached = cache.lookup(query, tier)
    if cached is None:
        return None
    result = _CacheResult(cached.get("model", "unknown"))
    request_id = _record(result, f.score, reasons + "; 语义缓存命中",
                         cached.get("usage", {}) or {}, "cache_hit",
                         query=query)
    cached["router"] = {**cached.get("router", {}),
                        "request_id": request_id, "tier": "cache",
                        "score": f.score}
    return JSONResponse(content=cached,
                        headers={"X-Router-Tier": "cache",
                                 "X-Router-Cache": "hit",
                                 "X-Router-Request-Id": str(request_id)})


async def _handle_stream(chain, payload, task, query, f, reasons):
    """流式路径: 直接走候选链 (级联与缓存只服务非流式)."""
    try:
        result, stream, usage_holder = await router.chat_stream(
            chain, payload, task)
    except UpstreamError as e:
        return JSONResponse(status_code=502,
                            content={"error": {"message": str(e)}})

    async def gen():
        async for chunk in stream:
            yield chunk
        if not usage_holder:
            usage_holder["prompt_tokens"] = _estimate_tokens(payload)
            usage_holder["completion_tokens"] = 0
        _record(result, f.score, reasons, usage_holder, result.status,
                query=query)
        _mem_record(query, result.model_name, 0.75)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers=_router_headers(result, f.score))


async def _handle_moa(tier, pinned, chain, payload, task, query, f, reasons):
    """MoA 竞技场: 开赛则返回冠军响应, 否则 None."""
    if not (MOA.get("enabled") and not pinned
            and tier in MOA.get("tiers", ["hard"]) and len(chain) >= 2):
        return None
    try:
        outcome = await run_moa(router, chain, payload, task,
                                fanout=MOA.get("fanout", 3))
    except RuntimeError as e:
        return JSONResponse(status_code=502,
                            content={"error": {"message": str(e)}})
    winner, request_id = outcome.winner, 0
    for b in outcome.members:  # 逐参赛者记账 + 竞赛学习
        if not b["ok"]:
            continue
        mcfg = cfg["models"][b["model_name"]]
        member = SimpleNamespace(
            model_name=b["model_name"], model=b["model"],
            provider=mcfg["provider"], tier=mcfg.get("tier", tier),
            latency_ms=b["latency_ms"])
        res_usage = b["usage"] or {}
        if not res_usage.get("prompt_tokens"):
            res_usage["prompt_tokens"] = _estimate_tokens(payload)
            res_usage["completion_tokens"] = 0
        rid = _record(member, f.score,
                      reasons + ("; MoA 冠军" if b["winner"] else "; MoA 参赛"),
                      res_usage,
                      "moa_winner" if b["winner"] else "moa_member",
                      query=query)
        if b["winner"]:
            request_id = rid
        reward = 0.9 if b["winner"] else 0.3
        learner.record(b["model_name"], task.get("tags") or [], reward)
        _mem_record(query, b["model_name"], reward)
    # 聚合模式: 冠军作为聚合器, 把全部提案合成最终答案
    aggregated = False
    if MOA.get("mode") == "aggregate" and not outcome.degraded:
        try:
            agg = await aggregate_proposals(
                router, winner.model_name, query, outcome.members,
                outcome.responses, task)
            agg_usage = agg.usage or {}
            _record(agg, f.score, reasons + "; MoA 聚合调用",
                    agg_usage, "moa_aggregate", query=query)
            final_body = dict(agg.response or {})
            aggregated = True
        except Exception:
            final_body = dict(winner.response or {})  # 聚合失败回退冠军答案
    else:
        final_body = dict(winner.response or {})
    final_body["router"] = {
        "request_id": request_id, "tier": winner.tier,
        "model_name": winner.model_name, "model": winner.model,
        "provider": winner.provider, "score": f.score,
        "reasons": f.reasons,
        "status": "moa_aggregate" if aggregated else "moa_winner",
        "moa": {"fanout": len(outcome.members), "mode": MOA.get("mode", "select"),
                "winner_judge_score": outcome.winner_judge_score,
                "degraded": outcome.degraded, "members": outcome.members},
    }
    if _cacheable(payload, task):
        cache.store(query, tier, final_body, winner.model)
    _launch_shadow(payload, task, query)
    return JSONResponse(content=final_body,
                        headers=_router_headers(winner, f.score, request_id,
                                                {"X-Router-MoA": "winner"}))


async def _try_cascade(tier, pinned, payload, task, query, f, reasons):
    """级联: 一次通过返回响应; 升级返回 (None, cascade_info); 未启用 (None, None)."""
    if not (CASCADE.get("enabled") and not pinned
            and cascade_start_tier(tier) is not None):
        return None, None
    start_tier = cascade_start_tier(tier)
    start_chain = router.candidate_chain(start_tier, None, task, query)
    if not start_chain:
        return None, None
    try:
        first = await router.chat(start_chain, payload, task)
    except UpstreamError:
        return None, None  # 便宜档全挂: 落到原计划链
    jscore, jreasons = await _judge(first.response, task, query)
    usage1 = first.usage or {}
    if not usage1.get("prompt_tokens"):
        usage1["prompt_tokens"] = _estimate_tokens(payload)
        usage1["completion_tokens"] = 0
    if should_escalate(jscore, CASCADE.get("judge_threshold", 0.55)):
        # 不合格: 差评反哺 + 升级到原计划链
        learner.record(first.model_name, task.get("tags") or [], 0.2)
        _mem_record(query, first.model_name, 0.2)
        _record(first, f.score,
                reasons + f"; 级联升级(评审{jscore}: "
                f"{'/'.join(jreasons) or '质量不足'})",
                usage1, "cascade_escalated", query=query)
        return None, {"started_tier": start_tier, "first_model": first.model,
                      "judge_score": jscore, "escalated": True}
    # 合格: 好评反哺, 直接交卷 (省下了高档的钱)
    learner.record(first.model_name, task.get("tags") or [], 0.9)
    _mem_record(query, first.model_name, 0.9)
    request_id = _record(first, f.score,
                         reasons + f"; 级联一次通过(评审{jscore})",
                         usage1, "cascade_accept", query=query)
    body = dict(first.response or {})
    body["router"] = {
        "request_id": request_id, "tier": first.tier,
        "model_name": first.model_name, "model": first.model,
        "provider": first.provider, "score": f.score,
        "reasons": f.reasons, "status": "cascade_accept",
        "cascade": {"started_tier": start_tier, "judge_score": jscore,
                    "escalated": False, "planned_tier": tier},
    }
    if _cacheable(payload, task):
        cache.store(query, tier, body, first.model)
    _launch_shadow(payload, task, query)
    return JSONResponse(content=body,
                        headers=_router_headers(first, f.score, request_id,
                                                {"X-Router-Cascade": "accept"})), None


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """编排骨架: 预算 -> 缓存 -> 流式 -> MoA -> 级联 -> 常规候选链."""
    payload = await request.json()
    tier, pinned, f = _plan(payload)
    _, budget_action = apply_budget(tier, stats.daily_spend(), BUDGET)
    if budget_action == "block":
        return JSONResponse(status_code=429, content={"error": {
            "message": f"日预算已用尽 (${BUDGET.get('daily_usd')}), "
                       f"明天零点重置或调大 budget.daily_usd"}})

    task = detect_task_profile(payload)
    query = _cache_query(payload)
    chain = router.candidate_chain(tier, pinned, task, query)
    reasons = "; ".join(f.reasons) or "无显著信号"

    resp = _handle_cache_hit(payload, task, tier, query, f, reasons)
    if resp is not None:
        return resp
    if payload.get("stream"):
        return await _handle_stream(chain, payload, task, query, f, reasons)
    resp = await _handle_moa(tier, pinned, chain, payload, task, query, f,
                             reasons)
    if resp is not None:
        return resp
    resp, cascade_info = await _try_cascade(tier, pinned, payload, task,
                                            query, f, reasons)
    if resp is not None:
        return resp

    # ---- 常规路径: 计划候选链 ----
    try:
        result = await router.chat(chain, payload, task)
    except UpstreamError as e:
        return JSONResponse(status_code=502, content={"error": {"message": str(e)}})

    usage = result.usage or {}
    if not usage.get("prompt_tokens"):
        usage["prompt_tokens"] = _estimate_tokens(payload)
        usage["completion_tokens"] = usage.get("completion_tokens", 0)
    request_id = _record(result, f.score, reasons, usage, result.status,
                         query=query)
    _mem_record(query, result.model_name, 0.75)

    body = dict(result.response or {})
    body["router"] = {
        "request_id": request_id,
        "tier": result.tier, "model_name": result.model_name,
        "model": result.model, "provider": result.provider,
        "score": f.score, "reasons": f.reasons,
        "status": result.status, "fell_back_from": result.fell_back_from,
    }
    if cascade_info:
        body["router"]["cascade"] = cascade_info
    # 写入语义缓存 (评审合格才存, 防止缓存劣质答案)
    if _cacheable(payload, task):
        jscore, _ = judge_response(body, task)
        if jscore >= CASCADE.get("judge_threshold", 0.55):
            cache.store(query, tier, body, result.model)
    _launch_shadow(payload, task, query)
    return JSONResponse(content=body,
                        headers=_router_headers(result, f.score, request_id))


@app.post("/v1/feedback")
async def feedback(request: Request):
    """显式反馈: 告诉路由器这次回答好不好 (0-1), 调度会越用越准.

    两种用法:
      {"request_id": 123, "score": 0.9}          # 引用某次请求 (推荐)
      {"model": "ds-r1", "score": 0.2, "tags": ["math"]}  # 直接指定
    """
    body = await request.json()
    score = body.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = -1.0
    if not (0 <= score <= 1):
        return JSONResponse(status_code=422,
                            content={"error": {"message": "score 必须是 0-1 的数字"}})
    model_name = body.get("model")
    rec = None
    if body.get("request_id") is not None:
        rec = stats.get_request(int(body["request_id"]))
        if rec is None:
            return JSONResponse(status_code=404,
                                content={"error": {"message": "request_id 不存在"}})
        # rec["model"] 形如 "provider/model-id", 反查池内名称
        for name, m in cfg["models"].items():
            if rec["model"] == f"{m['provider']}/{m['model']}":
                model_name = name
                break
    if not model_name or model_name not in cfg["models"]:
        return JSONResponse(status_code=422,
                            content={"error": {"message": "请提供有效的 model 或 request_id"}})
    tags = body.get("tags") or []
    learner.record(model_name, tags, float(score))
    # 显式反馈同步写入经验回忆 (优先用 request 记录里的原始问题)
    if memory is not None:
        q = body.get("query") or (rec.get("query", "") if rec else "")
        if q:
            memory.record(q, model_name, float(score))
    return {"ok": True, "model": model_name, "tags": tags or ["*"],
            "score": float(score), "learner": learner.summary().get(model_name)}


@app.get("/v1/stats")
async def get_stats():
    d = stats.summary()
    d["learner"] = learner.summary()
    d["latency"] = stats.latency_by_model()
    d["budget"] = {**BUDGET, "spent_today": round(stats.daily_spend(), 4)}
    d["cache"] = cache.stats() if cache else {"enabled": False}
    d["cascade"] = {"enabled": CASCADE.get("enabled", False)}
    d["memory"] = {"enabled": memory is not None,
                   "entries": memory.size() if memory else 0}
    d["moa"] = {"enabled": MOA.get("enabled", False)}
    return d


# ---------------- 运行时管理 ----------------

@app.get("/v1/admin/models")
async def admin_models():
    return {n: {**{k: m.get(k) for k in ("provider", "model", "tier", "enabled")},
                "disabled_at_runtime": n in router.disabled,
                "breaker_open": not router.breakers[n].available()}
            for n, m in cfg["models"].items()}


@app.post("/v1/admin/models/{name}/disable")
async def admin_disable(name: str):
    if name not in cfg["models"]:
        return JSONResponse(status_code=404,
                            content={"error": {"message": f"模型不存在: {name}"}})
    router.disabled.add(name)
    return {"ok": True, "disabled": sorted(router.disabled)}


@app.post("/v1/admin/models/{name}/enable")
async def admin_enable(name: str):
    router.disabled.discard(name)
    return {"ok": True, "disabled": sorted(router.disabled)}


@app.post("/v1/admin/models/{name}/promote")
async def admin_promote(name: str):
    """影子模型转正: 数据攒够后一键进入正常候选链."""
    if name not in cfg["models"]:
        return JSONResponse(status_code=404,
                            content={"error": {"message": f"模型不存在: {name}"}})
    router.promoted.add(name)
    return {"ok": True, "promoted": sorted(router.promoted),
            "learner": learner.summary().get(name)}


@app.post("/v1/admin/reload")
async def admin_reload():
    reloaded = _reload_if_changed(force=True)
    return {"ok": True, "reloaded": reloaded,
            "models": list(cfg["models"].keys())}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
