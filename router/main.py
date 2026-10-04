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
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .budget import apply_budget
from .cache import SemanticCache
from .classifier import classify, detect_task_profile, route
from .config import enabled_models, load_config
from .embedder import build_embedder
from .judge import cascade_start_tier, judge_response, should_escalate
from .learner import Learner
from .memory import ExperienceMemory
from .moa import run_moa
from .proxy import SmartRouter, UpstreamError
from .stats import Stats, calc_cost

CONFIG_PATH = os.environ.get("ROUTER_CONFIG")

# ---------------- 全局状态 (热重载时整体刷新) ----------------

cfg = load_config()
stats = Stats(cfg["stats"]["db_path"])
learner = Learner(cfg["stats"]["db_path"])


def _build_memory(cfg):
    m = cfg.get("memory", {})
    if not m.get("enabled"):
        return None
    return ExperienceMemory(cfg["stats"]["db_path"], build_embedder(cfg),
                            k=m.get("k", 5), min_sim=m.get("min_sim", 0.7),
                            max_entries=m.get("max_entries", 5000))


memory = _build_memory(cfg)
router = SmartRouter(cfg, learner=learner, memory=memory)


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
_config_mtime: float | None = None

app = FastAPI(title="DeepSeek Smart Router", version="3.0.0")


def _reload_if_changed() -> None:
    """配置文件变更时热重载: 模型池/阈值/预算/缓存/级联立即生效, 学习成果保留."""
    global cfg, router, cache, memory, TH, ALIASES, BUDGET, CASCADE, MOA
    global _config_mtime
    if not CONFIG_PATH or not Path(CONFIG_PATH).exists():
        return
    mtime = os.path.getmtime(CONFIG_PATH)
    if _config_mtime is None:
        _config_mtime = mtime
        return
    if mtime <= _config_mtime:
        return
    cfg = load_config(CONFIG_PATH)
    disabled = router.disabled  # 保留运行时禁用状态
    memory = memory if memory is not None else _build_memory(cfg)
    router = SmartRouter(cfg, learner=learner, memory=memory)
    router.disabled = disabled
    new_cache = _build_cache(cfg)
    if new_cache is not None or cache is None:
        cache = new_cache or cache  # 缓存对象复用, 保住命中率
    TH = cfg["thresholds"]
    ALIASES = cfg.get("aliases", {})
    BUDGET = cfg.get("budget", {})
    CASCADE = cfg.get("cascade", {})
    MOA = cfg.get("moa", {})
    _config_mtime = mtime


@app.middleware("http")
async def hot_reload_middleware(request: Request, call_next):
    _reload_if_changed()
    return await call_next(request)


# ---------------- 内部 ----------------

def _estimate_tokens(payload: dict) -> int:
    total = sum(len(str(m.get("content", ""))) for m in payload.get("messages", []))
    return max(1, total // 4)


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
        "status": "ok", "version": "3.0.0-omega",
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
                "tiers": MOA.get("tiers", [])},
        "quality_lambda": cfg["routing"].get("quality_lambda", 0.55),
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
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
    is_stream = bool(payload.get("stream"))

    # ---- Frontier 第一层: 语义缓存 (仅非流式安全请求) ----
    if _cacheable(payload, task):
        cached = cache.lookup(query, tier)
        if cached is not None:
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

    # ---- 流式: 直接走候选链 (级联与缓存只服务非流式) ----
    if is_stream:
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

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"X-Router-Tier": result.tier,
                     "X-Router-Model": result.model,
                     "X-Router-Score": str(f.score)})

    # ---- Omega: MoA 竞技场 (困难任务多模型并行竞赛, 评审选冠军) ----
    if (MOA.get("enabled") and not pinned
            and tier in MOA.get("tiers", ["hard"]) and len(chain) >= 2):
        try:
            outcome = await run_moa(router, chain, payload, task,
                                    fanout=MOA.get("fanout", 3))
        except RuntimeError as e:
            return JSONResponse(status_code=502,
                                content={"error": {"message": str(e)}})
        winner = outcome.winner
        request_id = 0
        for b in outcome.members:  # 逐参赛者记账 + 竞赛学习
            if not b["ok"]:
                continue
            res_usage = b["usage"] or {}
            if not res_usage.get("prompt_tokens"):
                res_usage["prompt_tokens"] = _estimate_tokens(payload)
                res_usage["completion_tokens"] = 0
            class _Member:
                model_name = b["model_name"]
                model = b["model"]; provider = cfg["models"][b["model_name"]]["provider"]
                tier = cfg["models"][b["model_name"]].get("tier", tier)
                latency_ms = b["latency_ms"]; fell_back_from = None
            rid = _record(_Member(), f.score,
                          reasons + ("; MoA 冠军" if b["winner"] else "; MoA 参赛"),
                          res_usage,
                          "moa_winner" if b["winner"] else "moa_member",
                          query=query)
            if b["winner"]:
                request_id = rid
            reward = 0.9 if b["winner"] else 0.3
            learner.record(b["model_name"], task.get("tags") or [], reward)
            _mem_record(query, b["model_name"], reward)
        body = dict(winner.response or {})
        body["router"] = {
            "request_id": request_id, "tier": winner.tier,
            "model_name": winner.model_name, "model": winner.model,
            "provider": winner.provider, "score": f.score,
            "reasons": f.reasons, "status": "moa_winner",
            "moa": {"fanout": len(outcome.members),
                    "winner_judge_score": outcome.winner_judge_score,
                    "degraded": outcome.degraded,
                    "members": outcome.members},
        }
        if _cacheable(payload, task):
            cache.store(query, tier, body, winner.model)
        return JSONResponse(
            content=body,
            headers={"X-Router-Tier": winner.tier,
                     "X-Router-Model": winner.model,
                     "X-Router-Score": str(f.score),
                     "X-Router-MoA": "winner",
                     "X-Router-Request-Id": str(request_id)})

    # ---- Frontier 第二层: 级联升级 (便宜档先答, 评审不合格再升级) ----
    cascade_info = None
    if (CASCADE.get("enabled") and not pinned
            and cascade_start_tier(tier) is not None):
        start_tier = cascade_start_tier(tier)
        start_chain = router.candidate_chain(start_tier, None, task, query)
        if start_chain:
            try:
                first = await router.chat(start_chain, payload, task)
                jscore, jreasons = judge_response(first.response, task)
                usage1 = first.usage or {}
                if not usage1.get("prompt_tokens"):
                    usage1["prompt_tokens"] = _estimate_tokens(payload)
                    usage1["completion_tokens"] = 0
                if should_escalate(jscore,
                                   CASCADE.get("judge_threshold", 0.55)):
                    # 不合格: 差评反哺 + 升级到原计划链
                    learner.record(first.model_name, task.get("tags") or [], 0.2)
                    _mem_record(query, first.model_name, 0.2)
                    _record(first, f.score,
                            reasons + f"; 级联升级(评审{jscore}: "
                            f"{'/'.join(jreasons) or '质量不足'})",
                            usage1, "cascade_escalated", query=query)
                    cascade_info = {"started_tier": start_tier,
                                    "first_model": first.model,
                                    "judge_score": jscore,
                                    "escalated": True}
                else:
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
                        "cascade": {"started_tier": start_tier,
                                    "judge_score": jscore,
                                    "escalated": False,
                                    "planned_tier": tier},
                    }
                    if _cacheable(payload, task):
                        cache.store(query, tier, body, first.model)
                    return JSONResponse(
                        content=body,
                        headers={"X-Router-Tier": first.tier,
                                 "X-Router-Model": first.model,
                                 "X-Router-Score": str(f.score),
                                 "X-Router-Cascade": "accept",
                                 "X-Router-Request-Id": str(request_id)})
            except UpstreamError:
                pass  # 便宜档全挂: 落到原计划链

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
    return JSONResponse(
        content=body,
        headers={"X-Router-Tier": result.tier,
                 "X-Router-Model": result.model,
                 "X-Router-Score": str(f.score),
                 "X-Router-Request-Id": str(request_id)})


@app.post("/v1/feedback")
async def feedback(request: Request):
    """显式反馈: 告诉路由器这次回答好不好 (0-1), 调度会越用越准.

    两种用法:
      {"request_id": 123, "score": 0.9}          # 引用某次请求 (推荐)
      {"model": "ds-r1", "score": 0.2, "tags": ["math"]}  # 直接指定
    """
    body = await request.json()
    score = body.get("score")
    if score is None or not (0 <= float(score) <= 1):
        return JSONResponse(status_code=422,
                            content={"error": {"message": "score 必须是 0-1"}})
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


@app.post("/v1/admin/reload")
async def admin_reload():
    global _config_mtime
    _config_mtime = None  # 强制下一次检查触发重载
    _reload_if_changed()
    return {"ok": True, "models": list(cfg["models"].keys())}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


@app.on_event("shutdown")
async def shutdown():
    stats.close()
    learner.close()
    if cache:
        cache.close()
    if memory:
        memory.close()
