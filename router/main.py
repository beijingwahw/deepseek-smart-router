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
from .classifier import classify, detect_task_profile, route
from .config import enabled_models, load_config
from .learner import Learner
from .proxy import SmartRouter, UpstreamError
from .stats import Stats, calc_cost

CONFIG_PATH = os.environ.get("ROUTER_CONFIG")

# ---------------- 全局状态 (热重载时整体刷新) ----------------

cfg = load_config()
stats = Stats(cfg["stats"]["db_path"])
learner = Learner(cfg["stats"]["db_path"])
router = SmartRouter(cfg, learner=learner)
TH = cfg["thresholds"]
ALIASES = cfg.get("aliases", {})
BUDGET = cfg.get("budget", {})
_config_mtime: float | None = None

app = FastAPI(title="DeepSeek Smart Router", version="1.0.0")


def _reload_if_changed() -> None:
    """配置文件变更时热重载: 模型池/阈值/预算立即生效, 学习成果保留."""
    global cfg, router, TH, ALIASES, BUDGET, _config_mtime
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
    router = SmartRouter(cfg, learner=learner)
    router.disabled = disabled
    TH = cfg["thresholds"]
    ALIASES = cfg.get("aliases", {})
    BUDGET = cfg.get("budget", {})
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
            status: str, error: str | None = None) -> int:
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
    )


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
        "status": "ok", "version": "1.0.0-genesis",
        "models": {n: {"provider": m["provider"], "model": m["model"],
                       "tier": m.get("tier"), "enabled": m.get("enabled", True),
                       "disabled_at_runtime": n in router.disabled}
                   for n, m in cfg["models"].items()},
        "aliases": ALIASES, "thresholds": TH,
        "budget": {**BUDGET, "spent_today": round(stats.daily_spend(), 4)},
        "learner": learner.summary(),
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
    chain = router.candidate_chain(tier, pinned, task)
    return {
        "score": f.score, "tier": tier, "pinned": pinned,
        "task_profile": task,
        "candidates": [r for r in router.suitability_report(tier, task)
                       if r["name"] in chain],
        "forced": f.forced, "reasons": f.reasons,
    }


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
    chain = router.candidate_chain(tier, pinned, task)
    reasons = "; ".join(f.reasons) or "无显著信号"

    if payload.get("stream"):
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
            _record(result, f.score, reasons, usage_holder, result.status)

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"X-Router-Tier": result.tier,
                     "X-Router-Model": result.model,
                     "X-Router-Score": str(f.score)})

    try:
        result = await router.chat(chain, payload, task)
    except UpstreamError as e:
        return JSONResponse(status_code=502, content={"error": {"message": str(e)}})

    usage = result.usage or {}
    if not usage.get("prompt_tokens"):
        usage["prompt_tokens"] = _estimate_tokens(payload)
        usage["completion_tokens"] = usage.get("completion_tokens", 0)
    request_id = _record(result, f.score, reasons, usage, result.status)

    body = dict(result.response or {})
    body["router"] = {
        "request_id": request_id,
        "tier": result.tier, "model_name": result.model_name,
        "model": result.model, "provider": result.provider,
        "score": f.score, "reasons": f.reasons,
        "status": result.status, "fell_back_from": result.fell_back_from,
    }
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
    return {"ok": True, "model": model_name, "tags": tags or ["*"],
            "score": float(score), "learner": learner.summary().get(model_name)}


@app.get("/v1/stats")
async def get_stats():
    d = stats.summary()
    d["learner"] = learner.summary()
    d["latency"] = stats.latency_by_model()
    d["budget"] = {**BUDGET, "spent_today": round(stats.daily_spend(), 4)}
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
