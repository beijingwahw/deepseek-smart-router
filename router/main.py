"""FastAPI 入口 v2: OpenAI 兼容端点 + 多供应商路由 + 成本看板.

启动: uvicorn router.main:app --host 0.0.0.0 --port 8355
Harness 接入: 把 base_url 指向 http://localhost:8355/v1 即可, 无需改其他配置.

请求 model 字段的含义:
  - "auto" / 任意未知名 -> 按难度自动路由
  - 别名 (cheap/fast/smart/reasoning/default) -> 锁定对应难度档
  - 模型池中的名称 (如 "ds-r1") -> 锁定该模型
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .classifier import classify, detect_task_profile, route
from .config import enabled_models, load_config
from .proxy import SmartRouter, UpstreamError
from .stats import Stats, calc_cost

cfg = load_config()
router = SmartRouter(cfg)
stats = Stats(cfg["stats"]["db_path"])
TH = cfg["thresholds"]
ALIASES = cfg.get("aliases", {})

app = FastAPI(title="DeepSeek Smart Router", version="0.2.0")


def _estimate_tokens(payload: dict) -> int:
    """上游未返回 usage 时的粗略估算 (4 字符 ≈ 1 token)."""
    total = sum(len(str(m.get("content", ""))) for m in payload.get("messages", []))
    return max(1, total // 4)


def _ref_price(tier: str) -> dict:
    """该档第一个启用模型的单价, 用作节省基线."""
    for m in enabled_models(cfg).values():
        if m.get("tier") == tier:
            return m["price"]
    return {"input": 0.0, "output": 0.0}


def _record(result, score: int, reasons: str, usage: dict,
            status: str, error: str | None = None) -> None:
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    price = cfg["models"].get(result.model_name, {}).get(
        "price", {"input": 0.0, "output": 0.0})
    stats.record(
        tier=result.tier, model=f"{result.provider}/{result.model}",
        score=score, reasons=reasons, prompt_tokens=pt, completion_tokens=ct,
        cost=calc_cost(price, pt, ct, cache_hit),
        baseline_hard=calc_cost(_ref_price("hard"), pt, ct),
        baseline_standard=calc_cost(_ref_price("standard"), pt, ct),
        latency_ms=result.latency_ms, status=status, error=error,
    )


def _plan(payload: dict) -> tuple[str, str | None, object]:
    """路由计划: (tier, pinned_model, features)."""
    f = classify(payload)
    requested = str(payload.get("model") or "auto")
    if requested in cfg["models"]:
        return cfg["models"][requested].get("tier", "standard"), requested, f
    alias = ALIASES.get(requested)
    if alias and alias != "auto":
        return alias, None, f
    return route(f.score, TH["trivial"], TH["hard"]), None, f


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {n: {"provider": m["provider"], "model": m["model"],
                       "tier": m.get("tier"), "enabled": m.get("enabled", True)}
                   for n, m in cfg["models"].items()},
        "aliases": ALIASES,
        "thresholds": TH,
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI 兼容的模型列表: 暴露别名 + 模型池名称, harness 可直接选择."""
    ids = list(ALIASES.keys()) + list(enabled_models(cfg).keys())
    return {"object": "list", "data": [
        {"id": i, "object": "model", "created": 0, "owned_by": "smart-router"}
        for i in ids]}


@app.post("/v1/route/preview")
async def route_preview(request: Request):
    """干跑模式: 只返回路由计划, 不调用上游 —— 调阈值、看理由都靠它.

    返回任务画像 (类型标签/硬性需求) + 全池适配度报告 (含被过滤模型及原因).
    """
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
    task = detect_task_profile(payload)
    chain = router.candidate_chain(tier, pinned, task)
    reasons = "; ".join(f.reasons) or "无显著信号"

    if payload.get("stream"):
        try:
            result, stream, usage_holder = await router.chat_stream(chain, payload)
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
        result = await router.chat(chain, payload)
    except UpstreamError as e:
        return JSONResponse(status_code=502, content={"error": {"message": str(e)}})

    usage = result.usage or {}
    if not usage.get("prompt_tokens"):
        usage["prompt_tokens"] = _estimate_tokens(payload)
        usage["completion_tokens"] = usage.get("completion_tokens", 0)
    _record(result, f.score, reasons, usage, result.status)

    body = dict(result.response or {})
    body["router"] = {
        "tier": result.tier, "model_name": result.model_name,
        "model": result.model, "provider": result.provider,
        "score": f.score, "reasons": f.reasons,
        "status": result.status, "fell_back_from": result.fell_back_from,
    }
    return JSONResponse(
        content=body,
        headers={"X-Router-Tier": result.tier,
                 "X-Router-Model": result.model,
                 "X-Router-Score": str(f.score)})


@app.get("/v1/stats")
async def get_stats():
    return stats.summary()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


@app.on_event("shutdown")
async def shutdown():
    stats.close()
