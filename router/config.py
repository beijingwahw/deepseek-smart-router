"""配置加载: YAML + 默认值 (v2 多供应商模型池)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "server": {"host": "0.0.0.0", "port": 8355},
    "thresholds": {"trivial": 25, "hard": 60},
    "routing": {
        # 同档内多模型的挑选策略:
        # best_fit(按任务类型x模型能力画像, 推荐) / priority / cheapest / round_robin
        "strategy": "best_fit",
        "fallback_enabled": True,
        # 某档全部不可用时的跨档降级顺序
        "cross_tier_fallback": {
            "hard": ["standard"],
            "standard": ["hard"],
            "trivial": ["standard"],
        },
        "circuit_breaker": {"failure_threshold": 3, "cooldown_seconds": 60},
    },
    "providers": {
        "deepseek": {
            "type": "openai",
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "ollama": {
            "type": "openai",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        },
    },
    # 模型池: key 为内部名称, tier 决定难度档, priority 同档内越小越优先
    "models": {
        "local-qwen": {
            "provider": "ollama", "model": "qwen2.5-coder:7b",
            "tier": "trivial", "priority": 1, "timeout": 60,
            "enabled": True, "price": {"input": 0.0, "output": 0.0},
        },
        "ds-v3": {
            "provider": "deepseek", "model": "deepseek-chat",
            "tier": "standard", "priority": 1, "timeout": 120,
            "enabled": True,
            "price": {"input": 0.27, "output": 1.10, "cache_hit_input": 0.07},
        },
        "ds-r1": {
            "provider": "deepseek", "model": "deepseek-reasoner",
            "tier": "hard", "priority": 1, "timeout": 300,
            "enabled": True,
            "price": {"input": 0.55, "output": 2.19, "cache_hit_input": 0.14},
        },
    },
    # 请求 model 字段的别名: auto=按难度路由, 其余直接锁定档位
    "aliases": {
        "auto": "auto",
        "cheap": "trivial", "fast": "trivial",
        "smart": "hard", "reasoning": "hard",
        "default": "standard",
    },
    "stats": {"db_path": "router_stats.db"},
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None) -> dict:
    """加载配置文件, 缺失时用默认值. 环境变量 ROUTER_CONFIG 可指定路径.

    始终返回深拷贝, 调用方修改不会污染全局默认配置.
    """
    path = path or os.environ.get("ROUTER_CONFIG")
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        return _merge(copy.deepcopy(DEFAULT_CONFIG), user_cfg)
    return copy.deepcopy(DEFAULT_CONFIG)


def resolve_api_key(provider_cfg: dict) -> str:
    env = provider_cfg.get("api_key_env") or ""
    if env:
        return os.environ.get(env, "")
    return provider_cfg.get("api_key", "")


def enabled_models(cfg: dict) -> dict[str, dict]:
    """所有 enabled 的模型 (保持配置书写顺序)."""
    return {n: m for n, m in cfg["models"].items() if m.get("enabled", True)}
