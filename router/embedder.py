"""文本向量化: 语义缓存与语义匹配的底座.

两种实现, 自动选择:
  - HashEmbedder: 内置零依赖. 字符 3-gram 哈希映射到固定维度,
    对"近似重复"问题 (语义缓存的主战场) 余弦相似度表现足够好,
    中英混合通用, 单次 <1ms.
  - APIEmbedder: 调用任意 OpenAI 兼容 provider 的 /embeddings
    (text-embedding-3 / bge / voyage 等), 精度更高, 失败自动降级 Hash.
"""

from __future__ import annotations

import hashlib

import numpy as np

EMBED_DIM = 512


class HashEmbedder:
    """字符 n-gram 哈希向量 (零依赖, 本地)."""

    name = "hash-ngram"

    def __init__(self, dim: int = EMBED_DIM, n: int = 3):
        self.dim = dim
        self.n = n

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        text = " ".join(text.lower().split())
        if not text:
            return vec
        for i in range(len(text) - self.n + 1):
            gram = text[i:i + self.n]
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class APIEmbedder:
    """OpenAI 兼容 /embeddings 端点, 失败降级 Hash."""

    name = "api"

    def __init__(self, base_url: str, api_key: str, model: str,
                 fallback: HashEmbedder | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.fallback = fallback or HashEmbedder()

    def embed(self, text: str) -> np.ndarray:
        try:
            import httpx
            resp = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": text[:8000]},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10)
            resp.raise_for_status()
            vec = np.array(resp.json()["data"][0]["embedding"],
                           dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception:
            return self.fallback.embed(text)


def build_embedder(cfg: dict):
    """cache 配置里有 embed 段则用 API, 否则内置 Hash."""
    cache_cfg = cfg.get("cache", {})
    embed_cfg = cache_cfg.get("embed") or {}
    provider_name = embed_cfg.get("provider")
    if provider_name and provider_name in cfg.get("providers", {}):
        from .config import resolve_api_key
        p = cfg["providers"][provider_name]
        return APIEmbedder(p["base_url"], resolve_api_key(p),
                           embed_cfg.get("model", "text-embedding-3-small"))
    return HashEmbedder()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
