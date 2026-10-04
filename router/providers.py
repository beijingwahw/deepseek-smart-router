"""上游协议适配层.

Harness 侧永远是 OpenAI chat/completions 格式;
本层负责把请求/响应/流式事件翻译成各厂商原生协议:

  - openai    : OpenAI 兼容协议, 覆盖 90% 厂商
                (DeepSeek / Qwen / GLM / Kimi / OpenRouter / Groq /
                 Mistral / Together / SiliconFlow / Ollama / vLLM ...)
  - azure     : Azure OpenAI (api-key 头 + api-version + deployment 路径)
  - anthropic : Claude 原生 Messages API (/v1/messages)
  - gemini    : Google 原生 generateContent / streamGenerateContent
"""

from __future__ import annotations

import json
import time
import uuid

from .config import resolve_api_key


# ---------------- OpenAI 格式构造辅助 ----------------

def _chunk(model: str, delta: dict | None = None, content: str | None = None,
           finish: str | None = None, usage: dict | None = None) -> dict:
    d = delta if delta is not None else ({"content": content} if content else {})
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": d, "finish_reason": finish}],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


def _sse(obj: dict | str) -> bytes:
    payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return f"data: {payload}\n\n".encode()


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text")
    return ""


class PassThroughStreamTranslator:
    """OpenAI 原生流: 原样透传, 按行重组为标准 SSE 事件."""

    def feed(self, line: bytes) -> list[bytes]:
        return [line.strip() + b"\n\n"] if line.strip() else []


# ---------------- 基类 ----------------

class BaseAdapter:
    type_name = "openai"

    def __init__(self, provider_cfg: dict, model_cfg: dict):
        self.provider = provider_cfg
        self.model_cfg = model_cfg

    def endpoint(self, stream: bool) -> str:
        return self.provider["base_url"].rstrip("/") + "/chat/completions"

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {resolve_api_key(self.provider)}",
                "Content-Type": "application/json"}

    def translate_request(self, payload: dict) -> dict:
        return payload

    def translate_response(self, data: dict) -> dict:
        return data

    def make_stream_translator(self):
        return PassThroughStreamTranslator()


class OpenAIAdapter(BaseAdapter):
    """OpenAI 兼容协议: 除 Authorization 头外全部原样透传."""


class AzureAdapter(BaseAdapter):
    """Azure OpenAI: api-key 认证 + deployment 路径 + api-version."""

    type_name = "azure"

    def endpoint(self, stream: bool) -> str:
        base = self.provider["base_url"].rstrip("/")
        deployment = self.model_cfg.get("deployment", self.model_cfg["model"])
        version = self.provider.get("api_version", "2024-10-21")
        return f"{base}/openai/deployments/{deployment}/chat/completions?api-version={version}"

    def headers(self) -> dict:
        return {"api-key": resolve_api_key(self.provider),
                "Content-Type": "application/json"}


# ---------------- Anthropic ----------------

_ANTHROPIC_FINISH = {"end_turn": "stop", "max_tokens": "length",
                     "tool_use": "tool_calls", "stop_sequence": "stop"}


class AnthropicAdapter(BaseAdapter):
    type_name = "anthropic"

    def endpoint(self, stream: bool) -> str:
        return self.provider["base_url"].rstrip("/") + "/v1/messages"

    def headers(self) -> dict:
        return {
            "x-api-key": resolve_api_key(self.provider),
            "anthropic-version": self.provider.get("anthropic_version", "2023-06-01"),
            "Content-Type": "application/json",
        }

    # ---- 请求: OpenAI -> Anthropic ----

    def _content_blocks(self, m: dict) -> list[dict]:
        content = m.get("content")
        blocks: list[dict] = []
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    blocks.append({"type": "text", "text": p.get("text", "")})
                elif p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    if url.startswith("data:") and ";base64," in url:
                        media, b64 = url[5:].split(";base64,", 1)
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": media, "data": b64}})
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    inp = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    inp = {}
                blocks.append({"type": "tool_use", "id": tc.get("id", ""),
                               "name": fn.get("name", ""), "input": inp})
        return blocks or [{"type": "text", "text": ""}]

    def translate_request(self, payload: dict) -> dict:
        body = {
            "model": self.model_cfg["model"],
            "max_tokens": (payload.get("max_tokens")
                           or payload.get("max_completion_tokens") or 8192),
            "stream": bool(payload.get("stream")),
        }
        system_parts, messages = [], []
        for m in payload.get("messages", []):
            role = m.get("role")
            if role == "system":
                system_parts.append(_text_of(m.get("content")))
            elif role == "tool":
                messages.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": _text_of(m.get("content"))}]})
            elif role in ("user", "assistant"):
                messages.append({"role": role,
                                 "content": self._content_blocks(m)})
        if system_parts:
            body["system"] = "\n".join(p for p in system_parts if p)
        body["messages"] = messages
        if payload.get("tools"):
            body["tools"] = [{
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"].get("parameters", {"type": "object"}),
            } for t in payload["tools"]]
        for src, dst in (("temperature", "temperature"), ("top_p", "top_p"),
                         ("stop", "stop_sequences")):
            if payload.get(src) is not None:
                body[dst] = payload[src]
        return body

    # ---- 响应: Anthropic -> OpenAI ----

    def translate_response(self, data: dict) -> dict:
        text_parts, tool_calls = [], []
        for b in data.get("content", []):
            if b.get("type") == "text":
                text_parts.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                tool_calls.append({
                    "id": b.get("id", ""), "type": "function",
                    "function": {"name": b.get("name", ""),
                                 "arguments": json.dumps(b.get("input", {}),
                                                         ensure_ascii=False)}})
        msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        usage = data.get("usage", {})
        pt, ct = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        return {
            "id": data.get("id", "chatcmpl-anthropic"),
            "object": "chat.completion", "created": int(time.time()),
            "model": data.get("model", self.model_cfg["model"]),
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": _ANTHROPIC_FINISH.get(
                             data.get("stop_reason"), "stop")}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                      "total_tokens": pt + ct},
        }

    def make_stream_translator(self):
        return AnthropicStreamTranslator(self.model_cfg["model"])


class AnthropicStreamTranslator:
    """Anthropic SSE 事件流 -> OpenAI chunk 流 (有状态)."""

    def __init__(self, model: str):
        self.model = model
        self.in_tokens = 0
        self.out_tokens = 0
        self.finish = "stop"
        self._tool_bufs: dict[int, dict] = {}

    def feed(self, line: bytes) -> list[bytes]:
        line = line.strip()
        if not line.startswith(b"data:"):
            return []
        try:
            ev = json.loads(line[5:].strip())
        except ValueError:
            return []
        t = ev.get("type")
        out: list[bytes] = []
        if t == "message_start":
            self.in_tokens = (ev.get("message", {}).get("usage", {})
                              .get("input_tokens", 0))
        elif t == "content_block_start":
            cb = ev.get("content_block", {})
            if cb.get("type") == "tool_use":
                self._tool_bufs[ev.get("index", 0)] = {
                    "id": cb.get("id", ""), "name": cb.get("name", ""), "json": ""}
        elif t == "content_block_delta":
            idx = ev.get("index", 0)
            d = ev.get("delta", {})
            if d.get("type") == "text_delta" and d.get("text"):
                out.append(_sse(_chunk(self.model, content=d["text"])))
            elif d.get("type") == "input_json_delta" and idx in self._tool_bufs:
                self._tool_bufs[idx]["json"] += d.get("partial_json", "")
        elif t == "content_block_stop":
            buf = self._tool_bufs.pop(ev.get("index", 0), None)
            if buf is not None:
                out.append(_sse(_chunk(self.model, delta={"tool_calls": [{
                    "index": 0, "id": buf["id"], "type": "function",
                    "function": {"name": buf["name"], "arguments": buf["json"]}}]})))
        elif t == "message_delta":
            self.out_tokens = ev.get("usage", {}).get("output_tokens",
                                                      self.out_tokens)
            sr = ev.get("delta", {}).get("stop_reason")
            if sr:
                self.finish = _ANTHROPIC_FINISH.get(sr, "stop")
        elif t == "message_stop":
            usage = {"prompt_tokens": self.in_tokens,
                     "completion_tokens": self.out_tokens,
                     "total_tokens": self.in_tokens + self.out_tokens}
            out.append(_sse(_chunk(self.model, finish=self.finish, usage=usage)))
            out.append(b"data: [DONE]\n\n")
        return out


# ---------------- Gemini ----------------

_GEMINI_FINISH = {"STOP": "stop", "MAX_TOKENS": "length",
                  "SAFETY": "content_filter"}


class GeminiAdapter(BaseAdapter):
    type_name = "gemini"

    def endpoint(self, stream: bool) -> str:
        base = self.provider["base_url"].rstrip("/")
        action = "streamGenerateContent" if stream else "generateContent"
        url = f"{base}/models/{self.model_cfg['model']}:{action}"
        return url + "?alt=sse" if stream else url

    def headers(self) -> dict:
        return {"x-goog-api-key": resolve_api_key(self.provider),
                "Content-Type": "application/json"}

    def translate_request(self, payload: dict) -> dict:
        contents, system = [], []
        for m in payload.get("messages", []):
            role, text = m.get("role"), _text_of(m.get("content"))
            if role == "system":
                system.append(text)
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            elif role == "tool":
                contents.append({"role": "user",
                                 "parts": [{"text": f"[工具返回] {text}"}]})
        body: dict = {"contents": contents}
        if any(system):
            body["systemInstruction"] = {
                "parts": [{"text": "\n".join(system)}]}
        gen = {}
        if payload.get("max_tokens"):
            gen["maxOutputTokens"] = payload["max_tokens"]
        if payload.get("temperature") is not None:
            gen["temperature"] = payload["temperature"]
        if payload.get("top_p") is not None:
            gen["topP"] = payload["top_p"]
        if gen:
            body["generationConfig"] = gen
        if payload.get("tools"):
            body["tools"] = [{"function_declarations": [{
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "parameters": t["function"].get("parameters", {"type": "object"}),
            } for t in payload["tools"]]}]
        return body

    @staticmethod
    def _parse_candidate(data: dict):
        cand = (data.get("candidates") or [{}])[0]
        parts = cand.get("content", {}).get("parts", []) or []
        text = "".join(p.get("text", "") for p in parts)
        calls = [{
            "id": f"call_{i}", "type": "function",
            "function": {"name": p["functionCall"].get("name", ""),
                         "arguments": json.dumps(p["functionCall"].get("args", {}),
                                                 ensure_ascii=False)},
        } for i, p in enumerate(parts) if "functionCall" in p]
        finish = _GEMINI_FINISH.get(cand.get("finishReason"),
                                    "stop" if cand.get("finishReason") else None)
        um = data.get("usageMetadata", {}) or {}
        usage = {"prompt_tokens": um.get("promptTokenCount", 0),
                 "completion_tokens": um.get("candidatesTokenCount", 0),
                 "total_tokens": um.get("totalTokenCount", 0)}
        return text, calls, finish, usage

    def translate_response(self, data: dict) -> dict:
        text, calls, finish, usage = self._parse_candidate(data)
        msg = {"role": "assistant", "content": text or None}
        if calls:
            msg["tool_calls"] = calls
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion", "created": int(time.time()),
            "model": self.model_cfg["model"],
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": finish or "stop"}],
            "usage": usage,
        }

    def make_stream_translator(self):
        return GeminiStreamTranslator(self.model_cfg["model"])


class GeminiStreamTranslator:
    """Gemini SSE (每行一个完整 JSON) -> OpenAI chunk 流."""

    def __init__(self, model: str):
        self.model = model

    def feed(self, line: bytes) -> list[bytes]:
        line = line.strip()
        if not line.startswith(b"data:"):
            return []
        try:
            data = json.loads(line[5:].strip())
        except ValueError:
            return []
        text, calls, finish, usage = GeminiAdapter._parse_candidate(data)
        out: list[bytes] = []
        if text:
            out.append(_sse(_chunk(self.model, content=text)))
        if calls:
            out.append(_sse(_chunk(self.model, delta={"tool_calls": calls})))
        if finish:
            out.append(_sse(_chunk(self.model, finish=finish, usage=usage)))
            out.append(b"data: [DONE]\n\n")
        return out


# ---------------- 工厂 ----------------

ADAPTERS: dict[str, type[BaseAdapter]] = {
    "openai": OpenAIAdapter,
    "azure": AzureAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
}


def build_adapter(provider_cfg: dict, model_cfg: dict) -> BaseAdapter:
    ptype = provider_cfg.get("type", "openai")
    cls = ADAPTERS.get(ptype)
    if cls is None:
        raise ValueError(f"未知 provider 类型: {ptype} "
                         f"(支持: {', '.join(ADAPTERS)})")
    return cls(provider_cfg, model_cfg)
