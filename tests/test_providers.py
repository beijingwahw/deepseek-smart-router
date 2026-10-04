"""协议适配层单元测试: Anthropic / Gemini / Azure 的翻译正确性."""

import json

from router.providers import (AnthropicAdapter, AzureAdapter, GeminiAdapter,
                              build_adapter)

ANTHROPIC = AnthropicAdapter(
    {"type": "anthropic", "base_url": "https://api.anthropic.com",
     "api_key": "sk-ant-test"},
    {"model": "claude-sonnet-4-5"})

GEMINI = GeminiAdapter(
    {"type": "gemini",
     "base_url": "https://generativelanguage.googleapis.com/v1beta",
     "api_key": "gm-test"},
    {"model": "gemini-2.0-flash"})


# ---------- 工厂 ----------

def test_build_adapter_types():
    a = build_adapter({"type": "openai", "base_url": "http://x", "api_key": "k"},
                      {"model": "m"})
    assert a.type_name == "openai"
    a = build_adapter({"type": "azure", "base_url": "http://x", "api_key": "k"},
                      {"model": "m"})
    assert a.type_name == "azure"
    try:
        build_adapter({"type": "unknown"}, {"model": "m"})
        assert False, "应抛异常"
    except ValueError as e:
        assert "未知 provider" in str(e)


# ---------- Azure ----------

def test_azure_endpoint_and_headers():
    a = AzureAdapter({"type": "azure", "base_url": "https://res.openai.azure.com",
                      "api_key": "az-key", "api_version": "2024-10-21"},
                     {"model": "gpt-4o"})
    url = a.endpoint(stream=False)
    assert url == ("https://res.openai.azure.com/openai/deployments/gpt-4o"
                   "/chat/completions?api-version=2024-10-21")
    assert a.headers()["api-key"] == "az-key"


# ---------- Anthropic 请求翻译 ----------

def test_anthropic_headers():
    h = ANTHROPIC.headers()
    assert h["x-api-key"] == "sk-ant-test"
    assert "anthropic-version" in h


def test_anthropic_request_system_and_max_tokens():
    body = ANTHROPIC.translate_request({
        "messages": [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
        ],
        "max_tokens": 1024,
    })
    assert body["system"] == "你是助手"
    assert body["max_tokens"] == 1024
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "你好"}]}]
    assert body["model"] == "claude-sonnet-4-5"


def test_anthropic_request_tools_and_tool_result():
    body = ANTHROPIC.translate_request({
        "messages": [
            {"role": "user", "content": "查天气"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": '{"city": "北京"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "晴 25°C"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "查天气",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}}}}}],
    })
    # assistant 的 tool_calls -> tool_use block
    asst = body["messages"][1]
    assert asst["content"][0]["type"] == "tool_use"
    assert asst["content"][0]["input"] == {"city": "北京"}
    # tool 角色 -> user + tool_result
    tool_msg = body["messages"][2]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"][0]["type"] == "tool_result"
    assert tool_msg["content"][0]["tool_use_id"] == "call_1"
    # tools -> input_schema
    assert body["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"


# ---------- Anthropic 响应翻译 ----------

def test_anthropic_response_text():
    out = ANTHROPIC.translate_response({
        "id": "msg_1", "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "你好"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}})
    choice = out["choices"][0]
    assert choice["message"]["content"] == "你好"
    assert choice["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 5,
                            "total_tokens": 15}


def test_anthropic_response_tool_use():
    out = ANTHROPIC.translate_response({
        "content": [{"type": "tool_use", "id": "tu_1", "name": "get_weather",
                     "input": {"city": "上海"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 3, "output_tokens": 7}})
    msg = out["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"city": "上海"}
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_anthropic_stream_translation():
    tr = ANTHROPIC.make_stream_translator()
    events = [
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":12}}}',
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你"}}'.encode(),
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"好"}}'.encode(),
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}',
        b'data: {"type":"message_stop"}',
    ]
    out = b""
    for e in events:
        for chunk in tr.feed(e):
            out += chunk
    text = out.decode()
    chunks = [json.loads(l[5:]) for l in text.split("\n\n")
              if l.startswith("data:") and "[DONE]" not in l]
    deltas = [c["choices"][0]["delta"].get("content") for c in chunks]
    assert "你" in deltas and "好" in deltas
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["prompt_tokens"] == 12
    assert final["usage"]["completion_tokens"] == 2
    assert "data: [DONE]" in text


def test_anthropic_stream_tool_call_buffered():
    tr = ANTHROPIC.make_stream_translator()
    events = [
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"f"}}',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"a\\":"}}',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"1}"}}',
        b'data: {"type":"content_block_stop","index":0}',
    ]
    out = b""
    for e in events:
        for chunk in tr.feed(e):
            out += chunk
    text = out.decode()
    chunks = [json.loads(l[5:]) for l in text.split("\n\n")
              if l.startswith("data:") and "[DONE]" not in l]
    tc = chunks[-1]["choices"][0]["delta"]["tool_calls"][0]
    # 分片的 JSON 参数被拼成完整的 tool_calls chunk
    assert tc["id"] == "tu_1"
    assert tc["function"]["name"] == "f"
    assert json.loads(tc["function"]["arguments"]) == {"a": 1}


# ---------- Gemini ----------

def test_gemini_endpoints():
    assert GEMINI.endpoint(stream=False).endswith(
        "/models/gemini-2.0-flash:generateContent")
    assert "streamGenerateContent?alt=sse" in GEMINI.endpoint(stream=True)
    assert GEMINI.headers()["x-goog-api-key"] == "gm-test"


def test_gemini_request():
    body = GEMINI.translate_request({
        "messages": [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好呀"},
            {"role": "user", "content": "1+1=?"},
        ],
        "temperature": 0.3, "max_tokens": 100})
    assert body["systemInstruction"]["parts"][0]["text"] == "你是助手"
    assert body["contents"][1]["role"] == "model"  # assistant -> model
    assert body["generationConfig"]["maxOutputTokens"] == 100
    assert body["generationConfig"]["temperature"] == 0.3


def test_gemini_response():
    out = GEMINI.translate_response({
        "candidates": [{"content": {"parts": [{"text": "答案是 2"}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4,
                          "totalTokenCount": 12}})
    assert out["choices"][0]["message"]["content"] == "答案是 2"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 12


def test_gemini_stream_translation():
    tr = GEMINI.make_stream_translator()
    out = b""
    for line in [
        'data: {"candidates":[{"content":{"parts":[{"text":"答"}]}}]}'.encode(),
        'data: {"candidates":[{"content":{"parts":[{"text":"案"}]}}]}'.encode(),
        b'data: {"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],'
        b'"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7}}',
    ]:
        for chunk in tr.feed(line):
            out += chunk
    text = out.decode()
    chunks = [json.loads(l[5:]) for l in text.split("\n\n")
              if l.startswith("data:") and "[DONE]" not in l]
    deltas = [c["choices"][0]["delta"].get("content") for c in chunks]
    assert "答" in deltas and "案" in deltas
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["total_tokens"] == 7
    assert "data: [DONE]" in text


# ---------- OpenAI 透传 ----------

def test_openai_passthrough():
    a = build_adapter({"type": "openai", "base_url": "https://api.deepseek.com/v1",
                       "api_key": "k"}, {"model": "deepseek-chat"})
    payload = {"messages": [{"role": "user", "content": "hi"}], "model": "x"}
    assert a.translate_request(payload) is payload
    assert a.endpoint(False) == "https://api.deepseek.com/v1/chat/completions"
    tr = a.make_stream_translator()
    out = tr.feed(b'data: {"choices":[]}')
    assert out == [b'data: {"choices":[]}\n\n']
