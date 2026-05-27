#!/usr/bin/env python3
"""standalone tests for dokurouter (in-house LLM gateway).

run: python3 tests/scripts/test_dokurouter.py
"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dokumen.providers.dokurouter import (
    DokuRouter,
    _strip_provider_prefix,
    _infer_provider,
    PROVIDER_API_KEY_ENV,
    PROVIDER_BASE_URLS,
)
from dokumen.config import ProviderConfig

passed = 0
failed = 0


def test(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}")


print("── provider prefix stripping ──")
test("strip openai/gpt-4o", _strip_provider_prefix("openai/gpt-4o") == "gpt-4o")
test("strip gemini/gemini-2.5-pro", _strip_provider_prefix("gemini/gemini-2.5-pro") == "gemini-2.5-pro")
test("no prefix passthrough", _strip_provider_prefix("gpt-4o") == "gpt-4o")
test("nested prefix", _strip_provider_prefix("a/b/c") == "b/c")

print("\n── provider inference ──")
test("claude → anthropic", _infer_provider("claude-sonnet-4-6") == "anthropic")
test("gpt → openai", _infer_provider("gpt-4o") == "openai")
test("o1 → openai", _infer_provider("o1-mini") == "openai")
test("o3 → openai", _infer_provider("o3-mini") == "openai")
test("gemini → gemini", _infer_provider("gemini-2.5-pro") == "gemini")
test("mistral → mistral", _infer_provider("mistral-large") == "mistral")
test("deepseek → deepseek", _infer_provider("deepseek-chat") == "deepseek")
test("explicit prefix", _infer_provider("openai/gpt-4o") == "openai")
test("unknown → custom", _infer_provider("llama-3-70b") == "custom")

print("\n── provider init ──")
p = DokuRouter(model="openai/gpt-4o", api_key="test")
test("model stored (stripped)", p.model == "gpt-4o")
test("provider inferred", p.provider_name == "openai")
test("api key stored", p.api_key == "test")

p2 = DokuRouter(model="gpt-4o", provider_name="openai")
test("provider_name explicit", p2.provider_name == "openai")
test("model stored", p2.model == "gpt-4o")

p3 = DokuRouter(model="gemini-2.5-pro")
test("gemini inferred", p3.provider_name == "gemini")
test("gemini base url", p3.api_base == PROVIDER_BASE_URLS["gemini"])

p4 = DokuRouter(api_base="http://localhost:8080/v1", model="my-model")
test("custom api_base", p4.api_base == "http://localhost:8080/v1")

print("\n── message preparation (openai format) ──")
provider = DokuRouter(model="gpt-4o", provider_name="openai", api_key="test")

msgs = [{"role": "user", "content": "hello"}]
result = provider._prepare_messages_openai(msgs)
test("simple user message", result == [{"role": "user", "content": "hello"}])

msgs = [{"role": "system", "content": "be helpful"}]
result = provider._prepare_messages_openai(msgs)
test("system message", result == [{"role": "system", "content": "be helpful"}])

msgs = [{"role": "tool", "tool_call_id": "c1", "content": "data"}]
result = provider._prepare_messages_openai(msgs)
test("tool message role", result[0]["role"] == "tool")
test("tool message id", result[0]["tool_call_id"] == "c1")

msgs = [{
    "role": "assistant",
    "content": "reading file",
    "tool_calls": [{"id": "c1", "name": "read_file", "arguments": {"path": "x.py"}}],
}]
result = provider._prepare_messages_openai(msgs)
tc = result[0]["tool_calls"][0]
test("tool call type", tc["type"] == "function")
test("tool call name", tc["function"]["name"] == "read_file")
test("tool call args", json.loads(tc["function"]["arguments"]) == {"path": "x.py"})

print("\n── tool formatting (openai) ──")
tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
result = provider._format_tools_openai(tools)
test("standard tool passes through", len(result) == 1)

server_tools = [{
    "type": "function",
    "function": {"name": "web_search", "parameters": {"_server_side": True}},
}]
result = provider._format_tools_openai(server_tools)
test("server-side tool filtered", len(result) == 0)

print("\n── response normalization (openai) ──")

# text response
data = {
    "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
}
r = provider._normalize_openai_response(data)
test("text content", r["content"] == "hello")
test("stop reason", r["stop_reason"] == "end_turn")
test("usage input", r["usage"]["input_tokens"] == 100)

# tool call response
data = {
    "choices": [{
        "message": {
            "content": None,
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "x.py"}'},
            }],
        },
        "finish_reason": "tool_calls",
    }],
    "usage": {"prompt_tokens": 50, "completion_tokens": 20},
}
r = provider._normalize_openai_response(data)
test("tool use extracted", len(r["tool_use"]) == 1)
test("tool use name", r["tool_use"][0]["name"] == "read_file")
test("tool use stop_reason", r["stop_reason"] == "tool_use")

# length → max_tokens
data = {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}], "usage": {}}
r = provider._normalize_openai_response(data)
test("length → max_tokens", r["stop_reason"] == "max_tokens")

# malformed args
data = {
    "choices": [{
        "message": {
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "t", "arguments": "not json"}}],
        },
        "finish_reason": "stop",
    }],
    "usage": {},
}
r = provider._normalize_openai_response(data)
test("malformed args wrapped", r["tool_use"][0]["input"] == {"raw": "not json"})

# empty response
r = provider._normalize_openai_response({"choices": [], "usage": {}})
test("empty response", r["content"] == "")

print("\n── gemini message preparation ──")
gp = DokuRouter(model="gemini-2.5-pro", api_key="test")

msgs = [
    {"role": "system", "content": "be helpful"},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
]
contents, sys_inst = gp._prepare_messages_gemini(msgs)
test("system extracted", sys_inst == "be helpful")
test("2 contents", len(contents) == 2)
test("user role", contents[0]["role"] == "user")
test("model role", contents[1]["role"] == "model")

# tool call in assistant message
msgs = [{
    "role": "assistant",
    "content": "",
    "tool_calls": [{"name": "read", "arguments": {"p": "x"}}],
}]
contents, _ = gp._prepare_messages_gemini(msgs)
test("gemini function call", "functionCall" in contents[0]["parts"][0])

# tool result
msgs = [{"role": "tool", "tool_call_id": "read", "content": "file data"}]
contents, _ = gp._prepare_messages_gemini(msgs)
test("gemini function response", "functionResponse" in contents[0]["parts"][0])

print("\n── gemini tool formatting ──")
tools = [{"type": "function", "function": {"name": "read_file", "description": "reads a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
result = gp._format_tools_gemini(tools)
test("gemini tool formatted", len(result) == 1)
test("gemini tool name", result[0]["name"] == "read_file")
test("gemini tool has params", "parameters" in result[0])

print("\n── gemini response normalization ──")
data = {
    "candidates": [{
        "content": {"parts": [{"text": "hello from gemini"}]},
        "finishReason": "STOP",
    }],
    "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10},
}
r = gp._normalize_gemini_response(data)
test("gemini text", r["content"] == "hello from gemini")
test("gemini stop reason", r["stop_reason"] == "end_turn")
test("gemini usage", r["usage"]["input_tokens"] == 50)

# function call response
data = {
    "candidates": [{
        "content": {"parts": [{"functionCall": {"name": "read", "args": {"path": "x"}}}]},
        "finishReason": "STOP",
    }],
    "usageMetadata": {},
}
r = gp._normalize_gemini_response(data)
test("gemini tool use", len(r["tool_use"]) == 1)
test("gemini tool name", r["tool_use"][0]["name"] == "read")

# empty
r = gp._normalize_gemini_response({"candidates": []})
test("gemini empty", r["content"] == "")

print("\n── create_provider routing ──")
from dokumen.test_builder import create_provider
from dokumen.providers.anthropic import AnthropicProvider

p = create_provider("anthropic", api_key="sk-test", model="claude-haiku-4-5-20251001")
test("anthropic uses native", isinstance(p, AnthropicProvider))

p = create_provider("openai", api_key="sk-test", model="gpt-4o")
test("openai uses dokurouter", isinstance(p, DokuRouter))
test("openai model stripped", p.model == "gpt-4o")

p = create_provider("google", api_key="test", model="gemini-2.5-pro")
test("google uses dokurouter", isinstance(p, DokuRouter))
test("google provider name", p.provider_name == "google")

p = create_provider("custom", api_key="test", model="my-model", api_base="http://localhost:8080/v1")
test("custom uses dokurouter", isinstance(p, DokuRouter))
test("custom api_base", p.api_base == "http://localhost:8080/v1")

test("none returns none", create_provider(None) is None)
test("empty returns none", create_provider("") is None)

print("\n── config provider names ──")
for name in ["openai", "google", "gemini", "mistral", "deepseek", "groq", "together", "bedrock", "vertex", "custom"]:
    c = ProviderConfig(name=name)
    test(f"{name} accepted", c.name == name)

try:
    ProviderConfig(name="litellm")
    test("litellm rejected", False)
except Exception:
    test("litellm rejected", True)

try:
    ProviderConfig(name="invalid_xyz")
    test("invalid rejected", False)
except Exception:
    test("invalid rejected", True)

c = ProviderConfig(name="openai", api_base="http://localhost:8080")
test("api_base field works", c.api_base == "http://localhost:8080")

print(f"\n{'='*50}")
print(f"dokurouter: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
