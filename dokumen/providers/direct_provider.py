"""Optional direct provider adapter for non-SDK execution paths.

The default executor and judge runtime uses the Claude Agent SDK. This adapter
keeps older direct provider flows available without making them part of the
primary public CLI story.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ..agent_object import Provider
from ..config import DEFAULT_FAST_MODEL
from ..logging_config import get_logger
from .retry import retry_with_exponential_backoff

logger = get_logger(__name__)

# provider → api key env var
PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
}

# provider → base url
PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
}

# providers that use openai-compatible chat/completions format
OPENAI_COMPATIBLE_PROVIDERS = {
    "openai",
    "mistral",
    "deepseek",
    "groq",
    "together",
    "custom",
}


def _strip_provider_prefix(model: str) -> str:
    """remove provider/ prefix if present. 'openai/gpt-4o' → 'gpt-4o'."""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def _infer_provider(model: str) -> str:
    """infer provider from model name."""
    if "/" in model:
        return model.split("/", 1)[0].lower()
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if model.startswith("gemini-"):
        return "gemini"
    if model.startswith("mistral-") or model.startswith("codestral"):
        return "mistral"
    if model.startswith("deepseek"):
        return "deepseek"
    return "custom"


class DirectProviderRouter(Provider):
    """Direct provider adapter. Uses provider APIs through httpx.

    usage:
        # explicit provider
        p = DirectProviderRouter(model="gpt-4o", provider_name="openai", api_key="sk-...")

        # inferred from model name
        p = DirectProviderRouter(model="openai/gpt-4o")

        # custom endpoint (any openai-compatible API)
        p = DirectProviderRouter(model="my-model", api_base="http://localhost:8080/v1", api_key="test")
    """

    def __init__(
        self,
        model: str = None,
        provider_name: str = None,
        api_key: str = None,
        api_base: str = None,
        **kwargs,
    ):
        raw_model = model or DEFAULT_FAST_MODEL
        self.provider_name = (provider_name or _infer_provider(raw_model)).lower()
        self.model = _strip_provider_prefix(raw_model)
        self.api_key = api_key
        self.api_base = api_base
        self.extra_kwargs = kwargs

        # resolve api key from env if not provided
        if not self.api_key:
            env_var = PROVIDER_API_KEY_ENV.get(self.provider_name)
            if env_var:
                self.api_key = os.environ.get(env_var)

        # resolve base url
        if not self.api_base:
            self.api_base = PROVIDER_BASE_URLS.get(self.provider_name)

        logger.info(
            "direct_provider.init",
            model=self.model,
            provider=self.provider_name,
            has_api_key=bool(self.api_key),
            has_api_base=bool(self.api_base),
        )

    async def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Route to the appropriate external provider API."""
        deadline = kwargs.pop("deadline", None)
        start = time.time()

        logger.info(
            "direct_provider.complete.entry",
            model=self.model,
            provider=self.provider_name,
            message_count=len(messages),
            has_tools=bool(tools),
        )

        try:
            if self.provider_name == "gemini" or self.provider_name == "google":
                result = await retry_with_exponential_backoff(
                    self._complete_gemini, messages, tools, deadline=deadline, **kwargs
                )
            elif self.provider_name in OPENAI_COMPATIBLE_PROVIDERS:
                result = await retry_with_exponential_backoff(
                    self._complete_openai, messages, tools, deadline=deadline, **kwargs
                )
            else:
                # fallback: try openai-compatible format
                result = await retry_with_exponential_backoff(
                    self._complete_openai, messages, tools, deadline=deadline, **kwargs
                )

            duration = time.time() - start
            logger.info(
                "direct_provider.complete.success",
                model=self.model,
                provider=self.provider_name,
                duration_ms=int(duration * 1000),
                input_tokens=result.get("usage", {}).get("input_tokens", 0),
                output_tokens=result.get("usage", {}).get("output_tokens", 0),
            )
            return result

        except Exception as e:
            duration = time.time() - start
            logger.error(
                "direct_provider.complete.error",
                model=self.model,
                provider=self.provider_name,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=int(duration * 1000),
            )
            raise

    # ── openai-compatible provider ─────────────────────────────────

    async def _complete_openai(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """openai chat/completions API (also works for groq, together, mistral, deepseek, custom)."""
        url = f"{self.api_base}/chat/completions"

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": self._prepare_messages_openai(messages),
            "max_tokens": kwargs.get("max_tokens", 16384),
        }

        if tools:
            formatted = self._format_tools_openai(tools)
            if formatted:
                body["tools"] = formatted

        for param in ("temperature", "top_p", "stop", "seed"):
            if param in kwargs:
                body[param] = kwargs[param]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return self._normalize_openai_response(data)

    def _prepare_messages_openai(self, messages: List[Dict]) -> List[Dict]:
        """prepare messages for openai format."""
        prepared = []
        for msg in messages:
            role = msg.get("role")

            if role == "tool":
                prepared.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                prepared.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content") or None,
                        "tool_calls": [
                            {
                                "id": tc.get("id", tc.get("name", "")),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name"),
                                    "arguments": (
                                        tc.get("arguments")
                                        if isinstance(tc.get("arguments"), str)
                                        else json.dumps(tc.get("arguments", {}))
                                    ),
                                },
                            }
                            for tc in msg["tool_calls"]
                        ],
                    }
                )
            else:
                prepared.append(
                    {
                        "role": role,
                        "content": msg.get("content", ""),
                    }
                )
        return prepared

    def _format_tools_openai(self, tools: List[Dict]) -> List[Dict]:
        """format tools for openai api."""
        formatted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                params = func.get("parameters", {"type": "object", "properties": {}})
                if params.get("_server_side"):
                    continue
                formatted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": func["name"],
                            "description": func.get("description", ""),
                            "parameters": params,
                        },
                    }
                )
        return formatted

    def _normalize_openai_response(self, data: Dict) -> Dict[str, Any]:
        """normalize openai response to internal format."""
        result: Dict[str, Any] = {"content": ""}

        choices = data.get("choices", [])
        if not choices:
            return result

        choice = choices[0]
        message = choice.get("message", {})

        if message.get("content"):
            result["content"] = message["content"]

        # tool calls
        if message.get("tool_calls"):
            tool_use = []
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                arguments = func.get("arguments", "{}")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"raw": arguments}
                tool_use.append(
                    {
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": arguments,
                    }
                )
            if tool_use:
                result["tool_use"] = tool_use

        # stop reason mapping
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            reason_map = {
                "stop": "end_turn",
                "tool_calls": "tool_use",
                "length": "max_tokens",
            }
            result["stop_reason"] = reason_map.get(finish_reason, finish_reason)

        # usage
        usage = data.get("usage", {})
        if usage:
            result["usage"] = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
            }

        return result

    # ── gemini provider ────────────────────────────────────────────

    async def _complete_gemini(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """google gemini generateContent API."""
        url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"

        contents, system_instruction = self._prepare_messages_gemini(messages)

        body: Dict[str, Any] = {"contents": contents}

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        # generation config
        gen_config: Dict[str, Any] = {}
        max_tokens = kwargs.get("max_tokens", 16384)
        if max_tokens:
            gen_config["maxOutputTokens"] = max_tokens
        if "temperature" in kwargs:
            gen_config["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            gen_config["topP"] = kwargs["top_p"]
        if gen_config:
            body["generationConfig"] = gen_config

        # tools
        if tools:
            formatted = self._format_tools_gemini(tools)
            if formatted:
                body["tools"] = [{"functionDeclarations": formatted}]

        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return self._normalize_gemini_response(data)

    def _prepare_messages_gemini(self, messages: List[Dict]):
        """convert messages to gemini format.

        gemini uses:
        - systemInstruction (separate from contents)
        - contents: [{role: "user"|"model", parts: [{text: "..."}]}]
        - function calls: parts with functionCall/functionResponse
        """
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                parts = []
                if content:
                    parts.append({"text": content})
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        args = tc.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                args = {"raw": args}
                        parts.append(
                            {
                                "functionCall": {
                                    "name": tc.get("name", ""),
                                    "args": args,
                                }
                            }
                        )
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.get("tool_call_id", ""),
                                    "response": {"result": content},
                                }
                            }
                        ],
                    }
                )

        return contents, system_instruction

    def _format_tools_gemini(self, tools: List[Dict]) -> List[Dict]:
        """convert openai tool format to gemini functionDeclarations."""
        declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                params = func.get("parameters", {"type": "object", "properties": {}})
                if params.get("_server_side"):
                    continue
                decl: Dict[str, Any] = {
                    "name": func["name"],
                    "description": func.get("description", ""),
                }
                # gemini doesn't want the top-level "type": "object" wrapper the same way
                if params.get("properties"):
                    decl["parameters"] = params
                declarations.append(decl)
        return declarations

    def _normalize_gemini_response(self, data: Dict) -> Dict[str, Any]:
        """normalize gemini response to internal format."""
        result: Dict[str, Any] = {"content": ""}

        candidates = data.get("candidates", [])
        if not candidates:
            return result

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])

        text_parts = []
        tool_use = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_use.append(
                    {
                        "id": fc.get("name", ""),
                        "name": fc.get("name", ""),
                        "input": fc.get("args", {}),
                    }
                )

        if text_parts:
            result["content"] = "\n".join(text_parts)

        if tool_use:
            result["tool_use"] = tool_use

        # stop reason
        finish_reason = candidate.get("finishReason", "")
        reason_map = {
            "STOP": "end_turn",
            "MAX_TOKENS": "max_tokens",
            "SAFETY": "end_turn",
        }
        result["stop_reason"] = reason_map.get(finish_reason, "end_turn")

        # usage
        usage_meta = data.get("usageMetadata", {})
        if usage_meta:
            result["usage"] = {
                "input_tokens": usage_meta.get("promptTokenCount", 0),
                "output_tokens": usage_meta.get("candidatesTokenCount", 0),
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
            }

        return result


# ── embedding support ──────────────────────────────────────────


async def embed_text(
    texts: List[str],
    model: str = "text-embedding-004",
    provider: str = "gemini",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> List[List[float]]:
    """embed text using provider APIs directly. no litellm.

    supports:
    - gemini: text-embedding-004, gemini-embedding-2-preview
    - openai: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    """
    # strip provider prefix
    if "/" in model:
        provider, model = model.split("/", 1)
    provider = provider.lower()

    api_key = api_key or os.environ.get(PROVIDER_API_KEY_ENV.get(provider, ""))

    if provider in ("gemini", "google"):
        return await _embed_gemini(texts, model, api_key, api_base)
    elif provider == "openai":
        return await _embed_openai(texts, model, api_key, api_base)
    else:
        # try openai-compatible
        return await _embed_openai(
            texts, model, api_key, api_base or PROVIDER_BASE_URLS.get(provider)
        )


async def _embed_gemini(
    texts: List[str],
    model: str,
    api_key: str,
    api_base: Optional[str] = None,
) -> List[List[float]]:
    """gemini embedding API."""
    base = api_base or "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base}/models/{model}:batchEmbedContents?key={api_key}"

    body = {
        "requests": [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in texts
        ]
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=body, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()

    return [emb["values"] for emb in data.get("embeddings", [])]


async def _embed_openai(
    texts: List[str],
    model: str,
    api_key: str,
    api_base: Optional[str] = None,
) -> List[List[float]]:
    """openai-compatible embedding API."""
    base = api_base or "https://api.openai.com/v1"
    url = f"{base}/embeddings"

    body = {"model": model, "input": texts}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    return [item["embedding"] for item in data.get("data", [])]


# ── completion helper for memory/skills (non-Provider interface) ──


async def direct_provider_completion(
    messages: List[Dict[str, str]],
    model: str = "gemini/gemini-2.0-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """simple completion call for internal use (memory extraction, skill extraction, etc.).

    returns the text content of the response. no tool calling support needed.
    """
    provider_name = _infer_provider(model)
    bare_model = _strip_provider_prefix(model)
    key = api_key or os.environ.get(PROVIDER_API_KEY_ENV.get(provider_name, ""), "")

    router = DirectProviderRouter(
        model=bare_model,
        provider_name=provider_name,
        api_key=key,
    )

    result = await router.complete(messages, temperature=temperature)
    return result.get("content", "")
