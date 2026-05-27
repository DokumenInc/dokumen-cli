"""
Anthropic Provider for the Skill Testing Framework.
"""

from typing import Any, Dict, List, Optional
import os

from ..agent_object import Provider
from ..config import DEFAULT_FAST_MODEL
from ..logging_config import get_logger
from .retry import retry_with_exponential_backoff

logger = get_logger(__name__)


class AnthropicProvider(Provider):
    """
    Anthropic Claude provider for real LLM calls.
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. If not provided, loads from 1Password
                     (if USE_1PASSWORD=true) or ANTHROPIC_API_KEY env var.
            model: Model to use (default: claude-haiku-4-5-20251001)
        """
        if api_key:
            self.api_key = api_key
        else:
            # Try loading from 1Password or env var via secrets module
            try:
                from ..secrets import get_anthropic_key

                self.api_key = get_anthropic_key()
            except (ImportError, ValueError):
                # Fallback to direct env var check
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")

        self.model = model or DEFAULT_FAST_MODEL
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import anthropic
            import httpx

            # Set a reasonable timeout (60 seconds connect, 120 seconds read)
            timeout = httpx.Timeout(60.0, read=120.0)
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key, timeout=timeout)
            # Security: Don't log any part of the API key, even prefixes
            # Truncated keys can still aid attackers in credential stuffing
            logger.debug("anthropic.client.init", has_api_key=bool(self.api_key), model=self.model)
        return self._client

    async def complete(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Send a completion request to Anthropic Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions (OpenAI format)
            **kwargs: Additional parameters

        Returns:
            Response dict with 'content' and optionally 'tool_use'
        """
        client = self._get_client()

        logger.info(
            "anthropic.complete.entry",
            model=self.model,
            message_count=len(messages),
            has_tools=bool(tools),
            max_tokens=kwargs.get("max_tokens", 16384),
        )

        # Extract system message and convert messages to Anthropic format
        # system_prompt can be passed as kwarg or as a system-role message
        system_prompt = kwargs.get("system_prompt", "")
        anthropic_messages = []

        # Track pending tool results to batch them
        pending_tool_results = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "tool":
                # Anthropic uses tool_result content blocks in a user message
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg["content"],
                    }
                )
            else:
                # Flush any pending tool results before adding non-tool message
                if pending_tool_results:
                    anthropic_messages.append({"role": "user", "content": pending_tool_results})
                    pending_tool_results = []

                # Handle assistant messages with tool_calls
                if msg["role"] == "assistant" and msg.get("tool_calls"):
                    # Convert to Anthropic format with tool_use content blocks
                    content_blocks = []
                    if msg.get("content"):
                        content_blocks.append({"type": "text", "text": msg["content"]})
                    for tc in msg["tool_calls"]:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", tc.get("name", "")),
                                "name": tc.get("name"),
                                "input": tc.get("arguments", {}),
                            }
                        )
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        # Flush any remaining tool results
        if pending_tool_results:
            anthropic_messages.append({"role": "user", "content": pending_tool_results})

        # Convert tools from OpenAI format to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = self._format_tools_for_api(tools)

        # Build request kwargs
        request_kwargs = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 64000),
            "messages": anthropic_messages,
        }
        if system_prompt:
            request_kwargs["system"] = system_prompt
        if anthropic_tools:
            request_kwargs["tools"] = anthropic_tools

        # structured outputs — pass output_config through to the API
        output_config = kwargs.get("output_config")
        if output_config:
            request_kwargs["output_config"] = output_config

        logger.debug(
            "anthropic.request.start",
            model=self.model,
            messages_count=len(anthropic_messages),
            tools_count=len(anthropic_tools) if anthropic_tools else 0,
        )

        # Extract deadline before building request (not an Anthropic API param)
        deadline = kwargs.pop("deadline", None)

        # Make the API call with retry on rate limit
        import asyncio
        import time as time_module

        start_time = time_module.time()
        try:
            logger.debug("anthropic.request.calling", model=self.model)
            # per-call timeout — 10min to handle large codebases and complex tool responses
            response = await asyncio.wait_for(
                retry_with_exponential_backoff(
                    client.messages.create,
                    **request_kwargs,
                    deadline=deadline,
                ),
                timeout=600.0,
            )
            duration = time_module.time() - start_time
            logger.info(
                "anthropic.request.complete",
                model=self.model,
                duration_ms=int(duration * 1000),
                input_tokens=getattr(response.usage, "input_tokens", 0),
                output_tokens=getattr(response.usage, "output_tokens", 0),
            )
        except asyncio.TimeoutError:
            duration = time_module.time() - start_time
            logger.error(
                "anthropic.request.timeout",
                model=self.model,
                timeout=600,
                duration_ms=int(duration * 1000),
            )
            raise
        except Exception as e:
            duration = time_module.time() - start_time
            logger.error(
                "anthropic.request.error",
                model=self.model,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=int(duration * 1000),
            )
            raise

        # Convert response to normalized format
        result = self._normalize_response(response)
        content_block_types = [block.type for block in response.content] if response.content else []
        logger.info(
            "anthropic.complete.response",
            model=self.model,
            stop_reason=getattr(response, "stop_reason", None),
            content_block_types=content_block_types,
            content_length=len(result.get("content", "")),
            tool_calls_count=len(result.get("tool_use", [])),
        )
        return result

    def _format_tools_for_api(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools from OpenAI/internal format to Anthropic API format.

        Server-side tools (with _server_tool_config) are formatted as
        web_search_20250305 type. Regular function tools use the standard
        Anthropic tool format.

        Args:
            tools: List of tool dicts in OpenAI format

        Returns:
            List of tool dicts in Anthropic API format
        """
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                # openai format: {"type": "function", "function": {"name": ..., "parameters": ...}}
                func = tool["function"]
                params = func.get("parameters", {"type": "object", "properties": {}})

                # Detect server-side tools
                if params.get("_server_side"):
                    server_config = tool.get("_server_tool_config", {})
                    server_tool: Dict[str, Any] = {
                        "type": server_config.get("type", "web_search_20250305"),
                        "name": "web_search",
                        "max_uses": server_config.get("max_uses", 20),
                    }
                    if "allowed_domains" in server_config:
                        server_tool["allowed_domains"] = server_config["allowed_domains"]
                    if "blocked_domains" in server_config:
                        server_tool["blocked_domains"] = server_config["blocked_domains"]
                    logger.info(
                        "anthropic.tool.server_side",
                        tool_name=func["name"],
                        server_type=server_tool["type"],
                        max_uses=server_tool["max_uses"],
                    )
                    anthropic_tools.append(server_tool)
                else:
                    anthropic_tools.append(
                        {
                            "name": func["name"],
                            "description": func.get("description", ""),
                            "input_schema": params,
                        }
                    )
            elif "name" in tool and "input_schema" in tool:
                # already in anthropic format — pass through directly
                anthropic_tools.append(
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": tool["input_schema"],
                    }
                )
        return anthropic_tools

    def _normalize_response(self, response) -> Dict[str, Any]:
        """
        Convert Anthropic response to normalized format.

        Args:
            response: Anthropic API response

        Returns:
            Dict with 'content', optionally 'tool_use', and 'usage'
        """
        result = {"content": "", "tool_use": []}

        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_use"].append(
                    {"id": block.id, "name": block.name, "input": block.input}
                )
            elif block.type == "server_tool_use":
                logger.info(
                    "anthropic.web_search.invoked",
                    tool_use_id=getattr(block, "id", None),
                    query=(
                        getattr(block, "input", {}).get("query", "")
                        if isinstance(getattr(block, "input", None), dict)
                        else ""
                    ),
                )
            elif block.type == "web_search_tool_result":
                content_items = getattr(block, "content", [])
                if isinstance(content_items, list):
                    for item in content_items:
                        if hasattr(item, "url"):
                            logger.debug(
                                "anthropic.web_search.result",
                                url=getattr(item, "url", ""),
                                title=getattr(item, "title", ""),
                            )

        # Remove empty tool_use list for cleaner response
        if not result["tool_use"]:
            del result["tool_use"]

        # Propagate stop_reason so agent loop can detect truncation
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason:
            result["stop_reason"] = stop_reason

        # Include token usage from response
        if hasattr(response, "usage") and response.usage:
            result["usage"] = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
                "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
            }

        return result
