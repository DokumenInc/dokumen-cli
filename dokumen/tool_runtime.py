"""Dokucode-style tool runtime for dokumen CLI.

This module introduces a registry/handler runtime so agent loops execute tools
through a single dispatch path, instead of iterating ad-hoc ToolDefinition
lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .logging_config import get_logger
from .tools_object import ToolDefinition, ToolResult

logger = get_logger(__name__)


@dataclass
class ToolContext:
    """Execution context passed to tool handlers.

    Modeled after Dokucode's tool context shape but adapted for the current
    Python runtime.
    """

    session_id: str = ""
    message_id: str = ""
    agent: str = ""
    call_id: Optional[str] = None
    abort: Any = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    metadata_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ask_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None

    def metadata(self, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.metadata_callback:
            return
        payload = {"title": title, "metadata": metadata or {}}
        self.metadata_callback(payload)

    async def ask(self, request: Dict[str, Any]) -> None:
        if not self.ask_callback:
            return
        await self.ask_callback(request)


@runtime_checkable
class ToolHandler(Protocol):
    """Protocol all registry handlers implement."""

    name: str
    description: str
    input_schema: Dict[str, Any]

    async def execute(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        ...


class LegacyToolAdapter:
    """Adapter to run legacy ToolDefinition handlers in the new runtime."""

    def __init__(self, tool: ToolDefinition):
        self._tool = tool
        self.name = tool.name
        self.description = tool.description
        self.input_schema = tool.parameters

    async def execute(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        # Legacy tools do not consume ToolContext yet.
        return await self._tool.handler(params)


class ToolRegistry:
    """Central tool registry and execution dispatcher."""

    def __init__(self) -> None:
        self._handlers: Dict[str, ToolHandler] = {}

    @property
    def tool_names(self) -> List[str]:
        return list(self._handlers.keys())

    def register(self, handler: ToolHandler) -> None:
        self._handlers[handler.name] = handler
        logger.debug("tool_registry.register", tool=handler.name)

    def unregister(self, tool_name: str) -> bool:
        if tool_name in self._handlers:
            del self._handlers[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> Optional[ToolHandler]:
        return self._handlers.get(tool_name)

    def to_openai_format(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": h.name,
                    "description": h.description,
                    "parameters": h.input_schema,
                },
            }
            for h in self._handlers.values()
        ]

    def to_anthropic_format(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": h.name,
                "description": h.description,
                "input_schema": h.input_schema,
            }
            for h in self._handlers.values()
        ]

    async def execute(self, tool_name: str, params: Dict[str, Any], context: Optional[ToolContext] = None) -> ToolResult:
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolResult(success=False, output=None, error=f"Tool not found: {tool_name}")

        context = context or ToolContext()

        try:
            result = await handler.execute(params, context)
            return result
        except Exception as e:
            logger.error("tool_registry.execute.error", tool=tool_name, error=str(e), exc_info=True)
            return ToolResult(success=False, output=None, error=str(e))

    @classmethod
    def from_legacy_tools(cls, tools: List[ToolDefinition]) -> "ToolRegistry":
        registry = cls()
        for tool in tools:
            registry.register(LegacyToolAdapter(tool))
        return registry
