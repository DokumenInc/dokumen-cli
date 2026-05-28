"""Framework-neutral tool contracts used across Dokumen.

These types describe Dokumen tool capabilities without tying them to a
specific agent runtime. Runtime adapters, such as the Claude Agent SDK adapter,
can map these definitions onto provider-native tools or expose them through
MCP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ToolResult:
    """Result returned by a Dokumen tool handler."""

    success: bool
    output: Any
    error: Optional[str] = None


@dataclass
class ToolCall:
    """Record of a tool invocation."""

    tool_name: str
    parameters: Dict[str, Any]
    result: ToolResult
    timestamp: datetime
    duration: float


@dataclass
class SubagentResult:
    """Result from a single subagent execution."""

    file_path: str
    start_line: int
    end_line: int
    goal: str
    success: bool
    response: str
    tool_calls: List[Dict[str, Any]]
    covered_lines: List[int] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "lines": f"{self.start_line}-{self.end_line}",
            "goal": self.goal,
            "success": self.success,
            "response": self.response,
            "tool_calls": self.tool_calls,
            "covered_lines": self.covered_lines,
            "error": self.error,
        }


ToolHandler = Callable[[Dict[str, Any]], Awaitable[ToolResult]]


@dataclass
class ToolDefinition:
    """Definition of a tool available to agents."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: ToolHandler


class ToolsObject:
    """In-memory registry for framework-neutral tool definitions."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool definition."""
        if tool.name in self._tools:
            raise ValueError(f"Tool already exists: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> bool:
        """Remove a tool from the registry."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._tools.get(tool_name)

    async def execute(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a registered tool by name."""
        logger.debug("tools.execute.start", tool_name=tool_name, params=params)
        if tool_name not in self._tools:
            logger.warning("tools.execute.not_found", tool_name=tool_name)
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool not found: {tool_name}",
            )

        tool = self._tools[tool_name]

        try:
            self._validate_params(tool.parameters, params)

            import time

            start = time.time()
            result = await tool.handler(params)
            duration = time.time() - start
            logger.info(
                "tools.execute.complete",
                tool_name=tool_name,
                success=result.success,
                duration_ms=int(duration * 1000),
            )
            return result
        except Exception as exc:
            logger.error("tools.execute.error", tool_name=tool_name, error=str(exc))
            return ToolResult(success=False, output=None, error=str(exc))

    def get_definitions(self) -> List[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def to_anthropic_format(self) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic tool use format."""
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
            for tool in self._tools.values()
        ]

    def to_fastmcp_format(self) -> List[Dict[str, Any]]:
        """Convert tools to fastMCP-compatible format."""
        return [
            {"name": tool.name, "description": tool.description, "inputSchema": tool.parameters}
            for tool in self._tools.values()
        ]

    @staticmethod
    def _validate_params(schema: Dict[str, Any], params: Dict[str, Any]) -> None:
        """Validate parameters against JSON schema.

        Args:
            schema: JSON schema for parameters
            params: Parameters to validate

        Raises:
            ValueError: If required parameter is missing
        """
        required = schema.get("required", [])
        for req in required:
            if req not in params:
                raise ValueError(f"Missing required parameter: {req}")
