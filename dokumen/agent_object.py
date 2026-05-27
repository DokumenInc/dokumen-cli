"""
Agent types and Provider ABC for the Dokumen CLI.

Contains the Provider abstract base class still used by the create agent
(to be migrated in issue #604), plus re-exports of the
canonical result types from sdk/types.py.

The legacy ExecutorOutput and JudgeResult classes have been replaced by
ExecutorResult and JudgeVerdict (defined in sdk/types.py). The old names
are kept as aliases for backward compatibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .logging_config import get_logger
from .sdk.types import ExecutorResult, JudgeVerdict

logger = get_logger(__name__)

# Backward-compatible aliases — existing code importing these names will
# continue to work without changes.
ExecutorOutput = ExecutorResult
JudgeResult = JudgeVerdict


class AgentType(Enum):
    """Type of agent."""

    EXECUTOR = "executor"
    JUDGE = "judge"


@dataclass
class LogEntry:
    """A single log entry during agent execution.

    Deprecated: No longer used by ExecutorResult. Kept for backward
    compatibility with any external code that imports it.
    """

    timestamp: datetime
    level: str  # "info", "warning", "error", "debug"
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "data": self.data,
        }


@dataclass
class ToolCall:
    """Record of a tool invocation.

    Deprecated: ExecutorResult uses List[Dict] for tool_calls instead.
    Kept for backward compatibility with test code that imports it.
    Note: This is NOT the same as tools.types.ToolCall.
    """

    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    timestamp: datetime
    duration: float  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        if self.result is None:
            result_str = ""
        elif hasattr(self.result, "output"):
            output = self.result.output
            result_str = str(output) if output is not None else ""
        else:
            result_str = str(self.result)
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "result": result_str,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
        }


class Provider(ABC):
    """Abstract base class for LLM providers.

    Still used by create_agent.py.
    Will be removed when those are migrated to the SDK path (issue #604).
    """

    @abstractmethod
    async def complete(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Send a completion request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            **kwargs: Additional provider-specific parameters

        Returns:
            Response dict with 'content' and optionally 'tool_calls'
        """
        pass
