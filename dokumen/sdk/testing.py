"""
Test helpers — message factories for MockQueryRunner.

Provides factory functions that create properly typed SDK message objects
for use in unit tests. All factories return real SDK types.
"""

import json
from typing import Any, Dict, List, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


def make_init(session_id: str = "test-session") -> SystemMessage:
    """Create a SystemMessage init event."""
    return SystemMessage(subtype="init", data={"session_id": session_id})


def make_assistant(
    text: str, tool_calls: Optional[List[Dict[str, Any]]] = None
) -> AssistantMessage:
    """Create an AssistantMessage with text and optional tool calls.

    Args:
        text: The assistant's text response.
        tool_calls: Optional list of dicts with 'id', 'name', 'input' keys.
    """
    blocks: list = []
    if text:
        blocks.append(TextBlock(text=text))
    if tool_calls:
        for tc in tool_calls:
            blocks.append(
                ToolUseBlock(
                    id=tc["id"],
                    name=tc["name"],
                    input=tc.get("input", {}),
                )
            )
    return AssistantMessage(content=blocks, model="claude-sonnet-4-6")


def make_tool_result(tool_use_id: str, content: str, is_error: bool = False) -> UserMessage:
    """Create a UserMessage containing a tool result."""
    return UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id=tool_use_id,
                content=content,
                is_error=is_error,
            )
        ]
    )


def make_result(
    text: str,
    is_error: bool = False,
    usage: Optional[Dict[str, Any]] = None,
    session_id: str = "test-session",
    duration_ms: int = 1000,
) -> ResultMessage:
    """Create a ResultMessage (final message in a query stream)."""
    return ResultMessage(
        subtype="error" if is_error else "success",
        duration_ms=duration_ms,
        duration_api_ms=int(duration_ms * 0.8),
        is_error=is_error,
        num_turns=1,
        session_id=session_id,
        stop_reason="end_turn" if not is_error else "error",
        total_cost_usd=0.01,
        usage=usage or {"input_tokens": 100, "output_tokens": 50},
        result=text,
        structured_output=None,
    )


def make_judge_pass(reason: str = "Correct") -> List[Any]:
    """Create a complete judge PASS message sequence."""
    verdict = json.dumps({"verdict": "PASS", "reason": reason})
    return [make_init(), make_assistant(verdict), make_result(verdict)]


def make_judge_fail(reason: str = "Incorrect") -> List[Any]:
    """Create a complete judge FAIL message sequence."""
    verdict = json.dumps({"verdict": "FAIL", "reason": reason})
    return [make_init(), make_assistant(verdict), make_result(verdict)]


def make_executor_simple(final_text: str) -> List[Any]:
    """Create a simple executor message sequence (no tool calls)."""
    return [make_init(), make_assistant(final_text), make_result(final_text)]


def make_executor_with_tools(tool_sequence: List[tuple], final_text: str) -> List[Any]:
    """Create an executor message sequence with tool calls.

    Args:
        tool_sequence: List of (name, input_dict, result_text) tuples.
        final_text: The final assistant response after all tools.

    Returns:
        List of messages simulating: init → [tool_call → tool_result]* → final → result
    """
    msgs: List[Any] = [make_init()]
    for name, inp, result_text in tool_sequence:
        tool_use_id = f"tc_{name}"
        msgs.append(
            make_assistant(
                "",
                tool_calls=[{"id": tool_use_id, "name": name, "input": inp}],
            )
        )
        msgs.append(make_tool_result(tool_use_id, result_text))
    msgs.append(make_assistant(final_text))
    msgs.append(make_result(final_text))
    return msgs
