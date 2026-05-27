"""
Message stream processing utilities.

Extracts structured data from Claude Agent SDK message streams
for populating result types and conversation logs.
"""

import logging
from typing import Any, Dict, List, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from .types import ExecutorResult

logger = logging.getLogger(__name__)


def extract_tool_calls(messages: List[Any]) -> List[Dict[str, Any]]:
    """Extract tool call records from a message stream.

    Walks through messages, pairing ToolUseBlocks with their
    corresponding ToolResultBlocks to produce a complete record.

    Args:
        messages: List of SDK messages (AssistantMessage, UserMessage, etc.)

    Returns:
        List of dicts with keys: tool_name, tool_input, tool_result, tool_use_id
    """
    tool_calls = []
    # Collect pending tool uses from assistant messages
    pending: Dict[str, Dict[str, Any]] = {}

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    pending[block.id] = {
                        "tool_name": block.name,
                        "tool_input": block.input,
                        "tool_use_id": block.id,
                        "tool_result": None,
                    }
        elif isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        if block.tool_use_id in pending:
                            pending[block.tool_use_id]["tool_result"] = block.content
                            tool_calls.append(pending.pop(block.tool_use_id))

    # Any remaining pending tool uses (no result received)
    for tc in pending.values():
        tool_calls.append(tc)

    return tool_calls


def build_conversation_log(messages: List[Any]) -> List[Dict[str, Any]]:
    """Build a conversation log from a message stream for UI display.

    Produces a list of log entries compatible with the existing
    conversation_log format used by test_object.py and the frontend.

    Args:
        messages: List of SDK messages.

    Returns:
        List of dicts with 'role', 'content', and optionally 'tool_calls'.
    """
    log: List[Dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            entry: Dict[str, Any] = {"role": "assistant"}
            text_parts = []
            tool_uses = []

            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tool_uses.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            if text_parts:
                entry["content"] = "\n".join(text_parts)
            if tool_uses:
                entry["tool_calls"] = tool_uses
            log.append(entry)

        elif isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, str):
                log.append({"role": "user", "content": content})
            elif isinstance(content, list):
                tool_results = []
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        result_content = block.content
                        if isinstance(result_content, list):
                            # Extract text from content blocks
                            result_content = " ".join(
                                b.get("text", str(b))
                                for b in result_content
                                if isinstance(b, dict)
                            )
                        tool_results.append(
                            {
                                "tool_use_id": block.tool_use_id,
                                "content": str(result_content) if result_content else "",
                                "is_error": block.is_error or False,
                            }
                        )
                if tool_results:
                    log.append({"role": "tool", "tool_results": tool_results})

    return log


def extract_usage(result: Optional[Any]) -> Dict[str, int]:
    """Extract token usage from a ResultMessage.

    Args:
        result: A ResultMessage or None.

    Returns:
        Dict with at minimum input_tokens and output_tokens.
    """
    if not result:
        return {"input_tokens": 0, "output_tokens": 0}
    if not isinstance(result, ResultMessage):
        return {"input_tokens": 0, "output_tokens": 0}
    if not result.usage:
        return {"input_tokens": 0, "output_tokens": 0}
    return dict(result.usage)


def build_judge_context(
    executor_result: ExecutorResult,
    judge_prompt: str,
    include_output: bool,
    max_response_chars: int = 0,
    include_tool_calls: bool = True,
    executor_system_prompt: str = "",
    executor_user_prompt: str = "",
) -> str:
    """Build the judge prompt context string.

    Constructs the full prompt that the judge agent receives,
    including the executor's task context, output, tool calls,
    and the evaluation criteria.

    Args:
        executor_result: The executor's result to evaluate.
        judge_prompt: The judge's evaluation criteria prompt.
        include_output: Whether to include the executor's final response.
        max_response_chars: Max chars for executor response. 0 = unlimited.
        include_tool_calls: Whether to include executor tool calls section.
        executor_system_prompt: The executor's system prompt (for judge context).
        executor_user_prompt: The executor's user prompt (for judge context).

    Returns:
        Combined context string for the judge.
    """
    parts = []

    # Executor task context
    if executor_system_prompt or executor_user_prompt:
        task_parts = []
        if executor_system_prompt:
            task_parts.append(f"**Instructions:** {executor_system_prompt[:500]}")
        if executor_user_prompt:
            task_parts.append(f"**Task:** {executor_user_prompt[:1000]}")
        parts.append("## Executor Task\n\n" + "\n\n".join(task_parts))
        logger.debug(
            "Judge context includes executor task",
            extra={
                "system_prompt_len": len(executor_system_prompt),
                "user_prompt_len": len(executor_user_prompt),
            },
        )

    # Executor output
    if include_output:
        response = executor_result.final_response
        if max_response_chars > 0 and len(response) > max_response_chars:
            original_len = len(response)
            response = (
                response[:max_response_chars]
                + f"\n\n[Response truncated at {max_response_chars:,} chars "
                f"(original: {original_len:,} chars)]"
            )
            logger.info(
                "Judge context truncated",
                extra={
                    "original_chars": original_len,
                    "truncated_to": max_response_chars,
                },
            )
        parts.append(f"## Executor Output\n\n{response}")

    # Executor tool calls
    if include_tool_calls and executor_result.tool_calls:
        tool_lines = []
        for i, tc in enumerate(executor_result.tool_calls, 1):
            name = tc.get("tool_name", "unknown")
            inp = str(tc.get("tool_input", ""))[:200]
            res = str(tc.get("tool_result", ""))[:500] if tc.get("tool_result") else "(no result)"
            tool_lines.append(f"{i}. **{name}**({inp}): {res}")
        parts.append("## Executor Tool Calls\n\n" + "\n".join(tool_lines))
        logger.debug(
            "Judge context includes tool calls",
            extra={"tool_call_count": len(executor_result.tool_calls)},
        )

    # Evaluation criteria
    if judge_prompt:
        parts.append(f"## Evaluation Criteria\n\n{judge_prompt}")

    return "\n\n".join(parts)
