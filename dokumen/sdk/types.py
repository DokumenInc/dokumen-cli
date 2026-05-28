"""
Canonical result types for the Dokumen CLI.

Defines the two core result types used throughout the codebase:
- ExecutorResult: output from running an executor agent
- JudgeVerdict: output from running a judge agent

These are the single source of truth — no adapters, no legacy types.
Backward-compatible aliases (ExecutorOutput, JudgeResult, SdkExecutorResult,
SdkJudgeResult) are provided so existing imports continue to work.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryResult:
    """Raw result from a query() call."""

    session_id: Optional[str]
    messages: List[
        Any
    ]  # List[AssistantMessage | UserMessage] — Any to avoid SDK import at module level
    result: Optional[Any]  # Optional[ResultMessage]


@dataclass
class Verdict:
    """Parsed judge verdict."""

    passed: bool
    reason: str


@dataclass
class ExecutorResult:
    """Canonical executor result type.

    Used by executor agents, test_object.py, and formatters. Replaces both the
    legacy ExecutorOutput and SdkExecutorResult.

    Fields:
        success: Whether the executor completed successfully.
        final_response: The executor's final text response.
        tool_calls: List of tool call dicts (tool_name, parameters, result).
        input_tokens: Input tokens consumed.
        output_tokens: Output tokens consumed.
        cache_creation_tokens: Cache creation tokens consumed.
        cache_read_tokens: Cache read tokens consumed.
        conversation_log: Full conversation log for UI display.
        system_prompt: System prompt used.
        user_prompt: User prompt sent (may include explore context).
        original_user_prompt: Original user prompt before explore injection.
        duration_ms: Execution duration in milliseconds.
        error: Error message if execution failed.
    """

    success: bool
    final_response: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    conversation_log: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    user_prompt: str = ""
    original_user_prompt: str = ""
    duration_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "success": self.success,
            "final_response": self.final_response,
            "tool_calls": self.tool_calls,
            "error": self.error,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "original_user_prompt": self.original_user_prompt,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
        }


@dataclass
class SubAssertion:
    """single binary sub-assertion within a decomposed judge verdict."""

    question: str
    passed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """serialize to dict for JSON storage."""
        return {
            "question": self.question,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass
class JudgeVerdict:
    """Canonical judge result type.

    Used by judge agents, test_object.py, and formatters. Replaces both the
    legacy JudgeResult and SdkJudgeResult.

    Fields:
        judge_id: Identifier for this judge (e.g., "accuracy", "completeness").
        passed: Whether the judge verdict is PASS.
        failure_reason: Reason for failure (None if passed).
        reason: Parsed verdict reason (always set when verdict is parsed).
        response: Full judge response text.
        tool_calls: List of tool call dicts made by the judge.
        assertion_text: The assertion/question text for this judge.
        input_tokens: Input tokens consumed.
        output_tokens: Output tokens consumed.
        cache_creation_tokens: Cache creation tokens consumed.
        cache_read_tokens: Cache read tokens consumed.
        conversation_log: Full conversation log for UI display.
        error: True when judge errored (timeout, exception) vs legitimate verdict.
    """

    judge_id: str
    passed: bool
    failure_reason: Optional[str] = None
    reason: Optional[str] = None
    response: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    assertion_text: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    conversation_log: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[bool] = None
    sub_assertions: List[SubAssertion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "judge_id": self.judge_id,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "reason": self.reason,
            "response": self.response,
            "tool_calls": self.tool_calls,
            "assertion_text": self.assertion_text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "error": self.error,
            "sub_assertions": [sa.to_dict() for sa in self.sub_assertions],
        }


# Backward-compatible aliases
SdkExecutorResult = ExecutorResult
SdkJudgeResult = JudgeVerdict
