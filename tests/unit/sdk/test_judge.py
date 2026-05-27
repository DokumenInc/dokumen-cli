"""Tests for dokumen.sdk.judge — JudgeAgent and parse_verdict."""

import asyncio
import json
from typing import Any, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

from dokumen.sdk.judge import (
    JUDGE_RETRY_PROMPT,
    JudgeAgent,
    _extract_verdict,
    parse_verdict,
)
from dokumen.sdk.query_runner import MockQueryRunner
from dokumen.sdk.testing import (
    make_assistant,
    make_init,
    make_judge_fail,
    make_judge_pass,
    make_result,
)
from dokumen.sdk.types import SdkExecutorResult, SdkJudgeResult, Verdict


def _make_executor_result(
    final_response: str = "The answer is 42.", success: bool = True
) -> SdkExecutorResult:
    """Helper to create an executor result for judge tests."""
    return SdkExecutorResult(
        success=success,
        final_response=final_response,
        tool_calls=[],
        input_tokens=100,
        output_tokens=50,
        conversation_log=[],
        duration_ms=1000,
    )


# ── parse_verdict unit tests ──────────────────────────────────────────


class TestParseVerdictDirectJson:
    def test_parse_verdict_direct_json(self):
        """parse_verdict with clean JSON."""
        text = json.dumps({"verdict": "PASS", "confidence": 0.95, "reason": "All correct"})
        verdict = parse_verdict(text)

        assert verdict is not None
        assert verdict.passed is True
        assert verdict.confidence == 0.95
        assert verdict.reason == "All correct"


class TestParseVerdictCodeFence:
    def test_parse_verdict_code_fence(self):
        """parse_verdict with ```json fence."""
        text = (
            "Here is my evaluation:\n\n"
            "```json\n"
            '{"verdict": "FAIL", "confidence": 0.7, "reason": "Missing details"}\n'
            "```\n"
        )
        verdict = parse_verdict(text)

        assert verdict is not None
        assert verdict.passed is False
        assert verdict.confidence == 0.7
        assert verdict.reason == "Missing details"


class TestParseVerdictInline:
    def test_parse_verdict_inline(self):
        """parse_verdict with text + inline JSON."""
        text = (
            'Based on my analysis, the verdict is: {"verdict": "PASS", '
            '"confidence": 0.85, "reason": "Mostly accurate"} and that is my conclusion.'
        )
        verdict = parse_verdict(text)

        assert verdict is not None
        assert verdict.passed is True
        assert verdict.confidence == 0.85
        assert verdict.reason == "Mostly accurate"


class TestParseVerdictInvalid:
    def test_parse_verdict_invalid(self):
        """parse_verdict returns None for garbage."""
        assert parse_verdict("This is just random text with no JSON.") is None

    def test_parse_verdict_none_input(self):
        """parse_verdict returns None for None."""
        assert parse_verdict(None) is None

    def test_parse_verdict_empty_string(self):
        """parse_verdict returns None for empty string."""
        assert parse_verdict("") is None


class TestParseVerdictInvalidVerdictValue:
    def test_parse_verdict_invalid_verdict_value(self):
        """parse_verdict returns None for bad verdict field."""
        text = json.dumps({"verdict": "MAYBE", "confidence": 0.5, "reason": "Uncertain"})
        verdict = parse_verdict(text)

        assert verdict is None


class TestExtractVerdictClampConfidence:
    def test_extract_verdict_clamp_high(self):
        """Confidence > 1.0 is clamped to 1.0."""
        verdict = _extract_verdict({"verdict": "PASS", "confidence": 1.5, "reason": "Sure"})

        assert verdict.confidence == 1.0

    def test_extract_verdict_clamp_low(self):
        """Confidence < 0.0 is clamped to 0.0."""
        verdict = _extract_verdict({"verdict": "FAIL", "confidence": -0.5, "reason": "Nope"})

        assert verdict.confidence == 0.0

    def test_extract_verdict_case_insensitive(self):
        """Verdict string is case-insensitive (lowered input works)."""
        verdict = _extract_verdict({"verdict": "pass", "confidence": 0.9, "reason": "OK"})

        assert verdict.passed is True


# ── JudgeAgent tests ──────────────────────────────────────────────────


class TestJudgePassVerdict:
    async def test_judge_pass_verdict(self):
        """Judge returns PASS -> passed=True."""
        messages = make_judge_pass(confidence=0.95, reason="Correct")
        runner = MockQueryRunner(messages)

        judge = JudgeAgent(
            id="accuracy",
            system_prompt="Evaluate accuracy.",
            user_prompt="Is the answer correct?",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await judge.run(_make_executor_result())

        assert isinstance(result, SdkJudgeResult)
        assert result.passed is True
        assert result.confidence == 0.95
        assert result.failure_reason is None  # PASS verdicts have no failure_reason
        assert result.reason == "Correct"  # reason always set from parsed verdict
        assert result.error is False
        assert result.judge_id == "accuracy"


class TestJudgeFailVerdict:
    async def test_judge_fail_verdict(self):
        """Judge returns FAIL -> passed=False."""
        messages = make_judge_fail(reason="Missing info", confidence=0.8)
        runner = MockQueryRunner(messages)

        judge = JudgeAgent(
            id="completeness",
            system_prompt="Evaluate completeness.",
            user_prompt="Is it complete?",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await judge.run(_make_executor_result())

        assert result.passed is False
        assert result.confidence == 0.8
        assert result.failure_reason == "Missing info"
        assert result.reason == "Missing info"  # reason always set from parsed verdict
        assert result.error is False


class TestJudgeConfidenceExtraction:
    async def test_judge_confidence_extraction(self):
        """Confidence extracted correctly from judge verdict."""
        verdict_json = json.dumps(
            {"verdict": "PASS", "confidence": 0.73, "reason": "Mostly right"}
        )
        messages = [make_init(), make_assistant(verdict_json), make_result(verdict_json)]
        runner = MockQueryRunner(messages)

        judge = JudgeAgent(
            id="conf-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await judge.run(_make_executor_result())

        assert result.confidence == 0.73


class TestJudgeMalformedRetry:
    async def test_judge_malformed_retry(self):
        """First response not valid JSON -> retries with session resume -> gets valid verdict."""

        call_count = 0

        class RetryRunner:
            """Runner that returns malformed on first call, valid on retry."""

            def __init__(self):
                self.calls = []

            async def run(
                self, prompt: str, options: Optional[ClaudeAgentOptions] = None
            ):
                nonlocal call_count
                call_count += 1
                self.calls.append(prompt)

                if call_count == 1:
                    # First call: return malformed verdict (not valid JSON)
                    yield make_init(session_id="sess-123")
                    yield make_assistant("I think the answer is correct but let me explain...")
                    yield make_result(
                        "I think the answer is correct but let me explain...",
                        session_id="sess-123",
                    )
                else:
                    # Retry call: return valid verdict JSON
                    valid = json.dumps(
                        {"verdict": "PASS", "confidence": 0.9, "reason": "Correct after retry"}
                    )
                    yield ResultMessage(
                        subtype="success",
                        duration_ms=500,
                        duration_api_ms=400,
                        is_error=False,
                        num_turns=1,
                        session_id="sess-123",
                        stop_reason="end_turn",
                        result=valid,
                    )

        runner = RetryRunner()
        judge = JudgeAgent(
            id="retry-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await judge.run(_make_executor_result())

        assert result.passed is True
        assert result.confidence == 0.9
        assert result.failure_reason is None  # PASS verdicts have no failure_reason
        assert result.error is False
        # Verify retry was called
        assert call_count == 2
        assert runner.calls[1] == JUDGE_RETRY_PROMPT


class TestJudgeMalformedRetryFails:
    async def test_judge_retry_also_fails(self):
        """First response malformed, retry also malformed -> error=True, passed=False."""

        call_count = 0

        class DoubleFailRunner:
            """Runner that returns malformed on both initial and retry calls."""

            def __init__(self):
                self.calls = []

            async def run(
                self, prompt: str, options: Optional[ClaudeAgentOptions] = None
            ):
                nonlocal call_count
                call_count += 1
                self.calls.append(prompt)

                if call_count == 1:
                    # First call: malformed verdict
                    yield make_init(session_id="sess-fail")
                    yield make_assistant("I cannot decide.")
                    yield make_result("I cannot decide.", session_id="sess-fail")
                else:
                    # Retry: also malformed
                    yield ResultMessage(
                        subtype="success",
                        duration_ms=500,
                        duration_api_ms=400,
                        is_error=False,
                        num_turns=1,
                        session_id="sess-fail",
                        stop_reason="end_turn",
                        result="Still cannot produce valid JSON verdict.",
                    )

        runner = DoubleFailRunner()
        judge = JudgeAgent(
            id="double-fail-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await judge.run(_make_executor_result())

        assert result.passed is False
        assert result.error is True
        assert result.failure_reason == "Failed to parse verdict"
        assert result.judge_id == "double-fail-judge"
        assert call_count == 2

    async def test_judge_retry_exception(self):
        """Retry raises exception -> error=True, passed=False."""

        call_count = 0

        class ExceptionRetryRunner:
            def __init__(self):
                self.calls = []

            async def run(
                self, prompt: str, options: Optional[ClaudeAgentOptions] = None
            ):
                nonlocal call_count
                call_count += 1
                self.calls.append(prompt)

                if call_count == 1:
                    yield make_init(session_id="sess-exc")
                    yield make_assistant("No JSON here.")
                    yield make_result("No JSON here.", session_id="sess-exc")
                else:
                    raise RuntimeError("Connection lost during retry")

        runner = ExceptionRetryRunner()
        judge = JudgeAgent(
            id="exc-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await judge.run(_make_executor_result())

        assert result.passed is False
        assert result.error is True
        assert result.judge_id == "exc-judge"
        assert call_count == 2


class TestJudgeTimeout:
    async def test_judge_timeout(self):
        """Judge times out -> error=True, passed=False."""

        class SlowRunner:
            def __init__(self):
                self.calls = []

            async def run(self, prompt, options=None):
                self.calls.append(prompt)
                await asyncio.sleep(10)
                yield make_init()  # pragma: no cover

        runner = SlowRunner()
        judge = JudgeAgent(
            id="timeout-judge",
            system_prompt="Slow judge.",
            user_prompt="Evaluate slowly.",
            sdk_tools=[],
            query_runner=runner,
            timeout=0.01,
        )
        result = await judge.run(_make_executor_result())

        assert result.passed is False
        assert result.error is True
        assert "timed out" in result.failure_reason
        assert result.judge_id == "timeout-judge"


class TestJudgeNoExecutorOutput:
    async def test_judge_no_executor_output(self):
        """include_executor_output=False works — executor output not in context."""
        messages = make_judge_pass()
        runner = MockQueryRunner(messages)

        judge = JudgeAgent(
            id="no-output-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate criteria only.",
            include_executor_output=False,
            sdk_tools=[],
            query_runner=runner,
        )
        result = await judge.run(_make_executor_result())

        assert result.passed is True
        # Verify the prompt sent to the runner doesn't include executor output
        sent_prompt = runner.calls[0].prompt
        assert "## Executor Output" not in sent_prompt
        assert "## Evaluation Criteria" in sent_prompt


class TestJudgeTruncation:
    """Tests for max_response_chars truncation in judge context."""

    async def test_judge_truncates_large_executor_output(self):
        """Large executor output is truncated in judge context."""
        messages = make_judge_pass()
        runner = MockQueryRunner(messages)

        judge = JudgeAgent(
            id="trunc-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            include_executor_output=True,
            max_response_chars=1000,
            sdk_tools=[],
            query_runner=runner,
        )

        large_output = "X" * 50_000
        executor = _make_executor_result(final_response=large_output)
        result = await judge.run(executor)

        assert result.passed is True
        # Verify the prompt sent to the runner was truncated
        sent_prompt = runner.calls[0].prompt
        assert large_output not in sent_prompt
        assert "X" * 1000 in sent_prompt
        assert "[Response truncated" in sent_prompt

    async def test_judge_no_truncation_when_zero(self):
        """max_response_chars=0 (default) means no truncation."""
        messages = make_judge_pass()
        runner = MockQueryRunner(messages)

        judge = JudgeAgent(
            id="no-trunc-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            include_executor_output=True,
            sdk_tools=[],
            query_runner=runner,
        )

        large_output = "Y" * 50_000
        executor = _make_executor_result(final_response=large_output)
        result = await judge.run(executor)

        assert result.passed is True
        sent_prompt = runner.calls[0].prompt
        assert large_output in sent_prompt
        assert "truncated" not in sent_prompt.lower()
