"""tests for decomposed binary judging."""

import json
from typing import Optional

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

from dokumen.sdk.judge import (
    JudgeAgent,
    parse_decomposed_verdict,
    parse_verdict,
)
from dokumen.sdk.query_runner import MockQueryRunner
from dokumen.sdk.testing import (
    make_assistant,
    make_init,
    make_result,
)
from dokumen.sdk.types import (
    ExecutorResult,
    JudgeVerdict,
    SubAssertion,
)


def _make_executor_result(
    final_response: str = "OAuth 2.0 and API keys are supported.",
    success: bool = True,
) -> ExecutorResult:
    return ExecutorResult(
        success=success,
        final_response=final_response,
        tool_calls=[],
        input_tokens=100,
        output_tokens=50,
        conversation_log=[],
        duration_ms=1000,
    )


# ── SubAssertion dataclass tests ─────────────────────────────────────


class TestSubAssertionDataclass:
    def test_sub_assertion_creation(self):
        sa = SubAssertion(
            question="did the executor mention OAuth 2.0?",
            passed=True,
            reason="yes, mentioned in first paragraph",
        )
        assert sa.question == "did the executor mention OAuth 2.0?"
        assert sa.passed is True
        assert sa.reason == "yes, mentioned in first paragraph"

    def test_sub_assertion_to_dict(self):
        sa = SubAssertion(
            question="is the response accurate?",
            passed=False,
            reason="contains factual error about token expiry",
        )
        d = sa.to_dict()
        assert d["question"] == "is the response accurate?"
        assert d["passed"] is False
        assert d["reason"] == "contains factual error about token expiry"


# ── JudgeVerdict with sub_assertions ─────────────────────────────────


class TestJudgeVerdictSubAssertions:
    def test_verdict_with_sub_assertions(self):
        subs = [
            SubAssertion(question="q1", passed=True, reason="yes"),
            SubAssertion(question="q2", passed=False, reason="no"),
        ]
        verdict = JudgeVerdict(
            judge_id="accuracy",
            passed=False,
            sub_assertions=subs,
            confidence=0.5,
        )
        assert len(verdict.sub_assertions) == 2
        assert verdict.sub_assertions[0].passed is True
        assert verdict.sub_assertions[1].passed is False

    def test_verdict_to_dict_includes_sub_assertions(self):
        subs = [
            SubAssertion(question="q1", passed=True, reason="ok"),
        ]
        verdict = JudgeVerdict(
            judge_id="test",
            passed=True,
            sub_assertions=subs,
            confidence=1.0,
        )
        d = verdict.to_dict()
        assert "sub_assertions" in d
        assert len(d["sub_assertions"]) == 1
        assert d["sub_assertions"][0]["question"] == "q1"

    def test_verdict_without_sub_assertions_backward_compat(self):
        """existing verdicts without sub_assertions still work."""
        verdict = JudgeVerdict(
            judge_id="old-judge",
            passed=True,
            confidence=0.9,
        )
        assert verdict.sub_assertions == []
        d = verdict.to_dict()
        assert d["sub_assertions"] == []


# ── parse_decomposed_verdict tests ───────────────────────────────────


class TestParseDecomposedVerdict:
    def test_parse_valid_decomposed_response(self):
        response = json.dumps({
            "sub_assertions": [
                {"question": "mentions OAuth?", "passed": True, "reason": "yes"},
                {"question": "mentions API keys?", "passed": True, "reason": "yes"},
                {"question": "accurate description?", "passed": False, "reason": "wrong expiry time"},
            ],
        })
        subs, confidence = parse_decomposed_verdict(response)
        assert len(subs) == 3
        assert subs[0].passed is True
        assert subs[2].passed is False
        assert confidence == pytest.approx(2 / 3)

    def test_parse_decomposed_in_code_fence(self):
        response = (
            "here is my evaluation:\n\n"
            "```json\n"
            + json.dumps({
                "sub_assertions": [
                    {"question": "q1", "passed": True, "reason": "ok"},
                    {"question": "q2", "passed": True, "reason": "ok"},
                ],
            })
            + "\n```"
        )
        subs, confidence = parse_decomposed_verdict(response)
        assert len(subs) == 2
        assert confidence == 1.0

    def test_parse_decomposed_all_fail(self):
        response = json.dumps({
            "sub_assertions": [
                {"question": "q1", "passed": False, "reason": "no"},
                {"question": "q2", "passed": False, "reason": "no"},
            ],
        })
        subs, confidence = parse_decomposed_verdict(response)
        assert confidence == 0.0

    def test_parse_decomposed_empty_list(self):
        response = json.dumps({"sub_assertions": []})
        subs, confidence = parse_decomposed_verdict(response)
        assert subs == []
        assert confidence == 0.0

    def test_parse_decomposed_returns_none_on_garbage(self):
        result = parse_decomposed_verdict("not json at all")
        assert result is None

    def test_parse_decomposed_returns_none_on_missing_key(self):
        response = json.dumps({"verdict": "PASS"})
        result = parse_decomposed_verdict(response)
        assert result is None

    def test_parse_decomposed_returns_none_on_none(self):
        result = parse_decomposed_verdict(None)
        assert result is None


# ── JudgeAgent decomposed mode ───────────────────────────────────────


class TestJudgeDecomposedMode:
    async def test_decomposed_judge_all_pass(self):
        """all sub-assertions pass -> verdict PASS, confidence 1.0."""
        decomposed_response = json.dumps({
            "sub_assertions": [
                {"question": "mentions OAuth?", "passed": True, "reason": "yes"},
                {"question": "mentions API keys?", "passed": True, "reason": "yes"},
            ],
        })
        messages = [
            make_init(),
            make_assistant(decomposed_response),
            make_result(decomposed_response),
        ]
        runner = MockQueryRunner(messages)

        judge = JudgeAgent(
            id="decomposed-pass",
            system_prompt="Evaluate accuracy.",
            user_prompt="Check auth methods.",
            sdk_tools=[],
            query_runner=runner,
            decomposed=True,
        )
        result = await judge.run(_make_executor_result())

        assert result.passed is True
        assert result.confidence == 1.0
        assert len(result.sub_assertions) == 2
        assert all(sa.passed for sa in result.sub_assertions)
        assert result.error is False

    async def test_decomposed_judge_partial_pass(self):
        """some sub-assertions fail -> verdict PASS (above threshold), confidence < 1."""
        decomposed_response = json.dumps({
            "sub_assertions": [
                {"question": "q1", "passed": True, "reason": "ok"},
                {"question": "q2", "passed": True, "reason": "ok"},
                {"question": "q3", "passed": False, "reason": "missing"},
            ],
        })
        messages = [
            make_init(),
            make_assistant(decomposed_response),
            make_result(decomposed_response),
        ]
        runner = MockQueryRunner(messages)

        judge = JudgeAgent(
            id="decomposed-partial",
            system_prompt="Evaluate.",
            user_prompt="Check things.",
            sdk_tools=[],
            query_runner=runner,
            decomposed=True,
            decomposed_threshold=0.5,
        )
        result = await judge.run(_make_executor_result())

        assert result.passed is True  # 2/3 = 0.667 > 0.5
        assert result.confidence == pytest.approx(2 / 3)
        assert len(result.sub_assertions) == 3

    async def test_decomposed_judge_below_threshold(self):
        """too many sub-assertions fail -> verdict FAIL."""
        decomposed_response = json.dumps({
            "sub_assertions": [
                {"question": "q1", "passed": False, "reason": "no"},
                {"question": "q2", "passed": False, "reason": "no"},
                {"question": "q3", "passed": True, "reason": "yes"},
            ],
        })
        messages = [
            make_init(),
            make_assistant(decomposed_response),
            make_result(decomposed_response),
        ]
        runner = MockQueryRunner(messages)

        judge = JudgeAgent(
            id="decomposed-fail",
            system_prompt="Evaluate.",
            user_prompt="Check things.",
            sdk_tools=[],
            query_runner=runner,
            decomposed=True,
            decomposed_threshold=0.5,
        )
        result = await judge.run(_make_executor_result())

        assert result.passed is False  # 1/3 = 0.333 < 0.5
        assert result.confidence == pytest.approx(1 / 3)
        assert result.failure_reason is not None

    async def test_decomposed_judge_custom_threshold(self):
        """custom threshold changes pass/fail boundary."""
        decomposed_response = json.dumps({
            "sub_assertions": [
                {"question": "q1", "passed": True, "reason": "ok"},
                {"question": "q2", "passed": False, "reason": "no"},
            ],
        })
        messages = [
            make_init(),
            make_assistant(decomposed_response),
            make_result(decomposed_response),
        ]

        # threshold 0.75 -> 0.5 < 0.75 -> FAIL
        runner = MockQueryRunner(messages)
        judge = JudgeAgent(
            id="threshold-high",
            system_prompt="Evaluate.",
            user_prompt="Check things.",
            sdk_tools=[],
            query_runner=runner,
            decomposed=True,
            decomposed_threshold=0.75,
        )
        result = await judge.run(_make_executor_result())
        assert result.passed is False

    async def test_decomposed_judge_malformed_falls_back(self):
        """if decomposed parse fails, falls back to regular verdict parsing."""
        # response is a regular verdict, not decomposed
        regular_verdict = json.dumps({
            "verdict": "PASS",
            "confidence": 0.9,
            "reason": "looks good",
        })
        messages = [
            make_init(),
            make_assistant(regular_verdict),
            make_result(regular_verdict),
        ]
        runner = MockQueryRunner(messages)

        judge = JudgeAgent(
            id="decomposed-fallback",
            system_prompt="Evaluate.",
            user_prompt="Check things.",
            sdk_tools=[],
            query_runner=runner,
            decomposed=True,
        )
        result = await judge.run(_make_executor_result())

        # should fall back to regular parsing
        assert result.passed is True
        assert result.confidence == 0.9
        assert result.sub_assertions == []

    async def test_decomposed_false_uses_regular_parsing(self):
        """decomposed=False (default) uses regular verdict parsing."""
        regular_verdict = json.dumps({
            "verdict": "PASS",
            "confidence": 0.85,
            "reason": "all good",
        })
        messages = [
            make_init(),
            make_assistant(regular_verdict),
            make_result(regular_verdict),
        ]
        runner = MockQueryRunner(messages)

        judge = JudgeAgent(
            id="regular-judge",
            system_prompt="Evaluate.",
            user_prompt="Check things.",
            sdk_tools=[],
            query_runner=runner,
            decomposed=False,
        )
        result = await judge.run(_make_executor_result())

        assert result.passed is True
        assert result.confidence == 0.85
        assert result.sub_assertions == []
