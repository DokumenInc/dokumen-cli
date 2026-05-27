"""
JudgeAgent — evaluates executor output via Claude Agent SDK.

Parses JSON verdicts from the judge response, with session-resume
retry on structural failure.
"""

import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

from .base import DokumenAgent
from .messages import (
    build_conversation_log,
    build_judge_context,
    extract_tool_calls,
    extract_usage,
)
from .query_runner import QueryRunner
from .types import ExecutorResult, JudgeVerdict, SubAssertion, Verdict

logger = logging.getLogger(__name__)

# Prompt sent on verdict retry (session resume)
JUDGE_RETRY_PROMPT = (
    "Your previous response could not be parsed as a valid verdict. "
    'Return ONLY a JSON object: {"verdict": "PASS" or "FAIL", '
    '"confidence": 0.0 to 1.0, "reason": "explanation"}'
)


def parse_verdict(text: Optional[str]) -> Optional[Verdict]:
    """Parse a verdict JSON from judge response text.

    Tries multiple strategies:
    1. Parse the entire text as JSON
    2. Extract JSON from a ```json code fence
    3. Find inline JSON with verdict key via regex

    Args:
        text: The judge's response text.

    Returns:
        Parsed Verdict or None if parsing fails.
    """
    if not text:
        return None

    # Strategy 1: Direct JSON parse
    try:
        data = json.loads(text.strip())
        return _extract_verdict(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract from ```json fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1).strip())
            return _extract_verdict(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Find inline JSON object with verdict key
    json_match = re.search(r'\{[^{}]*"verdict"\s*:\s*"[^"]*"[^{}]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return _extract_verdict(data)
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning(
        "Failed to parse verdict from judge response",
        extra={"response_preview": text[:200] if text else None},
    )
    return None


def _extract_verdict(data: dict) -> Verdict:
    """Extract a Verdict from a parsed JSON dict.

    Args:
        data: Parsed JSON dict with verdict, confidence, reason.

    Returns:
        Verdict dataclass.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    verdict_str = data.get("verdict", "").upper()
    if verdict_str not in ("PASS", "FAIL"):
        raise ValueError(f"Invalid verdict value: {data.get('verdict')}")

    confidence = float(data.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

    reason = str(data.get("reason", ""))

    return Verdict(
        passed=verdict_str == "PASS",
        confidence=confidence,
        reason=reason,
    )


def parse_decomposed_verdict(text: Optional[str]):
    """Parse a decomposed verdict response with sub_assertions.

    Tries to extract a JSON object with a "sub_assertions" key containing
    a list of {question, passed, reason} objects.

    Args:
        text: The judge's response text.

    Returns:
        Tuple of (list of SubAssertion, confidence float) or None if parsing fails.
    """
    if not text:
        return None

    def _try_parse(raw: str):
        try:
            data = json.loads(raw.strip())
            if isinstance(data, dict) and "sub_assertions" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    # strategy 1: direct json
    data = _try_parse(text)

    # strategy 2: code fence
    if data is None:
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            data = _try_parse(fence_match.group(1))

    if data is None:
        return None

    raw_subs = data.get("sub_assertions", [])
    if not isinstance(raw_subs, list):
        return None

    sub_assertions = []
    for item in raw_subs:
        if not isinstance(item, dict):
            continue
        sub_assertions.append(SubAssertion(
            question=str(item.get("question", "")),
            passed=bool(item.get("passed", False)),
            reason=str(item.get("reason", "")),
        ))

    if not sub_assertions:
        return sub_assertions, 0.0

    passed_count = sum(1 for sa in sub_assertions if sa.passed)
    confidence = passed_count / len(sub_assertions)
    return sub_assertions, confidence


class JudgeAgent(DokumenAgent):
    """Judge agent that evaluates executor output via the SDK.

    Features:
    - Verdict parsing with multiple JSON extraction strategies
    - Session-resume retry on structural failure (malformed JSON)
    - Timeout handling with error result
    - max_turns defaults to 3 (judge should be a quick evaluation)

    Usage:
        judge = JudgeAgent(
            id="accuracy",
            system_prompt="Evaluate the executor's response...",
            user_prompt="Check if all conditions are met...",
            include_executor_output=True,
        )
        result = await judge.run(executor_result)
    """

    def __init__(
        self,
        include_executor_output: bool = True,
        max_response_chars: int = 0,
        decomposed: bool = False,
        decomposed_threshold: float = 0.5,
        **kwargs,
    ):
        """Initialize a JudgeAgent.

        Args:
            include_executor_output: Whether to include executor output in context.
            max_response_chars: Max chars for executor response in context. 0 = unlimited.
            decomposed: Use decomposed binary sub-assertion judging.
            decomposed_threshold: Fraction of sub-assertions that must pass (0.0-1.0).
            **kwargs: Passed to DokumenAgent.__init__.
        """
        self.include_executor_output = include_executor_output
        self.max_response_chars = max_response_chars
        self.decomposed = decomposed
        self.decomposed_threshold = max(0.0, min(1.0, decomposed_threshold))
        # Judge defaults to max_turns=3 (single call + potential retry)
        kwargs.setdefault("max_turns", 3)
        super().__init__(**kwargs)

    async def run(
        self,
        executor_result: ExecutorResult,
        executor_system_prompt: str = "",
        executor_user_prompt: str = "",
    ) -> JudgeVerdict:
        """Evaluate executor output and return a verdict.

        Args:
            executor_result: The executor's result to evaluate.
            executor_system_prompt: The executor's system prompt (for context).
            executor_user_prompt: The executor's user prompt (for context).

        Returns:
            JudgeVerdict with verdict, confidence, and reason.
        """
        logger.info(
            "Judge starting evaluation",
            extra={
                "judge_id": self.id,
                "include_executor_output": self.include_executor_output,
                "executor_success": executor_result.success,
                "timeout": self.timeout,
                "has_executor_prompts": bool(executor_system_prompt or executor_user_prompt),
            },
        )

        context = build_judge_context(
            executor_result=executor_result,
            judge_prompt=self.user_prompt,
            include_output=self.include_executor_output,
            max_response_chars=self.max_response_chars,
            executor_system_prompt=executor_system_prompt,
            executor_user_prompt=executor_user_prompt,
        )

        # Run the judge query
        try:
            qr = await asyncio.wait_for(
                self._collect(context),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Judge timed out",
                extra={"judge_id": self.id, "timeout": self.timeout},
            )
            return self._timeout_result()

        # Parse verdict from result
        result_text = qr.result.result if qr.result else None
        verdict = None
        sub_assertions = []

        if self.decomposed:
            # try decomposed parsing first
            decomposed_result = parse_decomposed_verdict(result_text)
            if decomposed_result is not None:
                sub_assertions, confidence = decomposed_result
                passed = confidence >= self.decomposed_threshold
                failed_subs = [sa for sa in sub_assertions if not sa.passed]
                failure_reasons = "; ".join(
                    f"{sa.question}: {sa.reason}" for sa in failed_subs
                )
                verdict = Verdict(
                    passed=passed,
                    confidence=confidence,
                    reason=failure_reasons if not passed else "all sub-assertions passed",
                )
                logger.info(
                    "Decomposed verdict parsed",
                    extra={
                        "judge_id": self.id,
                        "sub_count": len(sub_assertions),
                        "passed_count": sum(1 for sa in sub_assertions if sa.passed),
                        "confidence": confidence,
                        "threshold": self.decomposed_threshold,
                    },
                )

        # fall back to regular verdict parsing (or if not decomposed mode)
        if verdict is None:
            verdict = parse_verdict(result_text)

        if verdict is None and qr.session_id:
            # Structural failure — retry with correction prompt
            logger.warning(
                "Judge verdict parse failed, retrying with session resume",
                extra={
                    "judge_id": self.id,
                    "session_id": qr.session_id,
                    "response_preview": result_text[:200] if result_text else None,
                },
            )
            verdict = await self._retry_verdict(qr.session_id)

        if verdict is None:
            logger.error(
                "Judge failed to produce valid verdict after retry",
                extra={"judge_id": self.id},
            )

        usage = extract_usage(qr.result)

        # Determine failure_reason:
        # - verdict is None (parse error): "Failed to parse verdict"
        # - verdict FAIL: the verdict's reason
        # - verdict PASS: None
        if verdict is None:
            failure_reason = "Failed to parse verdict"
        elif not verdict.passed:
            failure_reason = verdict.reason
        else:
            failure_reason = None

        result = JudgeVerdict(
            judge_id=self.id,
            passed=verdict.passed if verdict else False,
            failure_reason=failure_reason,
            reason=verdict.reason if verdict else None,
            confidence=verdict.confidence if verdict else 0.0,
            response=result_text,
            error=verdict is None,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            tool_calls=extract_tool_calls(qr.messages),
            conversation_log=build_conversation_log(qr.messages),
            sub_assertions=sub_assertions,
        )

        logger.info(
            "Judge completed",
            extra={
                "judge_id": self.id,
                "passed": result.passed,
                "confidence": result.confidence,
                "error": result.error,
            },
        )

        return result

    def _timeout_result(self) -> JudgeVerdict:
        """Create a timeout error result."""
        return JudgeVerdict(
            judge_id=self.id,
            passed=False,
            failure_reason=f"Judge timed out after {self.timeout}s",
            confidence=0.0,
            error=True,
        )

    async def _retry_verdict(self, session_id: str) -> Optional[Verdict]:
        """Resume session with strict JSON instruction.

        Args:
            session_id: The session to resume.

        Returns:
            Parsed Verdict or None if retry also fails.
        """
        logger.info(
            "Retrying verdict with session resume",
            extra={"judge_id": self.id, "session_id": session_id},
        )

        try:
            async for msg in self._runner.run(
                JUDGE_RETRY_PROMPT,
                ClaudeAgentOptions(resume=session_id),
            ):
                if isinstance(msg, ResultMessage):
                    verdict = parse_verdict(msg.result)
                    if verdict:
                        logger.info(
                            "Verdict retry succeeded",
                            extra={
                                "judge_id": self.id,
                                "passed": verdict.passed,
                            },
                        )
                    return verdict
        except Exception as e:
            logger.error(
                "Verdict retry failed with exception",
                extra={"judge_id": self.id, "error": str(e)},
            )
        return None
