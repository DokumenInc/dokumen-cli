"""Judge stage — runs all judges in parallel and aggregates results."""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger
from ..pipeline import PipelineContext, PipelineStage
from ..sdk.types import JudgeVerdict

logger = get_logger(__name__)

# Re-export for backward compat — JudgeResult is now JudgeVerdict
JudgeResult = JudgeVerdict


def _extract_report_markdown(response: Optional[str]) -> str:
    """Extract markdown report from verdict judge response.

    The verdict judge produces a markdown report followed by a JSON verdict block.
    This helper splits them, returning only the markdown report.

    Strategies:
    1. Look for ```json { ... "verdict" ... } ``` code fence at end
    2. Fallback: find last '{' that parses as JSON with "verdict" key
    3. If no JSON found, return full response
    """
    if not response:
        return ""

    # Strategy 1: Look for ```json ... ``` block containing "verdict"
    pattern = r'```json\s*\n?\s*(\{[^`]*?"verdict"[^`]*?\})\s*\n?\s*```'
    matches = list(re.finditer(pattern, response, re.DOTALL))
    if matches:
        last_match = matches[-1]
        report = response[: last_match.start()].rstrip()
        logger.debug("report.extract.json_fence", report_length=len(report))
        return report

    # Strategy 2: Find last '{' that parses as JSON with "verdict" key
    last_brace = response.rfind("{")
    while last_brace >= 0:
        try:
            candidate = response[last_brace:]
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "verdict" in parsed:
                report = response[:last_brace].rstrip()
                logger.debug("report.extract.inline_json",
                             report_length=len(report))
                return report
        except (json.JSONDecodeError, ValueError):
            pass
        last_brace = response.rfind("{", 0, last_brace)

    # Strategy 3: No JSON verdict found, return full response
    logger.debug("report.extract.no_json", response_length=len(response))
    return response


class JudgeStage(PipelineStage):
    """Run all judges in parallel and aggregate results.

    This stage:
    1. Runs all judges concurrently via asyncio.gather
    2. Fires on_judge_complete callbacks
    3. Extracts research reports from verdict judges
    4. Aggregates token usage
    5. Determines final pass/fail status
    """

    @property
    def name(self) -> str:
        return "judge"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute all judges in parallel.

        Args:
            ctx: The pipeline context (must have executor_output set).

        Returns:
            Updated context with judge_results populated.
        """
        logger.info("stage.judge.start", test_id=ctx.test_id,
                     judge_count=len(ctx.judges))

        async def run_single_judge(judge: Any) -> JudgeVerdict:
            """Run a single judge and handle exceptions."""
            logger.debug("stage.judge.run_one", test_id=ctx.test_id,
                         judge_id=judge.id)
            try:
                if ctx.executor_output and not getattr(ctx.executor_output, "final_response", None):
                    conversation_log = getattr(ctx.executor_output, "conversation_log", None) or []
                    last_assistant = None
                    for entry in reversed(conversation_log):
                        if isinstance(entry, dict) and entry.get("role") == "assistant":
                            content = entry.get("content")
                            if isinstance(content, str) and content.strip():
                                last_assistant = content.strip()
                                break
                    if last_assistant:
                        ctx.executor_output.final_response = last_assistant
                        logger.warning(
                            "stage.judge.reconstructed_executor_response",
                            test_id=ctx.test_id,
                        )

                judge_result = await judge.run(
                    executor_output=ctx.executor_output,
                    on_tool_call=ctx.on_tool_call,
                    on_conversation_message=ctx.on_conversation_message,
                    executor_system_prompt=ctx.executor.system_prompt,
                    executor_user_prompt=ctx.executor.user_prompt,
                )
                logger.info("stage.judge.complete_one", test_id=ctx.test_id,
                            judge_id=judge.id, passed=judge_result.passed)
                return judge_result
            except Exception as e:
                logger.error("stage.judge.error", test_id=ctx.test_id,
                             judge_id=judge.id, error=str(e),
                             error_type=type(e).__name__, exc_info=True)
                return JudgeVerdict(
                    judge_id=judge.id,
                    passed=False,
                    failure_reason=f"Judge error: {str(e)}",
                    assertion_text=judge._get_assertion_text(),
                )

        # Run all judges concurrently
        judge_results_list = await asyncio.gather(
            *[run_single_judge(judge) for judge in ctx.judges]
        )

        # Process results
        for judge, judge_result in zip(ctx.judges, judge_results_list):
            ctx.judge_results.append(judge_result)

            if ctx.on_judge_complete:
                ctx.on_judge_complete(judge_result)

            if not judge_result.passed:
                logger.debug("stage.judge.failed", test_id=ctx.test_id,
                             judge_id=judge.id,
                             reason=judge_result.failure_reason)
                ctx.failure_reasons.append(
                    f"Judge {judge.id} failed: {judge_result.failure_reason}"
                )

        # Summary
        passed_count = sum(1 for jr in ctx.judge_results if jr.passed)
        failed_count = len(ctx.judge_results) - passed_count
        logger.info("stage.judge.summary", test_id=ctx.test_id,
                     total=len(ctx.judge_results),
                     passed=passed_count, failed=failed_count)

        # Extract research reports from verdict judge
        if ctx.test_type == "research":
            for judge, judge_result in zip(ctx.judges, judge_results_list):
                if judge.id == "verdict" and judge_result.response:
                    report_markdown = _extract_report_markdown(
                        judge_result.response
                    )
                    if report_markdown:
                        report_path = os.path.join(
                            ".dokumen-cache", "output", ctx.test_id, "report.md"
                        )
                        os.makedirs(os.path.dirname(report_path), exist_ok=True)
                        with open(report_path, "w") as f:
                            f.write(report_markdown)
                        ctx.research_report_rel_path = "report.md"
                        logger.info("stage.judge.report_saved",
                                    test_id=ctx.test_id,
                                    report_path=report_path,
                                    size_bytes=len(report_markdown.encode("utf-8")))

        logger.info("stage.judge.complete", test_id=ctx.test_id,
                     passed=passed_count, failed=failed_count)
        return ctx
