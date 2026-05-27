"""
Output file writers for .dokumen-cache/ artifacts.

Writes spec-compliant output files after each test run:
- results.json - Test results with assertions (spec format)
- junit.xml - GitLab CI compatible JUnit XML
- coverage.json - File-level coverage metrics
- debug/{test-name}-{timestamp}.json - Debug traces (when --debug)
- explore/{test-id}.json - Per-test explore traces (full output, always)
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ..output_schemas import (
    ResultsJsonOutput,
    TestOutputResult,
    AssertionResult,
    ResultsSummary,
    CoverageSection,
    CoverageJsonOutput,
    CoverageFile,
    CoverageSummary,
    DebugTraceOutput,
    DebugExecutor,
    DebugJudge,
    DebugJudgeAssertion,
    DebugMessage,
    DebugToolCall,
    ExploreToolCall,
    TokenUsage,
    JudgePromptInfo,
    BrowserArtifact,
    ReportArtifact,
    OutputArtifact,
    ConversationToolCall,
    ConversationIteration,
    ExecutorConversationLog,
    JudgeConversationLog,
)

CACHE_DIR = ".dokumen-cache"


def _str_or_none(value: Any) -> str | None:
    """Return value if it's a string, otherwise None.

    Safely extracts Optional[str] from objects that may return
    non-string sentinel values (e.g. MagicMock in tests).
    """
    return value if isinstance(value, str) else None


def _dict_str_str_or_none(value: Any) -> dict[str, str] | None:
    """Return value if it's a dict of str→str, otherwise None.

    Safely extracts Optional[Dict[str, str]] from objects that may return
    non-dict sentinel values (e.g. MagicMock in tests).
    """
    if isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        return value
    return None


class OutputWriter:
    """Writes output files to .dokumen-cache/ directory."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        """Initialize with cache directory path.

        Args:
            cache_dir: Path to cache directory (default: .dokumen-cache)
        """
        self.cache_dir = Path(cache_dir)

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        results: Any,
        coverage_stats: Dict[str, Any],
        debug_enabled: bool = False,
    ) -> None:
        """Write all output files.

        Args:
            results: TestSuiteResults object
            coverage_stats: Coverage statistics dict from get_coverage_stats()
            debug_enabled: If True, also write debug trace files
        """
        self._ensure_cache_dir()

        # Pass coverage_stats to include coverage section in results.json
        self.write_results_json(results, coverage_stats=coverage_stats)
        self.write_junit_xml(results)
        self.write_coverage_json(coverage_stats)

        self.write_explore_traces(results)

        if debug_enabled:
            self.write_debug_traces(results)

    def write_results_json(
        self, results: Any, coverage_stats: Dict[str, Any] = None
    ) -> Path:
        """Write results.json per spec schema.

        Args:
            results: TestSuiteResults object
            coverage_stats: Optional coverage statistics dict from get_coverage_stats().
                           If provided, coverage section will be included in results.json
                           containing ALL files in the documentation scope.

        Returns:
            Path to written file
        """
        self._ensure_cache_dir()

        # Transform test results to spec format
        tests_output: List[TestOutputResult] = []
        for tr in results.test_results:
            # Extract assertions from judge results
            assertions: List[AssertionResult] = []
            has_judge_error = False
            for jr in getattr(tr, 'judge_results', []):
                # Per PHASE0-CLI-SPEC: assertion should be the question text, not judge_id
                assertion_text = getattr(jr, 'assertion_text', None) or jr.judge_id
                jr_error = getattr(jr, 'error', None)
                if jr_error:
                    has_judge_error = True
                jr_reason = getattr(jr, 'reason', None)
                jr_reason = jr_reason if isinstance(jr_reason, str) else None
                assertions.append(AssertionResult(
                    assertion=assertion_text,
                    passed=jr.passed,
                    reasoning=jr_reason or jr.failure_reason or jr.response or "",
                    error=jr_error,
                ))

            # Get error message if test failed
            error_msg = None
            failure_reasons = getattr(tr, 'failure_reasons', [])
            if failure_reasons and not tr.passed:
                error_msg = failure_reasons[0] if failure_reasons else None

            # Extract executor final response if available
            executor_final_response = None
            executor_output = getattr(tr, 'executor_output', None)
            if executor_output and hasattr(executor_output, 'final_response'):
                executor_final_response = executor_output.final_response or None

            # Extract explore results if available
            explore_output = getattr(tr, 'explore_output', None)
            explore_tool_calls_raw = getattr(tr, 'explore_tool_calls', None)
            explore_tool_calls = None
            if explore_tool_calls_raw:
                explore_tool_calls = [
                    ExploreToolCall(
                        tool=tc.get('tool', ''),
                        command=tc.get('command', ''),
                        output=tc.get('output', '')[:500]  # Limit output size
                    )
                    for tc in explore_tool_calls_raw
                    if isinstance(tc, dict)
                ]

            # Extract token usage per phase
            executor_tokens = TokenUsage(
                input_tokens=getattr(tr, 'executor_input_tokens', 0),
                output_tokens=getattr(tr, 'executor_output_tokens', 0),
                cache_creation_tokens=getattr(tr, 'executor_cache_creation_tokens', 0),
                cache_read_tokens=getattr(tr, 'executor_cache_read_tokens', 0),
            )
            judge_tokens = TokenUsage(
                input_tokens=getattr(tr, 'judge_input_tokens', 0),
                output_tokens=getattr(tr, 'judge_output_tokens', 0),
                cache_creation_tokens=getattr(tr, 'judge_cache_creation_tokens', 0),
                cache_read_tokens=getattr(tr, 'judge_cache_read_tokens', 0),
            )
            explore_tokens = TokenUsage(
                input_tokens=getattr(tr, 'explore_input_tokens', 0),
                output_tokens=getattr(tr, 'explore_output_tokens', 0),
                cache_creation_tokens=getattr(tr, 'explore_cache_creation_tokens', 0),
                cache_read_tokens=getattr(tr, 'explore_cache_read_tokens', 0),
            )

            # Extract executor prompts from executor_output object
            # Prefer original_user_prompt (before explore context injection) for UI display
            executor_system_prompt = None
            executor_user_prompt = None
            if executor_output and hasattr(executor_output, 'system_prompt'):
                executor_system_prompt = executor_output.system_prompt or None
            if executor_output:
                # Use original_user_prompt if available (shows the actual task instructions)
                # Falls back to user_prompt if original is not set
                original = getattr(executor_output, 'original_user_prompt', None)
                if original:
                    executor_user_prompt = original
                elif hasattr(executor_output, 'user_prompt'):
                    executor_user_prompt = executor_output.user_prompt or None

            # Extract judge prompts from judge_prompts field on TestResult
            judge_prompts_output = None
            judge_prompts_raw = getattr(tr, 'judge_prompts', None)
            if judge_prompts_raw:
                judge_prompts_output = [
                    JudgePromptInfo(
                        name=jp.get('name', ''),
                        system_prompt=jp.get('system_prompt'),
                        user_prompt=jp.get('user_prompt'),
                    )
                    for jp in judge_prompts_raw
                    if isinstance(jp, dict)
                ]

            # Extract browser artifacts (videos/screenshots) if available
            browser_artifacts_output = None
            browser_artifacts_raw = getattr(tr, 'browser_artifacts', None)
            if browser_artifacts_raw:
                browser_artifacts_output = [
                    BrowserArtifact(
                        type=ba.get('type', 'screenshot'),
                        path=ba.get('path', ''),
                        filename=ba.get('filename', ''),
                        size_bytes=ba.get('size_bytes')
                    )
                    for ba in browser_artifacts_raw
                    if isinstance(ba, dict)
                ]

            # Extract report artifacts (research reports) if available
            report_artifacts_output = None
            report_artifacts_raw = getattr(tr, 'report_artifacts', None)
            if report_artifacts_raw:
                report_artifacts_output = [
                    ReportArtifact(
                        type=ra.get('type', 'report'),
                        path=ra.get('path', ''),
                        filename=ra.get('filename', ''),
                        size_bytes=ra.get('size_bytes'),
                        content=ra.get('content')
                    )
                    for ra in report_artifacts_raw
                    if isinstance(ra, dict)
                ]

            # Extract output artifacts (files from output folder) if available
            output_artifacts_output = None
            output_artifacts_raw = getattr(tr, 'output_artifacts', None)
            if output_artifacts_raw:
                output_artifacts_output = [
                    OutputArtifact(
                        filename=oa.get('filename', ''),
                        path=oa.get('path', ''),
                        size_bytes=oa.get('size_bytes'),
                        content_type=oa.get('content_type', 'application/octet-stream'),
                        content=oa.get('content'),
                        source=oa.get('source'),
                    )
                    for oa in output_artifacts_raw
                    if isinstance(oa, dict)
                ]

            # Build executor conversation log if available
            executor_conversation_output = None
            executor_conv_raw = getattr(tr, 'executor_conversation_log', None)
            if executor_conv_raw:
                conv_iterations = [
                    ConversationIteration(
                        iteration=it.get('iteration', 0),
                        response_content=it.get('response_content'),
                        tool_calls=[
                            ConversationToolCall(
                                tool=tc.get('tool', ''),
                                input=tc.get('input', {}),
                                output=tc.get('output', ''),
                            )
                            for tc in it.get('tool_calls', [])
                        ]
                    )
                    for it in executor_conv_raw
                    if isinstance(it, dict)
                ]
                executor_conversation_output = ExecutorConversationLog(
                    iterations=conv_iterations,
                    total_iterations=len(conv_iterations),
                )

            # Build judge conversation logs if available
            judge_conversations_output = None
            judge_conv_raw = getattr(tr, 'judge_conversation_logs', None)
            if judge_conv_raw:
                judge_conversations_output = []
                for jc in judge_conv_raw:
                    if not isinstance(jc, dict):
                        continue
                    jc_iterations = [
                        ConversationIteration(
                            iteration=it.get('iteration', 0),
                            response_content=it.get('response_content'),
                            tool_calls=[
                                ConversationToolCall(
                                    tool=tc.get('tool', ''),
                                    input=tc.get('input', {}),
                                    output=tc.get('output', ''),
                                )
                                for tc in it.get('tool_calls', [])
                            ]
                        )
                        for it in jc.get('iterations', [])
                        if isinstance(it, dict)
                    ]
                    judge_conversations_output.append(JudgeConversationLog(
                        judge_name=jc.get('judge_name', ''),
                        iterations=jc_iterations,
                        total_iterations=len(jc_iterations),
                    ))

            # Read scaffold YAML content if source_path is available
            scaffold_yaml = None
            source_path = getattr(tr, 'source_path', None)
            if source_path and isinstance(source_path, str):
                try:
                    with open(source_path, 'r') as f:
                        scaffold_yaml = f.read()
                except Exception:
                    pass  # File may not be available in all contexts

            # Determine status: error if any judge errored, else passed/failed
            test_status = "error" if has_judge_error else ("passed" if tr.passed else "failed")

            tests_output.append(TestOutputResult(
                name=tr.test_id,
                status=test_status,
                duration_ms=int(tr.duration * 1000),
                files=getattr(tr, 'files', []),
                assertions=assertions,
                error=error_msg,
                executor_output=executor_final_response,
                explore_output=explore_output,
                explore_status=_str_or_none(getattr(tr, 'explore_status', None)),
                explore_tool_calls=explore_tool_calls,
                executor_model=_str_or_none(getattr(tr, 'executor_model', None)),
                judge_model=_str_or_none(getattr(tr, 'judge_model', None)),
                judge_models=_dict_str_str_or_none(getattr(tr, 'judge_models', None)),
                explore_model=_str_or_none(getattr(tr, 'explore_model', None)),
                executor_tokens=executor_tokens,
                judge_tokens=judge_tokens,
                explore_tokens=explore_tokens,
                executor_system_prompt=executor_system_prompt,
                executor_user_prompt=executor_user_prompt,
                judge_prompts=judge_prompts_output,
                browser_artifacts=browser_artifacts_output,
                report_artifacts=report_artifacts_output,
                output_artifacts=output_artifacts_output,
                executor_tools=getattr(tr, 'executor_tools', []),
                executor_conversation=executor_conversation_output,
                judge_conversations=judge_conversations_output,
                scaffold_yaml=scaffold_yaml,
            ))

        # Build coverage section if stats provided
        # This includes ALL files in doc scope (from dokumen.yaml coverage patterns)
        coverage_section = None
        if coverage_stats:
            files_detail = coverage_stats.get('files_detail', {})
            coverage_files: List[CoverageFile] = []

            for file_path, detail in files_detail.items():
                status = detail.get('status', 'uncovered')
                test_ids = detail.get('test_ids', [])
                # Per PHASE0-CLI-SPEC: "covered" means tested (passed OR failed)
                coverage_files.append(CoverageFile(
                    path=file_path,
                    covered=(status in ('passed', 'failed')),
                    tests=test_ids
                ))

            coverage_section = CoverageSection(
                total_files=coverage_stats.get('total', len(files_detail)),
                covered_files=coverage_stats.get('passed', 0),
                percentage=coverage_stats.get('percentage', 0.0),
                files=coverage_files
            )

        # Calculate total token usage across all tests
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_creation_tokens = 0
        total_cache_read_tokens = 0
        for tr in results.test_results:
            total_input_tokens += getattr(tr, 'executor_input_tokens', 0)
            total_input_tokens += getattr(tr, 'judge_input_tokens', 0)
            total_input_tokens += getattr(tr, 'explore_input_tokens', 0)
            total_output_tokens += getattr(tr, 'executor_output_tokens', 0)
            total_output_tokens += getattr(tr, 'judge_output_tokens', 0)
            total_output_tokens += getattr(tr, 'explore_output_tokens', 0)
            total_cache_creation_tokens += getattr(tr, 'executor_cache_creation_tokens', 0)
            total_cache_creation_tokens += getattr(tr, 'judge_cache_creation_tokens', 0)
            total_cache_creation_tokens += getattr(tr, 'explore_cache_creation_tokens', 0)
            total_cache_read_tokens += getattr(tr, 'executor_cache_read_tokens', 0)
            total_cache_read_tokens += getattr(tr, 'judge_cache_read_tokens', 0)
            total_cache_read_tokens += getattr(tr, 'explore_cache_read_tokens', 0)

        # Count error-status tests
        error_count = sum(1 for t in tests_output if t.status == "error")

        # Build output object
        output = ResultsJsonOutput(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_ms=int(results.duration * 1000),
            tests=tests_output,
            summary=ResultsSummary(
                total=results.total_tests,
                passed=results.passed,
                failed=results.failed,
                skipped=getattr(results, 'skipped', 0),
                error=error_count,
            ),
            coverage=coverage_section,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cache_creation_tokens=total_cache_creation_tokens,
            total_cache_read_tokens=total_cache_read_tokens,
        )

        # Write file
        output_path = self.cache_dir / "results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output.model_dump(), f, indent=2)

        return output_path

    def write_junit_xml(self, results: Any) -> Path:
        """Write junit.xml for GitLab CI.

        Args:
            results: TestSuiteResults object

        Returns:
            Path to written file
        """
        self._ensure_cache_dir()

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="dokumen" tests="{results.total_tests}" '
            f'failures="{results.failed + results.error}" time="{results.duration:.2f}">',
        ]

        for tr in results.test_results:
            duration = getattr(tr, 'duration', 0)
            if tr.passed:
                lines.append(f'  <testcase name="{tr.test_id}" time="{duration:.2f}"/>')
            else:
                lines.append(f'  <testcase name="{tr.test_id}" time="{duration:.2f}">')
                failure_reasons = getattr(tr, 'failure_reasons', ['Unknown failure'])
                message = "; ".join(failure_reasons) if isinstance(failure_reasons, list) else str(failure_reasons)
                # Escape XML special characters
                message = self._escape_xml(message)
                lines.append(f'    <failure message="{message}"/>')
                lines.append('  </testcase>')

        lines.append('</testsuite>')

        output_path = self.cache_dir / "junit.xml"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        return output_path

    def write_coverage_json(self, coverage_stats: Dict[str, Any]) -> Path:
        """Write coverage.json per spec schema.

        Args:
            coverage_stats: Coverage statistics dict from get_coverage_stats()

        Returns:
            Path to written file
        """
        self._ensure_cache_dir()

        # Build files list from files_detail
        files_detail = coverage_stats.get('files_detail', {})
        files_output: List[CoverageFile] = []

        for file_path, detail in files_detail.items():
            status = detail.get('status', 'uncovered')
            test_ids = detail.get('test_ids', [])
            # Per PHASE0-CLI-SPEC: "covered" means tested (passed OR failed)
            # Only files with status='uncovered' should have covered=False
            files_output.append(CoverageFile(
                path=file_path,
                covered=(status in ('passed', 'failed')),
                tests=test_ids
            ))

        # Calculate summary
        total = coverage_stats.get('total', len(files_detail))
        covered = coverage_stats.get('passed', 0)
        percentage = coverage_stats.get('percentage', 0.0)

        output = CoverageJsonOutput(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            summary=CoverageSummary(
                total_files=total,
                covered_files=covered,
                percentage=percentage
            ),
            files=files_output
        )

        output_path = self.cache_dir / "coverage.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output.model_dump(), f, indent=2)

        return output_path

    def write_explore_traces(self, results: Any) -> None:
        """Write per-test explore trace files to .dokumen-cache/explore/.

        Each test with explore data gets a dedicated JSON file containing
        the full untruncated ExploreToolCall data (unlike results.json which
        truncates output to 500 chars).

        Args:
            results: TestSuiteResults object
        """
        logger = logging.getLogger(__name__)

        explore_dir = self.cache_dir / "explore"
        explore_dir.mkdir(parents=True, exist_ok=True)

        for tr in results.test_results:
            explore_tool_calls_raw = getattr(tr, 'explore_tool_calls', None)
            if not explore_tool_calls_raw:
                continue

            trace = {
                "test_id": tr.test_id,
                "explore_status": getattr(tr, 'explore_status', None),
                "explore_model": getattr(tr, 'explore_model', None),
                "explore_output": getattr(tr, 'explore_output', None),
                "tool_calls": [
                    {
                        "tool": tc.get('tool', ''),
                        "command": tc.get('command', ''),
                        "output": tc.get('output', ''),
                    }
                    for tc in explore_tool_calls_raw
                    if isinstance(tc, dict)
                ],
                "tokens": {
                    "input": getattr(tr, 'explore_input_tokens', 0),
                    "output": getattr(tr, 'explore_output_tokens', 0),
                },
            }

            safe_id = tr.test_id.replace("/", "_").replace("..", "_")
            output_path = explore_dir / f"{safe_id}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(trace, f, indent=2)

            logger.info(
                "Wrote explore trace",
                extra={"test_id": tr.test_id, "path": str(output_path), "tool_calls": len(trace["tool_calls"])}
            )

    def write_debug_traces(self, results: Any) -> None:
        """Write debug trace files for each test.

        Args:
            results: TestSuiteResults object
        """
        debug_dir = self.cache_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for tr in results.test_results:
            self._write_single_debug_trace(tr, debug_dir)

    def _write_single_debug_trace(self, test_result: Any, debug_dir: Path) -> Path:
        """Write debug trace for a single test.

        Args:
            test_result: TestResult object
            debug_dir: Directory to write trace file

        Returns:
            Path to written file
        """
        # Extract executor data
        executor_output = getattr(test_result, 'executor_output', None)
        executor = DebugExecutor(
            # Per PHASE0-CLI-SPEC: Debug traces must include executor's prompts
            system_prompt=getattr(executor_output, 'system_prompt', '') if executor_output else "",
            user_prompt=getattr(executor_output, 'user_prompt', '') if executor_output else "",
            messages=[],
            final_output=executor_output.final_response if executor_output else ""
        )

        # Extract tool calls if available
        # Per PHASE0-CLI-SPEC: messages should use role="assistant" with tool_calls
        if executor_output and hasattr(executor_output, 'tool_calls'):
            tool_calls = executor_output.tool_calls
            if tool_calls:
                # Group all tool calls into one assistant message
                debug_tool_calls = [
                    DebugToolCall(
                        tool=getattr(tc, 'tool_name', ''),
                        input=getattr(tc, 'parameters', {}),
                        output=str(getattr(tc, 'result', ''))
                    )
                    for tc in tool_calls
                ]
                executor.messages.append(DebugMessage(
                    role="assistant",
                    content="Executing tools to complete the task...",
                    tool_calls=debug_tool_calls
                ))

        # Extract judge data
        judge_assertions: List[DebugJudgeAssertion] = []
        for jr in getattr(test_result, 'judge_results', []):
            # Per PHASE0-CLI-SPEC: assertion should be the question text, not judge_id
            assertion_text = getattr(jr, 'assertion_text', None) or jr.judge_id
            judge_assertions.append(DebugJudgeAssertion(
                assertion=assertion_text,
                evaluation=AssertionResult(
                    assertion=assertion_text,
                    passed=jr.passed,
                    reasoning=jr.failure_reason or jr.response or ""
                )
            ))

        judge = DebugJudge(assertions=judge_assertions)

        # Build output
        timestamp = getattr(test_result, 'timestamp', datetime.now())
        if timestamp is None:
            timestamp = datetime.now()

        output = DebugTraceOutput(
            test_name=test_result.test_id,
            started_at=timestamp.isoformat(),
            completed_at=datetime.now().isoformat(),
            executor=executor,
            judge=judge
        )

        # Generate filename with test name and timestamp
        ts_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{test_result.test_id}-{ts_str}.json"
        output_path = debug_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output.model_dump(), f, indent=2)

        return output_path

    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters.

        Args:
            text: Text to escape

        Returns:
            Escaped text safe for XML attributes
        """
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))
