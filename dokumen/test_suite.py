"""
Test Suite module for the Skill Testing Framework.

Manages collections of tests, caching, and coverage reporting.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Any, TYPE_CHECKING
from datetime import datetime
import json
import os
import time

from .debug import debug
from .file_object import (
    FileObject,
    FileMetrics,
    FileStatus,
    IncorrectLine,
    LineCoverage,
    normalize_path,
)
from .logging_config import get_logger
from .test_object import FailureAnalysis, TestObject, TestResult

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .coverage_agent import CoverageAgent
    from .sandbox import SandboxConfig
    from .history import HistoryManager

# Type alias for progress callbacks
ProgressCallback = Callable[[str, str, Optional[Any]], None]

# Type alias for tool call callbacks (fired when executor calls a tool)
ToolCallCallback = Callable[[str, dict, Any], None]


def _log_tool_provenance(test_id: str, provenance) -> None:
    """Log detailed tool provenance to stderr (always visible in CI).

    Supplements the condensed tool_provenance.built structured log line
    with per-tool source details so CI pipelines show exactly where
    each tool came from regardless of output format.
    """
    if not provenance:
        return
    prov = provenance.to_dict()

    # Group executor tools by source
    executor = prov.get("executor_tools", {})
    if executor:
        groups: dict[str, list[str]] = {}
        for tool_name, source in executor.items():
            groups.setdefault(source, []).append(tool_name)
        parts = [f"{', '.join(sorted(tools))} ({source})" for source, tools in groups.items()]
        logger.info("Executor tools", test_id=test_id, tools="; ".join(parts))

    # Log judge tools
    for judge_name, tools in prov.get("judge_tools", {}).items():
        if tools:
            parts = [f"{t} ({s})" for t, s in tools.items()]
            logger.info("Judge tools", test_id=test_id, judge=judge_name, tools=", ".join(parts))

    # Log explore tools
    explore = prov.get("explore_tools", {})
    if explore:
        sources = sorted(set(explore.values()))
        logger.info(
            "Explore tools",
            test_id=test_id,
            tools=", ".join(sorted(explore.keys())),
            source=", ".join(sources),
        )

    # Log overrides/removals
    if prov.get("overrides_active"):
        logger.info("Tool overrides active", test_id=test_id)
    if prov.get("removed_tools"):
        logger.info("Tools filtered out", test_id=test_id, removed=prov["removed_tools"])


@dataclass
class TestSuiteConfig:
    """Configuration options for the test suite."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    name: str
    cache_path: str = ".dokumen-cache"
    parallel_execution: bool = False
    max_concurrency: int = 4
    coverage_agent: Optional["CoverageAgent"] = None  # Agent for line-level coverage
    sandbox_config: Optional["SandboxConfig"] = None  # Sandbox config for isolated test execution


@dataclass
class TestSuiteResults:
    """Aggregated results from running the test suite."""

    __test__ = False  # Tell pytest this is not a test class
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float  # seconds
    test_results: List[TestResult]
    cached_results: int
    error: int = 0  # tests with status="error" (judge timeout/crash, infrastructure failures)


@dataclass
class CoverageReport:
    """Coverage statistics for knowledge files."""

    total_files: int
    covered_files: int
    coverage_percentage: float
    file_details: List[FileMetrics]


class TestSuite:
    """Manages a collection of test objects."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(self, config: TestSuiteConfig):
        """
        Initialize a test suite.

        Args:
            config: Configuration options for the suite
        """
        self.name = config.name
        self.config = config
        self.tests: List[TestObject] = []
        self._cached_results: Dict[str, TestResult] = {}
        self._last_results: Optional[TestSuiteResults] = None
        self._file_registry: Dict[str, FileObject] = {}
        self._file_status: Dict[str, FileStatus] = {}  # Track file status
        self._failure_analysis: Dict[str, Dict[str, FailureAnalysis]] = (
            {}
        )  # file_path -> {test_id -> analysis}
        self._coverage_attempted: set = set()  # Track files where coverage inference was attempted

    def add_test(self, test: TestObject) -> None:
        """
        Add a test to the suite.

        Args:
            test: TestObject to add
        """
        self.tests.append(test)

    def remove_test(self, test_id: str) -> bool:
        """
        Remove a test from the suite by ID.

        Args:
            test_id: ID of test to remove

        Returns:
            True if test was removed, False if not found
        """
        for i, test in enumerate(self.tests):
            if test.id == test_id:
                self.tests.pop(i)
                return True
        return False

    def get_test(self, test_id: str) -> Optional[TestObject]:
        """
        Get a test by ID.

        Args:
            test_id: ID of test to find

        Returns:
            TestObject if found, None otherwise
        """
        for test in self.tests:
            if test.id == test_id:
                return test
        return None

    async def run(
        self,
        on_progress: Optional[ProgressCallback] = None,
        on_tool_call: Optional[ToolCallCallback] = None,
        on_conversation_message: Optional[Callable] = None,
        on_executor_complete: Optional[Callable] = None,
        on_judge_complete: Optional[Callable] = None,
        history_manager: Optional["HistoryManager"] = None,
        run_id: Optional[str] = None,
    ) -> TestSuiteResults:
        """
        Execute all tests in the suite.

        Args:
            on_progress: Optional callback called with (event, test_id, data)
                         Events: 'start', 'complete', 'cached'
            on_tool_call: Optional callback fired after each tool execution.
                         Signature: (tool_name: str, params: dict, result: Any) -> None
            on_conversation_message: Optional callback for streaming conversation.
                         Signature: (agent_type: str, message_type: str, content: str) -> None
            on_executor_complete: Optional callback fired when executor finishes.
                         Signature: (test_id: str, executor_output: ExecutorOutput) -> None
            on_judge_complete: Optional callback fired when each judge finishes.
                         Signature: (test_id: str, judge_result: JudgeResult) -> None
            history_manager: Optional HistoryManager for incremental history saves
            run_id: Optional unique identifier for this run (for history/debug filenames)

        Returns:
            TestSuiteResults with aggregated results
        """
        start_time = time.time()
        results: List[TestResult] = []
        cached_count = 0

        logger.info(
            "suite.run.start", test_count=len(self.tests), parallel=self.config.parallel_execution
        )

        if self.config.parallel_execution:
            logger.info(
                "suite.parallel.start",
                max_concurrency=self.config.max_concurrency,
                test_count=len(self.tests),
            )
            results, cached_count = await self._run_concurrent(
                on_progress,
                on_tool_call,
                on_conversation_message,
                on_executor_complete,
                on_judge_complete,
                history_manager,
                run_id,
            )
        else:
            results, cached_count = await self._run_sequential(
                on_progress,
                on_tool_call,
                on_conversation_message,
                on_executor_complete,
                on_judge_complete,
                history_manager,
                run_id,
            )

        duration = time.time() - start_time
        logger.info(
            "suite.run.complete",
            total=len(results),
            cached=cached_count,
            duration_ms=int(duration * 1000),
        )
        self._last_results = self._aggregate_results(results, duration, cached_count)
        return self._last_results

    async def _run_sequential(
        self,
        on_progress: Optional[ProgressCallback] = None,
        on_tool_call: Optional[ToolCallCallback] = None,
        on_conversation_message: Optional[Callable] = None,
        on_executor_complete: Optional[Callable] = None,
        on_judge_complete: Optional[Callable] = None,
        history_manager: Optional["HistoryManager"] = None,
        run_id: Optional[str] = None,
    ) -> tuple:
        """
        Run tests sequentially.

        Args:
            on_progress: Optional callback called with (event, test_id, data)
            on_tool_call: Optional callback fired after each tool execution.
                         Signature: (tool_name: str, params: dict, result: Any) -> None
            on_conversation_message: Optional callback for streaming conversation.
                         Signature: (agent_type: str, message_type: str, content: str) -> None
            on_executor_complete: Optional callback fired when executor finishes.
                         Signature: (test_id: str, executor_output: ExecutorOutput) -> None
            on_judge_complete: Optional callback fired when each judge finishes.
                         Signature: (test_id: str, judge_result: JudgeResult) -> None
            history_manager: Optional HistoryManager for incremental history saves
            run_id: Optional unique identifier for this run (for history/debug filenames)

        Returns:
            Tuple of (results list, cached count)
        """
        from .debug import is_debug, get_debug_session

        results = []
        cached_count = 0

        for test in self.tests:
            # Check cache
            cached = self._cached_results.get(test.id)
            if cached and not test.is_stale():
                logger.info("cache.hit", test_id=test.id)
                if on_progress:
                    on_progress("cached", test.id, cached)
                results.append(cached)
                cached_count += 1
                continue
            elif cached and test.is_stale():
                logger.info("cache.stale", test_id=test.id)
            else:
                logger.info("cache.miss", test_id=test.id)

            # Log test header with important configuration
            model_name = (
                getattr(test.executor.provider, "model", "unknown")
                if test.executor.provider
                else "unknown"
            )
            judge_ids = [j.id for j in test.judges] if test.judges else []
            tool_names = [t.name for t in test.executor.tools] if test.executor.tools else []

            # Notify test start (pass tool names and provenance so CLI can print them)
            progress_data = {"tools": tool_names}
            if test.tool_provenance:
                progress_data["tool_provenance"] = test.tool_provenance.to_dict()
            if on_progress:
                on_progress("start", test.id, progress_data)
            _log_tool_provenance(test.id, test.tool_provenance)

            logger.info(
                "test.starting",
                test_id=test.id,
                reason=test.reason,
                model=model_name,
                executor_id=test.executor.id,
                judges=judge_ids,
                timeout=test.timeout,
                retries=test.retries,
                tools=tool_names,
                files=[str(f) for f in test.files],
            )

            # Start debug tracking for this test
            if is_debug():
                session = get_debug_session()
                if session:
                    session.start_test(test.id)
                    session.start_executor()

            # Create wrapper callbacks that include test_id
            def make_executor_complete_callback(test_id):
                def callback(executor_output):
                    if on_executor_complete:
                        on_executor_complete(test_id, executor_output)

                return callback

            def make_judge_complete_callback(test_id):
                def callback(judge_result):
                    if on_judge_complete:
                        on_judge_complete(test_id, judge_result)

                return callback

            # Run the test (no per-test sandbox - runs directly in host environment)
            result = await test.run(
                coverage_agent=self.config.coverage_agent,
                sandbox=None,
                on_tool_call=on_tool_call,
                on_conversation_message=on_conversation_message,
                on_executor_complete=make_executor_complete_callback(test.id),
                on_judge_complete=make_judge_complete_callback(test.id),
            )
            results.append(result)

            # Finish debug tracking for this test
            if is_debug():
                session = get_debug_session()
                if session:
                    # Finish executor tracking
                    if result.executor_output:
                        session.finish_executor(result.executor_output.to_dict())
                    else:
                        session.finish_executor({})

                    # Track judge results
                    for judge_result in result.judge_results:
                        session.start_judge(judge_result.judge_id)
                        session.finish_judge(judge_result.to_dict())

                    # Finish test tracking
                    session.finish_test(
                        {
                            "test_id": result.test_id,
                            "passed": result.passed,
                            "duration": result.duration,
                            "failure_reasons": result.failure_reasons,
                        }
                    )

            # Notify test complete
            if on_progress:
                on_progress("complete", test.id, result)

            logger.info(
                "test.complete",
                test_id=test.id,
                passed=result.passed,
                duration_ms=int(result.duration * 1000),
            )

            # Post-test processing (coverage, file status, cache)
            self._process_test_result(test, result)

            # Incremental saves after each test
            if history_manager and run_id:
                history_manager.save_test_result(result, run_id)

            if is_debug():
                session = get_debug_session()
                if session:
                    session.flush_test(test.id)

            # Incremental cache save
            await self._save_cache_incremental(test.id, result)

        return results, cached_count

    async def _run_concurrent(
        self,
        on_progress: Optional[ProgressCallback] = None,
        on_tool_call: Optional[ToolCallCallback] = None,
        on_conversation_message: Optional[Callable] = None,
        on_executor_complete: Optional[Callable] = None,
        on_judge_complete: Optional[Callable] = None,
        history_manager: Optional["HistoryManager"] = None,
        run_id: Optional[str] = None,
    ) -> tuple:
        """
        Run tests concurrently with semaphore-controlled concurrency.

        Args:
            on_progress: Optional callback called with (event, test_id, data)
            on_tool_call: Optional callback fired after each tool execution.
            on_conversation_message: Optional callback for streaming conversation.
            on_executor_complete: Optional callback fired when executor finishes.
            on_judge_complete: Optional callback fired when each judge finishes.
            history_manager: Optional HistoryManager for incremental history saves
            run_id: Optional unique identifier for this run

        Returns:
            Tuple of (results list, cached count)
        """
        import asyncio
        from .debug import is_debug, get_debug_session

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        cache_lock = asyncio.Lock()
        registry_lock = asyncio.Lock()
        results_lock = asyncio.Lock()

        results = []
        cached_count = 0

        async def run_single(test):
            nonlocal cached_count

            # Check cache (read-only dict, populated before run)
            cached = self._cached_results.get(test.id)
            if cached and not test.is_stale():
                logger.info("cache.hit", test_id=test.id)
                if on_progress:
                    on_progress("cached", test.id, cached)
                async with results_lock:
                    results.append(cached)
                    cached_count += 1
                return

            async with semaphore:
                # Log test header
                model_name = (
                    getattr(test.executor.provider, "model", "unknown")
                    if test.executor.provider
                    else "unknown"
                )
                judge_ids = [j.id for j in test.judges] if test.judges else []
                tool_names = [t.name for t in test.executor.tools] if test.executor.tools else []

                # Notify test start (pass tool names and provenance so CLI can print them)
                progress_data = {"tools": tool_names}
                if test.tool_provenance:
                    progress_data["tool_provenance"] = test.tool_provenance.to_dict()
                if on_progress:
                    on_progress("start", test.id, progress_data)
                _log_tool_provenance(test.id, test.tool_provenance)

                logger.info(
                    "test.starting",
                    test_id=test.id,
                    reason=test.reason,
                    model=model_name,
                    executor_id=test.executor.id,
                    judges=judge_ids,
                    timeout=test.timeout,
                    retries=test.retries,
                    tools=tool_names,
                    files=[str(f) for f in test.files],
                    mode="parallel",
                )

                # Start debug tracking for this test
                if is_debug():
                    session = get_debug_session()
                    if session:
                        session.start_test(test.id)
                        session.start_executor()

                # Create wrapper callbacks that include test_id
                def make_executor_complete_callback(test_id):
                    def callback(executor_output):
                        if on_executor_complete:
                            on_executor_complete(test_id, executor_output)

                    return callback

                def make_judge_complete_callback(test_id):
                    def callback(judge_result):
                        if on_judge_complete:
                            on_judge_complete(test_id, judge_result)

                    return callback

                # Run the test
                result = await test.run(
                    coverage_agent=self.config.coverage_agent,
                    sandbox=None,
                    on_tool_call=on_tool_call,
                    on_conversation_message=on_conversation_message,
                    on_executor_complete=make_executor_complete_callback(test.id),
                    on_judge_complete=make_judge_complete_callback(test.id),
                )

                # Finish debug tracking for this test
                if is_debug():
                    session = get_debug_session()
                    if session:
                        if result.executor_output:
                            session.finish_executor(result.executor_output.to_dict())
                        else:
                            session.finish_executor({})

                        for judge_result in result.judge_results:
                            session.start_judge(judge_result.judge_id)
                            session.finish_judge(judge_result.to_dict())

                        session.finish_test(
                            {
                                "test_id": result.test_id,
                                "passed": result.passed,
                                "duration": result.duration,
                                "failure_reasons": result.failure_reasons,
                            }
                        )

                # Notify test complete
                if on_progress:
                    on_progress("complete", test.id, result)

                logger.info(
                    "test.complete",
                    test_id=test.id,
                    passed=result.passed,
                    duration_ms=int(result.duration * 1000),
                )

                # Post-test processing (coverage, file status) under lock
                async with registry_lock:
                    self._process_test_result(test, result)

                async with results_lock:
                    results.append(result)

                # Incremental saves under lock
                if history_manager and run_id:
                    history_manager.save_test_result(result, run_id)

                if is_debug():
                    session = get_debug_session()
                    if session:
                        session.flush_test(test.id)

                # Cache write under lock
                async with cache_lock:
                    await self._save_cache_incremental(test.id, result)

        # Launch all tests concurrently (semaphore limits actual concurrency)
        await asyncio.gather(*[run_single(test) for test in self.tests])

        logger.info("suite.parallel.complete", total=len(results), cached=cached_count)
        return results, cached_count

    def _process_test_result(self, test: TestObject, result: TestResult) -> None:
        """
        Process post-test coverage and file status updates.

        Shared by both _run_sequential and _run_concurrent to avoid duplication.
        Updates file registry, file status, and cached results based on test outcome.

        Args:
            test: The test that was run
            result: The result of the test
        """
        logger.debug("test.result.processing", test_id=test.id, passed=result.passed)

        # Extract accessed files from tool calls
        accessed_files = self._extract_accessed_files(result)

        # Extract coverage from subagents
        subagent_coverage = self._extract_subagent_coverage(result)
        if subagent_coverage:
            debug(f"[COVERAGE] Extracted subagent coverage for {len(subagent_coverage)} files")
            for file_path, line_cov in subagent_coverage.items():
                if file_path in result.line_coverage:
                    result.line_coverage[file_path] = result.line_coverage[file_path].merge(
                        line_cov
                    )
                else:
                    result.line_coverage[file_path] = line_cov
                if file_path not in accessed_files:
                    accessed_files.append(file_path)

        # Combine accessed files with scaffold-declared files
        scaffold_files = [normalize_path(f) for f in test.files] if test.files else []
        all_covered_files = set(accessed_files) | set(scaffold_files)

        if result.passed:
            self._cached_results[test.id] = result
            test.set_cached_hash(test.get_hash())
            for file_path in all_covered_files:
                if file_path not in self._file_registry:
                    self._file_registry[file_path] = FileObject(path=file_path)
                self._file_registry[file_path].increment_pass_count()
                if self._file_status.get(file_path) != FileStatus.FAILED:
                    self._file_status[file_path] = FileStatus.PASSED
        else:
            for file_path in all_covered_files:
                if file_path not in self._file_registry:
                    self._file_registry[file_path] = FileObject(path=file_path)
                self._file_status[file_path] = FileStatus.FAILED

                if file_path in result.line_coverage:
                    lc = result.line_coverage[file_path]
                    if file_path in self._file_registry:
                        existing = self._file_registry[file_path].metrics.line_coverage
                        if existing:
                            self._file_registry[file_path].metrics.line_coverage = existing.merge(
                                lc
                            )
                        else:
                            self._file_registry[file_path].metrics.line_coverage = lc

        logger.debug(
            "test.result.processed",
            test_id=test.id,
            status="passed" if result.passed else "failed",
            files_updated=len(all_covered_files),
        )

    async def process_test_coverage(self, test: TestObject, result: TestResult) -> None:
        """
        Process coverage data for a completed test.

        This extracts accessed files, runs the coverage agent, and updates
        file status. Called after each test completes.

        Args:
            test: The test that was run
            result: The result of the test
        """
        # Extract accessed files from tool calls
        accessed_files = self._extract_accessed_files(result)

        # Extract coverage from subagents (if spawn_subagent was used)
        subagent_coverage = self._extract_subagent_coverage(result)
        if subagent_coverage:
            debug(f"[COVERAGE] Extracted subagent coverage for {len(subagent_coverage)} files")
            for file_path, line_cov in subagent_coverage.items():
                debug(
                    f"[COVERAGE] Subagent coverage for {file_path}: {len(line_cov.covered_lines)} lines"
                )
                if file_path in result.line_coverage:
                    result.line_coverage[file_path] = result.line_coverage[file_path].merge(
                        line_cov
                    )
                else:
                    result.line_coverage[file_path] = line_cov
                if file_path not in accessed_files:
                    accessed_files.append(file_path)

        # Run coverage agent to get line-level coverage for each accessed file
        if not self.config.coverage_agent:
            debug("[COVERAGE] No coverage agent configured")
        elif not accessed_files:
            debug(f"[COVERAGE] No accessed files for test '{test.id}'")

        if self.config.coverage_agent and accessed_files:
            debug(
                f"[COVERAGE] Running coverage agent for test '{test.id}' on {len(accessed_files)} files"
            )
            for file_path in accessed_files:
                # Skip files that already have subagent coverage
                if file_path in result.line_coverage:
                    debug(f"[COVERAGE] Skipping {file_path} - already has subagent coverage")
                    continue

                debug(f"[COVERAGE] Processing file: {file_path}")
                if file_path not in self._file_registry:
                    self._file_registry[file_path] = FileObject(path=file_path)
                file_obj = self._file_registry[file_path]

                if result.passed:
                    debug("[COVERAGE] Calling infer_coverage for passed test")
                    # Mark that we attempted coverage for this file (even if it fails)
                    self._coverage_attempted.add(file_path)
                    line_cov = await self.config.coverage_agent.infer_coverage(
                        result.executor_output, file_obj, test.id, goal=test.executor.user_prompt
                    )
                    if line_cov:
                        debug(
                            f"[COVERAGE] infer_coverage returned {len(line_cov.covered_lines)} covered lines"
                        )
                        result.line_coverage[file_path] = line_cov
                    else:
                        debug(
                            "[COVERAGE] infer_coverage returned None (coverage attempted but failed)"
                        )
                else:
                    debug("[COVERAGE] Calling analyze_failure for failed test")
                    failure_output = await self.config.coverage_agent.analyze_failure(
                        result.executor_output, file_obj, test.id, "; ".join(result.failure_reasons)
                    )
                    if failure_output:
                        debug(
                            f"[COVERAGE] analyze_failure returned {len(failure_output.referenced_lines)} referenced lines"
                        )
                        try:
                            content = await file_obj.read()
                            total_lines = len(content.splitlines())
                        except Exception as e:
                            logger.debug("total_lines.fallback", test_id=test.id, error=str(e))
                            total_lines = (
                                max(failure_output.referenced_lines)
                                if failure_output.referenced_lines
                                else 0
                            )

                        line_cov = LineCoverage(
                            file_path=file_path,
                            total_lines=total_lines,
                            covered_lines=set(),
                            failed_lines=set(failure_output.referenced_lines),
                            source_test_ids={
                                ln: {test.id} for ln in failure_output.referenced_lines
                            },
                        )
                        result.line_coverage[file_path] = line_cov

                        incorrect_lines = [
                            IncorrectLine(
                                line_number=il.line_number, reason=il.reason, test_id=test.id
                            )
                            for il in failure_output.incorrect_lines
                        ]

                        if file_path not in self._failure_analysis:
                            self._failure_analysis[file_path] = {}
                        self._failure_analysis[file_path][test.id] = FailureAnalysis(
                            file_path=file_path,
                            referenced_lines=failure_output.referenced_lines,
                            incorrect_lines=incorrect_lines,
                            analysis=failure_output.analysis,
                        )

        # Update file status
        if result.passed:
            self._cached_results[test.id] = result
            test.set_cached_hash(test.get_hash())
            for file_path in accessed_files:
                if file_path not in self._file_registry:
                    self._file_registry[file_path] = FileObject(path=file_path)
                self._file_registry[file_path].increment_pass_count()
                if self._file_status.get(file_path) != FileStatus.FAILED:
                    self._file_status[file_path] = FileStatus.PASSED
        else:
            for file_path in accessed_files:
                if file_path not in self._file_registry:
                    self._file_registry[file_path] = FileObject(path=file_path)
                self._file_status[file_path] = FileStatus.FAILED

                if file_path in result.line_coverage:
                    lc = result.line_coverage[file_path]
                    if file_path in self._file_registry:
                        existing = self._file_registry[file_path].metrics.line_coverage
                        if existing:
                            self._file_registry[file_path].metrics.line_coverage = existing.merge(
                                lc
                            )
                        else:
                            self._file_registry[file_path].metrics.line_coverage = lc

    def _extract_accessed_files(self, result: TestResult) -> List[str]:
        """
        Extract file paths from tool calls that read files.

        Supports:
        - read_file tool calls
        - bash commands that read files (cat, head, tail, less, more)

        Args:
            result: The test result containing executor output with tool calls

        Returns:
            List of unique file paths that were accessed
        """
        accessed = set()
        if result.executor_output and result.executor_output.tool_calls:
            for tc in result.executor_output.tool_calls:
                tool_name = getattr(tc, "tool_name", None)
                parameters = getattr(tc, "parameters", None)
                if isinstance(tc, dict):
                    tool_name = tc.get("tool_name") or tc.get("name")
                    parameters = tc.get("parameters") or tc.get("input") or {}
                if not isinstance(parameters, dict):
                    parameters = {}

                if tool_name == "read_file":
                    path = parameters.get("path")
                    if path:
                        normalized = normalize_path(path)
                        accessed.add(normalized)
                elif tool_name == "run_shell_command":
                    command = parameters.get("command", "")
                    accessed.update(self._extract_files_from_shell_command(command))
        return list(accessed)

    def _extract_files_from_shell_command(self, command: str) -> set:
        """
        Extract file paths from shell commands that read files.

        Recognizes: cat, head, tail, less, more, grep, sed, awk

        Args:
            command: The shell command string

        Returns:
            Set of file paths extracted from the command
        """
        import re

        files = set()

        # Pattern to match file-reading commands followed by file paths
        # Matches: cat file.txt, cat ./path/file.md, head -n 10 file.txt, etc.
        read_commands = r"(?:cat|head|tail|less|more)\s+"

        # Find all potential file paths after read commands
        # This regex handles:
        # - cat file.txt
        # - cat ./docs/file.md
        # - cat /absolute/path/file.md
        # - head -n 100 file.txt (flags before file)
        # - cat file.txt | grep ... (pipes)
        pattern = read_commands + r"(?:-[a-zA-Z0-9]+\s+)*([^\s|><;]+)"

        for match in re.finditer(pattern, command):
            path = match.group(1)
            # Skip if it looks like a flag or special char
            if path.startswith("-") or path in ["2>&1", "2>/dev/null"]:
                continue
            # Normalize path (remove ./ prefix, handle relative paths)
            if path.startswith("./"):
                path = path[2:]
            # Only include if it looks like a real file path
            if "." in path or "/" in path:
                normalized = normalize_path(path)
                files.add(normalized)
                debug(f"[COVERAGE] Extracted file from shell command: {normalized}")

        return files

    def _extract_subagent_coverage(self, result: TestResult) -> Dict[str, LineCoverage]:
        """
        Extract coverage from delegate_to_agent tool calls.

        When delegate_to_agent is used, each delegated agent returns covered_lines
        indicating which lines in their section were important. This method
        extracts and aggregates that coverage.

        Args:
            result: The test result containing executor output with tool calls

        Returns:
            Dict mapping file paths to LineCoverage objects with covered lines
        """
        coverage_by_file: Dict[str, LineCoverage] = {}

        if not result.executor_output or not result.executor_output.tool_calls:
            return coverage_by_file

        for tc in result.executor_output.tool_calls:
            tool_name = getattr(tc, "tool_name", None)
            tc_result = getattr(tc, "result", None)
            if isinstance(tc, dict):
                tool_name = tc.get("tool_name") or tc.get("name")
                tc_result = tc.get("result")
            if tool_name != "delegate_to_agent":
                continue

            subagent_results = tc_result
            if not isinstance(subagent_results, list):
                continue

            for subagent_result in subagent_results:
                if not isinstance(subagent_result, dict):
                    continue

                file_path = subagent_result.get("file_path")
                covered_lines = subagent_result.get("covered_lines", [])

                if not file_path or not covered_lines:
                    continue

                # Normalize path
                normalized_path = normalize_path(file_path)

                # Get or create LineCoverage for this file
                if normalized_path not in coverage_by_file:
                    # Need to get total lines - read file or estimate
                    try:
                        # Use sync file read since we're in a potentially non-async context
                        import os

                        if os.path.exists(normalized_path):
                            with open(normalized_path, "r") as f:
                                total_lines = len(f.readlines())
                        else:
                            total_lines = max(covered_lines) if covered_lines else 0
                    except Exception as e:
                        logger.debug("total_lines.fallback", path=normalized_path, error=str(e))
                        total_lines = max(covered_lines) if covered_lines else 0

                    coverage_by_file[normalized_path] = LineCoverage(
                        file_path=normalized_path,
                        total_lines=total_lines,
                        covered_lines=set(),
                        source_test_ids={},
                    )

                # Add covered lines from this subagent
                lc = coverage_by_file[normalized_path]
                for ln in covered_lines:
                    lc.covered_lines.add(ln)
                    if ln not in lc.source_test_ids:
                        lc.source_test_ids[ln] = set()
                    lc.source_test_ids[ln].add(result.test_id)

        return coverage_by_file

    def _aggregate_results(
        self, results: List[TestResult], duration: float, cached_count: int
    ) -> TestSuiteResults:
        """
        Aggregate individual test results.

        Args:
            results: List of TestResult objects
            duration: Total duration in seconds
            cached_count: Number of cached results used

        Returns:
            TestSuiteResults with aggregated data
        """
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed and getattr(r, "status", "") != "error")
        error = sum(1 for r in results if getattr(r, "status", "") == "error")

        return TestSuiteResults(
            total_tests=len(results),
            passed=passed,
            failed=failed,
            error=error,
            skipped=0,
            duration=duration,
            test_results=results,
            cached_results=cached_count,
        )

    async def run_test(self, test_id: str) -> TestResult:
        """
        Run a single test by ID.

        Args:
            test_id: ID of test to run

        Returns:
            TestResult from the test

        Raises:
            ValueError: If test not found
        """
        for test in self.tests:
            if test.id == test_id:
                result = await test.run()
                if result.passed:
                    self._cached_results[test.id] = result
                    test.set_cached_hash(test.get_hash())
                return result
        raise ValueError(f"Test not found: {test_id}")

    def get_results(self) -> Optional[TestSuiteResults]:
        """
        Return results from the last run.

        Returns:
            TestSuiteResults or None if not run yet
        """
        return self._last_results

    async def load_cache(self) -> None:
        """Load cached results, line coverage, file status, and failure analysis from file system."""
        cache_file = os.path.join(self.config.cache_path, "cache.json")

        if not os.path.exists(cache_file):
            return

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Deserialize cached results
                for test_id, result_data in data.get("results", {}).items():
                    # Find matching test and set hash
                    for test in self.tests:
                        if test.id == test_id:
                            test.set_cached_hash(result_data.get("hash", ""))
                            # Create minimal TestResult for cache
                            if result_data.get("passed"):
                                self._cached_results[test_id] = TestResult(
                                    test_id=test_id,
                                    passed=True,
                                    executor_passed=True,
                                    judge_results=[],
                                    executor_output=None,
                                    duration=0.0,
                                    timestamp=datetime.fromisoformat(
                                        result_data.get("timestamp", datetime.now().isoformat())
                                    ),
                                    line_coverage={},
                                )
                            break

                # Load aggregated line coverage into file registry
                for file_path, coverage_data in data.get("line_coverage", {}).items():
                    normalized_path = normalize_path(file_path)
                    if normalized_path in self._file_registry:
                        coverage = LineCoverage.from_dict(coverage_data)
                        self._file_registry[normalized_path].metrics.line_coverage = coverage

                # Load file status
                for file_path, status_value in data.get("file_status", {}).items():
                    normalized_path = normalize_path(file_path)
                    try:
                        self._file_status[normalized_path] = FileStatus(status_value)
                    except ValueError as e:
                        logger.debug("cache.status.invalid", value=status_value, error=str(e))

                # Load failure analysis
                for file_path, analyses in data.get("failure_analysis", {}).items():
                    normalized_path = normalize_path(file_path)
                    self._failure_analysis[normalized_path] = {}
                    for test_id, analysis_data in analyses.items():
                        incorrect_lines = [
                            IncorrectLine(
                                line_number=il["line_number"],
                                reason=il["reason"],
                                test_id=il["test_id"],
                            )
                            for il in analysis_data.get("incorrect_lines", [])
                        ]
                        self._failure_analysis[normalized_path][test_id] = FailureAnalysis(
                            file_path=analysis_data["file_path"],
                            referenced_lines=analysis_data.get("referenced_lines", []),
                            incorrect_lines=incorrect_lines,
                            analysis=analysis_data.get("analysis", ""),
                        )
        except (json.JSONDecodeError, IOError) as e:
            logger.debug("cache.load.failed", error=str(e))

    async def save_cache(self) -> None:
        """Save cached results, line coverage, file status, and failure analysis to file system."""
        os.makedirs(self.config.cache_path, exist_ok=True)
        cache_file = os.path.join(self.config.cache_path, "cache.json")

        # Load existing cache to preserve results from previous runs
        cache_data = {
            "version": "3.0",
            "generated": datetime.now().isoformat(),
            "results": {},
            "line_coverage": {},  # Aggregated line coverage per file
            "file_status": {},  # File status (covered/failed/uncovered)
            "failure_analysis": {},  # Failure analysis per file per test
            "coverage_attempted": [],  # Files where line coverage inference was attempted
        }
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    # Preserve existing results, line_coverage, file_status, failure_analysis
                    cache_data["results"] = existing.get("results", {})
                    cache_data["line_coverage"] = existing.get("line_coverage", {})
                    cache_data["file_status"] = existing.get("file_status", {})
                    cache_data["failure_analysis"] = existing.get("failure_analysis", {})
                    cache_data["coverage_attempted"] = existing.get("coverage_attempted", [])
                    debug(
                        f"[CACHE] Loaded {len(cache_data['results'])} existing results: {list(cache_data['results'].keys())}"
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.debug("cache.read.failed", path=cache_file, error=str(e))

        for test in self.tests:
            if test.id in self._cached_results:
                result = self._cached_results[test.id]
                cache_data["results"][test.id] = {
                    "passed": result.passed,
                    "executor_passed": result.executor_passed,
                    "timestamp": result.timestamp.isoformat(),
                    "hash": test.get_hash(),
                    "duration": result.duration,
                    "failure_reasons": result.failure_reasons,
                    "executor_output": (
                        result.executor_output.final_response if result.executor_output else None
                    ),
                    "judge_results": [
                        {
                            "judge_id": jr.judge_id,
                            "passed": jr.passed,
                            "failure_reason": jr.failure_reason,
                            "response": jr.response,
                        }
                        for jr in result.judge_results
                    ],
                }

                # Aggregate line coverage (union of covered lines per file)
                for file_path, coverage in result.line_coverage.items():
                    if file_path in cache_data["line_coverage"]:
                        # Merge with existing coverage
                        existing = LineCoverage.from_dict(cache_data["line_coverage"][file_path])
                        merged = existing.merge(coverage)
                        cache_data["line_coverage"][file_path] = merged.to_dict()
                    else:
                        cache_data["line_coverage"][file_path] = coverage.to_dict()

        # Also save line coverage from file registry (includes failed test coverage)
        for file_path, file_obj in self._file_registry.items():
            if file_obj.metrics.line_coverage:
                coverage = file_obj.metrics.line_coverage
                if file_path in cache_data["line_coverage"]:
                    # Merge with existing coverage
                    existing = LineCoverage.from_dict(cache_data["line_coverage"][file_path])
                    merged = existing.merge(coverage)
                    cache_data["line_coverage"][file_path] = merged.to_dict()
                else:
                    cache_data["line_coverage"][file_path] = coverage.to_dict()

        # Save file status
        for file_path, status in self._file_status.items():
            cache_data["file_status"][file_path] = status.value

        # Save coverage_attempted (merge with existing)
        existing_attempted = set(cache_data.get("coverage_attempted", []))
        all_attempted = existing_attempted.union(self._coverage_attempted)
        cache_data["coverage_attempted"] = sorted(list(all_attempted))

        # Save failure analysis
        for file_path, analyses in self._failure_analysis.items():
            cache_data["failure_analysis"][file_path] = {}
            for test_id, analysis in analyses.items():
                cache_data["failure_analysis"][file_path][test_id] = {
                    "file_path": analysis.file_path,
                    "referenced_lines": analysis.referenced_lines,
                    "incorrect_lines": [
                        {"line_number": il.line_number, "reason": il.reason, "test_id": il.test_id}
                        for il in analysis.incorrect_lines
                    ],
                    "analysis": analysis.analysis,
                }

        debug(
            f"[CACHE] Saving {len(cache_data['results'])} results: {list(cache_data['results'].keys())}"
        )
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

    async def _save_cache_incremental(self, test_id: str, result: TestResult) -> None:
        """
        Save cache incrementally after each test completion.

        Args:
            test_id: The test that just completed
            result: The result of the test
        """
        if not self.config.cache_path:
            return

        os.makedirs(self.config.cache_path, exist_ok=True)
        cache_file = os.path.join(self.config.cache_path, "cache.json")

        # Load existing cache
        cache_data = {
            "version": "3.0",
            "generated": datetime.now().isoformat(),
            "results": {},
            "line_coverage": {},
            "file_status": {},
            "failure_analysis": {},
            "coverage_attempted": [],
        }

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    cache_data["results"] = existing.get("results", {})
                    cache_data["line_coverage"] = existing.get("line_coverage", {})
                    cache_data["file_status"] = existing.get("file_status", {})
                    cache_data["failure_analysis"] = existing.get("failure_analysis", {})
                    cache_data["coverage_attempted"] = existing.get("coverage_attempted", [])
            except (json.JSONDecodeError, IOError) as e:
                logger.debug("cache.incremental.read.failed", error=str(e))

        # Update with this test's result (save all results, not just passed)
        test_obj = self.get_test(test_id)
        cache_data["results"][test_id] = {
            "passed": result.passed,
            "executor_passed": result.executor_passed,
            "timestamp": (
                result.timestamp.isoformat() if result.timestamp else datetime.now().isoformat()
            ),
            "hash": test_obj.get_hash() if test_obj else "",
            "duration": result.duration,
            "failure_reasons": result.failure_reasons,
            "executor_output": (
                result.executor_output.final_response if result.executor_output else None
            ),
            "judge_results": [
                {
                    "judge_id": jr.judge_id,
                    "passed": jr.passed,
                    "failure_reason": jr.failure_reason,
                    "response": jr.response,
                }
                for jr in result.judge_results
            ],
        }

        # Update line coverage from this result
        for file_path, coverage in result.line_coverage.items():
            if file_path in cache_data["line_coverage"]:
                existing = LineCoverage.from_dict(cache_data["line_coverage"][file_path])
                merged = existing.merge(coverage)
                cache_data["line_coverage"][file_path] = merged.to_dict()
            else:
                cache_data["line_coverage"][file_path] = coverage.to_dict()

        # Update file status
        for file_path, status in self._file_status.items():
            cache_data["file_status"][file_path] = status.value

        # Update coverage_attempted (merge with existing)
        existing_attempted = set(cache_data.get("coverage_attempted", []))
        all_attempted = existing_attempted.union(self._coverage_attempted)
        cache_data["coverage_attempted"] = sorted(list(all_attempted))

        # Update failure analysis for this test
        for file_path, analyses in self._failure_analysis.items():
            if file_path not in cache_data["failure_analysis"]:
                cache_data["failure_analysis"][file_path] = {}
            if test_id in analyses:
                analysis = analyses[test_id]
                cache_data["failure_analysis"][file_path][test_id] = {
                    "file_path": analysis.file_path,
                    "referenced_lines": analysis.referenced_lines,
                    "incorrect_lines": [
                        {"line_number": il.line_number, "reason": il.reason, "test_id": il.test_id}
                        for il in analysis.incorrect_lines
                    ],
                    "analysis": analysis.analysis,
                }

        cache_data["generated"] = datetime.now().isoformat()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

    async def clear_cache(self) -> None:
        """Clear all cached results."""
        import shutil

        self._cached_results.clear()
        if os.path.exists(self.config.cache_path):
            shutil.rmtree(self.config.cache_path)

    def get_coverage(self) -> CoverageReport:
        """
        Generate coverage report for all files accessed during tests.

        Returns:
            CoverageReport with file coverage statistics
        """
        # Use files tracked via tool calls
        all_files = list(self._file_registry.values())

        # Calculate coverage based on file status
        covered = sum(1 for f in all_files if self._file_status.get(f.path) == FileStatus.PASSED)
        total = len(all_files)

        return CoverageReport(
            total_files=total,
            covered_files=covered,
            coverage_percentage=(covered / total * 100) if total > 0 else 0.0,
            file_details=[f.metrics for f in all_files],
        )

    def get_uncovered_files(self) -> List[str]:
        """
        Get list of files with no passing tests.

        Returns:
            List of file paths with zero coverage
        """
        return [path for path, file in self._file_registry.items() if file.metrics.coverage == 0]

    def get_line_coverage(self) -> Dict[str, Any]:
        """
        Get aggregated line coverage statistics.

        Returns:
            Dictionary with line coverage data:
            {
                "total_lines": int,
                "covered_lines": int,
                "percentage": float,
                "files": {
                    "path": {
                        "total_lines": int,
                        "covered_lines": [list of ints],
                        "covered_count": int,
                        "percentage": float,
                        "source_tests": {line: [test_ids]}
                    }
                }
            }
        """
        total_lines = 0
        covered_lines = 0
        files_data = {}

        for path, file in self._file_registry.items():
            if file.metrics.line_coverage:
                lc = file.metrics.line_coverage
                total_lines += lc.total_lines
                covered_lines += lc.covered_count
                files_data[path] = {
                    "total_lines": lc.total_lines,
                    "covered_lines": sorted(list(lc.covered_lines)),
                    "covered_count": lc.covered_count,
                    "percentage": lc.coverage_percentage,
                    "source_tests": {
                        str(ln): sorted(list(tests)) for ln, tests in lc.source_test_ids.items()
                    },
                }

        return {
            "total_lines": total_lines,
            "covered_lines": covered_lines,
            "percentage": (covered_lines / total_lines * 100) if total_lines > 0 else 0.0,
            "files": files_data,
        }

    def get_all_files(self) -> List[FileObject]:
        """
        Get all files tracked by the suite.

        Returns:
            List of FileObject instances
        """
        return list(self._file_registry.values())

    def get_file_status(self, file_path: str) -> FileStatus:
        """
        Get the status of a file.

        Args:
            file_path: Path to the file

        Returns:
            FileStatus (UNCOVERED, COVERED, or FAILED)
        """
        return self._file_status.get(file_path, FileStatus.UNCOVERED)

    def get_all_file_statuses(self) -> Dict[str, FileStatus]:
        """
        Get status of all tracked files.

        Returns:
            Dictionary mapping file path to status
        """
        # Include all files from registry, defaulting to UNCOVERED
        statuses = {}
        for path in self._file_registry:
            statuses[path] = self._file_status.get(path, FileStatus.UNCOVERED)
        return statuses

    def get_failed_files(self) -> List[str]:
        """
        Get list of files with at least one failing test.

        Returns:
            List of file paths with failed status
        """
        return [path for path, status in self._file_status.items() if status == FileStatus.FAILED]

    def get_failure_analysis(self, file_path: str) -> Dict[str, FailureAnalysis]:
        """
        Get failure analysis for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary mapping test_id to FailureAnalysis, or empty dict if no failures
        """
        return self._failure_analysis.get(file_path, {})

    def get_all_failure_analysis(self) -> Dict[str, Dict[str, FailureAnalysis]]:
        """
        Get all failure analysis data.

        Returns:
            Dictionary mapping file_path -> {test_id -> FailureAnalysis}
        """
        return self._failure_analysis.copy()

    def __len__(self) -> int:
        return len(self.tests)

    def __iter__(self):
        return iter(self.tests)

    def __repr__(self) -> str:
        return f"TestSuite(name={self.name}, tests={len(self.tests)})"
