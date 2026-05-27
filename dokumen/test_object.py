"""
Test Object module for the Skill Testing Framework.

Represents a single test case that orchestrates executor and judge agents.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
import hashlib
import json
import time

import os
import shutil
import yaml

from .file_object import LineCoverage, IncorrectLine
from .sdk.types import ExecutorResult, JudgeVerdict
from .explore_agent import ExploreAgent, ExploreResult
from .debug import is_debug, debug
from .logging_config import get_logger
from .playwright_tools import BROWSER_TOOL_NAMES

# Backward-compatible aliases for type references in this module.
ExecutorOutput = ExecutorResult
JudgeResult = JudgeVerdict

logger = get_logger(__name__)


def _prompt_hash(prompt: Optional[str]) -> str:
    """Compute a short SHA256 hash of a prompt for observability logging.

    Returns the first 12 hex characters of the SHA256 digest, or "none"
    if the prompt is None or not a string. This provides a compact
    fingerprint to verify which prompt was actually applied without
    logging full prompt text.

    Args:
        prompt: The prompt text to hash, or None.

    Returns:
        12-char hex hash string, or "none" for None/non-string input.
    """
    if prompt is None or not isinstance(prompt, str):
        return "none"
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def clear_output_dir(output_dir: str) -> None:
    """Clear the output directory to remove stale recordings from previous runs.

    This prevents duplicate browser videos when a test runs multiple times.
    The directory is completely removed if it exists.

    Args:
        output_dir: Path to the test's recording output directory.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        logger.info("browser.output_dir.cleared", output_dir=output_dir)


def collect_browser_artifacts(output_dir: str) -> List[Dict[str, Any]]:
    """Collect video and screenshot artifacts from test output directory.

    Scans the output directory (typically .dokumen-cache/recordings/{test_id})
    for video files (.webm, .mp4) and screenshots (.png, .jpg, .jpeg).

    Args:
        output_dir: Path to the test's recording output directory.

    Returns:
        List of artifact dicts with type, path, filename, size_bytes.
        Path is relative to output_dir for portability.
    """
    artifacts = []

    if not os.path.exists(output_dir):
        return artifacts

    for root, _, files in os.walk(output_dir):
        for f in files:
            full_path = os.path.join(root, f)
            # Path relative to output_dir (e.g., "video-1.webm" or "screenshots/page-123.png")
            rel_path = os.path.relpath(full_path, output_dir)

            artifact_type = None
            if f.lower().endswith(('.webm', '.mp4')):
                artifact_type = "video"
            elif f.lower().endswith(('.png', '.jpg', '.jpeg')):
                artifact_type = "screenshot"

            if artifact_type:
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    size = 0

                artifacts.append({
                    "type": artifact_type,
                    "path": rel_path,
                    "filename": f,
                    "size_bytes": size
                })
                logger.info("browser.artifact.collected",
                           type=artifact_type, path=rel_path, size=size)

    return artifacts

# Internal files written by Playwright MCP - only excluded within browser dirs
_BROWSER_INTERNAL_FILES = {'click-indicator.js'}


def collect_output_artifacts(output_dir: str, skip_inline_dirs: Optional[set] = None) -> List[Dict[str, Any]]:
    """Collect all files from the test output directory.

    Scans the output directory for deliverables written by the executor or judge.
    Text files under 100KB have their content inlined for easy access in the UI.

    Args:
        output_dir: Path to the test's output directory
            (typically .dokumen-cache/output/{test_id}).
        skip_inline_dirs: Optional set of directory names (relative to output_dir)
            where inline content should be skipped and browser internal files excluded.

    Returns:
        List of artifact dicts with filename, path, size_bytes, content_type, content.
    """
    artifacts = []
    if not os.path.exists(output_dir):
        return artifacts

    CONTENT_TYPE_MAP = {
        '.py': 'text/x-python',
        '.md': 'text/markdown',
        '.txt': 'text/plain',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.yaml': 'text/yaml',
        '.yml': 'text/yaml',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.webm': 'video/webm',
        '.mp4': 'video/mp4',
    }
    MAX_INLINE_SIZE = 100 * 1024  # 100KB

    for root, _, files in os.walk(output_dir):
        rel_dir = os.path.relpath(root, output_dir)
        in_skip_dir = skip_inline_dirs and any(
            rel_dir == d or rel_dir.startswith(d + os.sep)
            for d in skip_inline_dirs
        )

        for f in files:
            # Only exclude browser internal files within browser dirs
            if in_skip_dir and f in _BROWSER_INTERNAL_FILES:
                continue

            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, output_dir)
            try:
                size = os.path.getsize(full_path)
            except OSError:
                size = 0

            # Skip empty files (e.g. 0-byte videos from Playwright context init)
            if size == 0:
                logger.debug("test.output_artifact.skip_empty", filename=f, path=rel_path)
                continue

            ext = os.path.splitext(f)[1].lower()
            content_type = CONTENT_TYPE_MAP.get(ext, 'application/octet-stream')

            # Inline content for small text files (skip for browser dirs)
            content = None
            if not in_skip_dir:
                if content_type.startswith('text/') or content_type == 'application/json':
                    if size <= MAX_INLINE_SIZE:
                        try:
                            with open(full_path, 'r', encoding='utf-8') as fh:
                                content = fh.read()
                        except (UnicodeDecodeError, OSError):
                            pass
                # Inline small images as base64 for UI preview
                elif content_type.startswith('image/') and size <= MAX_INLINE_SIZE:
                    try:
                        import base64
                        with open(full_path, 'rb') as fh:
                            content = base64.b64encode(fh.read()).decode('ascii')
                        logger.info("test.output_artifact.image_inlined",
                                   filename=f, size=size, content_type=content_type)
                    except OSError:
                        pass

            artifacts.append({
                'filename': f,
                'path': rel_path,
                'size_bytes': size,
                'content_type': content_type,
                'content': content,
            })
            logger.info("test.output_artifact.collected", filename=f, path=rel_path, size=size)

    return artifacts


def _extract_report_markdown(response: Optional[str]) -> str:
    """Extract markdown report from verdict judge response.

    The verdict judge produces a markdown report followed by a JSON verdict block.
    This helper splits them, returning only the markdown report.

    Strategies:
    1. Look for ```json { ... "verdict" ... } ``` code fence at end
    2. Fallback: find last '{' that parses as JSON with "verdict" key
    3. If no JSON found, return full response

    Args:
        response: Full judge response text

    Returns:
        Markdown report content (without JSON verdict block)
    """
    if not response:
        return ""

    import re

    # Strategy 1: Look for ```json ... ``` block containing "verdict"
    # Match the last ```json block in the response
    pattern = r'```json\s*\n?\s*(\{[^`]*?"verdict"[^`]*?\})\s*\n?\s*```'
    matches = list(re.finditer(pattern, response, re.DOTALL))
    if matches:
        last_match = matches[-1]
        report = response[:last_match.start()].rstrip()
        logger.debug("report.extract.json_fence", report_length=len(report))
        return report

    # Strategy 2: Find last '{' that parses as JSON with "verdict" key
    last_brace = response.rfind('{')
    while last_brace >= 0:
        try:
            candidate = response[last_brace:]
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "verdict" in parsed:
                report = response[:last_brace].rstrip()
                logger.debug("report.extract.inline_json", report_length=len(report))
                return report
        except (json.JSONDecodeError, ValueError):
            pass
        last_brace = response.rfind('{', 0, last_brace)

    # Strategy 3: No JSON verdict found, return full response
    logger.debug("report.extract.no_json", response_length=len(response))
    return response

DEFAULT_BROWSER_VIEWPORT = os.environ.get("DOKUMEN_BROWSER_VIEWPORT", "1512x982")
DEFAULT_BROWSER_VIDEO_SIZE = os.environ.get("DOKUMEN_BROWSER_VIDEO_SIZE", DEFAULT_BROWSER_VIEWPORT)
DEFAULT_BROWSER_HEADLESS = False


def resolve_browser_headless(config_headless: Optional[bool] = None) -> bool:
    """Resolve browser headless mode based on priority.

    Priority order:
    1. Explicit config (from YAML) - highest priority
    2. DOKUMEN_BROWSER_HEADLESS env var (true/1/yes)
    3. CI environment detection (CI or GITLAB_CI env vars)
    4. DEFAULT_BROWSER_HEADLESS (False) - lowest priority

    Args:
        config_headless: Explicit headless setting from BrowserConfig

    Returns:
        True for headless mode, False for headful mode
    """
    # Priority 1: Explicit config overrides everything
    if config_headless is not None:
        return config_headless

    # Priority 2: DOKUMEN_BROWSER_HEADLESS env var
    env_headless = os.environ.get("DOKUMEN_BROWSER_HEADLESS", "").lower()
    if env_headless in ("1", "true", "yes"):
        return True

    # Priority 3: CI environment detection (no display server available)
    if os.environ.get("CI") or os.environ.get("GITLAB_CI"):
        logger.info("test.browser.headless_auto", reason="CI environment detected")
        return True

    # Priority 4: Default (headful for local development)
    return DEFAULT_BROWSER_HEADLESS

if TYPE_CHECKING:
    from .config import ExploreConfig
    from .coverage_agent import CoverageAgent
    from .pipeline import PipelineContext
    from .sandbox import Sandbox, SandboxConfig
    from .tool_resolver import ToolProvenance
    from .user_tool_overrides import ToolOverridesResult


@dataclass
class FailureAnalysis:
    """Analysis of a test failure identifying potentially incorrect documentation."""
    file_path: str
    referenced_lines: List[int]
    incorrect_lines: List[IncorrectLine]
    analysis: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "file_path": self.file_path,
            "referenced_lines": self.referenced_lines,
            "incorrect_lines": [
                {
                    "line_number": il.line_number,
                    "reason": il.reason,
                    "test_id": il.test_id
                }
                for il in self.incorrect_lines
            ],
            "analysis": self.analysis
        }


@dataclass
class TestResult:
    """Complete result of running a test."""
    __test__ = False  # Tell pytest this is not a test class
    test_id: str
    passed: bool
    executor_passed: bool
    judge_results: List[JudgeResult]
    executor_output: Optional[ExecutorOutput]
    duration: float  # seconds
    timestamp: datetime
    files: List[str] = field(default_factory=list)  # files covered by this test
    failure_reasons: List[str] = field(default_factory=list)
    line_coverage: Dict[str, LineCoverage] = field(default_factory=dict)  # file_path -> LineCoverage
    failure_analysis: Dict[str, FailureAnalysis] = field(default_factory=dict)  # file_path -> FailureAnalysis
    explore_output: Optional[str] = None  # Natural language summary from explore phase
    explore_tool_calls: Optional[List[Dict[str, Any]]] = None  # Tool calls made during explore
    executor_model: Optional[str] = None  # Model used for executor
    judge_model: Optional[str] = None  # Model used for judges (first judge, backward compat)
    judge_models: Optional[Dict[str, str]] = None  # Per-judge model map {name: model_id}
    explore_model: Optional[str] = None  # Model used for explore phase
    # Token usage per phase
    executor_input_tokens: int = 0
    executor_output_tokens: int = 0
    executor_cache_creation_tokens: int = 0
    executor_cache_read_tokens: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_cache_creation_tokens: int = 0
    judge_cache_read_tokens: int = 0
    explore_input_tokens: int = 0
    explore_output_tokens: int = 0
    explore_cache_creation_tokens: int = 0
    explore_cache_read_tokens: int = 0
    # Judge prompts for display in UI (list of dicts with 'name' and 'system_prompt')
    judge_prompts: Optional[List[Dict[str, Any]]] = None
    # Browser test artifacts (videos and screenshots)
    browser_artifacts: Optional[List[Dict[str, Any]]] = None
    # Research report artifacts (markdown reports from verdict judge)
    report_artifacts: Optional[List[Dict[str, Any]]] = None
    # Explore phase status: "pass", "fail", or None if explore was not run
    explore_status: Optional[str] = None
    # Output artifacts (files written by executor/judge to output folder)
    output_artifacts: Optional[List[Dict[str, Any]]] = None
    # Executor tool names for config visibility in results
    executor_tools: List[str] = field(default_factory=list)
    # Test status: "passed", "failed", or "error" (judge timeout/crash)
    status: str = "passed"
    # Conversation logs for UI display
    executor_conversation_log: Optional[List[Dict[str, Any]]] = None
    judge_conversation_logs: Optional[List[Dict[str, Any]]] = None
    # Path to the source YAML scaffold file
    source_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "test_id": self.test_id,
            "passed": self.passed,
            "executor_passed": self.executor_passed,
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
            "failure_reasons": self.failure_reasons,
            "executor_output": self.executor_output.to_dict() if self.executor_output else None,
            "judge_results": [jr.to_dict() for jr in self.judge_results],
            "line_coverage": {
                path: cov.to_dict() for path, cov in self.line_coverage.items()
            },
            "failure_analysis": {
                path: fa.to_dict() for path, fa in self.failure_analysis.items()
            },
            "explore_output": self.explore_output,
            "explore_tool_calls": self.explore_tool_calls,
            "executor_model": self.executor_model,
            "judge_model": self.judge_model,
            "judge_models": self.judge_models,
            "explore_model": self.explore_model,
            "executor_input_tokens": self.executor_input_tokens,
            "executor_output_tokens": self.executor_output_tokens,
            "executor_cache_creation_tokens": self.executor_cache_creation_tokens,
            "executor_cache_read_tokens": self.executor_cache_read_tokens,
            "judge_input_tokens": self.judge_input_tokens,
            "judge_output_tokens": self.judge_output_tokens,
            "judge_cache_creation_tokens": self.judge_cache_creation_tokens,
            "judge_cache_read_tokens": self.judge_cache_read_tokens,
            "explore_input_tokens": self.explore_input_tokens,
            "explore_output_tokens": self.explore_output_tokens,
            "explore_cache_creation_tokens": self.explore_cache_creation_tokens,
            "explore_cache_read_tokens": self.explore_cache_read_tokens,
            "judge_prompts": self.judge_prompts,
            "browser_artifacts": self.browser_artifacts,
            "report_artifacts": self.report_artifacts,
            "output_artifacts": self.output_artifacts,
            "executor_tools": self.executor_tools,
            "explore_status": self.explore_status,
        }


@dataclass
class TestConfig:
    """Configuration for a test object."""
    id: str
    reason: str
    executor: Any
    judges: List[Any]
    timeout: float = 60.0
    retries: int = 0


@dataclass
class BrowserConfig:
    """Browser configuration for Playwright MCP sessions."""
    headless: Optional[bool] = None
    save_video: Optional[str] = None
    viewport_size: Optional[str] = None


class TestObject:
    """
    Represents a single test case.

    Orchestrates the execution of an executor agent to perform a task,
    then evaluates the results using one or more judge agents.
    """
    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(
        self,
        id: str,
        reason: str,
        executor: Any,
        judges: List[Any],
        timeout: float = 60.0,
        retries: int = 0,
        sandbox_config: Optional['SandboxConfig'] = None,
        source_path: Optional[str] = None,
        files: Optional[List[str]] = None,
        explore_config: Optional['ExploreConfig'] = None,
        browser_config: Optional[BrowserConfig] = None,
        test_type: Optional[str] = None,
        tool_overrides: Optional['ToolOverridesResult'] = None,
        tool_provenance: Optional['ToolProvenance'] = None,
        setup_steps: Optional[List] = None,
        agent: Optional[str] = None,
        outputs: Optional[List[str]] = None,
        user_dirs: Optional[List] = None,
        resolved_skills: Optional[Dict] = None,
        compaction_config: Optional[Any] = None,
        coordinator_config: Optional[Any] = None,
        tasks_config: Optional[Any] = None,
    ):
        """
        Initialize a test object.

        Args:
            id: Unique identifier for the test
            reason: Human-readable description of what the test validates
            executor: The agent that performs the task being tested
            judges: One or more agents that evaluate the executor's work
            timeout: Maximum time (seconds) for test execution
            retries: Number of retries on executor failure
            sandbox_config: Optional sandbox configuration for this test
            source_path: Optional path to the source YAML file
            files: Optional list of file paths covered by this test
            explore_config: Optional explore phase configuration
            test_type: Optional test type (e.g., 'research', 'browser') — deprecated, use agent
            tool_overrides: Optional pre-loaded tool overrides (avoids re-reading filesystem)
            tool_provenance: Optional provenance tracking for tool observability
            setup_steps: Optional list of SetupStep to run before executor
            agent: Optional pre-built agent name (replaces test_type)
            outputs: Optional list of output file paths the executor should produce
            user_dirs: Optional list of directories for user-defined agents
            resolved_skills: Optional dict of resolved skills {name: {content, source, used_by}}
            compaction_config: Optional CompactionConfig for context compaction
            coordinator_config: Optional CoordinatorConfig for multi-agent mode
            tasks_config: Optional TasksConfig for task tracking
        """
        self.id = id
        self.reason = reason
        self.executor = executor
        self.judges = judges
        self.timeout = timeout
        self.retries = retries
        self.sandbox_config: Optional['SandboxConfig'] = sandbox_config
        self.source_path = source_path
        self.files: List[str] = files or []
        self.explore_config: Optional['ExploreConfig'] = explore_config
        self.browser_config: Optional[BrowserConfig] = browser_config
        self.test_type: Optional[str] = test_type
        self.tool_overrides = tool_overrides
        self.tool_provenance = tool_provenance
        self.setup_steps = setup_steps or []
        self.agent: Optional[str] = agent
        self.outputs: Optional[List[str]] = outputs
        self.user_dirs: Optional[List] = user_dirs
        self.resolved_skills: Optional[Dict] = resolved_skills
        self.compaction_config = compaction_config
        self.coordinator_config = coordinator_config
        self.tasks_config = tasks_config
        self._cached_hash: Optional[str] = None

    async def run(
        self,
        coverage_agent: Optional['CoverageAgent'] = None,
        sandbox: Optional['Sandbox'] = None,
        on_tool_call=None,
        on_conversation_message=None,
        on_executor_complete=None,
        on_judge_complete=None,
        on_explore_event=None
    ) -> TestResult:
        """
        Execute the test by running executor and all judges.

        Delegates to a TestPipeline composed of independently testable stages:
        BrowserSetupStage, SetupStage, ExploreStage, ExecutorStage, JudgeStage,
        ArtifactStage. Each stage reads/writes a shared PipelineContext.

        Args:
            coverage_agent: Optional coverage agent for line-level tracking.
                           Only runs if the test passes.
            sandbox: Optional sandbox for isolated execution. When provided,
                    executor tools are re-resolved with sandbox context,
                    allowing write_file, bash, etc. to execute in isolation.
            on_tool_call: Optional callback fired after each tool execution.
                         Signature: (tool_name: str, params: dict, result: Any) -> None
            on_conversation_message: Optional callback for streaming conversation.
                         Signature: (agent_type: str, message_type: str, content: str) -> None
            on_executor_complete: Optional callback fired when executor finishes.
                         Signature: (executor_output: ExecutorOutput) -> None
            on_judge_complete: Optional callback fired when each judge finishes.
                         Signature: (judge_result: JudgeResult) -> None
            on_explore_event: Optional callback for explore phase events.
                         Signature: (event_type: str, data: dict) -> None
                         Events: 'start', 'file_found', 'complete'

        Returns:
            TestResult with complete test results
        """
        from .pipeline import PipelineContext, TestPipeline
        from .stages import (
            BrowserSetupStage,
            SetupStage,
            ExploreStage,
            ExecutorStage,
            JudgeStage,
            ArtifactStage,
            MemoryStage,
            CompactionStage,
            CoordinatorStage,
        )

        start_time = time.time()
        logger.info("test.run.start", test_id=self.id, timeout=self.timeout, files_count=len(self.files))

        # Write resolved agent definition to .dokumen-cache/agents/ for CI observability
        if self.agent:
            self._write_agent_artifact()

        # Write resolved skills to .dokumen-cache/skills/ for CI observability
        if self.resolved_skills:
            self._write_skills_artifact()

        # If sandbox provided, re-resolve tools with sandbox context
        original_executor_tools = None
        original_judge_tools = {}
        if sandbox:
            debug(f"[DEBUG TEST] Re-resolving tools with sandbox id={id(sandbox)}")
            original_executor_tools = self.executor.tools
            self.executor.tools = self._resolve_tools_with_sandbox(sandbox)
            debug(f"[DEBUG TEST] Executor tools resolved: {[t.name for t in self.executor.tools]}")
            for judge in self.judges:
                if judge.tools:
                    original_judge_tools[judge.id] = judge.tools
                    judge.tools = self._resolve_tools_with_sandbox(sandbox, judge.tools)
                    debug(f"[DEBUG TEST] Judge '{judge.id}' tools resolved: {[t.name for t in judge.tools]}")

        # Build the pipeline context
        ctx = PipelineContext(
            test_id=self.id,
            reason=self.reason,
            executor=self.executor,
            judges=self.judges,
            files=self.files,
            timeout=self.timeout,
            retries=self.retries,
            browser_config=self.browser_config,
            explore_config=self.explore_config,
            sandbox=sandbox,
            sandbox_config=self.sandbox_config,
            source_path=self.source_path,
            test_type=self.test_type,
            on_tool_call=on_tool_call,
            on_conversation_message=on_conversation_message,
            on_executor_complete=on_executor_complete,
            on_judge_complete=on_judge_complete,
            on_explore_event=on_explore_event,
            setup_steps=self.setup_steps,
            agent=self.agent,
            outputs=self.outputs,
            user_dirs=self.user_dirs,
            resolved_skills=self.resolved_skills,
        )

        # Build stage list — coordinator replaces executor when enabled
        use_coordinator = (
            self.coordinator_config is not None
            and getattr(self.coordinator_config, 'enabled', False)
        )

        stages = [
            BrowserSetupStage(),
            SetupStage(),
            ExploreStage(),
        ]

        if use_coordinator:
            stages.append(CoordinatorStage(coordinator_config=self.coordinator_config))
        else:
            stages.append(ExecutorStage())

        # compaction runs after executor/coordinator, before judges
        stages.append(CompactionStage(compaction_config=self.compaction_config))

        stages.extend([
            JudgeStage(),
            MemoryStage(),
            ArtifactStage(),
        ])

        # Cleanup callbacks — always run regardless of success/failure
        async def _cleanup(ctx):
            """Clean up resources after pipeline execution."""
            # Clean up setup background processes
            if ctx.setup_runner:
                try:
                    await ctx.setup_runner.cleanup()
                except Exception as e:
                    logger.error("test.setup.cleanup_error",
                                 test_id=ctx.test_id, error=str(e))

            # Restore original tools if we modified them
            if original_executor_tools is not None:
                self.executor.tools = original_executor_tools
            for judge in self.judges:
                if judge.id in original_judge_tools:
                    judge.tools = original_judge_tools[judge.id]

            # Restore executor and judge prompts after output folder injection
            for judge in self.judges:
                if judge.id in ctx.original_judge_prompts:
                    judge.system_prompt = ctx.original_judge_prompts[judge.id]

        pipeline = TestPipeline(stages=stages, cleanup_callbacks=[_cleanup])
        ctx = await pipeline.run(ctx)

        # === Assemble TestResult from PipelineContext ===
        return self._build_result(ctx, start_time, original_executor_tools, original_judge_tools)

    def _build_result(
        self,
        ctx: 'PipelineContext',
        start_time: float,
        original_executor_tools: Optional[List],
        original_judge_tools: Dict[str, Any],
    ) -> 'TestResult':
        """Assemble a TestResult from the completed PipelineContext.

        Extracts model info, token usage, conversation logs, artifacts,
        and determines the final pass/fail/error status.

        Args:
            ctx: The completed pipeline context.
            start_time: Epoch timestamp when the test started.
            original_executor_tools: Executor tools before sandbox resolution (or None).
            original_judge_tools: Judge tools before sandbox resolution.

        Returns:
            A fully populated TestResult.
        """
        duration = time.time() - start_time

        # Extract model names from providers
        executor_model = None
        if self.executor.provider and hasattr(self.executor.provider, 'model'):
            executor_model = self.executor.provider.model

        judge_models_map: Dict[str, str] = {}
        for j in self.judges:
            if j.provider and hasattr(j.provider, 'model'):
                judge_models_map[j.id] = j.provider.model
        judge_model = next(iter(judge_models_map.values()), None)

        # Determine executor_passed — True only if executor ran successfully
        executor_passed = bool(
            ctx.executor_output and ctx.executor_output.success
        )

        # Collect executor token usage
        executor_input_tokens = 0
        executor_output_tokens = 0
        executor_cache_creation_tokens = 0
        executor_cache_read_tokens = 0
        executor_conversation_log = None
        if ctx.executor_output:
            executor_input_tokens = ctx.executor_output.input_tokens
            executor_output_tokens = ctx.executor_output.output_tokens
            executor_cache_creation_tokens = getattr(
                ctx.executor_output, 'cache_creation_tokens', 0
            )
            executor_cache_read_tokens = getattr(
                ctx.executor_output, 'cache_read_tokens', 0
            )
            if ctx.executor_output.conversation_log:
                executor_conversation_log = ctx.executor_output.conversation_log

        # Aggregate judge token usage
        judge_input_tokens = 0
        judge_output_tokens = 0
        judge_cache_creation_tokens = 0
        judge_cache_read_tokens = 0
        for jr in ctx.judge_results:
            judge_input_tokens += jr.input_tokens
            judge_output_tokens += jr.output_tokens
            judge_cache_creation_tokens += getattr(jr, 'cache_creation_tokens', 0)
            judge_cache_read_tokens += getattr(jr, 'cache_read_tokens', 0)

        # Extract judge prompts for display
        judge_prompts = [
            {
                'name': judge.id,
                'system_prompt': judge.system_prompt,
                'user_prompt': getattr(judge, 'user_prompt', None) or None,
            }
            for judge in self.judges
        ]

        # Extract judge conversation logs
        judge_conversation_logs = None
        judge_conv_list = []
        for judge, jr in zip(self.judges, ctx.judge_results):
            if jr.conversation_log:
                judge_conv_list.append({
                    'judge_name': judge.id,
                    'iterations': jr.conversation_log,
                })
        if judge_conv_list:
            judge_conversation_logs = judge_conv_list

        # Extract explore data
        explore_output = None
        explore_tool_calls = None
        if ctx.explore_result:
            explore_output = ctx.explore_result.summary
            explore_tool_calls = ctx.explore_result.tool_history

        # Extract output artifacts from context (set by ArtifactStage)
        output_artifacts = ctx.output_artifacts or None
        browser_artifacts = None
        report_artifacts = None
        if output_artifacts:
            browser_artifacts = [
                {
                    'type': (
                        'video' if a.get('content_type', '').startswith('video/')
                        else 'screenshot'
                    ),
                    'path': a['path'],
                    'filename': a['filename'],
                    'size_bytes': a['size_bytes'],
                }
                for a in output_artifacts if a.get('source') == 'browser'
            ]
            report_artifacts = [
                {
                    'type': 'report',
                    'path': a['path'],
                    'filename': a['filename'],
                    'size_bytes': a['size_bytes'],
                    'content': a.get('content'),
                }
                for a in output_artifacts if a.get('source') == 'report'
            ]

        # Determine final pass/fail/error status
        has_judge_error = any(
            getattr(jr, 'error', False) for jr in ctx.judge_results
        )
        if has_judge_error:
            passed = False
            status = "error"
            logger.info("test.result.judge_error_detected", test_id=self.id,
                        error_judges=[
                            jr.judge_id for jr in ctx.judge_results
                            if getattr(jr, 'error', False)
                        ])
        elif ctx.failed:
            passed = False
            status = "failed"
        elif executor_passed and all(jr.passed for jr in ctx.judge_results):
            passed = True
            status = "passed"
        else:
            passed = False
            status = "failed"

        result = TestResult(
            test_id=self.id,
            passed=passed,
            executor_passed=executor_passed,
            judge_results=list(ctx.judge_results),
            executor_output=ctx.executor_output,
            duration=duration,
            timestamp=datetime.now(),
            files=self.files,
            failure_reasons=list(ctx.failure_reasons),
            line_coverage={},
            executor_model=executor_model,
            judge_model=judge_model,
            judge_models=judge_models_map or None,
            explore_model=ctx.explore_model,
            executor_tools=[t.name for t in self.executor.tools] if self.executor.tools else [],
            source_path=self.source_path,
            # Token usage
            executor_input_tokens=executor_input_tokens,
            executor_output_tokens=executor_output_tokens,
            executor_cache_creation_tokens=executor_cache_creation_tokens,
            executor_cache_read_tokens=executor_cache_read_tokens,
            judge_input_tokens=judge_input_tokens,
            judge_output_tokens=judge_output_tokens,
            judge_cache_creation_tokens=judge_cache_creation_tokens,
            judge_cache_read_tokens=judge_cache_read_tokens,
            explore_input_tokens=ctx.explore_input_tokens,
            explore_output_tokens=ctx.explore_output_tokens,
            explore_cache_creation_tokens=ctx.explore_cache_creation_tokens,
            explore_cache_read_tokens=ctx.explore_cache_read_tokens,
            # Explore data
            explore_output=explore_output,
            explore_tool_calls=explore_tool_calls,
            explore_status=ctx.explore_status,
            # Prompts and conversation logs
            judge_prompts=judge_prompts,
            executor_conversation_log=executor_conversation_log,
            judge_conversation_logs=judge_conversation_logs,
            # Artifacts
            output_artifacts=output_artifacts,
            browser_artifacts=browser_artifacts or None,
            report_artifacts=report_artifacts or None,
            # Status
            status=status,
        )

        logger.info("test.run.complete", test_id=self.id, passed=result.passed,
                     duration_ms=int(result.duration * 1000),
                     judges_passed=sum(1 for jr in result.judge_results if jr.passed),
                     judges_total=len(result.judge_results),
                     status=status)

        return result

    def _uses_browser_tools(self) -> bool:
        """Check if this test uses any browser automation tools."""
        # Check executor tools
        for tool in self.executor.tools:
            if tool.name in BROWSER_TOOL_NAMES:
                return True
        # Check judge tools
        for judge in self.judges:
            if judge.tools:  # Skip if tools is None
                for tool in judge.tools:
                    if tool.name in BROWSER_TOOL_NAMES:
                        return True
        return False

    def _write_agent_artifact(self) -> None:
        """Write resolved agent definitions to .dokumen-cache/agents/ for CI observability.

        Creates YAML files for executor and judge agents so CI pipelines
        can inspect which agents ran with what configuration.
        """
        import re
        from pathlib import Path
        from dokumen_schema.agent_defs import load_agent as _load_agent

        def _write_single_agent(agent_name: str, prefix: str = "") -> None:
            """Write a single agent definition to the cache."""
            if not agent_name:
                return
            if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", agent_name):
                logger.warning(
                    "agent.definition.invalid_name",
                    agent_name=agent_name,
                )
                return

            agent_def = _load_agent(agent_name, user_dirs=self.user_dirs)
            if agent_def:
                # Use project-root .dokumen-cache/ (consistent with other artifacts)
                agents_dir = Path(".dokumen-cache") / "agents"
                agents_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{prefix}{agent_name}.agent.yaml" if prefix else f"{agent_name}.agent.yaml"
                agent_path = agents_dir / filename
                agent_path.write_text(
                    yaml.dump(agent_def.model_dump(), default_flow_style=False, sort_keys=False)
                )
                logger.info(
                    "agent.definition.artifact_written",
                    path=str(agent_path),
                    agent_name=agent_name,
                )

        try:
            # Write executor agent
            _write_single_agent(self.agent)

            # Write judge agents
            for judge in self.judges:
                judge_agent = getattr(judge, 'agent_name', None)
                if judge_agent:
                    _write_single_agent(judge_agent, prefix=f"judge-{judge.id}-")
        except Exception as e:
            logger.warning(
                "agent.definition.artifact_write_failed",
                agent_name=self.agent,
                error=str(e),
            )

    def _write_skills_artifact(self) -> None:
        """Write resolved skills to .dokumen-cache/skills/{test-id}/ for CI observability.

        Creates one file per skill so CI pipelines can inspect what skills
        were available to the executor and judges.
        """
        from pathlib import Path
        try:
            skills_dir = Path(".dokumen-cache") / "skills" / self.id
            skills_dir.mkdir(parents=True, exist_ok=True)

            for skill_name, skill_info in self.resolved_skills.items():
                # Write the skill content
                skill_path = skills_dir / f"{skill_name}.md"
                skill_path.write_text(skill_info["content"])

                # Write metadata
                meta_path = skills_dir / f"{skill_name}.meta.yaml"
                meta_content = (
                    f"name: {skill_name}\n"
                    f"source: {skill_info.get('source', 'unknown')}\n"
                    f"used_by: {skill_info.get('used_by', 'unknown')}\n"
                    f"content_length: {len(skill_info['content'])}\n"
                )
                meta_path.write_text(meta_content)

            logger.info(
                "skills.artifact_written",
                test_id=self.id,
                skill_count=len(self.resolved_skills),
                skill_names=list(self.resolved_skills.keys()),
                path=str(skills_dir),
            )
        except Exception as e:
            logger.warning(
                "skills.artifact_write_failed",
                test_id=self.id,
                error=str(e),
            )

    def _resolve_tools_with_sandbox(self, sandbox: 'Sandbox', tools: List = None) -> List:
        """
        Re-resolve tools with sandbox context.

        Gets the tool names from the provided tools (or executor tools) and resolves them
        with the sandbox, enabling sandbox-aware tools like write_file, bash, http_request, etc.
        Also passes provider context for context-dependent tools like spawn_subagent.

        Args:
            sandbox: Sandbox instance for tool execution
            tools: Optional list of tools to resolve (defaults to executor tools)

        Returns:
            List of ToolDefinition objects with sandbox context
        """
        from .loader import resolve_tools, find_project_root

        # Extract tool names from provided tools or executor tools
        source_tools = tools if tools is not None else self.executor.tools
        tool_names = [tool.name for tool in source_tools]

        # Use project root as base_dir for file path resolution
        # Files in test YAML are relative to project root (where dokumen.yaml is)
        if self.source_path:
            base_dir = find_project_root(self.source_path)
        else:
            base_dir = "."

        # Re-resolve with sandbox context and provider for context-dependent tools
        return resolve_tools(
            tool_names,
            base_dir=base_dir,
            sandbox=sandbox,
            provider=self.executor.provider,
            parent_tools=source_tools
        )

    def is_stale(self) -> bool:
        """
        Check if cached result is stale.

        Returns True if any of the following changed since the result was cached:
        - Test configuration (id, reason, timeout, executor, judges)
        - File contents (any referenced file was modified)

        Returns:
            True if cache should be invalidated
        """
        current_hash = self.get_hash()
        return current_hash != self._cached_hash

    def get_hash(self) -> str:
        """
        Generate hash of test configuration for cache validation.

        The hash includes the test configuration and prompts.
        This ensures the cache is invalidated when:
        - Test configuration changes
        - Executor or judge prompts change

        Returns:
            SHA256 hash string
        """
        hash_data = {
            "id": self.id,
            "reason": self.reason,
            "executor_id": self.executor.id,
            "executor_system_prompt": self.executor.system_prompt,
            "executor_user_prompt": self.executor.user_prompt,
            "judge_ids": [j.id for j in self.judges],
            "judge_prompts": [j.system_prompt for j in self.judges],
            "timeout": self.timeout,
            "setup_steps": [
                {"name": s.name, "command": s.command,
                 "working_dir": s.working_dir, "timeout": s.timeout,
                 "background": s.background, "ready_url": s.ready_url,
                 "ready_timeout": s.ready_timeout}
                for s in self.setup_steps
            ] if self.setup_steps else None,
        }
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def set_cached_hash(self, hash_value: str) -> None:
        """
        Set the cached hash value.

        Args:
            hash_value: Hash to cache
        """
        self._cached_hash = hash_value

    async def _run_explore(self, on_progress=None) -> Optional[ExploreResult]:
        """
        Run the explore phase to discover relevant documentation.

        Uses the SDK-based ExploreAgent with read-only tools (Read, Glob, Grep).

        Args:
            on_progress: Optional callback for explore events
                        Signature: (event_type: str, data: dict) -> None

        Returns:
            ExploreResult with discovered files, or None if explore fails
        """
        # Create default ExploreConfig if not provided but files are required
        explore_config = self.explore_config
        if not explore_config and self.files:
            # Import here to avoid circular dependency
            from .config import ExploreConfig
            explore_config = ExploreConfig()  # Use defaults
            if is_debug():
                debug("[TEST] Created default ExploreConfig for test with required files")

        if not explore_config:
            return None

        base_dir = self._get_base_dir()

        # Determine the model for the explore agent
        explore_model = explore_config.model

        logger.info(
            "explore.run",
            test_name=self.id,
            explore_model=explore_model,
            max_files=explore_config.max_files,
            timeout=explore_config.timeout,
        )

        explore_agent = ExploreAgent(
            base_dir=base_dir,
            max_files=explore_config.max_files,
            max_turns=explore_config.max_iterations,
            timeout=float(explore_config.timeout),
            model=explore_model,
        )

        # Use full executor user_prompt as exploration goal
        goal = self.executor.user_prompt

        return await explore_agent.explore(goal=goal, on_progress=on_progress)

    def _inject_explore_context(self, explore_result: ExploreResult) -> None:
        """
        Inject explore results into executor's user_prompt.

        Prefers natural language summary if available, falls back to files list.

        Args:
            explore_result: Results from exploration phase
        """
        if not explore_result:
            return

        # Check if there's anything to inject (summary or files)
        if not explore_result.summary and not explore_result.files:
            return

        # Format context block (to_context_block prefers summary)
        context_block = explore_result.to_context_block()
        if context_block:
            # Prepend context to user_prompt
            self.executor.user_prompt = f"{context_block}\n\n---\n\n{self.executor.user_prompt}"

    def _check_files_on_disk(self, missing_files: List[str], explore_result: ExploreResult) -> List[str]:
        """
        Deterministic fallback: check filesystem directly for files the explore AI missed.

        When the explore agent fails to discover required files (non-deterministic AI search),
        this method checks whether those files actually exist on disk. Files that exist are
        added to the explore result and removed from the missing list.

        Args:
            missing_files: File paths that explore didn't find
            explore_result: Explore result to augment with recovered files

        Returns:
            List of file paths that truly don't exist on disk
        """
        from .explore_agent import FileDiscovery

        base_dir = self._get_base_dir()
        still_missing = []

        for file_path in missing_files:
            full_path = os.path.join(base_dir, file_path)
            normalized = os.path.normpath(full_path)

            if os.path.isfile(normalized):
                logger.info("test.explore.deterministic_recovery",
                            test_id=self.id, file_path=file_path,
                            message="File exists on disk but was missed by explore agent")
                explore_result.files.append(
                    FileDiscovery(
                        path=file_path,
                        summary="(recovered by deterministic filesystem check)",
                        relevance=0.5,
                    )
                )
            else:
                logger.info("test.explore.deterministic_confirmed_missing",
                            test_id=self.id, file_path=file_path,
                            message="File confirmed missing from disk")
                still_missing.append(file_path)

        if len(still_missing) < len(missing_files):
            recovered_count = len(missing_files) - len(still_missing)
            logger.info("test.explore.deterministic_recovery_summary",
                        test_id=self.id,
                        recovered=recovered_count,
                        still_missing=len(still_missing))

        return still_missing

    def _verify_explore_found_files(self, explore_result: ExploreResult) -> List[str]:
        """
        Check that required files appear in explore result.

        Args:
            explore_result: Result from explore phase

        Returns:
            List of missing file paths (empty if all found)
        """
        if not self.files:
            return []

        missing = []
        summary = explore_result.summary or ""

        # Build set of found file paths from files list (raw + normalized)
        found_paths = set()
        found_paths_normalized = set()
        if explore_result.files:
            for f in explore_result.files:
                found_paths.add(f.path)
                normalized = os.path.normpath(f.path).lstrip('./')
                found_paths_normalized.add(normalized)

        for file_path in self.files:
            normalized_required = os.path.normpath(file_path).lstrip('./')
            # Check raw and normalized paths in both summary and files list
            if (file_path not in summary and normalized_required not in summary
                    and file_path not in found_paths and normalized_required not in found_paths_normalized):
                logger.info(f"[VERIFY] Missing file: {file_path!r}, normalized: {normalized_required!r}, found_normalized: {found_paths_normalized}")
                missing.append(file_path)

        return missing

    def _fail_with_missing_files_error(
        self,
        missing_files: List[str],
        explore_result: ExploreResult
    ) -> TestResult:
        """
        Create detailed failure result for missing required files.

        Args:
            missing_files: List of file paths that were not found
            explore_result: Result from explore phase

        Returns:
            TestResult with failure status and verbose error message
        """
        error_lines = [
            "EXPLORE PHASE FAILED: Required files not found",
            "",
            "Required files (from test scaffold):",
        ]

        for f in self.files:
            status = "FOUND" if f not in missing_files else "MISSING"
            error_lines.append(f"  [{status}] {f}")

        # Add diagnostic information for debugging
        error_lines.extend([
            "",
            "Explore diagnostics:",
            f"  success: {explore_result.success}",
            f"  error: {explore_result.error or '(none)'}",
            f"  tool_calls_count: {explore_result.tool_calls_count}",
            f"  files_found: {len(explore_result.files)}",
            "",
            "Explore summary:",
            explore_result.summary if explore_result.summary else "(no summary)",
            "",
        ])

        # Include file paths if any were found
        if explore_result.files:
            error_lines.append("Files discovered by explore agent:")
            for f in explore_result.files:
                error_lines.append(f"  - {f.path}")
            error_lines.append("")

        # Include tool history if available (limited)
        if explore_result.tool_history:
            error_lines.append(f"Tool calls ({len(explore_result.tool_history)} total):")
            for tc in explore_result.tool_history[:5]:  # Show first 5
                tool_name = tc.get('tool', 'unknown')
                command = str(tc.get('command', ''))[:50]
                error_lines.append(f"  - {tool_name}: {command}")
            if len(explore_result.tool_history) > 5:
                error_lines.append(f"  ... and {len(explore_result.tool_history) - 5} more")
            error_lines.append("")

        error_lines.extend([
            "The explore agent must discover the required files from docs/.",
            "Ensure the files exist and the explore agent's search finds them.",
        ])

        return TestResult(
            test_id=self.id,
            passed=False,
            executor_passed=False,
            judge_results=[],
            executor_output=None,
            duration=explore_result.duration,
            timestamp=datetime.now(),
            files=self.files,
            failure_reasons=["\n".join(error_lines)],
            line_coverage={},
            explore_output=explore_result.summary,
            explore_tool_calls=explore_result.tool_history,
            explore_status="fail",
        )

    def _get_base_dir(self) -> str:
        """Get base directory for explore operations.

        Returns the project root (where dokumen.yaml lives) so that paths like
        docs/policies/margin-policy.md resolve correctly regardless of where
        the test YAML file is located.
        """
        if self.source_path:
            import os
            # Search upward from source_path to find project root (dokumen.yaml)
            from .loader import find_project_root
            try:
                return find_project_root(self.source_path)
            except FileNotFoundError:
                # Fall back to source file directory if no dokumen.yaml found
                return os.path.dirname(self.source_path)
        return "."

    def __repr__(self) -> str:
        return f"TestObject(id={self.id}, judges={len(self.judges)})"
