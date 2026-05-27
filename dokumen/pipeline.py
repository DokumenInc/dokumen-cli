"""
Pipeline framework for test execution.

Provides the PipelineContext, PipelineStage ABC, and TestPipeline runner
that orchestrate test execution through independently testable stages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineContext:
    """Shared context passed through all pipeline stages.

    Each stage reads from and writes to this context. When a stage
    fails, it sets ``failed=True`` and appends a reason so the
    pipeline runner can short-circuit.

    Attributes:
        test_id: Unique identifier for the test.
        reason: Human-readable description of what the test validates.
        executor: The agent that performs the task being tested.
        judges: One or more agents that evaluate the executor's work.
        files: File paths covered by this test.
        timeout: Maximum time (seconds) for test execution.
        retries: Number of retries on executor failure.
        failed: Whether the pipeline has failed.
        failure_reasons: List of failure reason strings.
        executor_output: Result from the executor stage.
        judge_results: Results from the judge stage.
        explore_result: Result from the explore stage.
        browser_config: Optional browser configuration.
        explore_config: Optional explore configuration.
        sandbox: Optional sandbox instance.
        sandbox_config: Optional sandbox configuration.
        source_path: Optional path to the source YAML file.
        test_type: Optional test type (e.g., 'research', 'browser').
        output_dir: Path to the unified output directory.
        setup_runner: Setup runner instance (set by SetupStage).
        original_user_prompt: User prompt before explore injection.
        original_executor_tools: Executor tools before sandbox resolution.
        original_judge_tools: Judge tools before sandbox resolution.
        original_judge_prompts: Judge prompts before output folder injection.
        research_report_rel_path: Relative path to research report (if any).
        on_tool_call: Optional callback for tool calls.
        on_conversation_message: Optional callback for conversation messages.
        on_executor_complete: Optional callback for executor completion.
        on_judge_complete: Optional callback for judge completion.
        on_explore_event: Optional callback for explore events.
        tool_overrides: Optional tool overrides.
        tool_provenance: Optional tool provenance tracking.
        setup_steps: Optional list of setup steps.
        agent: Optional agent name.
        outputs: Optional list of output file paths.
        user_dirs: Optional user agent directories.
        resolved_skills: Optional resolved skills dict.
        output_artifacts: Output artifacts collected by ArtifactStage.
    """

    # Core test config
    test_id: str
    reason: str
    executor: Any
    judges: List[Any]
    files: List[str]
    timeout: float
    retries: int

    # Pipeline state
    failed: bool = False
    failure_reasons: List[str] = field(default_factory=list)
    executor_output: Any = None  # ExecutorResult
    judge_results: List[Any] = field(default_factory=list)  # List[JudgeVerdict]
    explore_result: Any = None  # Optional[ExploreResult]

    # Optional config
    browser_config: Any = None  # Optional[BrowserConfig]
    explore_config: Any = None  # Optional[ExploreConfig]
    sandbox: Any = None  # Optional[Sandbox]
    sandbox_config: Any = None  # Optional[SandboxConfig]
    source_path: Optional[str] = None
    test_type: Optional[str] = None

    # Runtime state set by stages
    output_dir: str = ""
    setup_runner: Any = None  # Optional[SetupRunner]
    original_user_prompt: str = ""
    original_executor_tools: Any = None
    original_judge_tools: Dict[str, Any] = field(default_factory=dict)
    original_judge_prompts: Dict[str, str] = field(default_factory=dict)
    research_report_rel_path: Optional[str] = None

    # Callbacks
    on_tool_call: Any = None
    on_conversation_message: Any = None
    on_executor_complete: Any = None
    on_judge_complete: Any = None
    on_explore_event: Any = None

    # Additional test object fields
    tool_overrides: Any = None
    tool_provenance: Any = None
    setup_steps: List[Any] = field(default_factory=list)
    agent: Optional[str] = None
    outputs: Optional[List[str]] = None
    user_dirs: Optional[List] = None
    resolved_skills: Optional[Dict] = None

    # Output artifacts (set by ArtifactStage)
    output_artifacts: List[Dict[str, Any]] = field(default_factory=list)

    # Memory system (optional, set when memory.enabled=True in config)
    memory_store: Any = None  # Optional[MemoryStore]
    embedding_provider: Any = None  # Optional[EmbeddingProvider]
    memory_config: Any = None  # Optional[MemoryConfig]

    # Token usage accumulators
    explore_input_tokens: int = 0
    explore_output_tokens: int = 0
    explore_cache_creation_tokens: int = 0
    explore_cache_read_tokens: int = 0
    explore_model: Optional[str] = None
    explore_status: Optional[str] = None

    def fail(self, reason: str) -> None:
        """Mark the pipeline as failed with a reason.

        Args:
            reason: Human-readable description of the failure.
        """
        self.failed = True
        self.failure_reasons.append(reason)
        logger.info("pipeline.stage.failed", test_id=self.test_id, reason=reason)


class PipelineStage(ABC):
    """Abstract base class for pipeline stages.

    Each stage performs a discrete unit of work during test execution.
    Stages receive a PipelineContext, perform work, and return the
    (possibly modified) context.
    """

    @abstractmethod
    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this pipeline stage.

        Args:
            ctx: The shared pipeline context.

        Returns:
            The updated pipeline context.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...


class TestPipeline:
    """Runs a sequence of pipeline stages on a PipelineContext.

    Stages are executed in order. If any stage marks the context
    as failed, subsequent stages are skipped. Cleanup callbacks
    are always called regardless of success or failure.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        cleanup_callbacks: Optional[List[Callable]] = None,
    ):
        """Initialize the pipeline.

        Args:
            stages: Ordered list of stages to execute.
            cleanup_callbacks: Optional callbacks to run after all stages
                (or after short-circuit), regardless of success/failure.
        """
        self.stages = stages
        self.cleanup_callbacks = cleanup_callbacks or []

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute all stages in order, short-circuiting on failure.

        Args:
            ctx: The shared pipeline context.

        Returns:
            The final pipeline context after all stages (or short-circuit).
        """
        logger.info("pipeline.run.start", test_id=ctx.test_id,
                     stage_count=len(self.stages))

        try:
            for stage in self.stages:
                if ctx.failed:
                    logger.info("pipeline.stage.skipped", test_id=ctx.test_id,
                                stage=stage.name, reason="pipeline already failed")
                    break

                logger.info("pipeline.stage.start", test_id=ctx.test_id,
                            stage=stage.name)
                try:
                    ctx = await stage.run(ctx)
                    logger.info("pipeline.stage.complete", test_id=ctx.test_id,
                                stage=stage.name, failed=ctx.failed)
                except Exception as e:
                    logger.error("pipeline.stage.exception", test_id=ctx.test_id,
                                 stage=stage.name, error=str(e),
                                 error_type=type(e).__name__, exc_info=True)
                    ctx.fail(f"Stage '{stage.name}' raised {type(e).__name__}: {e}")
                    break

        finally:
            for callback in self.cleanup_callbacks:
                try:
                    await callback(ctx)
                except Exception as e:
                    logger.error("pipeline.cleanup.error", test_id=ctx.test_id,
                                 error=str(e), exc_info=True)

        logger.info("pipeline.run.complete", test_id=ctx.test_id,
                     failed=ctx.failed,
                     failure_count=len(ctx.failure_reasons))

        return ctx
