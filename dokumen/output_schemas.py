"""
Pydantic models for spec-compliant output file schemas.

These models define the structure of output files written to .dokumen-cache/:
- results.json - Test results with assertions
- coverage.json - File-level coverage metrics
- debug traces - Per-test conversation logs
"""

import os
from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel, Field

# =============================================================================
# Results JSON Schema (results.json)
# =============================================================================


class TokenUsage(BaseModel):
    """Token usage statistics for an AI call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


class AssertionResult(BaseModel):
    """Individual assertion result from a judge."""

    assertion: str
    passed: bool
    reasoning: str
    error: Optional[bool] = None  # True when judge errored (vs legitimate FAIL)


class ExploreToolCall(BaseModel):
    """Tool call made during exploration phase."""

    tool: str
    command: str
    output: str


class JudgePromptInfo(BaseModel):
    """Prompt information for a judge."""

    name: str
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None


class OutputArtifact(BaseModel):
    """Unified output artifact — base for all test deliverables."""

    filename: str
    path: str  # Relative to output/{test-id}/
    size_bytes: Optional[int] = None  # None for backward compat
    content_type: str = "application/octet-stream"
    content: Optional[str] = None  # inline text < 100KB, base64 images < 100KB
    source: Optional[str] = None  # "browser", "report", "output"
    download_url: Optional[str] = None  # optional external artifact URL


class BrowserArtifact(OutputArtifact):
    """Browser artifact (backward compat subclass)."""

    type: Literal["video", "screenshot"] = "screenshot"
    source: Optional[str] = "browser"


class ReportArtifact(OutputArtifact):
    """Report artifact (backward compat subclass)."""

    type: Literal["report"] = "report"
    source: Optional[str] = "report"
    content: Optional[str] = None  # Markdown content (stored in DB)


class ConversationToolCall(BaseModel):
    """A tool call within a conversation iteration."""

    tool: str
    input: Dict[str, Any] = Field(default_factory=dict)
    output: str


class ConversationIteration(BaseModel):
    """A single iteration in an executor or judge conversation."""

    iteration: int
    response_content: Optional[str] = None
    tool_calls: List[ConversationToolCall] = Field(default_factory=list)


class ExecutorConversationLog(BaseModel):
    """Conversation log for the executor phase."""

    iterations: List[ConversationIteration] = Field(default_factory=list)
    total_iterations: int = 0


class JudgeConversationLog(BaseModel):
    """Conversation log for a single judge."""

    judge_name: str
    iterations: List[ConversationIteration] = Field(default_factory=list)
    total_iterations: int = 0


# MIME inference for legacy artifacts (BrowserArtifact/ReportArtifact lack content_type)
_MIME_MAP: Dict[str, str] = {
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".py": "text/x-python",
    ".csv": "text/csv",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
}


def _infer_content_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return _MIME_MAP.get(ext, "application/octet-stream")


def _to_unified(artifact, source_hint: str) -> OutputArtifact:
    """Convert a legacy BrowserArtifact/ReportArtifact/dict to OutputArtifact."""
    if isinstance(artifact, OutputArtifact) and artifact.content_type != "application/octet-stream":
        if artifact.source is None:
            artifact.source = source_hint
        return artifact  # Already unified with real content_type
    # Extract fields from Pydantic model or dict
    fn = artifact.filename if hasattr(artifact, "filename") else artifact.get("filename", "")
    path = artifact.path if hasattr(artifact, "path") else artifact.get("path", "")
    size = artifact.size_bytes if hasattr(artifact, "size_bytes") else artifact.get("size_bytes")
    content = artifact.content if hasattr(artifact, "content") else artifact.get("content")
    ct = _infer_content_type(fn)
    # For reports, always use text/markdown
    if source_hint == "report":
        ct = "text/markdown"
    return OutputArtifact(
        filename=fn,
        path=path,
        size_bytes=size,
        content_type=ct,
        content=content,
        source=source_hint,
    )


class TestOutputResult(BaseModel):
    """Individual test result in spec format."""

    name: str
    status: Literal["passed", "failed", "error"]
    duration_ms: int
    files: List[str] = Field(default_factory=list)
    assertions: List[AssertionResult] = Field(default_factory=list)
    error: Optional[str] = None
    executor_output: Optional[str] = None
    explore_output: Optional[str] = None  # Natural language summary from explore phase
    explore_status: Optional[str] = None  # Explore phase pass/fail status
    explore_tool_calls: Optional[List[ExploreToolCall]] = None  # Tool calls during explore
    executor_model: Optional[str] = None  # Model used for executor
    judge_model: Optional[str] = None  # Model used for judges (first judge, backward compat)
    judge_models: Optional[Dict[str, str]] = None  # Per-judge model map {name: model_id}
    explore_model: Optional[str] = None  # Model used for explore phase
    # Token usage per phase
    executor_tokens: Optional[TokenUsage] = None
    judge_tokens: Optional[TokenUsage] = None
    explore_tokens: Optional[TokenUsage] = None
    # Executor and judge prompts for display in UI
    executor_system_prompt: Optional[str] = None
    executor_user_prompt: Optional[str] = None
    judge_prompts: Optional[List[JudgePromptInfo]] = None
    # Browser test artifacts (videos and screenshots)
    browser_artifacts: Optional[List[BrowserArtifact]] = None
    # Research report artifacts (markdown reports from verdict judge)
    report_artifacts: Optional[List[ReportArtifact]] = None
    # Output artifacts (files written by executor/judge to output folder)
    output_artifacts: Optional[List[OutputArtifact]] = None
    # Executor tool names for config visibility in results
    executor_tools: List[str] = Field(default_factory=list)
    # Conversation logs for executor and judges
    executor_conversation: Optional[ExecutorConversationLog] = None
    judge_conversations: Optional[List[JudgeConversationLog]] = None
    # Raw YAML content of the test scaffold file
    scaffold_yaml: Optional[str] = None

    @property
    def all_artifacts(self) -> List[OutputArtifact]:
        """Single merged view with dedup and MIME-typed OutputArtifact instances."""
        # New format detection: output_artifacts contain 'source' field
        if self.output_artifacts and any(
            (a.source if isinstance(a, OutputArtifact) else a.get("source"))
            for a in self.output_artifacts
        ):
            return list(self.output_artifacts)

        # Old/mixed format: merge all three, dedup by path, convert to OutputArtifact
        seen: set = set()
        merged: List[OutputArtifact] = []
        # Browser artifacts -> source='browser' with inferred MIME
        for a in self.browser_artifacts or []:
            path = a.path if hasattr(a, "path") else a.get("path", "")
            if path not in seen:
                seen.add(path)
                merged.append(_to_unified(a, "browser"))
        # Report artifacts -> source='report' with text/markdown
        for a in self.report_artifacts or []:
            path = a.path if hasattr(a, "path") else a.get("path", "")
            if path not in seen:
                seen.add(path)
                merged.append(_to_unified(a, "report"))
        # Output artifacts -> source='output' with stored or inferred content_type
        for a in self.output_artifacts or []:
            path = a.path if hasattr(a, "path") else a.get("path", "")
            if path not in seen:
                seen.add(path)
                merged.append(_to_unified(a, "output"))
        return merged


class ResultsSummary(BaseModel):
    """Summary statistics for test run."""

    total: int
    passed: int
    failed: int
    skipped: int = 0
    error: int = 0


class CoverageSection(BaseModel):
    """Coverage data embedded in results.json.

    Contains all files in the documentation scope (from dokumen.yaml coverage patterns),
    not just files covered by tests.
    """

    total_files: int
    covered_files: int
    percentage: float
    files: List["CoverageFile"]  # Forward reference, defined below


class ResultsJsonOutput(BaseModel):
    """Schema for .dokumen-cache/results.json per Phase 0 spec."""

    timestamp: str  # ISO 8601 format with Z suffix
    duration_ms: int
    tests: List[TestOutputResult]
    summary: ResultsSummary
    coverage: Optional[CoverageSection] = None  # All files in doc scope
    # Aggregate token usage across all tests
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0


# =============================================================================
# Coverage JSON Schema (coverage.json)
# =============================================================================


class CoverageFile(BaseModel):
    """Coverage data for a single file."""

    path: str
    covered: bool
    tests: List[str] = Field(default_factory=list)


class CoverageSummary(BaseModel):
    """Summary of coverage metrics."""

    total_files: int
    covered_files: int
    percentage: float


class CoverageJsonOutput(BaseModel):
    """Schema for .dokumen-cache/coverage.json per Phase 0 spec."""

    timestamp: str  # ISO 8601 format with Z suffix
    summary: CoverageSummary
    files: List[CoverageFile]


# =============================================================================
# Debug Trace Schema (debug/{test-name}-{timestamp}.json)
# =============================================================================


class DebugToolCall(BaseModel):
    """Record of a tool call during execution."""

    tool: str
    input: Dict[str, Any]
    output: str


class DebugMessage(BaseModel):
    """A message in the conversation trace."""

    role: str
    content: str
    tool_calls: Optional[List[DebugToolCall]] = None


class DebugExecutor(BaseModel):
    """Executor trace data."""

    system_prompt: str
    user_prompt: str
    messages: List[DebugMessage] = Field(default_factory=list)
    final_output: str


class DebugJudgeAssertion(BaseModel):
    """Judge assertion evaluation."""

    assertion: str
    evaluation: AssertionResult


class DebugJudge(BaseModel):
    """Judge trace data."""

    assertions: List[DebugJudgeAssertion] = Field(default_factory=list)


class DebugTraceOutput(BaseModel):
    """Schema for .dokumen-cache/debug/{test-name}-{timestamp}.json."""

    test_name: str
    started_at: str  # ISO 8601 format
    completed_at: str  # ISO 8601 format
    executor: DebugExecutor
    judge: DebugJudge
