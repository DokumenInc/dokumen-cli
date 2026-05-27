"""Tests for explore_agent module — SDK-based ExploreAgent.

Tests cover the SDK-based ExploreAgent which uses MockQueryRunner
instead of the legacy Provider ABC.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage

from dokumen.sdk.query_runner import MockQueryRunner


# ---------------------------------------------------------------------------
# Helpers to build SDK messages for MockQueryRunner
# ---------------------------------------------------------------------------

def _make_text_block(text: str):
    """Create a TextBlock-like object with a .text attribute."""
    block = MagicMock()
    block.text = text
    return block


def make_assistant(content: str) -> AssistantMessage:
    """Create an AssistantMessage with realistic list[ContentBlock] content."""
    msg = MagicMock(spec=AssistantMessage)
    msg.content = [_make_text_block(content)]
    # Make isinstance check work
    msg.__class__ = AssistantMessage
    return msg


def make_result(
    num_turns: int = 1,
    duration_ms: int = 500,
    is_error: bool = False,
) -> ResultMessage:
    """Create a ResultMessage."""
    msg = MagicMock(spec=ResultMessage)
    msg.num_turns = num_turns
    msg.duration_ms = duration_ms
    msg.is_error = is_error
    msg.__class__ = ResultMessage
    return msg


# ---------------------------------------------------------------------------
# Type tests (shared dokumen_explore types, still valid)
# ---------------------------------------------------------------------------

class TestFileDiscovery:
    """Tests for FileDiscovery dataclass."""

    def test_creation(self):
        """FileDiscovery can be created."""
        from dokumen.explore_agent import FileDiscovery

        discovery = FileDiscovery(
            path="docs/api/auth.md",
            summary="Authentication documentation",
            relevance=0.9,
        )

        assert discovery.path == "docs/api/auth.md"
        assert discovery.summary == "Authentication documentation"
        assert discovery.relevance == 0.9

    def test_to_dict(self):
        """FileDiscovery serializes to dict."""
        from dokumen.explore_agent import FileDiscovery

        discovery = FileDiscovery(
            path="docs/policies/refund.md",
            summary="Refund policy document",
            relevance=0.85,
        )

        d = discovery.to_dict()

        assert d["path"] == "docs/policies/refund.md"
        assert d["summary"] == "Refund policy document"
        assert d["relevance"] == 0.85


class TestExploreResult:
    """Tests for ExploreResult dataclass."""

    def test_creation(self):
        """ExploreResult can be created."""
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        files = [
            FileDiscovery(path="docs/api.md", summary="API docs", relevance=0.9),
            FileDiscovery(path="docs/auth.md", summary="Auth docs", relevance=0.8),
        ]

        result = ExploreResult(
            files=files,
            duration=2.5,
            tool_calls_count=5,
            success=True,
        )

        assert len(result.files) == 2
        assert result.duration == 2.5
        assert result.tool_calls_count == 5
        assert result.success is True
        assert result.error is None

    def test_creation_with_error(self):
        """ExploreResult can include error."""
        from dokumen.explore_agent import ExploreResult

        result = ExploreResult(
            files=[],
            duration=1.0,
            tool_calls_count=1,
            success=False,
            error="Provider timeout",
        )

        assert result.success is False
        assert result.error == "Provider timeout"

    def test_to_dict(self):
        """ExploreResult serializes to dict."""
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        result = ExploreResult(
            files=[FileDiscovery(path="f.md", summary="s", relevance=0.5)],
            duration=1.0,
            tool_calls_count=2,
            success=True,
            summary="Found 1 file",
        )

        d = result.to_dict()

        assert d["success"] is True
        assert len(d["files"]) == 1
        assert d["summary"] == "Found 1 file"

    def test_to_context_block(self):
        """ExploreResult produces context block with summary."""
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        result = ExploreResult(
            files=[FileDiscovery(path="docs/api.md", summary="API docs", relevance=0.9)],
            duration=1.0,
            tool_calls_count=1,
            success=True,
            summary="Found the API documentation in docs/api.md",
        )

        block = result.to_context_block()
        assert "Pre-discovered Documentation" in block
        assert "docs/api.md" in block

    def test_to_context_block_empty(self):
        """ExploreResult with no files and no summary returns empty string."""
        from dokumen.explore_agent import ExploreResult

        result = ExploreResult(
            files=[],
            duration=1.0,
            tool_calls_count=0,
            success=True,
        )

        assert result.to_context_block() == ""


# ---------------------------------------------------------------------------
# ExploreAgent creation tests
# ---------------------------------------------------------------------------

class TestExploreAgent:
    """Tests for ExploreAgent with SDK integration."""

    def test_creation(self):
        """ExploreAgent can be created with a query runner."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner)

        assert agent._runner is runner
        assert agent.max_files == 20
        assert agent.max_turns == 50
        assert agent.timeout == 60.0
        assert agent.explore_type == "docs"

    def test_default_values(self):
        """ExploreAgent has sensible defaults."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner)

        assert agent.base_dir == "."
        assert agent.max_files == 20
        assert agent.max_turns == 50
        assert agent.timeout == 60.0
        assert agent.model is None

    def test_custom_model(self):
        """ExploreAgent accepts a model override."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner, model="claude-haiku-4-5-20251001")

        assert agent.model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_explore_returns_result(self):
        """ExploreAgent.explore() returns ExploreResult on success."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("Found relevant file: docs/api.md — API documentation"),
            make_result(num_turns=3, duration_ms=1000),
        ])

        agent = ExploreAgent(query_runner=runner, timeout=10.0)
        result = await agent.explore("Find API docs")

        assert result.success is True
        assert result.summary is not None
        assert len(result.files) >= 1
        assert any("api.md" in f.path for f in result.files)

    @pytest.mark.asyncio
    async def test_explore_calls_progress_callback(self):
        """ExploreAgent.explore() calls progress callbacks."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("docs/policy.md — found relevant doc"),
            make_result(num_turns=1),
        ])

        events = []
        def on_progress(event_type, data):
            events.append((event_type, data))

        agent = ExploreAgent(query_runner=runner, timeout=10.0)
        await agent.explore("Find policy docs", on_progress=on_progress)

        event_types = [e[0] for e in events]
        assert "start" in event_types
        assert "complete" in event_types

    @pytest.mark.asyncio
    async def test_explore_limits_file_count(self):
        """ExploreAgent.explore() limits results to max_files."""
        from dokumen.explore_agent import ExploreAgent

        # Response with many files
        many_files = "\n".join([f"docs/file{i}.md — file {i}" for i in range(30)])
        runner = MockQueryRunner([
            make_assistant(many_files),
            make_result(num_turns=1),
        ])

        agent = ExploreAgent(query_runner=runner, max_files=5, timeout=10.0)
        result = await agent.explore("Find docs")

        assert len(result.files) <= 5

    @pytest.mark.asyncio
    async def test_explore_handles_timeout(self):
        """ExploreAgent.explore() handles asyncio.TimeoutError gracefully."""
        from dokumen.explore_agent import ExploreAgent

        # Create a runner that hangs forever
        class HangingRunner:
            async def run(self, prompt, options):
                await asyncio.sleep(100)
                yield make_result()  # pragma: no cover (never reached)

        agent = ExploreAgent(query_runner=HangingRunner(), timeout=0.1)
        result = await agent.explore("Find stuff")

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_explore_handles_runner_error(self):
        """ExploreAgent.explore() handles errors from the query runner."""
        from dokumen.explore_agent import ExploreAgent

        # Create a runner that raises
        class ErrorRunner:
            async def run(self, prompt, options):
                raise RuntimeError("SDK query failed")
                yield  # pragma: no cover (makes it a generator)

        agent = ExploreAgent(query_runner=ErrorRunner(), timeout=10.0)
        result = await agent.explore("Find stuff")

        assert result.success is False
        assert "SDK query failed" in result.error

    @pytest.mark.asyncio
    async def test_explore_system_prompt_contains_instructions(self):
        """ExploreAgent uses appropriate system prompt."""
        from dokumen.explore_agent import ExploreAgent, EXPLORE_SYSTEM_PROMPT

        runner = MockQueryRunner([
            make_assistant("No relevant files found."),
            make_result(num_turns=1),
        ])

        agent = ExploreAgent(query_runner=runner, timeout=10.0)
        await agent.explore("Find docs")

        # Check that the runner was called with correct system prompt
        assert len(runner.calls) == 1
        options = runner.calls[0].options
        assert "documentation explorer" in options.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_explore_includes_goal_in_prompt(self):
        """ExploreAgent passes the goal in the user prompt."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("No files found."),
            make_result(num_turns=1),
        ])

        agent = ExploreAgent(query_runner=runner, timeout=10.0)
        await agent.explore("Find refund policy docs")

        assert len(runner.calls) == 1
        prompt = runner.calls[0].prompt
        assert "refund policy" in prompt.lower()

    @pytest.mark.asyncio
    async def test_explore_uses_read_only_tools(self):
        """ExploreAgent only allows read-only SDK tools (Read, Glob, Grep)."""
        from dokumen.explore_agent import ExploreAgent, EXPLORE_SDK_TOOLS

        runner = MockQueryRunner([
            make_assistant("Found docs/api.md"),
            make_result(num_turns=1),
        ])

        agent = ExploreAgent(query_runner=runner, timeout=10.0)
        await agent.explore("Find API docs")

        options = runner.calls[0].options
        assert set(options.allowed_tools) == set(EXPLORE_SDK_TOOLS)
        # Ensure no write tools
        assert "Write" not in options.allowed_tools
        assert "Bash" not in options.allowed_tools
        assert "Edit" not in options.allowed_tools

    @pytest.mark.asyncio
    async def test_explore_passes_model(self):
        """ExploreAgent passes model to SDK options."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("Found docs/api.md"),
            make_result(num_turns=1),
        ])

        agent = ExploreAgent(
            query_runner=runner,
            model="claude-haiku-4-5-20251001",
            timeout=10.0,
        )
        await agent.explore("Find docs")

        options = runner.calls[0].options
        assert options.model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_explore_passes_max_turns(self):
        """ExploreAgent passes max_turns to SDK options."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("Found docs/api.md"),
            make_result(num_turns=1),
        ])

        agent = ExploreAgent(
            query_runner=runner,
            max_turns=25,
            timeout=10.0,
        )
        await agent.explore("Find docs")

        options = runner.calls[0].options
        assert options.max_turns == 25


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

class TestExploreSystemPrompt:
    """Tests for system prompts."""

    def test_prompt_exists(self):
        """EXPLORE_SYSTEM_PROMPT is defined."""
        from dokumen.explore_agent import EXPLORE_SYSTEM_PROMPT
        assert len(EXPLORE_SYSTEM_PROMPT) > 100

    def test_prompt_includes_output_format(self):
        """System prompt has output format instructions."""
        from dokumen.explore_agent import EXPLORE_SYSTEM_PROMPT
        assert "Output Format" in EXPLORE_SYSTEM_PROMPT

    def test_prompt_includes_conciseness_rule(self):
        """System prompt has conciseness rule."""
        from dokumen.explore_agent import EXPLORE_SYSTEM_PROMPT
        assert "500 words" in EXPLORE_SYSTEM_PROMPT

    def test_code_prompt_exists(self):
        """EXPLORE_CODE_SYSTEM_PROMPT is defined."""
        from dokumen.explore_agent import EXPLORE_CODE_SYSTEM_PROMPT
        assert "code explorer" in EXPLORE_CODE_SYSTEM_PROMPT.lower()

    def test_both_prompt_exists(self):
        """EXPLORE_BOTH_SYSTEM_PROMPT is defined."""
        from dokumen.explore_agent import EXPLORE_BOTH_SYSTEM_PROMPT
        assert "documentation and code" in EXPLORE_BOTH_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Explore type tests
# ---------------------------------------------------------------------------

class TestExploreType:
    """Tests for explore_type handling."""

    def test_invalid_explore_type_raises(self):
        """ExploreAgent raises ValueError for invalid explore_type."""
        from dokumen.explore_agent import ExploreAgent

        with pytest.raises(ValueError, match="explore_type must be one of"):
            ExploreAgent(query_runner=MockQueryRunner([]), explore_type="invalid")

    @pytest.mark.asyncio
    async def test_code_explore_type_uses_code_prompt(self):
        """Code explore type uses code system prompt."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("Found src/main.py"),
            make_result(num_turns=1),
        ])

        agent = ExploreAgent(
            query_runner=runner,
            explore_type="code",
            timeout=10.0,
        )
        await agent.explore("Find main entry point")

        options = runner.calls[0].options
        assert "code explorer" in options.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_both_explore_type_uses_both_prompt(self):
        """Both explore type uses both system prompt."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("Found docs and code files"),
            make_result(num_turns=1),
        ])

        agent = ExploreAgent(
            query_runner=runner,
            explore_type="both",
            timeout=10.0,
        )
        await agent.explore("Find auth implementation")

        options = runner.calls[0].options
        assert "documentation and code" in options.system_prompt.lower()


# ---------------------------------------------------------------------------
# Response parsing tests (shared logic from dokumen_explore)
# ---------------------------------------------------------------------------

class TestParseExploreResponse:
    """Tests for _parse_explore_response (delegates to dokumen_explore)."""

    def test_extracts_pdf_file_paths(self):
        """Extracts file paths from response text."""
        from dokumen.explore_agent import ExploreAgent

        agent = ExploreAgent(query_runner=MockQueryRunner([]), timeout=10.0)
        content = "Found docs/api.md — the main API reference"
        files, summary = agent._parse_explore_response(content)

        assert any("api.md" in f.path for f in files)

    def test_extracts_mixed_file_types(self):
        """Extracts various file types."""
        from dokumen.explore_agent import ExploreAgent

        agent = ExploreAgent(query_runner=MockQueryRunner([]), timeout=10.0)
        content = "Found docs/policy.yaml and src/main.py and README.md"
        files, summary = agent._parse_explore_response(content)

        paths = [f.path for f in files]
        assert any("policy.yaml" in p for p in paths)
        assert any("main.py" in p for p in paths)

    def test_empty_content_returns_empty(self):
        """Empty content returns no files."""
        from dokumen.explore_agent import ExploreAgent

        agent = ExploreAgent(query_runner=MockQueryRunner([]), timeout=10.0)
        files, summary = agent._parse_explore_response("")

        assert files == []
        assert summary == ""


# ---------------------------------------------------------------------------
# None runner (no API key) guard
# ---------------------------------------------------------------------------

class TestExploreAgentNoneRunner:
    """Tests for ExploreAgent when _runner is None (safety net guard).

    Since __init__ now defaults to SDKQueryRunner(), _runner=None should
    not occur in production. These tests verify the guard by setting
    _runner=None directly after construction.
    """

    def test_default_runner_is_sdk_query_runner(self):
        """ExploreAgent defaults to SDKQueryRunner when no runner is passed."""
        from dokumen.explore_agent import ExploreAgent
        from dokumen.sdk.query_runner import SDKQueryRunner

        agent = ExploreAgent()
        assert isinstance(agent._runner, SDKQueryRunner)

    @pytest.mark.asyncio
    async def test_explore_with_none_runner_returns_error_result(self):
        """None runner returns error ExploreResult immediately."""
        from dokumen.explore_agent import ExploreAgent

        agent = ExploreAgent(query_runner=MockQueryRunner([]))
        agent._runner = None  # Simulate internal failure
        result = await agent.explore("Find docs")

        assert result.success is False
        assert result.error is not None
        assert "ANTHROPIC_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_explore_with_none_runner_includes_helpful_message(self):
        """None runner error includes helpful message."""
        from dokumen.explore_agent import ExploreAgent

        agent = ExploreAgent(query_runner=MockQueryRunner([]))
        agent._runner = None  # Simulate internal failure
        result = await agent.explore("Find docs")

        assert "no llm provider configured" in result.summary.lower()


# ---------------------------------------------------------------------------
# System prompt index reference tests
# ---------------------------------------------------------------------------

class TestExploreSystemPromptIndexReference:
    """Tests that system prompt references DOKUMEN_SUMMARIES_INDEX.md."""

    def test_prompt_mentions_summaries_index(self):
        """System prompt mentions DOKUMEN_SUMMARIES_INDEX.md."""
        from dokumen.explore_agent import EXPLORE_SYSTEM_PROMPT
        assert "DOKUMEN_SUMMARIES_INDEX.md" in EXPLORE_SYSTEM_PROMPT

    def test_prompt_instructs_reading_index_first(self):
        """System prompt instructs to check index first."""
        from dokumen.explore_agent import EXPLORE_SYSTEM_PROMPT
        assert "FIRST" in EXPLORE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# SDK tool configuration tests
# ---------------------------------------------------------------------------

class TestExploreSDKTools:
    """Tests for SDK tool configuration."""

    def test_explore_sdk_tools_are_read_only(self):
        """EXPLORE_SDK_TOOLS contains only read-only tools."""
        from dokumen.explore_agent import EXPLORE_SDK_TOOLS

        assert "Read" in EXPLORE_SDK_TOOLS
        assert "Glob" in EXPLORE_SDK_TOOLS
        assert "Grep" in EXPLORE_SDK_TOOLS
        assert "Write" not in EXPLORE_SDK_TOOLS
        assert "Bash" not in EXPLORE_SDK_TOOLS
        assert "Edit" not in EXPLORE_SDK_TOOLS

    def test_explore_sdk_tools_count(self):
        """EXPLORE_SDK_TOOLS has exactly 3 tools."""
        from dokumen.explore_agent import EXPLORE_SDK_TOOLS
        assert len(EXPLORE_SDK_TOOLS) == 3


# ---------------------------------------------------------------------------
# Multiple assistant messages test
# ---------------------------------------------------------------------------

class TestMultipleAssistantMessages:
    """Tests for handling multiple assistant messages (tool call + final)."""

    @pytest.mark.asyncio
    async def test_uses_last_assistant_message(self):
        """ExploreAgent uses the last assistant message content."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("Looking for files..."),  # intermediate
            make_assistant("Found docs/api.md — API docs\nFound docs/auth.md — Auth docs"),  # final
            make_result(num_turns=5, duration_ms=2000),
        ])

        agent = ExploreAgent(query_runner=runner, timeout=10.0)
        result = await agent.explore("Find API docs")

        assert result.success is True
        # The final response is used
        assert any("api.md" in f.path for f in result.files)

    @pytest.mark.asyncio
    async def test_empty_response_returns_success_with_no_files(self):
        """Empty assistant response returns success with no files."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant(""),
            make_result(num_turns=1),
        ])

        agent = ExploreAgent(query_runner=runner, timeout=10.0)
        result = await agent.explore("Find nonexistent docs")

        assert result.success is True
        assert len(result.files) == 0


# ---------------------------------------------------------------------------
# Model and duration tracking
# ---------------------------------------------------------------------------

class TestResultMetadata:
    """Tests for result metadata tracking."""

    @pytest.mark.asyncio
    async def test_result_includes_model(self):
        """ExploreResult includes the model used."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("Found docs/api.md"),
            make_result(num_turns=1),
        ])

        agent = ExploreAgent(
            query_runner=runner,
            model="claude-haiku-4-5-20251001",
            timeout=10.0,
        )
        result = await agent.explore("Find docs")

        assert result.model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_result_includes_duration(self):
        """ExploreResult includes duration."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("Found docs/api.md"),
            make_result(num_turns=1, duration_ms=500),
        ])

        agent = ExploreAgent(query_runner=runner, timeout=10.0)
        result = await agent.explore("Find docs")

        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_result_includes_tool_calls_count(self):
        """ExploreResult tracks tool calls (num_turns)."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("Found docs/api.md"),
            make_result(num_turns=7),
        ])

        agent = ExploreAgent(query_runner=runner, timeout=10.0)
        result = await agent.explore("Find docs")

        assert result.tool_calls_count == 7


# ---------------------------------------------------------------------------
# Default SDKQueryRunner creation (CRITICAL #1)
# ---------------------------------------------------------------------------

class TestExploreAgentDefaultRunner:
    """Tests that ExploreAgent defaults to SDKQueryRunner when none is provided."""

    def test_default_runner_is_sdk_query_runner(self):
        """ExploreAgent creates SDKQueryRunner when no runner is passed."""
        from dokumen.explore_agent import ExploreAgent
        from dokumen.sdk.query_runner import SDKQueryRunner

        agent = ExploreAgent()
        assert isinstance(agent._runner, SDKQueryRunner)

    def test_explicit_runner_overrides_default(self):
        """ExploreAgent uses the provided runner instead of SDKQueryRunner."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner)
        assert agent._runner is runner


# ---------------------------------------------------------------------------
# SDK is_error handling (CRITICAL #2)
# ---------------------------------------------------------------------------

class TestExploreAgentSDKError:
    """Tests that ExploreAgent handles is_error from SDK ResultMessage."""

    @pytest.mark.asyncio
    async def test_sdk_error_returns_failure_result(self):
        """SDK error (is_error=True) produces a failed ExploreResult."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("Found docs/api.md before error"),
            make_result(num_turns=2, is_error=True),
        ])

        agent = ExploreAgent(query_runner=runner, timeout=10.0)
        result = await agent.explore("Find docs")

        assert result.success is False
        assert "SDK query error" in result.error

    @pytest.mark.asyncio
    async def test_sdk_error_preserves_partial_files(self):
        """SDK error still extracts files from the response text."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([
            make_assistant("Found docs/api.md — API documentation"),
            make_result(num_turns=2, is_error=True),
        ])

        agent = ExploreAgent(query_runner=runner, timeout=10.0)
        result = await agent.explore("Find docs")

        assert result.success is False
        assert len(result.files) >= 1
        assert any("api.md" in f.path for f in result.files)


# ---------------------------------------------------------------------------
# Partial results on timeout (MAJOR #1)
# ---------------------------------------------------------------------------

class TestExploreAgentPartialResults:
    """Tests that partial explore results are preserved on timeout/error."""

    @pytest.mark.asyncio
    async def test_timeout_preserves_partial_files(self):
        """Timeout recovers files discovered before the timeout fired."""
        from dokumen.explore_agent import ExploreAgent

        class SlowRunnerWithPartialResults:
            """Runner that yields an assistant message then hangs."""
            async def run(self, prompt, options):
                # First yield a message with file paths
                yield make_assistant("Found docs/policy.md — Policy documentation")
                # Then hang to trigger timeout
                await asyncio.sleep(100)
                yield make_result()  # pragma: no cover

        agent = ExploreAgent(query_runner=SlowRunnerWithPartialResults(), timeout=0.2)
        result = await agent.explore("Find policy docs")

        assert result.success is False
        assert "timed out" in result.error.lower()
        # The key assertion: partial files are recovered
        assert len(result.files) >= 1
        assert any("policy.md" in f.path for f in result.files)

    @pytest.mark.asyncio
    async def test_error_preserves_partial_files(self):
        """Generic error recovers files discovered before the error."""
        from dokumen.explore_agent import ExploreAgent

        class PartialThenErrorRunner:
            """Runner that yields files then raises an error."""
            async def run(self, prompt, options):
                yield make_assistant("Found docs/setup.md — Setup guide")
                raise RuntimeError("Connection lost")

        agent = ExploreAgent(query_runner=PartialThenErrorRunner(), timeout=10.0)
        result = await agent.explore("Find setup docs")

        assert result.success is False
        assert "Connection lost" in result.error
        # Partial files should be recovered
        assert len(result.files) >= 1
        assert any("setup.md" in f.path for f in result.files)

    @pytest.mark.asyncio
    async def test_timeout_with_no_partial_results(self):
        """Timeout with no assistant messages yields empty files."""
        from dokumen.explore_agent import ExploreAgent

        class HangingRunner:
            async def run(self, prompt, options):
                await asyncio.sleep(100)
                yield make_result()  # pragma: no cover

        agent = ExploreAgent(query_runner=HangingRunner(), timeout=0.1)
        result = await agent.explore("Find stuff")

        assert result.success is False
        assert result.files == []

    @pytest.mark.asyncio
    async def test_last_assistant_text_reset_between_runs(self):
        """_last_assistant_text is reset between explore() calls."""
        from dokumen.explore_agent import ExploreAgent

        # First run: discovers files then hangs
        class FirstRunner:
            async def run(self, prompt, options):
                yield make_assistant("Found docs/first.md — First file")
                await asyncio.sleep(100)
                yield make_result()  # pragma: no cover

        agent = ExploreAgent(query_runner=FirstRunner(), timeout=0.2)
        result1 = await agent.explore("Find first")
        assert any("first.md" in f.path for f in result1.files)

        # Second run with a different runner that hangs immediately
        class EmptyHangRunner:
            async def run(self, prompt, options):
                await asyncio.sleep(100)
                yield make_result()  # pragma: no cover

        agent._runner = EmptyHangRunner()
        result2 = await agent.explore("Find second")
        # Should NOT contain files from the first run
        assert result2.files == []
