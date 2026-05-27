"""
Tests for dokumen ask command.

TDD tests for the ask command, including:
- Basic execution (ask questions, get answers)
- Test matching (find relevant tests, extract success criteria)
- Output formats (text, json)
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import List, Optional

import pytest
from click.testing import CliRunner


def extract_json_from_output(output: str) -> dict:
    """Extract JSON from CLI output, ignoring log lines."""
    # Find the first '{' which starts JSON
    start_idx = output.find('{')
    if start_idx == -1:
        return json.loads(output)

    # Try to parse from the start position
    decoder = json.JSONDecoder()
    try:
        result, end_idx = decoder.raw_decode(output[start_idx:])
        return result
    except json.JSONDecodeError:
        return json.loads(output)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def runner():
    """Click CLI test runner with separate stderr."""
    return CliRunner()


@pytest.fixture
def mock_explore_result():
    """Factory for creating mock ExploreResult objects."""
    def _make(
        files: List[dict] = None,
        summary: str = "Found relevant documentation.",
        success: bool = True,
        duration: float = 1.5
    ):
        result = MagicMock()
        result.files = files or [
            MagicMock(path="docs/policy.md", summary="Policy doc", relevance=0.9)
        ]
        result.summary = summary
        result.success = success
        result.duration = duration
        result.tool_calls_count = 3
        result.tool_history = []
        result.error = None
        return result
    return _make


@pytest.fixture
def mock_test_scaffold():
    """Factory for creating mock test scaffold objects."""
    def _make(
        test_id: str = "test-validation",
        name: str = "test-validation",
        reason: str = "Validate test documentation",
        files: List[str] = None,
        executor_user_prompt: str = "Validate the documentation...",
        judge_system_prompts: List[str] = None
    ):
        scaffold = MagicMock()
        scaffold.name = name
        scaffold.id = test_id
        scaffold.reason = reason
        scaffold.files = [MagicMock(path=f) for f in (files or ["docs/test.md"])]
        scaffold.executor = MagicMock()
        scaffold.executor.user_prompt = executor_user_prompt
        scaffold.judges = [
            MagicMock(system_prompt=prompt)
            for prompt in (judge_system_prompts or ["Must be accurate"])
        ]
        return scaffold
    return _make


@pytest.fixture
def project_with_docs(tmp_path: Path) -> Path:
    """Create a project with documentation and tests."""
    # Create docs
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "policy.md").write_text("# Policy\n\nThe margin requirement is 25%.")

    # Create config
    config = tmp_path / "dokumen.yaml"
    config.write_text("""
version: "1.0"
provider:
  name: anthropic
  model: claude-sonnet-4-5-20250929
""")

    # Create tests dir with scaffold
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "policy-test.test.yaml").write_text("""
name: policy-test
reason: Validate margin policy documentation
files:
  - path: docs/policy.md
executor:
  system_prompt: You are validating documentation.
  user_prompt: Check the margin policy document.
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: Must mention 25% margin requirement.
""")

    return tmp_path


# =============================================================================
# Ask Command Exists Tests
# =============================================================================


class TestAskCommandExists:
    """Tests that the ask command is properly registered."""

    def test_ask_command_registered(self, runner):
        """Ask command should be registered in CLI."""
        from dokumen.cli import cli

        result = runner.invoke(cli, ["ask", "--help"])

        assert result.exit_code == 0
        assert "Ask a question" in result.output or "question" in result.output.lower()

    def test_ask_without_question_enters_interactive_mode(self, runner):
        """Ask command without question should enter interactive mode."""
        from dokumen.cli import cli

        # When no question is provided, the command should enter interactive mode.
        # This is now valid behavior (not an error). We need to mock the interactive
        # session to avoid blocking.
        with patch('dokumen.cli.commands.ask._run_interactive_session') as mock_interactive:
            # Make the async function return immediately
            async def mock_session(*args, **kwargs):
                pass
            mock_interactive.return_value = mock_session()

            with patch('dokumen.cli.commands.ask.run_async') as mock_run_async:
                result = runner.invoke(cli, ["ask"])

                # The command should attempt to run interactive mode
                # (not fail with missing argument)
                # It may still fail due to mocking but should not be a "Missing argument" error
                combined = (result.output or "") + (result.stderr or "")
                assert "Missing argument" not in combined


# =============================================================================
# Basic Ask Execution Tests
# =============================================================================


class TestAskBasicExecution:
    """Tests for basic ask command execution."""

    def test_ask_returns_answer(self, runner, project_with_docs):
        """Ask command should return an answer to a question."""
        from dokumen.cli import cli

        # Patch at the source module where classes are defined
        with patch('dokumen.ask_agent.AskAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}), \
             patch('dokumen.tools_object.create_bash_tool', return_value=MagicMock()), \
             patch('dokumen.tools_object.create_grep_tool', return_value=MagicMock()):

            mock_provider.return_value = MagicMock()

            # Setup mock ask agent
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            # Create mock result
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "The margin requirement is 25%."
            mock_result.sources = ["docs/policy.md"]
            mock_result.confidence = "High"
            mock_result.matched_tests = []
            mock_result.explore_summary = "Found docs/policy.md"
            mock_result.duration = 2.5
            mock_result.tool_calls_count = 5
            mock_result.error = None

            mock_agent.ask = AsyncMock(return_value=mock_result)

            os.chdir(project_with_docs)
            result = runner.invoke(cli, ["ask", "What is the margin requirement?"])

            assert result.exit_code == 0
            assert "25%" in result.output or "margin" in result.output.lower()

    def test_ask_with_timeout_option(self, runner, project_with_docs):
        """Ask command should respect timeout option."""
        from dokumen.cli import cli

        with patch('dokumen.ask_agent.AskAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}), \
             patch('dokumen.tools_object.create_bash_tool', return_value=MagicMock()), \
             patch('dokumen.tools_object.create_grep_tool', return_value=MagicMock()):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "Answer here"
            mock_result.sources = []
            mock_result.confidence = "Medium"
            mock_result.matched_tests = []
            mock_result.explore_summary = None
            mock_result.duration = 1.0
            mock_result.tool_calls_count = 2
            mock_result.error = None

            mock_agent.ask = AsyncMock(return_value=mock_result)

            os.chdir(project_with_docs)
            result = runner.invoke(cli, ["ask", "Question?", "--timeout", "30"])

            assert result.exit_code == 0
            # Verify timeout was passed to agent constructor
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args
            assert call_kwargs.kwargs.get('timeout') == 30.0 or \
                   (call_kwargs.args and 30.0 in call_kwargs.args)


# =============================================================================
# Test Matching Tests
# =============================================================================


class TestAskTestMatching:
    """Tests for test matching functionality."""

    @pytest.mark.asyncio
    async def test_ask_uses_explore_for_tests(self):
        """AskAgent should use ExploreAgent to find relevant tests."""
        from dokumen.ask_agent import AskAgent, MatchedTest

        with patch('dokumen.ask_agent.ExploreAgent') as mock_explore_class:
            # Setup mock explore agent
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore

            # Mock explore result with test files
            mock_result = MagicMock()
            mock_result.files = [
                MagicMock(path="margin-test.test.yaml", relevance=0.85)
            ]
            mock_explore.explore = AsyncMock(return_value=mock_result)

            agent = AskAgent(
                provider=MagicMock(),
                base_dir=".",
                tests_dir="tests"
            )

            # Patch _extract_test_criteria to return test data
            with patch.object(agent, '_extract_test_criteria', new=AsyncMock(return_value={
                'name': 'margin-test',
                'reason': 'Validate margin documentation',
                'success_criteria': 'Must mention 25%',
                'files_covered': ['docs/margin.md'],
                'user_prompt': 'Check margin policy',
            })):
                matches = await agent._explore_tests("What are the margin requirements?")

            # Should have found the test
            assert len(matches) > 0
            assert any(m.test_name == "margin-test" for m in matches)

    def test_deprecated_match_tests_returns_empty(self):
        """Deprecated _match_tests should return empty list."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(
            provider=MagicMock(),
            base_dir=".",
            tests_dir="tests"
        )

        # The deprecated method should return empty list
        matches = agent._match_tests("question", [], set())
        assert matches == []

    def test_deprecated_calculate_relevance_returns_zero(self):
        """Deprecated _calculate_test_relevance should return 0."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(
            provider=MagicMock(),
            base_dir=".",
            tests_dir="tests"
        )

        # The deprecated method should return 0
        score = agent._calculate_test_relevance()
        assert score == 0.0


# =============================================================================
# Output Format Tests
# =============================================================================


class TestAskOutputFormats:
    """Tests for different output formats."""

    def test_ask_json_output(self, runner, project_with_docs):
        """Ask command should support JSON output format."""
        from dokumen.cli import cli

        with patch('dokumen.ask_agent.AskAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}), \
             patch('dokumen.tools_object.create_bash_tool', return_value=MagicMock()), \
             patch('dokumen.tools_object.create_grep_tool', return_value=MagicMock()):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "The margin is 25%"
            mock_result.sources = ["docs/policy.md"]
            mock_result.confidence = "High"
            mock_result.matched_tests = []
            mock_result.explore_summary = "Found docs"
            mock_result.duration = 2.0
            mock_result.tool_calls_count = 4
            mock_result.error = None
            mock_result.to_dict = MagicMock(return_value={
                "success": True,
                "answer": "The margin is 25%",
                "sources": ["docs/policy.md"],
                "confidence": "High"
            })

            mock_agent.ask = AsyncMock(return_value=mock_result)

            os.chdir(project_with_docs)
            result = runner.invoke(
                cli,
                ["ask", "What is margin?", "--output", "json"]
            )

            assert result.exit_code == 0
            # Output should be valid JSON (extract from output, ignoring log lines)
            output = extract_json_from_output(result.output)
            assert output["success"] == True
            assert "answer" in output

    def test_ask_text_output_includes_sources(self, runner, project_with_docs):
        """Text output should include sources and confidence."""
        from dokumen.cli import cli

        with patch('dokumen.ask_agent.AskAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}), \
             patch('dokumen.tools_object.create_bash_tool', return_value=MagicMock()), \
             patch('dokumen.tools_object.create_grep_tool', return_value=MagicMock()):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "The margin requirement is 25%."
            mock_result.sources = ["docs/policy.md"]
            mock_result.confidence = "High"
            mock_result.matched_tests = []
            mock_result.explore_summary = None
            mock_result.duration = 2.0
            mock_result.tool_calls_count = 4
            mock_result.error = None

            mock_agent.ask = AsyncMock(return_value=mock_result)

            os.chdir(project_with_docs)
            result = runner.invoke(cli, ["ask", "What is margin?"])

            assert result.exit_code == 0
            # Should include sources section
            assert "Sources" in result.output or "sources" in result.output.lower()
            # Should include confidence
            assert "High" in result.output or "confidence" in result.output.lower()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestAskErrorHandling:
    """Tests for error handling in ask command."""

    def test_ask_handles_api_error(self, runner, project_with_docs):
        """Ask command should handle API errors gracefully."""
        from dokumen.cli import cli

        with patch('dokumen.ask_agent.AskAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}), \
             patch('dokumen.tools_object.create_bash_tool', return_value=MagicMock()), \
             patch('dokumen.tools_object.create_grep_tool', return_value=MagicMock()):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            mock_result = MagicMock()
            mock_result.success = False
            mock_result.answer = ""
            mock_result.sources = []
            mock_result.confidence = "Low"
            mock_result.matched_tests = []
            mock_result.explore_summary = None
            mock_result.duration = 0.5
            mock_result.tool_calls_count = 0
            mock_result.error = "API rate limit exceeded"

            mock_agent.ask = AsyncMock(return_value=mock_result)

            os.chdir(project_with_docs)
            result = runner.invoke(cli, ["ask", "What is margin?"])

            # Should exit with non-zero code on error
            assert result.exit_code != 0
            # Error output may be in output or stderr
            combined = (result.output or "") + (result.stderr or "")
            assert "error" in combined.lower() or "failed" in combined.lower()


# =============================================================================
# AskAgent Dataclass Tests
# =============================================================================


class TestAskDataclasses:
    """Tests for AskAgent dataclasses."""

    def test_matched_test_dataclass(self):
        """MatchedTest dataclass should have all required fields."""
        from dokumen.ask_agent import MatchedTest

        matched = MatchedTest(
            test_id="margin-test",
            test_name="margin-test",
            reason="Validate margin docs",
            relevance_score=0.85,
            success_criteria="Must mention 25%",
            files_covered=["docs/margin.md"],
            user_prompt="Check margin policy"
        )

        assert matched.test_id == "margin-test"
        assert matched.relevance_score == 0.85
        assert "25%" in matched.success_criteria

    def test_ask_result_dataclass(self):
        """AskResult dataclass should have all required fields."""
        from dokumen.ask_agent import AskResult

        result = AskResult(
            success=True,
            answer="The margin is 25%",
            sources=["docs/margin.md"],
            confidence="High",
            matched_tests=[],
            explore_summary="Found docs",
            duration=2.5,
            tool_calls_count=5,
            error=None
        )

        assert result.success == True
        assert result.confidence == "High"
        assert result.duration == 2.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestAskIntegration:
    """Integration tests for ask command with explore phase."""

    def test_ask_runs_explore_phase(self, runner, project_with_docs):
        """Ask command should run explore phase before answering."""
        from dokumen.cli import cli

        with patch('dokumen.ask_agent.AskAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.explore_agent.ExploreAgent') as mock_explore_class, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}), \
             patch('dokumen.tools_object.create_bash_tool', return_value=MagicMock()), \
             patch('dokumen.tools_object.create_grep_tool', return_value=MagicMock()):

            mock_provider.return_value = MagicMock()

            # Setup explore agent mock
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore_result = MagicMock()
            mock_explore_result.success = True
            mock_explore_result.files = [MagicMock(path="docs/policy.md")]
            mock_explore_result.summary = "Found policy docs"
            mock_explore.explore = AsyncMock(return_value=mock_explore_result)

            # Setup ask agent mock
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "Answer here"
            mock_result.sources = ["docs/policy.md"]
            mock_result.confidence = "High"
            mock_result.matched_tests = []
            mock_result.explore_summary = "Found policy docs"
            mock_result.duration = 3.0
            mock_result.tool_calls_count = 6
            mock_result.error = None

            mock_agent.ask = AsyncMock(return_value=mock_result)

            os.chdir(project_with_docs)
            result = runner.invoke(cli, ["ask", "What is the policy?"])

            assert result.exit_code == 0
            # Verify ask was called (explore is integrated into ask)
            mock_agent.ask.assert_called_once()


# =============================================================================
# AskResult.to_dict() Tests
# =============================================================================


class TestAskResultToDict:
    """Tests for AskResult.to_dict() serialization."""

    def test_to_dict_basic(self):
        """to_dict should serialize basic fields."""
        from dokumen.ask_agent import AskResult

        result = AskResult(
            success=True,
            answer="The margin is 25%",
            sources=["docs/margin.md"],
            confidence="High",
            matched_tests=[],
            explore_summary="Found docs",
            duration=2.5,
            tool_calls_count=5,
            error=None
        )

        d = result.to_dict()

        assert d["success"] == True
        assert d["answer"] == "The margin is 25%"
        assert d["sources"] == ["docs/margin.md"]
        assert d["confidence"] == "High"
        assert d["explore_summary"] == "Found docs"
        assert d["duration"] == 2.5
        assert d["tool_calls_count"] == 5
        assert d["error"] is None

    def test_to_dict_with_matched_tests(self):
        """to_dict should serialize matched tests."""
        from dokumen.ask_agent import AskResult, MatchedTest

        matched = MatchedTest(
            test_id="margin-test",
            test_name="margin-test",
            reason="Validate margin docs",
            relevance_score=0.85,
            success_criteria="Must mention 25%",
            files_covered=["docs/margin.md"],
            user_prompt="Check margin"
        )

        result = AskResult(
            success=True,
            answer="Answer",
            sources=[],
            confidence="High",
            matched_tests=[matched],
            explore_summary=None,
            duration=1.0,
            tool_calls_count=1,
            error=None
        )

        d = result.to_dict()

        assert len(d["matched_tests"]) == 1
        assert d["matched_tests"][0]["test_id"] == "margin-test"
        assert d["matched_tests"][0]["relevance_score"] == 0.85


# =============================================================================
# Internal Method Tests
# =============================================================================


class TestAskAgentInternalMethods:
    """Tests for AskAgent internal helper methods."""

    def test_extract_words(self):
        """_extract_words should extract significant words."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(provider=MagicMock(), base_dir=".", tests_dir="tests")

        words = agent._extract_words("What are the margin requirements for trading?")

        assert "margin" in words
        assert "requirements" in words
        assert "trading" in words
        # Short words should be excluded
        assert "are" not in words
        assert "the" not in words
        assert "for" not in words

    def test_extract_words_empty_string(self):
        """_extract_words should handle empty string."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(provider=MagicMock(), base_dir=".", tests_dir="tests")

        words = agent._extract_words("")

        assert words == set()

    def test_extract_confidence_high(self):
        """_extract_confidence should detect high confidence."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(provider=MagicMock(), base_dir=".", tests_dir="tests")

        text = "Here is the answer.\n\n**Confidence:** High"
        assert agent._extract_confidence(text) == "High"

    def test_extract_confidence_medium(self):
        """_extract_confidence should detect medium confidence."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(provider=MagicMock(), base_dir=".", tests_dir="tests")

        text = "Here is the answer.\n\nConfidence: Medium"
        assert agent._extract_confidence(text) == "Medium"

    def test_extract_confidence_low(self):
        """_extract_confidence should detect low confidence."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(provider=MagicMock(), base_dir=".", tests_dir="tests")

        text = "Here is the answer.\n\nConfidence: Low"
        assert agent._extract_confidence(text) == "Low"

    def test_extract_confidence_default(self):
        """_extract_confidence should default to Medium."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(provider=MagicMock(), base_dir=".", tests_dir="tests")

        text = "Here is the answer with no confidence specified."
        assert agent._extract_confidence(text) == "Medium"

    def test_extract_sources(self):
        """_extract_sources should extract file paths from text."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(provider=MagicMock(), base_dir=".", tests_dir="tests")

        text = """
        Here is the answer.

        **Sources:**
        - `docs/margin.md`: Margin requirements
        - `docs/policy.md`: General policy
        """

        sources = agent._extract_sources(text)

        assert "docs/margin.md" in sources
        assert "docs/policy.md" in sources

    def test_extract_sources_pdf(self):
        """_extract_sources should extract PDF file paths from text."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(provider=MagicMock(), base_dir=".", tests_dir="tests")

        text = """
        Here is the answer based on the PDF.

        **Sources:**
        - `docs/guide.pdf`: User guide
        - `docs/api.md`: API documentation
        - `docs/manual.pdf`: Manual
        """

        sources = agent._extract_sources(text)

        assert "docs/guide.pdf" in sources
        assert "docs/api.md" in sources
        assert "docs/manual.pdf" in sources

    def test_build_context(self):
        """_build_context should build formatted context string."""
        from dokumen.ask_agent import AskAgent, MatchedTest

        agent = AskAgent(provider=MagicMock(), base_dir=".", tests_dir="tests")

        # Create mock explore result
        explore_result = MagicMock()
        explore_result.summary = "Found margin documentation"
        explore_result.files = []

        matched = MatchedTest(
            test_id="margin-test",
            test_name="margin-test",
            reason="Validate margin docs",
            relevance_score=0.85,
            success_criteria="Must mention 25%",
            files_covered=["docs/margin.md"],
            user_prompt="Check margin"
        )

        context = agent._build_context(
            question="What is the margin requirement?",
            explore_result=explore_result,
            matched_tests=[matched]
        )

        assert "What is the margin requirement?" in context
        assert "Found margin documentation" in context
        assert "margin-test" in context
        assert "85%" in context
        assert "Must mention 25%" in context

    def test_format_tools_for_provider(self):
        """_format_tools_for_provider should format tools correctly."""
        from dokumen.ask_agent import AskAgent
        from dokumen.tools_object import ToolDefinition

        mock_tool = MagicMock(spec=ToolDefinition)
        mock_tool.name = "read_file"
        mock_tool.description = "Read a file"
        mock_tool.parameters = {"type": "object"}

        agent = AskAgent(
            provider=MagicMock(),
            base_dir=".",
            tests_dir="tests",
            tools=[mock_tool]
        )

        formatted = agent._format_tools_for_provider()

        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "read_file"
        assert formatted[0]["function"]["description"] == "Read a file"

    def test_load_scaffolds_empty_dir(self):
        """_load_scaffolds should return empty list for non-existent dir."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(
            provider=MagicMock(),
            base_dir=".",
            tests_dir="__nonexistent__"
        )

        scaffolds = agent._load_scaffolds()
        assert scaffolds == []


# =============================================================================
# Async Method Tests
# =============================================================================


class TestAskAgentAsyncMethods:
    """Tests for AskAgent async methods."""

    @pytest.mark.asyncio
    async def test_run_explore(self):
        """_run_explore should use ExploreAgent."""
        from dokumen.ask_agent import AskAgent

        with patch('dokumen.ask_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.files = [MagicMock(path="docs/test.md")]
            mock_result.summary = "Found test docs"
            mock_explore.explore = AsyncMock(return_value=mock_result)

            agent = AskAgent(
                provider=MagicMock(),
                base_dir=".",
                tests_dir="tests",
                tools=[]
            )

            result = await agent._run_explore("What is the margin?")

            assert result.success
            mock_explore.explore.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """_execute_tool should execute tool and return result."""
        from dokumen.ask_agent import AskAgent

        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_result = MagicMock()
        mock_result.output = "File contents"
        mock_tool.handler = AsyncMock(return_value=mock_result)

        agent = AskAgent(
            provider=MagicMock(),
            base_dir=".",
            tests_dir="tests",
            tools=[mock_tool]
        )

        result = await agent._execute_tool("read_file", {"file_path": "test.md"})

        assert result == "File contents"

    @pytest.mark.asyncio
    async def test_execute_tool_unknown(self):
        """_execute_tool should handle unknown tool."""
        from dokumen.ask_agent import AskAgent

        agent = AskAgent(
            provider=MagicMock(),
            base_dir=".",
            tests_dir="tests",
            tools=[]
        )

        result = await agent._execute_tool("unknown_tool", {})

        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_execute_tool_error(self):
        """_execute_tool should handle tool errors."""
        from dokumen.ask_agent import AskAgent

        mock_tool = MagicMock()
        mock_tool.name = "failing_tool"
        mock_tool.handler = AsyncMock(side_effect=Exception("Tool failed"))

        agent = AskAgent(
            provider=MagicMock(),
            base_dir=".",
            tests_dir="tests",
            tools=[mock_tool]
        )

        result = await agent._execute_tool("failing_tool", {})

        assert "Error executing" in result

    @pytest.mark.asyncio
    async def test_execute_tool_returns_error_when_output_is_none(self):
        """_execute_tool should return error message when output is None but error exists."""
        from dokumen.ask_agent import AskAgent

        mock_tool = MagicMock()
        mock_tool.name = "create_test"
        mock_result = MagicMock()
        mock_result.output = None
        mock_result.error = "No provider configured. Set DOKUMEN_API_KEY."
        mock_tool.handler = AsyncMock(return_value=mock_result)

        agent = AskAgent(
            provider=MagicMock(),
            base_dir=".",
            tests_dir="tests",
            tools=[mock_tool]
        )

        result = await agent._execute_tool("create_test", {"goal": "Test goal"})

        assert "Error:" in result
        assert "No provider configured" in result

    @pytest.mark.asyncio
    async def test_execute_tool_returns_message_when_no_output_no_error(self):
        """_execute_tool should return helpful message when output and error are both None."""
        from dokumen.ask_agent import AskAgent

        mock_tool = MagicMock()
        mock_tool.name = "empty_tool"
        mock_result = MagicMock()
        mock_result.output = None
        mock_result.error = None
        mock_tool.handler = AsyncMock(return_value=mock_result)

        agent = AskAgent(
            provider=MagicMock(),
            base_dir=".",
            tests_dir="tests",
            tools=[mock_tool]
        )

        result = await agent._execute_tool("empty_tool", {})

        assert "Tool completed but returned no output" in result

    @pytest.mark.asyncio
    async def test_run_agent_loop_no_tools(self):
        """_run_agent_loop should handle response without tool calls."""
        from dokumen.ask_agent import AskAgent

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value={
            "content": "The margin is 25%.\n\n**Confidence:** High",
            "tool_calls": []
        })

        agent = AskAgent(
            provider=mock_provider,
            base_dir=".",
            tests_dir="tests",
            tools=[]
        )

        answer, sources, confidence = await agent._run_agent_loop(
            system_prompt="System",
            user_prompt="Question"
        )

        assert "25%" in answer
        assert confidence == "High"

    @pytest.mark.asyncio
    async def test_run_agent_loop_with_tool_calls(self):
        """_run_agent_loop should handle tool calls."""
        from dokumen.ask_agent import AskAgent

        mock_provider = MagicMock()
        # First call returns tool call, second returns answer
        mock_provider.complete = AsyncMock(side_effect=[
            {
                "content": "Let me check that file.",
                "tool_calls": [
                    {"id": "1", "name": "read_file", "arguments": {"file_path": "docs/margin.md"}}
                ]
            },
            {
                "content": "The margin is 25%.\n\n**Confidence:** High",
                "tool_calls": []
            }
        ])

        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_result = MagicMock()
        mock_result.output = "Margin: 25%"
        mock_tool.handler = AsyncMock(return_value=mock_result)

        agent = AskAgent(
            provider=mock_provider,
            base_dir=".",
            tests_dir="tests",
            tools=[mock_tool]
        )

        answer, sources, confidence = await agent._run_agent_loop(
            system_prompt="System",
            user_prompt="Question"
        )

        assert "25%" in answer
        assert "docs/margin.md" in sources
        assert agent._tool_calls_count == 1

    @pytest.mark.asyncio
    async def test_ask_with_exception(self):
        """ask should handle exceptions gracefully."""
        from dokumen.ask_agent import AskAgent

        with patch('dokumen.ask_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(side_effect=Exception("API error"))

            agent = AskAgent(
                provider=MagicMock(),
                base_dir=".",
                tests_dir="tests",
                tools=[]
            )

            result = await agent.ask("What is the margin?")

            assert result.success == False
            assert "API error" in result.error

    @pytest.mark.asyncio
    async def test_ask_success_flow(self):
        """ask should complete successfully with all steps."""
        from dokumen.ask_agent import AskAgent

        with patch('dokumen.ask_agent.ExploreAgent') as mock_explore_class:
            # Mock explore
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore_result = MagicMock()
            mock_explore_result.success = True
            mock_explore_result.files = [MagicMock(path="docs/margin.md")]
            mock_explore_result.summary = "Found margin docs"
            mock_explore.explore = AsyncMock(return_value=mock_explore_result)

            # Mock provider
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value={
                "content": "The margin is 25%.\n\n**Confidence:** High",
                "tool_calls": []
            })

            agent = AskAgent(
                provider=mock_provider,
                base_dir=".",
                tests_dir="__skip__",  # Skip test loading
                tools=[]
            )

            result = await agent.ask("What is the margin?")

            assert result.success == True
            assert "25%" in result.answer
            assert result.confidence == "High"

    @pytest.mark.asyncio
    async def test_ask_with_progress_callback(self):
        """ask should call progress callback at each step."""
        from dokumen.ask_agent import AskAgent

        with patch('dokumen.ask_agent.ExploreAgent') as mock_explore_class:
            # Mock explore
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore_result = MagicMock()
            mock_explore_result.success = True
            mock_explore_result.files = [MagicMock(path="docs/test.md")]
            mock_explore_result.summary = "Found docs"
            mock_explore.explore = AsyncMock(return_value=mock_explore_result)

            # Mock provider
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value={
                "content": "Answer here.\n\n**Confidence:** High",
                "tool_calls": []
            })

            progress_calls = []

            def on_progress(event, data):
                progress_calls.append((event, data))

            agent = AskAgent(
                provider=mock_provider,
                base_dir=".",
                tests_dir="__skip__",
                tools=[]
            )

            await agent.ask("What is the policy?", on_progress=on_progress)

            # Check that progress was reported
            events = [e[0] for e in progress_calls]
            assert "explore_start" in events
            assert "explore_end" in events
            assert "match_tests_start" in events
