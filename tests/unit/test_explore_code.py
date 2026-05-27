"""Tests for code repository exploration support in explore_agent.

Tests the explore_type parameter on ExploreAgent and explore decision changes.
Updated for SDK-based ExploreAgent (issue #604).
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from dokumen.sdk.query_runner import MockQueryRunner
from claude_agent_sdk import AssistantMessage, ResultMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_assistant(content: str) -> AssistantMessage:
    """Create an AssistantMessage with text content."""
    msg = MagicMock(spec=AssistantMessage)
    msg.content = content
    msg.__class__ = AssistantMessage
    return msg


def make_result(num_turns: int = 1, duration_ms: int = 500) -> ResultMessage:
    """Create a ResultMessage."""
    msg = MagicMock(spec=ResultMessage)
    msg.num_turns = num_turns
    msg.duration_ms = duration_ms
    msg.is_error = False
    msg.__class__ = ResultMessage
    return msg


# ---------------------------------------------------------------------------
# explore_type parameter tests
# ---------------------------------------------------------------------------

class TestExploreTypeParameter:
    """Test ExploreAgent accepts explore_type parameter."""

    def test_default_explore_type_is_docs(self):
        """ExploreAgent defaults to explore_type='docs'."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner)
        assert agent.explore_type == "docs"

    def test_accepts_explore_type_docs(self):
        """ExploreAgent accepts explore_type='docs' explicitly."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner, explore_type="docs")
        assert agent.explore_type == "docs"

    def test_accepts_explore_type_code(self):
        """ExploreAgent accepts explore_type='code'."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner, explore_type="code")
        assert agent.explore_type == "code"

    def test_accepts_explore_type_both(self):
        """ExploreAgent accepts explore_type='both'."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner, explore_type="both")
        assert agent.explore_type == "both"

    def test_invalid_explore_type_raises(self):
        """ExploreAgent rejects invalid explore_type values."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        with pytest.raises(ValueError, match="explore_type"):
            ExploreAgent(query_runner=runner, explore_type="invalid")

    def test_explore_type_logged_on_init(self):
        """ExploreAgent logs explore_type during initialization."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        with patch("dokumen.explore_agent.logger") as mock_logger:
            ExploreAgent(query_runner=runner, explore_type="code")
            # Should log at debug level with explore_type
            mock_logger.debug.assert_called()
            call_args = mock_logger.debug.call_args
            assert "explore_type" in str(call_args)


class TestExploreTypeSystemPrompt:
    """Test system prompt varies by explore_type."""

    def test_docs_explore_uses_default_system_prompt(self):
        """explore_type='docs' uses the standard EXPLORE_SYSTEM_PROMPT."""
        from dokumen.explore_agent import ExploreAgent, EXPLORE_SYSTEM_PROMPT

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner, explore_type="docs")
        prompt = agent._get_system_prompt()
        assert prompt == EXPLORE_SYSTEM_PROMPT

    def test_code_explore_uses_code_system_prompt(self):
        """explore_type='code' uses a system prompt mentioning code files."""
        from dokumen.explore_agent import ExploreAgent, EXPLORE_CODE_SYSTEM_PROMPT

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner, explore_type="code")
        prompt = agent._get_system_prompt()
        assert prompt == EXPLORE_CODE_SYSTEM_PROMPT

    def test_code_system_prompt_mentions_code(self):
        """Code system prompt should mention source code exploration."""
        from dokumen.explore_agent import EXPLORE_CODE_SYSTEM_PROMPT

        assert "code" in EXPLORE_CODE_SYSTEM_PROMPT.lower()
        assert "implementation" in EXPLORE_CODE_SYSTEM_PROMPT.lower()

    def test_both_explore_uses_combined_system_prompt(self):
        """explore_type='both' uses a combined system prompt."""
        from dokumen.explore_agent import ExploreAgent, EXPLORE_BOTH_SYSTEM_PROMPT

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner, explore_type="both")
        prompt = agent._get_system_prompt()
        assert prompt == EXPLORE_BOTH_SYSTEM_PROMPT

    def test_both_system_prompt_mentions_docs_and_code(self):
        """Combined system prompt should mention both documentation and code."""
        from dokumen.explore_agent import EXPLORE_BOTH_SYSTEM_PROMPT

        lower = EXPLORE_BOTH_SYSTEM_PROMPT.lower()
        assert "documentation" in lower
        assert "code" in lower


class TestExploreMethodUsesCorrectPrompt:
    """Test that the explore() method uses the prompt from _get_system_prompt()."""

    @pytest.mark.asyncio
    async def test_explore_passes_system_prompt_to_sdk(self):
        """explore() should use _get_system_prompt() in SDK options."""
        from dokumen.explore_agent import ExploreAgent, EXPLORE_CODE_SYSTEM_PROMPT

        runner = MockQueryRunner([
            make_assistant("Found relevant code files."),
            make_result(num_turns=1),
        ])

        agent = ExploreAgent(
            query_runner=runner,
            explore_type="code",
            timeout=5.0,
        )

        await agent.explore("find authentication implementation")

        # Verify SDK query was called with correct system prompt
        assert len(runner.calls) == 1
        options = runner.calls[0].options
        assert options.system_prompt == EXPLORE_CODE_SYSTEM_PROMPT


class TestExploreCodePathExtraction:
    """Test that code file extensions are extracted from responses."""

    def test_extracts_python_file_paths(self):
        """_parse_explore_response should extract .py file paths."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner, explore_type="code")

        content = "Found relevant files:\n- **src/auth/handler.py** - Authentication handler\n- **src/utils/helpers.py** - Helper functions"
        files, summary = agent._parse_explore_response(content)

        paths = [f.path for f in files]
        assert "src/auth/handler.py" in paths
        assert "src/utils/helpers.py" in paths

    def test_extracts_typescript_file_paths(self):
        """_parse_explore_response should extract .ts and .tsx file paths."""
        from dokumen.explore_agent import ExploreAgent

        runner = MockQueryRunner([])
        agent = ExploreAgent(query_runner=runner, explore_type="code")

        content = "Found:\n- `src/components/Auth.tsx` - Auth component\n- `src/lib/api.ts` - API client"
        files, summary = agent._parse_explore_response(content)

        paths = [f.path for f in files]
        assert "src/components/Auth.tsx" in paths
        assert "src/lib/api.ts" in paths


# NOTE: Backend chat explore tests (TestChatExploreDecision) are in
# backend/tests/unit/api/chat/test_explore_code.py because they import
# from backend.api.chat.explore which is not available in the CLI context.
