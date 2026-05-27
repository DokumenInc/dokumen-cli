"""Tests for AskAgent core logic.

Tests the key methods of AskAgent including session management,
confidence extraction, source parsing, and context building.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from pathlib import Path
import tempfile
import yaml

from dokumen.ask_agent import AskAgent, AskResult, MatchedTest
from dokumen.explore_agent import ExploreResult, FileDiscovery


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    provider = MagicMock()
    provider.complete = AsyncMock(return_value={
        "content": "This is a test answer.\n\n**Sources:**\n- `docs/api.md`\n\n**Confidence:** High",
        "tool_use": [],
    })
    return provider


@pytest.fixture
def mock_explore_result():
    """Create a mock ExploreResult."""
    files = [
        FileDiscovery(path="docs/api.md", summary="API documentation", relevance=0.9),
        FileDiscovery(path="docs/auth.md", summary="Auth documentation", relevance=0.7),
    ]
    return ExploreResult(
        files=files,
        duration=1.5,
        tool_calls_count=3,
        success=True,
        summary="Found API and auth documentation",
        tool_history=[],
    )


@pytest.fixture
def ask_agent(mock_provider):
    """Create an AskAgent instance for testing."""
    return AskAgent(
        provider=mock_provider,
        base_dir=".",
        timeout=60.0,
        tests_dir="__skip__",  # Skip test exploration
    )


# =============================================================================
# Session Management Tests
# =============================================================================


class TestInitializeSession:
    """Tests for initialize_session() method."""

    @pytest.mark.asyncio
    async def test_initialize_session_stores_explore_result(self, ask_agent, mock_explore_result):
        """initialize_session stores the explore result for reuse."""
        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            result = await ask_agent.initialize_session(topic="API documentation")

            assert ask_agent._explore_result is not None
            assert ask_agent._session_initialized is True
            assert result.summary == "Found API and auth documentation"

    @pytest.mark.asyncio
    async def test_initialize_session_sets_flag(self, ask_agent, mock_explore_result):
        """initialize_session sets the session initialized flag."""
        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            assert ask_agent.is_session_initialized is False

            await ask_agent.initialize_session()

            assert ask_agent.is_session_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_session_clears_conversation_history(self, ask_agent, mock_explore_result):
        """initialize_session clears previous conversation history."""
        ask_agent._conversation_history = [{"role": "user", "content": "old message"}]

        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            await ask_agent.initialize_session()

            assert ask_agent._conversation_history == []

    @pytest.mark.asyncio
    async def test_initialize_session_with_progress_callback(self, ask_agent, mock_explore_result):
        """initialize_session calls progress callback at appropriate stages."""
        progress_events = []

        def on_progress(event, data):
            progress_events.append((event, data))

        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            await ask_agent.initialize_session(topic="API docs", on_progress=on_progress)

            event_names = [e[0] for e in progress_events]
            assert "explore_start" in event_names
            assert "explore_end" in event_names


class TestResetSession:
    """Tests for reset_session() method."""

    @pytest.mark.asyncio
    async def test_reset_session_clears_state(self, ask_agent, mock_explore_result):
        """reset_session clears all session state."""
        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            await ask_agent.initialize_session()
            assert ask_agent.is_session_initialized is True

            ask_agent.reset_session()

            assert ask_agent.is_session_initialized is False
            assert ask_agent._explore_result is None
            assert ask_agent._matched_tests is None
            assert ask_agent._conversation_history == []


class TestConversationHistory:
    """Tests for conversation history management."""

    @pytest.mark.asyncio
    async def test_ask_updates_conversation_history_in_session_mode(
        self, ask_agent, mock_provider, mock_explore_result
    ):
        """ask() updates conversation history when in session mode."""
        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            await ask_agent.initialize_session(topic="test")

            # First question
            result = await ask_agent.ask("What is the API?")

            assert len(ask_agent.conversation_history) == 2
            assert ask_agent.conversation_history[0]["role"] == "user"
            assert ask_agent.conversation_history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_ask_does_not_update_history_without_session(
        self, ask_agent, mock_provider, mock_explore_result
    ):
        """ask() does not update history when not in session mode."""
        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            # Not in session mode
            result = await ask_agent.ask("What is the API?")

            assert len(ask_agent.conversation_history) == 0


# =============================================================================
# Ask Method Tests
# =============================================================================


class TestAskMethod:
    """Tests for the ask() method."""

    @pytest.mark.asyncio
    async def test_ask_uses_session_explore_when_initialized(
        self, ask_agent, mock_provider, mock_explore_result
    ):
        """ask() reuses session explore result when initialized."""
        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            await ask_agent.initialize_session()

            # Reset mock to verify it's not called again
            mock_explore.reset_mock()

            await ask_agent.ask("What is the API?")

            # Explore should not be called again
            mock_explore.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_runs_fresh_explore_without_session(
        self, ask_agent, mock_provider, mock_explore_result
    ):
        """ask() runs fresh explore when not in session mode."""
        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            await ask_agent.ask("What is the API?")

            mock_explore.assert_called_once()

    @pytest.mark.asyncio
    async def test_ask_returns_result_with_sources(self, ask_agent, mock_provider, mock_explore_result):
        """ask() returns result with sources extracted."""
        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            result = await ask_agent.ask("What is the API?")

            assert result.success is True
            assert "docs/api.md" in result.sources

    @pytest.mark.asyncio
    async def test_ask_returns_error_on_exception(self, ask_agent, mock_provider, mock_explore_result):
        """ask() returns error result on exception."""
        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.side_effect = Exception("API connection failed")

            result = await ask_agent.ask("What is the API?")

            assert result.success is False
            assert "API connection failed" in result.error


# =============================================================================
# Explore Tests Method Tests
# =============================================================================


class TestExploreTests:
    """Tests for _explore_tests() method."""

    @pytest.mark.asyncio
    async def test_explore_tests_skips_when_skip_flag_set(self, mock_provider):
        """_explore_tests returns empty when tests_dir is __skip__."""
        agent = AskAgent(
            provider=mock_provider,
            tests_dir="__skip__",
        )

        matched = await agent._explore_tests("test question")

        assert matched == []

    @pytest.mark.asyncio
    async def test_explore_tests_returns_empty_when_dir_missing(self, mock_provider, tmp_path):
        """_explore_tests returns empty when tests directory doesn't exist."""
        agent = AskAgent(
            provider=mock_provider,
            tests_dir=str(tmp_path / "nonexistent"),
        )

        matched = await agent._explore_tests("test question")

        assert matched == []


# =============================================================================
# Build Context Tests
# =============================================================================


class TestBuildContext:
    """Tests for _build_context() method."""

    def test_build_context_includes_question(self, ask_agent):
        """_build_context includes the user's question."""
        explore_result = ExploreResult(
            files=[],
            duration=0.1,
            tool_calls_count=0,
            success=True,
            summary="summary",
        )

        context = ask_agent._build_context("What is the refund policy?", explore_result, [])

        assert "What is the refund policy?" in context
        assert "## User's Question" in context

    def test_build_context_includes_explore_summary(self, ask_agent):
        """_build_context includes explore summary."""
        explore_result = ExploreResult(
            files=[],
            duration=0.1,
            tool_calls_count=0,
            success=True,
            summary="Found refund policy in docs/policies/",
        )

        context = ask_agent._build_context("question", explore_result, [])

        assert "Found refund policy in docs/policies/" in context
        assert "## Discovered Documentation" in context

    def test_build_context_includes_matched_tests(self, ask_agent):
        """_build_context includes matched test information."""
        explore_result = ExploreResult(
            files=[],
            duration=0.1,
            tool_calls_count=0,
            success=True,
            summary="summary",
        )
        matched_tests = [
            MatchedTest(
                test_id="refund-test",
                test_name="refund-policy-test",
                reason="Validate refund calculations",
                relevance_score=0.85,
                success_criteria="Must correctly calculate refunds",
                files_covered=["docs/refund.md"],
                user_prompt="Test refund",
            )
        ]

        context = ask_agent._build_context("question", explore_result, matched_tests)

        assert "refund-policy-test" in context
        assert "85%" in context  # Relevance score formatted
        assert "Must correctly calculate refunds" in context
        assert "docs/refund.md" in context


# =============================================================================
# Extract Confidence Tests
# =============================================================================


class TestExtractConfidence:
    """Tests for _extract_confidence() method."""

    def test_extract_confidence_high(self, ask_agent):
        """_extract_confidence extracts 'High' confidence."""
        text = "Here is my answer.\n\n**Confidence:** High"
        assert ask_agent._extract_confidence(text) == "High"

    def test_extract_confidence_medium(self, ask_agent):
        """_extract_confidence extracts 'Medium' confidence."""
        text = "The answer is...\n\nConfidence: Medium"
        assert ask_agent._extract_confidence(text) == "Medium"

    def test_extract_confidence_low(self, ask_agent):
        """_extract_confidence extracts 'Low' confidence."""
        text = "I'm not sure.\n\n**Confidence:** Low"
        assert ask_agent._extract_confidence(text) == "Low"

    def test_extract_confidence_defaults_to_medium(self, ask_agent):
        """_extract_confidence defaults to Medium when not found."""
        text = "Here is my answer without confidence."
        assert ask_agent._extract_confidence(text) == "Medium"

    def test_extract_confidence_case_insensitive(self, ask_agent):
        """_extract_confidence is case insensitive."""
        text = "**Confidence:** HIGH"
        assert ask_agent._extract_confidence(text) == "High"


# =============================================================================
# Extract Sources Tests
# =============================================================================


class TestExtractSources:
    """Tests for _extract_sources() method."""

    def test_extract_sources_from_backticks(self, ask_agent):
        """_extract_sources extracts paths in backticks."""
        text = "See `docs/api.md` for details."
        sources = ask_agent._extract_sources(text)
        assert "docs/api.md" in sources

    def test_extract_sources_from_markdown_list(self, ask_agent):
        """_extract_sources extracts paths from markdown lists."""
        text = "**Sources:**\n- docs/policy.md\n- docs/terms.md"
        sources = ask_agent._extract_sources(text)
        assert "docs/policy.md" in sources
        assert "docs/terms.md" in sources

    def test_extract_sources_multiple_extensions(self, ask_agent):
        """_extract_sources handles various file extensions."""
        text = "`config.yaml` and `data.json` and `script.py`"
        sources = ask_agent._extract_sources(text)
        assert "config.yaml" in sources
        assert "data.json" in sources
        assert "script.py" in sources

    def test_extract_sources_deduplicates(self, ask_agent):
        """_extract_sources removes duplicate paths."""
        text = "See `docs/api.md` and then `docs/api.md` again."
        sources = ask_agent._extract_sources(text)
        assert sources.count("docs/api.md") == 1

    def test_extract_sources_empty_text(self, ask_agent):
        """_extract_sources returns empty list for text without sources."""
        text = "No file paths here."
        sources = ask_agent._extract_sources(text)
        assert sources == []


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in AskAgent."""

    @pytest.mark.asyncio
    async def test_ask_handles_provider_timeout(self, mock_provider, mock_explore_result):
        """ask() handles provider timeout gracefully."""
        import asyncio

        mock_provider.complete = AsyncMock(side_effect=asyncio.TimeoutError("Timeout"))

        agent = AskAgent(
            provider=mock_provider,
            tests_dir="__skip__",
        )

        with patch.object(agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            result = await agent.ask("question")

            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_ask_handles_malformed_response(self, mock_provider, mock_explore_result):
        """ask() handles malformed provider response."""
        mock_provider.complete = AsyncMock(return_value={})  # Missing content

        agent = AskAgent(
            provider=mock_provider,
            tests_dir="__skip__",
        )

        with patch.object(agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            result = await agent.ask("question")

            # Should still return a result (possibly empty)
            assert result is not None


# =============================================================================
# Extract Test Criteria Tests
# =============================================================================


class TestExtractTestCriteria:
    """Tests for _extract_test_criteria() method."""

    @pytest.mark.asyncio
    async def test_extract_test_criteria_valid_file(self, mock_provider, tmp_path):
        """_extract_test_criteria extracts from valid test file."""
        test_file = tmp_path / "test.test.yaml"
        test_content = {
            "name": "refund-test",
            "reason": "Validate refund policy",
            "files": [{"path": "docs/refund.md"}],
            "executor": {"user_prompt": "Check refund policy"},
            "judges": [{"system_prompt": "Verify refund calculations"}],
        }
        test_file.write_text(yaml.dump(test_content))

        agent = AskAgent(provider=mock_provider, tests_dir=str(tmp_path))

        criteria = await agent._extract_test_criteria("test.test.yaml")

        assert criteria is not None
        assert criteria["name"] == "refund-test"
        assert criteria["reason"] == "Validate refund policy"
        assert "docs/refund.md" in criteria["files_covered"]
        assert "Verify refund calculations" in criteria["success_criteria"]

    @pytest.mark.asyncio
    async def test_extract_test_criteria_missing_file(self, mock_provider, tmp_path):
        """_extract_test_criteria returns None for missing file."""
        agent = AskAgent(provider=mock_provider, tests_dir=str(tmp_path))

        criteria = await agent._extract_test_criteria("nonexistent.test.yaml")

        assert criteria is None

    @pytest.mark.asyncio
    async def test_extract_test_criteria_invalid_yaml(self, mock_provider, tmp_path):
        """_extract_test_criteria returns None for invalid YAML."""
        test_file = tmp_path / "invalid.test.yaml"
        test_file.write_text("{ invalid yaml [")

        agent = AskAgent(provider=mock_provider, tests_dir=str(tmp_path))

        criteria = await agent._extract_test_criteria("invalid.test.yaml")

        assert criteria is None


# =============================================================================
# AskResult Tests
# =============================================================================


class TestSessionHistoryEmptyAnswer:
    """Tests for conversation history with empty assistant answers."""

    @pytest.mark.asyncio
    async def test_session_history_with_empty_answer_uses_placeholder(
        self, ask_agent, mock_provider, mock_explore_result
    ):
        """When answer is empty (tool-only response), a placeholder is stored."""
        # Provider returns empty content (simulating tool-only response)
        mock_provider.complete = AsyncMock(return_value={
            "content": "",
            "tool_use": [],
        })

        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            await ask_agent.initialize_session(topic="test")

            result = await ask_agent.ask("Fix the docs")

            # Should have user + assistant in history
            assert len(ask_agent.conversation_history) == 2
            assert ask_agent.conversation_history[0]["role"] == "user"
            assert ask_agent.conversation_history[1]["role"] == "assistant"
            # Assistant content must NOT be empty
            assert ask_agent.conversation_history[1]["content"] != ""
            assert ask_agent.conversation_history[1]["content"] == "[Tool operations completed]"

    @pytest.mark.asyncio
    async def test_session_history_with_nonempty_answer_stores_answer(
        self, ask_agent, mock_provider, mock_explore_result
    ):
        """When answer is non-empty, the actual answer is stored."""
        with patch.object(ask_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            await ask_agent.initialize_session(topic="test")

            result = await ask_agent.ask("What is the API?")

            assert len(ask_agent.conversation_history) == 2
            assert ask_agent.conversation_history[1]["role"] == "assistant"
            # Should contain the actual answer, not the placeholder
            assert "test answer" in ask_agent.conversation_history[1]["content"].lower()


class TestConversationHistoryFiltering:
    """Tests for conversation history filtering in _run_agent_loop."""

    @pytest.mark.asyncio
    async def test_empty_content_in_history_gets_placeholder(self, ask_agent, mock_provider):
        """Empty assistant content in history is replaced with placeholder."""
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": ""},  # Empty - should get placeholder
        ]

        answer, sources, confidence = await ask_agent._run_agent_loop(
            system_prompt="You are helpful.",
            user_prompt="second question",
            conversation_history=history,
        )

        # Verify the provider was called with non-empty content for all messages
        call_args = mock_provider.complete.call_args
        messages = call_args[0][0]

        # Find assistant message in history (not the system or final user message)
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        for msg in assistant_msgs:
            assert msg["content"] != "", "Assistant message content must not be empty"

    @pytest.mark.asyncio
    async def test_nonempty_content_in_history_preserved(self, ask_agent, mock_provider):
        """Non-empty assistant content in history is preserved as-is."""
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "Here is the answer."},
        ]

        answer, sources, confidence = await ask_agent._run_agent_loop(
            system_prompt="You are helpful.",
            user_prompt="second question",
            conversation_history=history,
        )

        call_args = mock_provider.complete.call_args
        messages = call_args[0][0]

        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert any(m["content"] == "Here is the answer." for m in assistant_msgs)


class TestAskResult:
    """Tests for AskResult dataclass."""

    def test_ask_result_to_dict(self):
        """AskResult.to_dict() returns correct structure."""
        result = AskResult(
            success=True,
            answer="Test answer",
            sources=["docs/test.md"],
            confidence="High",
            matched_tests=[],
            explore_summary="Found docs",
            duration=1.5,
            tool_calls_count=3,
            error=None,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["answer"] == "Test answer"
        assert d["sources"] == ["docs/test.md"]
        assert d["confidence"] == "High"
        assert d["duration"] == 1.5
        assert d["tool_calls_count"] == 3


class TestMatchedTest:
    """Tests for MatchedTest dataclass."""

    def test_matched_test_to_dict(self):
        """MatchedTest.to_dict() returns correct structure."""
        test = MatchedTest(
            test_id="test-1",
            test_name="my-test",
            reason="Test reason",
            relevance_score=0.85,
            success_criteria="Must pass",
            files_covered=["docs/api.md"],
            user_prompt="Test prompt",
        )

        d = test.to_dict()

        assert d["test_id"] == "test-1"
        assert d["test_name"] == "my-test"
        assert d["relevance_score"] == 0.85
        assert d["files_covered"] == ["docs/api.md"]


# =============================================================================
# Agent Loop Follow-Up Tests
# =============================================================================


class TestAgentLoopFollowUp:
    """Tests for follow-up LLM call when agent loop produces empty answer after tools."""

    @pytest.mark.asyncio
    async def test_agent_loop_follows_up_when_answer_empty_after_tools(self, mock_provider):
        """When provider returns tool calls then empty content, agent makes follow-up call."""
        mock_provider.complete = AsyncMock(side_effect=[
            # Iteration 1: tool call returned
            {
                "content": "",
                "tool_use": [{"name": "read_file", "id": "tc_1", "input": {"file_path": "docs/api.md"}}],
            },
            # Iteration 2: no tool calls, empty content → answer = ""
            {
                "content": "",
                "tool_use": [],
            },
            # Follow-up call (without tools): returns summary text
            {
                "content": "Based on my research, the API uses OAuth authentication.",
            },
        ])

        agent = AskAgent(
            provider=mock_provider,
            tests_dir="__skip__",
        )

        answer, sources, confidence = await agent._run_agent_loop(
            system_prompt="You are helpful.",
            user_prompt="How does auth work?",
        )

        # Follow-up should produce a non-empty answer
        assert answer != ""
        assert "OAuth" in answer
        # Provider should be called 3 times (initial + after tool + follow-up)
        assert mock_provider.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_agent_loop_follow_up_emits_chunk_event(self, mock_provider):
        """Follow-up call emits chunk events via on_progress."""
        mock_provider.complete = AsyncMock(side_effect=[
            {
                "content": "",
                "tool_use": [{"name": "read_file", "id": "tc_1", "input": {"file_path": "docs/api.md"}}],
            },
            {
                "content": "",
                "tool_use": [],
            },
            {
                "content": "Summary of findings.",
            },
        ])

        agent = AskAgent(
            provider=mock_provider,
            tests_dir="__skip__",
        )

        progress_events = []

        def on_progress(event, data):
            progress_events.append((event, data))

        answer, _, _ = await agent._run_agent_loop(
            system_prompt="You are helpful.",
            user_prompt="question",
            on_progress=on_progress,
        )

        chunk_events = [e for e in progress_events if e[0] == "chunk"]
        assert len(chunk_events) > 0
        assert chunk_events[0][1]["content"] == "Summary of findings."

    @pytest.mark.asyncio
    async def test_agent_loop_no_follow_up_when_answer_present(self, mock_provider):
        """No follow-up when provider returns text content directly."""
        mock_provider.complete = AsyncMock(return_value={
            "content": "Direct answer without tools.",
            "tool_use": [],
        })

        agent = AskAgent(
            provider=mock_provider,
            tests_dir="__skip__",
        )

        answer, _, _ = await agent._run_agent_loop(
            system_prompt="You are helpful.",
            user_prompt="question",
        )

        assert answer == "Direct answer without tools."
        # Only 1 call, no follow-up needed
        assert mock_provider.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_agent_loop_follow_up_handles_provider_error(self, mock_provider):
        """Follow-up gracefully handles provider errors without crashing."""
        mock_provider.complete = AsyncMock(side_effect=[
            {
                "content": "",
                "tool_use": [{"name": "read_file", "id": "tc_1", "input": {"file_path": "docs/api.md"}}],
            },
            {
                "content": "",
                "tool_use": [],
            },
            Exception("API rate limit exceeded"),
        ])

        agent = AskAgent(
            provider=mock_provider,
            tests_dir="__skip__",
        )

        # Should not raise
        answer, _, _ = await agent._run_agent_loop(
            system_prompt="You are helpful.",
            user_prompt="question",
        )

        # Answer should be empty string (follow-up failed), but no crash
        assert isinstance(answer, str)


class TestSystemPromptFinalAnswer:
    """Tests for final-answer instruction in system prompt."""

    def test_system_prompt_includes_final_answer_instruction(self):
        """UNIFIED_SYSTEM_PROMPT includes instruction about always providing a text response."""
        from dokumen.ask_agent import UNIFIED_SYSTEM_PROMPT

        prompt_lower = UNIFIED_SYSTEM_PROMPT.lower()
        # Must contain an instruction about providing a final text answer after tool use
        assert "always provide" in prompt_lower or "must always" in prompt_lower, (
            "System prompt must instruct the model to always provide a text response"
        )
