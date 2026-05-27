"""Tests for cache token tracking across the CLI pipeline.

Verifies that cache_creation_input_tokens and cache_read_input_tokens
from Anthropic API responses are properly extracted, accumulated, and
propagated through the output schemas.
"""
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from dokumen.output_schemas import TokenUsage, ResultsJsonOutput, ResultsSummary, TestOutputResult
from dokumen.providers.anthropic import AnthropicProvider
from dokumen.agent_object import AgentType, ExecutorOutput, JudgeResult


# =============================================================================
# TokenUsage schema tests
# =============================================================================

class TestTokenUsageCacheFields:
    """TokenUsage model must accept and default cache token fields."""

    def test_cache_creation_tokens_defaults_to_zero(self):
        """cache_creation_tokens defaults to 0."""
        usage = TokenUsage()
        assert usage.cache_creation_tokens == 0

    def test_cache_read_tokens_defaults_to_zero(self):
        """cache_read_tokens defaults to 0."""
        usage = TokenUsage()
        assert usage.cache_read_tokens == 0

    def test_cache_tokens_accepted_in_constructor(self):
        """TokenUsage accepts cache token values."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_tokens=200,
            cache_read_tokens=150,
        )
        assert usage.cache_creation_tokens == 200
        assert usage.cache_read_tokens == 150

    def test_cache_tokens_in_serialization(self):
        """Cache tokens appear in model_dump() output."""
        usage = TokenUsage(
            input_tokens=10,
            output_tokens=20,
            cache_creation_tokens=30,
            cache_read_tokens=40,
        )
        data = usage.model_dump()
        assert data["cache_creation_tokens"] == 30
        assert data["cache_read_tokens"] == 40

    def test_backward_compat_without_cache_tokens(self):
        """TokenUsage without cache tokens still works (backward compat)."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_creation_tokens == 0
        assert usage.cache_read_tokens == 0


# =============================================================================
# ResultsJsonOutput cache token aggregate tests
# =============================================================================

class TestResultsJsonOutputCacheTokens:
    """ResultsJsonOutput must include aggregate cache token fields."""

    def test_total_cache_creation_tokens_default(self):
        """total_cache_creation_tokens defaults to 0."""
        output = ResultsJsonOutput(
            timestamp="2026-03-13T00:00:00Z",
            duration_ms=1000,
            tests=[],
            summary=ResultsSummary(total=0, passed=0, failed=0),
        )
        assert output.total_cache_creation_tokens == 0

    def test_total_cache_read_tokens_default(self):
        """total_cache_read_tokens defaults to 0."""
        output = ResultsJsonOutput(
            timestamp="2026-03-13T00:00:00Z",
            duration_ms=1000,
            tests=[],
            summary=ResultsSummary(total=0, passed=0, failed=0),
        )
        assert output.total_cache_read_tokens == 0

    def test_total_cache_tokens_accepted(self):
        """ResultsJsonOutput accepts total cache token values."""
        output = ResultsJsonOutput(
            timestamp="2026-03-13T00:00:00Z",
            duration_ms=1000,
            tests=[],
            summary=ResultsSummary(total=0, passed=0, failed=0),
            total_cache_creation_tokens=500,
            total_cache_read_tokens=300,
        )
        assert output.total_cache_creation_tokens == 500
        assert output.total_cache_read_tokens == 300


# =============================================================================
# AnthropicProvider._normalize_response() cache token extraction tests
# =============================================================================

class TestNormalizeResponseCacheTokens:
    """_normalize_response() must extract cache tokens from API response."""

    def _make_response(self, cache_creation=0, cache_read=0, input_tokens=100, output_tokens=50):
        """Helper to create a mock Anthropic response with cache tokens."""
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello"
        mock_response.content = [mock_text_block]

        # Set up usage with cache token attributes
        mock_response.usage.input_tokens = input_tokens
        mock_response.usage.output_tokens = output_tokens
        mock_response.usage.cache_creation_input_tokens = cache_creation
        mock_response.usage.cache_read_input_tokens = cache_read

        return mock_response

    def test_extracts_cache_creation_tokens(self):
        """Extracts cache_creation_input_tokens from response."""
        provider = AnthropicProvider(api_key="sk-test")
        response = self._make_response(cache_creation=250)

        result = provider._normalize_response(response)

        assert result["usage"]["cache_creation_tokens"] == 250

    def test_extracts_cache_read_tokens(self):
        """Extracts cache_read_input_tokens from response."""
        provider = AnthropicProvider(api_key="sk-test")
        response = self._make_response(cache_read=180)

        result = provider._normalize_response(response)

        assert result["usage"]["cache_read_tokens"] == 180

    def test_cache_tokens_default_to_zero(self):
        """When response has no cache token attrs, defaults to 0."""
        provider = AnthropicProvider(api_key="sk-test")
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello"
        mock_response.content = [mock_text_block]

        # Usage without cache token attributes (simulate missing attrs)
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        # Delete cache attrs so getattr falls back to default
        del mock_response.usage.cache_creation_input_tokens
        del mock_response.usage.cache_read_input_tokens

        result = provider._normalize_response(mock_response)

        assert result["usage"]["cache_creation_tokens"] == 0
        assert result["usage"]["cache_read_tokens"] == 0

    def test_all_usage_fields_present(self):
        """All four usage fields are present in normalized response."""
        provider = AnthropicProvider(api_key="sk-test")
        response = self._make_response(
            input_tokens=100, output_tokens=50,
            cache_creation=200, cache_read=150,
        )

        result = provider._normalize_response(response)

        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50
        assert result["usage"]["cache_creation_tokens"] == 200
        assert result["usage"]["cache_read_tokens"] == 150


# =============================================================================
# ExecutorOutput / JudgeResult cache token field tests
# =============================================================================

class TestExecutorOutputCacheTokenFields:
    """ExecutorOutput dataclass must have cache token fields."""

    def test_cache_creation_tokens_defaults_to_zero(self):
        """ExecutorOutput.cache_creation_tokens defaults to 0."""
        output = ExecutorOutput(
            tool_calls=[], final_response="test", success=True,
        )
        assert output.cache_creation_tokens == 0

    def test_cache_read_tokens_defaults_to_zero(self):
        """ExecutorOutput.cache_read_tokens defaults to 0."""
        output = ExecutorOutput(
            tool_calls=[], final_response="test", success=True,
        )
        assert output.cache_read_tokens == 0

    def test_cache_tokens_in_to_dict(self):
        """Cache tokens appear in ExecutorOutput.to_dict()."""
        output = ExecutorOutput(
            tool_calls=[], final_response="test", success=True,
            cache_creation_tokens=100, cache_read_tokens=50,
        )
        d = output.to_dict()
        assert d["cache_creation_tokens"] == 100
        assert d["cache_read_tokens"] == 50


class TestJudgeResultCacheTokenFields:
    """JudgeResult dataclass must have cache token fields."""

    def test_cache_creation_tokens_defaults_to_zero(self):
        """JudgeResult.cache_creation_tokens defaults to 0."""
        result = JudgeResult(judge_id="test", passed=True)
        assert result.cache_creation_tokens == 0

    def test_cache_read_tokens_defaults_to_zero(self):
        """JudgeResult.cache_read_tokens defaults to 0."""
        result = JudgeResult(judge_id="test", passed=True)
        assert result.cache_read_tokens == 0

    def test_cache_tokens_in_to_dict(self):
        """Cache tokens appear in JudgeResult.to_dict()."""
        result = JudgeResult(
            judge_id="test", passed=True,
            cache_creation_tokens=200, cache_read_tokens=150,
        )
        d = result.to_dict()
        assert d["cache_creation_tokens"] == 200
        assert d["cache_read_tokens"] == 150


# =============================================================================
# Agent accumulation tests (executor loop)
# =============================================================================

# =============================================================================
# Explore agent cache token tests
# =============================================================================

class TestExploreCacheTokenAccumulation:
    """Explore agent must track cache tokens."""

    @pytest.mark.asyncio
    async def test_explore_loop_accumulates_cache_tokens(self):
        """_run_exploration_loop accumulates cache tokens from responses."""
        from dokumen.explore_agent import ExploreAgent

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value={
            "content": "Found relevant file: docs/api.md - API documentation",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_tokens": 300,
                "cache_read_tokens": 120,
            },
        })

        agent = ExploreAgent(
            provider=mock_provider,
            base_dir=".",
            max_files=10,
            max_iterations=5,
            timeout=30.0,
        )

        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ]

        files, input_tokens, output_tokens, cache_creation, cache_read = (
            await agent._run_exploration_loop(messages, [], None, 5)
        )

        assert cache_creation == 300
        assert cache_read == 120
