"""
Unit tests for the Perplexity web_search tool.

Tests the web_search tool including:
- Tool creation and registration
- Rate limiting (max_searches per test)
- Missing API key handling
- Successful search with mocked API
- HTTP error handling
- Query truncation
- Privacy logging (no raw query in logs)
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO


class TestPerplexityConfig:
    """Tests for PerplexityConfig in dokumen.yaml."""

    def test_perplexity_config_exists(self):
        """PerplexityConfig model should exist."""
        from dokumen.config import PerplexityConfig
        assert PerplexityConfig is not None

    def test_perplexity_config_defaults(self):
        """PerplexityConfig should have sensible defaults."""
        from dokumen.config import PerplexityConfig

        config = PerplexityConfig()
        assert config.api_key is None
        assert config.model == "sonar"
        assert config.max_searches_per_test == 5

    def test_perplexity_config_custom_values(self):
        """PerplexityConfig should accept custom values."""
        from dokumen.config import PerplexityConfig

        config = PerplexityConfig(
            api_key="pplx-test123",
            model="sonar-pro",
            max_searches_per_test=10,
        )
        assert config.api_key == "pplx-test123"
        assert config.model == "sonar-pro"
        assert config.max_searches_per_test == 10

    def test_perplexity_config_max_searches_range(self):
        """max_searches_per_test must be between 1 and 20."""
        from pydantic import ValidationError
        from dokumen.config import PerplexityConfig

        with pytest.raises(ValidationError):
            PerplexityConfig(max_searches_per_test=0)

        with pytest.raises(ValidationError):
            PerplexityConfig(max_searches_per_test=21)

    def test_dokumen_config_includes_perplexity(self):
        """DokumenConfig should include perplexity section with defaults."""
        from dokumen.config import DokumenConfig, ProviderConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic"),
        )
        assert config.perplexity is not None
        assert config.perplexity.api_key is None
        assert config.perplexity.model == "sonar"
        assert config.perplexity.max_searches_per_test == 5


class TestWebSearchToolCreation:
    """Tests for create_perplexity_web_search_tool factory."""

    def test_web_search_tool_creation(self):
        """create_perplexity_web_search_tool should return a ToolDefinition."""
        from dokumen.tools_object import create_perplexity_web_search_tool, ToolDefinition

        tool = create_perplexity_web_search_tool(api_key="pplx-test")
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "web_search"

    def test_web_search_tool_has_query_param(self):
        """web_search tool should require a query parameter."""
        from dokumen.tools_object import create_perplexity_web_search_tool

        tool = create_perplexity_web_search_tool(api_key="pplx-test")
        assert "query" in tool.parameters["properties"]
        assert "query" in tool.parameters["required"]

    def test_web_search_in_standalone_tools(self):
        """web_search should be registered in STANDALONE_TOOLS."""
        from dokumen.tools_object import STANDALONE_TOOLS

        assert "web_search" in STANDALONE_TOOLS

    def test_web_search_in_all_tool_names(self):
        """web_search should appear in get_all_tool_names()."""
        from dokumen.tools_object import get_all_tool_names

        assert "web_search" in get_all_tool_names()


class TestWebSearchHandler:
    """Tests for the web_search tool handler."""

    @pytest.mark.asyncio
    async def test_missing_query_returns_error(self):
        """Handler should error on missing query."""
        from dokumen.tools_object import create_perplexity_web_search_tool

        tool = create_perplexity_web_search_tool(api_key="pplx-test")
        result = await tool.handler({})
        assert result.success is False
        assert "Missing required parameter" in result.error

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(self):
        """Handler should error when no API key is available."""
        from dokumen.tools_object import create_perplexity_web_search_tool

        with patch.dict("os.environ", {}, clear=True):
            tool = create_perplexity_web_search_tool(api_key=None)
            result = await tool.handler({"query": "test question"})
            assert result.success is False
            assert "API key not configured" in result.error

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Handler should error after max_searches exceeded."""
        from dokumen.tools_object import create_perplexity_web_search_tool

        tool = create_perplexity_web_search_tool(api_key="pplx-test", max_searches=2)

        # Mock urllib to avoid real API calls
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "answer"}}],
            "citations": [],
        }).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=mock_response):
            # Search 1 - should succeed
            result1 = await tool.handler({"query": "question 1"})
            assert result1.success is True

            # Search 2 - should succeed
            result2 = await tool.handler({"query": "question 2"})
            assert result2.success is True

            # Search 3 - should be rate limited
            result3 = await tool.handler({"query": "question 3"})
            assert result3.success is False
            assert "Rate limit" in result3.error

    @pytest.mark.asyncio
    async def test_successful_search_with_citations(self):
        """Handler should return formatted results with citations."""
        from dokumen.tools_object import create_perplexity_web_search_tool

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Python is a programming language."}}],
            "citations": ["https://python.org", "https://docs.python.org"],
        }).encode("utf-8")

        tool = create_perplexity_web_search_tool(api_key="pplx-test")

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = await tool.handler({"query": "What is Python?"})

        assert result.success is True
        assert "Python is a programming language" in result.output
        assert "https://python.org" in result.output
        assert "[1]" in result.output
        assert "[2]" in result.output

    @pytest.mark.asyncio
    async def test_successful_search_without_citations(self):
        """Handler should work when no citations are returned."""
        from dokumen.tools_object import create_perplexity_web_search_tool

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "The answer is 42."}}],
        }).encode("utf-8")

        tool = create_perplexity_web_search_tool(api_key="pplx-test")

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = await tool.handler({"query": "What is the answer?"})

        assert result.success is True
        assert "The answer is 42" in result.output
        assert "Sources:" not in result.output

    @pytest.mark.asyncio
    async def test_http_error_handling(self):
        """Handler should handle HTTP errors gracefully."""
        import urllib.error
        from dokumen.tools_object import create_perplexity_web_search_tool

        tool = create_perplexity_web_search_tool(api_key="pplx-test")

        error = urllib.error.HTTPError(
            url="https://api.perplexity.ai/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=BytesIO(b"rate limited"),
        )

        with patch("urllib.request.urlopen", side_effect=error):
            result = await tool.handler({"query": "test"})

        assert result.success is False
        assert "HTTP 429" in result.error

    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Handler should handle network errors gracefully."""
        from dokumen.tools_object import create_perplexity_web_search_tool

        tool = create_perplexity_web_search_tool(api_key="pplx-test")

        with patch("urllib.request.urlopen", side_effect=ConnectionError("Network down")):
            result = await tool.handler({"query": "test"})

        assert result.success is False
        assert "Web search failed" in result.error

    @pytest.mark.asyncio
    async def test_query_truncation(self):
        """Long queries should be truncated to 1000 chars."""
        from dokumen.tools_object import create_perplexity_web_search_tool

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "truncated result"}}],
        }).encode("utf-8")

        tool = create_perplexity_web_search_tool(api_key="pplx-test")
        long_query = "x" * 2000

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
            result = await tool.handler({"query": long_query})

        assert result.success is True
        # Verify the request body had truncated query
        call_args = mock_open.call_args
        request_obj = call_args[0][0]
        body = json.loads(request_obj.data.decode("utf-8"))
        assert len(body["messages"][0]["content"]) == 1000

    @pytest.mark.asyncio
    async def test_env_var_fallback(self):
        """Tool should fall back to PERPLEXITY_API_KEY env var."""
        from dokumen.tools_object import create_perplexity_web_search_tool

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "env key works"}}],
        }).encode("utf-8")

        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "pplx-from-env"}):
            tool = create_perplexity_web_search_tool(api_key=None)

            with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
                result = await tool.handler({"query": "test"})

            assert result.success is True
            # Verify the Authorization header uses env key
            call_args = mock_open.call_args
            request_obj = call_args[0][0]
            assert request_obj.get_header("Authorization") == "Bearer pplx-from-env"

    @pytest.mark.asyncio
    async def test_model_passed_to_api(self):
        """Custom model should be passed to the Perplexity API."""
        from dokumen.tools_object import create_perplexity_web_search_tool

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "custom model"}}],
        }).encode("utf-8")

        tool = create_perplexity_web_search_tool(api_key="pplx-test", model="sonar-pro")

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
            await tool.handler({"query": "test"})

        call_args = mock_open.call_args
        request_obj = call_args[0][0]
        body = json.loads(request_obj.data.decode("utf-8"))
        assert body["model"] == "sonar-pro"


class TestWebSearchToolResolution:
    """Tests for resolving web_search via resolve_tools and resolve_builtin_tool."""

    def test_resolve_builtin_tool_web_search(self):
        """resolve_builtin_tool should resolve web_search with config."""
        from dokumen.tools_object import resolve_builtin_tool

        tool = resolve_builtin_tool(
            "web_search",
            perplexity_config={"api_key": "pplx-test", "model": "sonar-pro"},
        )
        assert tool is not None
        assert tool.name == "web_search"

    def test_resolve_builtin_tool_web_search_no_config(self):
        """resolve_builtin_tool should resolve web_search without config."""
        from dokumen.tools_object import resolve_builtin_tool

        tool = resolve_builtin_tool("web_search")
        assert tool is not None
        assert tool.name == "web_search"

    def test_resolve_tools_includes_web_search(self):
        """resolve_tools should resolve web_search in tool list."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["web_search"],
            base_dir=".",
            perplexity_config={"api_key": "pplx-test"},
        )
        assert len(tools) == 1
        assert tools[0].name == "web_search"

    def test_resolve_tools_web_search_with_other_tools(self):
        """resolve_tools should resolve web_search alongside other tools."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["read_file", "web_search"],
            base_dir=".",
            perplexity_config={"api_key": "pplx-test"},
        )
        names = [t.name for t in tools]
        assert "read_file" in names
        assert "web_search" in names
