"""Tests for explore and ask tool factories and their integration with resolve_tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dokumen.tools_object import ToolDefinition, ToolResult


class TestCreateExploreTool:
    """Tests for create_explore_tool() factory."""

    def test_returns_tool_definition(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()
        tool = create_explore_tool(config=config, project_root="/tmp/test")
        assert isinstance(tool, ToolDefinition)

    def test_tool_name_is_explore(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()
        tool = create_explore_tool(config=config, project_root="/tmp/test")
        assert tool.name == "explore"

    def test_tool_has_description(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()
        tool = create_explore_tool(config=config, project_root="/tmp/test")
        assert len(tool.description) > 0

    def test_parameters_has_query_required(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()
        tool = create_explore_tool(config=config, project_root="/tmp/test")
        assert "query" in tool.parameters["properties"]
        assert "query" in tool.parameters["required"]

    def test_parameters_has_explore_type(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()
        tool = create_explore_tool(config=config, project_root="/tmp/test")
        assert "explore_type" in tool.parameters["properties"]
        assert tool.parameters["properties"]["explore_type"]["enum"] == ["docs", "code", "both"]

    def test_handler_is_callable(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()
        tool = create_explore_tool(config=config, project_root="/tmp/test")
        assert callable(tool.handler)

    @pytest.mark.asyncio
    async def test_handler_missing_query_returns_error(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()
        tool = create_explore_tool(config=config, project_root="/tmp/test")
        result = await tool.handler({})
        assert not result.success
        assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handler_none_query_returns_error(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()
        tool = create_explore_tool(config=config, project_root="/tmp/test")
        result = await tool.handler({"query": None})
        assert not result.success
        assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handler_empty_query_returns_error(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()
        tool = create_explore_tool(config=config, project_root="/tmp/test")
        result = await tool.handler({"query": ""})
        assert not result.success
        assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handler_invalid_explore_type_returns_error(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()
        tool = create_explore_tool(config=config, project_root="/tmp/test")
        result = await tool.handler({"query": "test", "explore_type": "invalid"})
        assert not result.success
        assert "explore_type" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handler_api_key_missing_returns_error(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()
        config.explore.model = "test-model"
        config.provider.model = "test-model"

        with patch.dict("os.environ", {}, clear=True):
            with patch("dokumen.explore_agent.ExploreAgent"):
                tool = create_explore_tool(config=config, project_root="/tmp/test")
                result = await tool.handler({"query": "test query"})
                assert not result.success
                assert "ANTHROPIC_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_handler_calls_explore_agent(self):
        from dokumen.tools_object import create_explore_tool
        from dokumen.explore_agent import ExploreResult

        config = MagicMock()
        mock_explore_result = ExploreResult(
            files=[],
            duration=1.0,
            tool_calls_count=2,
            success=True,
            summary="Found 0 files",
        )

        with patch("dokumen.explore_agent.ExploreAgent") as MockExploreAgent:
            mock_agent = AsyncMock()
            mock_agent.explore = AsyncMock(return_value=mock_explore_result)
            MockExploreAgent.return_value = mock_agent

            with patch("dokumen.tools_object._create_explore_provider") as mock_provider_factory:
                mock_provider_factory.return_value = MagicMock()

                tool = create_explore_tool(config=config, project_root="/tmp/test")
                result = await tool.handler({"query": "find auth docs"})

                assert result.success
                mock_agent.explore.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_returns_error_on_explore_failure(self):
        from dokumen.tools_object import create_explore_tool
        from dokumen.explore_agent import ExploreResult

        config = MagicMock()
        mock_explore_result = ExploreResult(
            files=[],
            duration=1.0,
            tool_calls_count=0,
            success=False,
            error="Provider not configured",
            summary="Exploration failed",
        )

        with patch("dokumen.explore_agent.ExploreAgent") as MockExploreAgent:
            mock_agent = AsyncMock()
            mock_agent.explore = AsyncMock(return_value=mock_explore_result)
            MockExploreAgent.return_value = mock_agent

            with patch("dokumen.tools_object._create_explore_provider") as mock_provider_factory:
                mock_provider_factory.return_value = MagicMock()

                tool = create_explore_tool(config=config, project_root="/tmp/test")
                result = await tool.handler({"query": "find auth docs"})

                assert not result.success
                assert "Provider not configured" in result.error

    @pytest.mark.asyncio
    async def test_handler_catches_exception(self):
        from dokumen.tools_object import create_explore_tool

        config = MagicMock()

        with patch("dokumen.explore_agent.ExploreAgent") as MockExploreAgent:
            MockExploreAgent.side_effect = RuntimeError("Unexpected error")

            with patch("dokumen.tools_object._create_explore_provider") as mock_provider_factory:
                mock_provider_factory.return_value = MagicMock()

                tool = create_explore_tool(config=config, project_root="/tmp/test")
                result = await tool.handler({"query": "find auth docs"})

                assert not result.success
                assert "Unexpected error" in result.error


class TestCreateAskTool:
    """Tests for create_ask_tool() factory."""

    def test_returns_tool_definition(self):
        from dokumen.tools_object import create_ask_tool

        config = MagicMock()
        tool = create_ask_tool(config=config, project_root="/tmp/test")
        assert isinstance(tool, ToolDefinition)

    def test_tool_name_is_ask(self):
        from dokumen.tools_object import create_ask_tool

        config = MagicMock()
        tool = create_ask_tool(config=config, project_root="/tmp/test")
        assert tool.name == "ask"

    def test_tool_has_description(self):
        from dokumen.tools_object import create_ask_tool

        config = MagicMock()
        tool = create_ask_tool(config=config, project_root="/tmp/test")
        assert len(tool.description) > 0

    def test_parameters_has_question_required(self):
        from dokumen.tools_object import create_ask_tool

        config = MagicMock()
        tool = create_ask_tool(config=config, project_root="/tmp/test")
        assert "question" in tool.parameters["properties"]
        assert "question" in tool.parameters["required"]

    def test_handler_is_callable(self):
        from dokumen.tools_object import create_ask_tool

        config = MagicMock()
        tool = create_ask_tool(config=config, project_root="/tmp/test")
        assert callable(tool.handler)

    @pytest.mark.asyncio
    async def test_handler_missing_question_returns_error(self):
        from dokumen.tools_object import create_ask_tool

        config = MagicMock()
        tool = create_ask_tool(config=config, project_root="/tmp/test")
        result = await tool.handler({})
        assert not result.success
        assert "question" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handler_none_question_returns_error(self):
        from dokumen.tools_object import create_ask_tool

        config = MagicMock()
        tool = create_ask_tool(config=config, project_root="/tmp/test")
        result = await tool.handler({"question": None})
        assert not result.success
        assert "question" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handler_empty_question_returns_error(self):
        from dokumen.tools_object import create_ask_tool

        config = MagicMock()
        tool = create_ask_tool(config=config, project_root="/tmp/test")
        result = await tool.handler({"question": ""})
        assert not result.success
        assert "question" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handler_calls_ask_agent(self):
        from dokumen.tools_object import create_ask_tool
        from dokumen.ask_agent import AskResult

        config = MagicMock()
        mock_ask_result = AskResult(
            success=True,
            answer="The auth system uses OAuth2.",
            sources=["docs/auth.md"],
            confidence="High",
            matched_tests=[],
            explore_summary="Found auth docs",
            duration=2.0,
            tool_calls_count=3,
        )

        with patch("dokumen.ask_agent.AskAgent") as MockAskAgent:
            mock_agent = AsyncMock()
            mock_agent.ask = AsyncMock(return_value=mock_ask_result)
            MockAskAgent.return_value = mock_agent

            with patch("dokumen.tools_object._create_explore_provider") as mock_provider_factory:
                mock_provider_factory.return_value = MagicMock()

                tool = create_ask_tool(config=config, project_root="/tmp/test")
                result = await tool.handler({"question": "How does auth work?"})

                assert result.success
                mock_agent.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_returns_error_on_ask_failure(self):
        from dokumen.tools_object import create_ask_tool
        from dokumen.ask_agent import AskResult

        config = MagicMock()
        mock_ask_result = AskResult(
            success=False,
            answer="",
            sources=[],
            confidence="Low",
            matched_tests=[],
            explore_summary=None,
            duration=1.0,
            tool_calls_count=0,
            error="No provider configured",
        )

        with patch("dokumen.ask_agent.AskAgent") as MockAskAgent:
            mock_agent = AsyncMock()
            mock_agent.ask = AsyncMock(return_value=mock_ask_result)
            MockAskAgent.return_value = mock_agent

            with patch("dokumen.tools_object._create_explore_provider") as mock_provider_factory:
                mock_provider_factory.return_value = MagicMock()

                tool = create_ask_tool(config=config, project_root="/tmp/test")
                result = await tool.handler({"question": "How does auth work?"})

                assert not result.success
                assert "No provider configured" in result.error

    @pytest.mark.asyncio
    async def test_handler_catches_exception(self):
        from dokumen.tools_object import create_ask_tool

        config = MagicMock()

        with patch("dokumen.ask_agent.AskAgent") as MockAskAgent:
            MockAskAgent.side_effect = RuntimeError("Unexpected error")

            with patch("dokumen.tools_object._create_explore_provider") as mock_provider_factory:
                mock_provider_factory.return_value = MagicMock()

                tool = create_ask_tool(config=config, project_root="/tmp/test")
                result = await tool.handler({"question": "How does auth work?"})

                assert not result.success
                assert "Unexpected error" in result.error


class TestAgentToolsDict:
    """Tests for AGENT_TOOLS dictionary."""

    def test_agent_tools_contains_explore(self):
        from dokumen.tools_object import AGENT_TOOLS

        assert "explore" in AGENT_TOOLS

    def test_agent_tools_contains_ask(self):
        from dokumen.tools_object import AGENT_TOOLS

        assert "ask" in AGENT_TOOLS

    def test_agent_tools_factories_are_callable(self):
        from dokumen.tools_object import AGENT_TOOLS

        for name, factory in AGENT_TOOLS.items():
            assert callable(factory), f"Factory for '{name}' is not callable"


class TestResolveToolsAgentTools:
    """Tests for resolve_tools() handling of explore and ask tools."""

    def test_resolve_explore_tool(self):
        from dokumen.loader import resolve_tools

        config = MagicMock()
        with patch("dokumen.tool_resolver._get_agent_tool_config") as mock_config:
            mock_config.return_value = (config, "/tmp/test")
            tools = resolve_tools(["explore"], base_dir="/tmp/test")

        assert len(tools) == 1
        assert tools[0].name == "explore"

    def test_resolve_ask_tool(self):
        from dokumen.loader import resolve_tools

        config = MagicMock()
        with patch("dokumen.tool_resolver._get_agent_tool_config") as mock_config:
            mock_config.return_value = (config, "/tmp/test")
            tools = resolve_tools(["ask"], base_dir="/tmp/test")

        assert len(tools) == 1
        assert tools[0].name == "ask"

    def test_resolve_explore_with_other_tools(self):
        from dokumen.loader import resolve_tools

        config = MagicMock()
        with patch("dokumen.tool_resolver._get_agent_tool_config") as mock_config:
            mock_config.return_value = (config, "/tmp/test")
            tools = resolve_tools(["read_file", "explore"], base_dir="/tmp/test")

        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "read_file" in names
        assert "explore" in names
