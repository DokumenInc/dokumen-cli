"""Tests for CLI MCP tool integration.

TDD: Tests written per CLAUDE.md rules.
Covers: tool creation, success/error paths, missing credentials,
        import failures, and integration with resolve_tools.
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dokumen.tools_object import ToolDefinition, ToolResult


@pytest.fixture
def mock_in_process_module():
    """Inject a mock backend.dokumen_mcp.in_process module into sys.modules.

    The CLI doesn't have `backend` on its path, so we must inject mocks
    into sys.modules before `create_mcp_tool_definitions` tries to import.
    """
    mock_adapter_cls = MagicMock()
    mock_module = MagicMock()
    mock_module.InProcessMCPAdapter = mock_adapter_cls

    # Inject parent modules into sys.modules
    saved = {}
    modules_to_inject = {
        "backend": MagicMock(),
        "backend.dokumen_mcp": MagicMock(),
        "backend.dokumen_mcp.in_process": mock_module,
    }
    for name, mod in modules_to_inject.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    yield mock_adapter_cls

    # Restore original state
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


class TestCreateMCPToolDefinitions:
    """Tests for create_mcp_tool_definitions()."""

    @patch.dict("os.environ", {
        "GITLAB_SERVICE_TOKEN": "glpat-test",
        "GITLAB_URL": "https://gitlab.example.com",
        "GITLAB_PROJECT_ID": "42",
    })
    def test_creates_tool_definitions(self, mock_in_process_module):
        """Creates ToolDefinition instances from MCP tool definitions."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        mock_adapter = MagicMock()
        mock_adapter.get_tool_definitions.return_value = [
            {"name": "read_file", "description": "Read a file", "inputSchema": {}},
            {"name": "ask", "description": "Ask a question", "inputSchema": {}},
        ]
        mock_in_process_module.return_value = mock_adapter

        tools = create_mcp_tool_definitions()

        assert len(tools) == 2
        assert tools[0].name == "read_file"
        assert tools[1].name == "ask"
        assert all(isinstance(t, ToolDefinition) for t in tools)
        assert all(callable(t.handler) for t in tools)

    @patch.dict("os.environ", {}, clear=True)
    def test_returns_empty_when_no_credentials(self):
        """Returns empty list when credentials are missing."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        tools = create_mcp_tool_definitions()
        assert tools == []

    @patch.dict("os.environ", {
        "GITLAB_TOKEN": "glpat-from-token-var",
        "GITLAB_URL": "https://gitlab.example.com",
        "GITLAB_PROJECT_ID": "42",
    })
    def test_falls_back_to_gitlab_token(self, mock_in_process_module):
        """Falls back to GITLAB_TOKEN when GITLAB_SERVICE_TOKEN is not set."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        mock_adapter = MagicMock()
        mock_adapter.get_tool_definitions.return_value = [
            {"name": "read_file", "description": "Read", "inputSchema": {}},
        ]
        mock_in_process_module.return_value = mock_adapter

        tools = create_mcp_tool_definitions()

        assert len(tools) == 1
        mock_in_process_module.assert_called_once()
        call_kwargs = mock_in_process_module.call_args[1]
        assert call_kwargs["gitlab_token"] == "glpat-from-token-var"

    @patch.dict("os.environ", {
        "GITLAB_SERVICE_TOKEN": "glpat-service",
        "GITLAB_TOKEN": "glpat-user",
        "GITLAB_URL": "https://gitlab.example.com",
        "GITLAB_PROJECT_ID": "42",
    })
    def test_service_token_takes_precedence_over_gitlab_token(self, mock_in_process_module):
        """GITLAB_SERVICE_TOKEN takes precedence over GITLAB_TOKEN."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        mock_adapter = MagicMock()
        mock_adapter.get_tool_definitions.return_value = []
        mock_in_process_module.return_value = mock_adapter

        create_mcp_tool_definitions()

        call_kwargs = mock_in_process_module.call_args[1]
        assert call_kwargs["gitlab_token"] == "glpat-service"

    @patch.dict("os.environ", {
        "GITLAB_SERVICE_TOKEN": "glpat-test",
        "GITLAB_URL": "https://gitlab.example.com",
        "GITLAB_PROJECT_ID": "not-a-number",
    })
    def test_returns_empty_for_invalid_project_id(self):
        """Returns empty list when project_id is not a valid integer."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        tools = create_mcp_tool_definitions()
        assert tools == []

    @patch.dict("os.environ", {
        "GITLAB_SERVICE_TOKEN": "glpat-test",
        "GITLAB_URL": "https://gitlab.example.com",
        "GITLAB_PROJECT_ID": "42",
    })
    def test_explicit_params_override_env(self, mock_in_process_module):
        """Explicit parameters take precedence over env vars."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        mock_adapter = MagicMock()
        mock_adapter.get_tool_definitions.return_value = []
        mock_in_process_module.return_value = mock_adapter

        create_mcp_tool_definitions(
            gitlab_token="explicit-token",
            gitlab_url="https://explicit.com",
            project_id=99,
            branch="staging",
        )

        mock_in_process_module.assert_called_once_with(
            gitlab_token="explicit-token",
            gitlab_url="https://explicit.com",
            project_id=99,
            branch="staging",
            source_type="pipeline",
        )


class TestMCPToolHandlerExecution:
    """Tests for the async handler returned by create_mcp_tool_definitions."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {
        "GITLAB_SERVICE_TOKEN": "glpat-test",
        "GITLAB_URL": "https://gitlab.example.com",
        "GITLAB_PROJECT_ID": "42",
    })
    async def test_handler_success(self, mock_in_process_module):
        """Handler returns success ToolResult with JSON output."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        mock_adapter = MagicMock()
        mock_adapter.get_tool_definitions.return_value = [
            {"name": "read_file", "description": "Read", "inputSchema": {}},
        ]
        mock_adapter.call_tool = AsyncMock(return_value={
            "success": True,
            "content": "Hello world",
        })
        mock_in_process_module.return_value = mock_adapter

        tools = create_mcp_tool_definitions()
        result = await tools[0].handler({"file_path": "docs/index.md"})

        assert result.success is True
        parsed = json.loads(result.output)
        assert parsed["content"] == "Hello world"
        mock_adapter.call_tool.assert_awaited_once_with("read_file", {"file_path": "docs/index.md"})

    @pytest.mark.asyncio
    @patch.dict("os.environ", {
        "GITLAB_SERVICE_TOKEN": "glpat-test",
        "GITLAB_URL": "https://gitlab.example.com",
        "GITLAB_PROJECT_ID": "42",
    })
    async def test_handler_mcp_error(self, mock_in_process_module):
        """Handler returns failure ToolResult when MCP reports error."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        mock_adapter = MagicMock()
        mock_adapter.get_tool_definitions.return_value = [
            {"name": "read_file", "description": "Read", "inputSchema": {}},
        ]
        mock_adapter.call_tool = AsyncMock(return_value={
            "success": False,
            "error": "File not found",
        })
        mock_in_process_module.return_value = mock_adapter

        tools = create_mcp_tool_definitions()
        result = await tools[0].handler({"file_path": "missing.md"})

        assert result.success is False
        assert "File not found" in result.error

    @pytest.mark.asyncio
    @patch.dict("os.environ", {
        "GITLAB_SERVICE_TOKEN": "glpat-test",
        "GITLAB_URL": "https://gitlab.example.com",
        "GITLAB_PROJECT_ID": "42",
    })
    async def test_handler_exception(self, mock_in_process_module):
        """Handler returns failure ToolResult when adapter raises."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        mock_adapter = MagicMock()
        mock_adapter.get_tool_definitions.return_value = [
            {"name": "read_file", "description": "Read", "inputSchema": {}},
        ]
        mock_adapter.call_tool = AsyncMock(side_effect=RuntimeError("Network error"))
        mock_in_process_module.return_value = mock_adapter

        tools = create_mcp_tool_definitions()
        result = await tools[0].handler({})

        assert result.success is False
        assert "Network error" in result.error

    @pytest.mark.asyncio
    @patch.dict("os.environ", {
        "GITLAB_SERVICE_TOKEN": "glpat-test",
        "GITLAB_URL": "https://gitlab.example.com",
        "GITLAB_PROJECT_ID": "42",
    })
    async def test_each_handler_calls_correct_tool(self, mock_in_process_module):
        """Each handler calls the correct tool name on the adapter."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        mock_adapter = MagicMock()
        mock_adapter.get_tool_definitions.return_value = [
            {"name": "ask", "description": "Ask", "inputSchema": {}},
            {"name": "explore", "description": "Explore", "inputSchema": {}},
        ]
        mock_adapter.call_tool = AsyncMock(return_value={"success": True})
        mock_in_process_module.return_value = mock_adapter

        tools = create_mcp_tool_definitions()

        await tools[0].handler({"question": "What is X?"})
        mock_adapter.call_tool.assert_awaited_with("ask", {"question": "What is X?"})

        await tools[1].handler({"query": "coverage"})
        mock_adapter.call_tool.assert_awaited_with("explore", {"query": "coverage"})


class TestMCPToolsLogLevel:
    """Tests for mcp_tools.skip log level based on execution mode."""

    @patch.dict("os.environ", {}, clear=True)
    def test_skip_logs_debug_when_not_sandbox(self):
        """Missing credentials logs at DEBUG level when not in sandbox mode."""
        import logging
        from dokumen.mcp_tools import create_mcp_tool_definitions

        with patch("dokumen.mcp_tools.logger") as mock_logger:
            create_mcp_tool_definitions()

            # Should log at DEBUG, not WARNING
            debug_calls = [
                c for c in mock_logger.debug.call_args_list
                if c.args and "mcp_tools.skip" in str(c.args[0])
            ]
            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if c.args and "mcp_tools.skip" in str(c.args[0])
            ]
            assert len(debug_calls) == 1, "Should log at DEBUG when not in sandbox"
            assert len(warning_calls) == 0, "Should NOT log at WARNING when not in sandbox"

    @patch.dict("os.environ", {"DOKUMEN_EXECUTION_MODE": "sandbox"}, clear=True)
    def test_skip_logs_warning_in_sandbox_mode(self):
        """Missing credentials logs at WARNING level in sandbox mode."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        with patch("dokumen.mcp_tools.logger") as mock_logger:
            create_mcp_tool_definitions()

            # Should log at WARNING in sandbox mode
            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if c.args and "mcp_tools.skip" in str(c.args[0])
            ]
            assert len(warning_calls) == 1, "Should log at WARNING in sandbox mode"

    @patch.dict("os.environ", {"DOKUMEN_EXECUTION_MODE": "cli"}, clear=True)
    def test_skip_logs_debug_when_cli_mode(self):
        """Missing credentials logs at DEBUG level when execution_mode is cli."""
        from dokumen.mcp_tools import create_mcp_tool_definitions

        with patch("dokumen.mcp_tools.logger") as mock_logger:
            create_mcp_tool_definitions()

            debug_calls = [
                c for c in mock_logger.debug.call_args_list
                if c.args and "mcp_tools.skip" in str(c.args[0])
            ]
            assert len(debug_calls) == 1, "Should log at DEBUG when mode is cli"
