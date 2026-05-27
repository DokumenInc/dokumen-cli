"""Tests for SDK tool mapping layer (dokumen/sdk/tools.py)."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import fields

from dokumen.sdk.tools import (
    SDK_MAPPING,
    UNSUPPORTED_SDK_TOOLS,
    PLAYWRIGHT_MCP_PREFIX,
    ResolvedTools,
    resolve_sdk_tools,
    resolve_dokumen_tool,
    create_dokumen_mcp_server,
    get_playwright_mcp_config,
    _wrap_tool_definition,
)
from dokumen.tools_object import ToolDefinition, ToolResult


# --- Fixtures ---


def _make_tool_def(name: str = "test_tool") -> ToolDefinition:
    """Create a ToolDefinition for testing."""
    async def handler(params):
        return ToolResult(success=True, output=f"ran {name}", error=None)

    return ToolDefinition(
        name=name,
        description=f"Test tool: {name}",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )


# Fake BROWSER_TOOLS dict matching the real module's keys
FAKE_BROWSER_TOOLS = {
    "browser_navigate": MagicMock(),
    "browser_click": MagicMock(),
    "browser_type": MagicMock(),
    "browser_screenshot": MagicMock(),
    "browser_take_screenshot": MagicMock(),
    "browser_snapshot": MagicMock(),
    "browser_wait": MagicMock(),
    "browser_close": MagicMock(),
}


# --- ResolvedTools dataclass ---


class TestResolvedToolsDataclass:
    """Test ResolvedTools dataclass structure."""

    def test_resolved_tools_fields_correct_types(self):
        """ResolvedTools has the expected fields with correct defaults."""
        rt = ResolvedTools(
            sdk_tool_names=["Read", "Bash"],
            dokumen_mcp_tools=[],
        )
        assert rt.sdk_tool_names == ["Read", "Bash"]
        assert rt.dokumen_mcp_tools == []
        assert rt.playwright_mcp_config is None
        assert rt.playwright_tool_names == []

    def test_resolved_tools_with_all_fields(self):
        """ResolvedTools accepts all fields explicitly."""
        tool_def = _make_tool_def()
        rt = ResolvedTools(
            sdk_tool_names=["Read"],
            dokumen_mcp_tools=[tool_def],
            playwright_mcp_config={"some": "config"},
            playwright_tool_names=["mcp__playwright__browser_click"],
        )
        assert len(rt.dokumen_mcp_tools) == 1
        assert rt.playwright_mcp_config == {"some": "config"}
        assert rt.playwright_tool_names == ["mcp__playwright__browser_click"]


# --- SDK_MAPPING ---


class TestSdkMappingStandardTools:
    """Each key in SDK_MAPPING resolves to the correct SDK tool."""

    @pytest.mark.parametrize(
        "dokumen_name,expected_sdk_name",
        [
            ("read_file", "Read"),
            ("write_file", "Write"),
            ("glob", "Glob"),
            ("search_file_content", "Grep"),
            ("list_directory", "Glob"),
            ("run_shell_command", "Bash"),
            ("web_fetch", "WebFetch"),
            ("web_search", "WebSearch"),
        ],
    )
    def test_sdk_mapping_standard_tools(self, dokumen_name, expected_sdk_name):
        """SDK_MAPPING maps Dokumen tool name to correct SDK built-in."""
        assert SDK_MAPPING[dokumen_name] == expected_sdk_name


# --- Unsupported tools ---


class TestUnsupportedTools:
    """Unsupported tools raise ValueError with descriptive message."""

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", {})
    def test_unsupported_tool_raises_error(self):
        """anthropic_web_search raises ValueError."""
        with pytest.raises(ValueError, match="anthropic_web_search is not supported"):
            resolve_sdk_tools(["anthropic_web_search"])


# --- resolve_sdk_tools ---


class TestResolveSdkTools:
    """Tests for the main resolve_sdk_tools function."""

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", {})
    def test_resolve_sdk_tools_empty(self):
        """Empty list returns empty ResolvedTools."""
        result = resolve_sdk_tools([])
        assert result.sdk_tool_names == []
        assert result.dokumen_mcp_tools == []
        assert result.playwright_mcp_config is None
        assert result.playwright_tool_names == []

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", {})
    def test_resolve_sdk_tools_single_mapped(self):
        """Single SDK-mapped tool resolves correctly."""
        result = resolve_sdk_tools(["read_file"])
        assert result.sdk_tool_names == ["Read"]
        assert result.dokumen_mcp_tools == []

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", {})
    def test_resolve_sdk_tools_dedup(self):
        """Multiple Dokumen tools mapping to the same SDK tool are deduped."""
        # Both list_directory and glob map to Glob
        result = resolve_sdk_tools(["glob", "list_directory"])
        assert result.sdk_tool_names == ["Glob"]
        assert result.sdk_tool_names.count("Glob") == 1

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", {})
    def test_resolve_sdk_tools_multiple_unique(self):
        """Multiple Dokumen tools mapping to distinct SDK tools all appear."""
        result = resolve_sdk_tools(["read_file", "run_shell_command", "web_fetch"])
        assert "Read" in result.sdk_tool_names
        assert "Bash" in result.sdk_tool_names
        assert "WebFetch" in result.sdk_tool_names
        assert len(result.sdk_tool_names) == 3

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", FAKE_BROWSER_TOOLS)
    def test_resolve_sdk_tools_browser_tools(self):
        """Browser tools create playwright tool names and auto-inject Read."""
        result = resolve_sdk_tools(["browser_navigate", "browser_click"])
        assert f"{PLAYWRIGHT_MCP_PREFIX}browser_navigate" in result.playwright_tool_names
        assert f"{PLAYWRIGHT_MCP_PREFIX}browser_click" in result.playwright_tool_names
        assert len(result.playwright_tool_names) == 2
        # Read is auto-injected for browser tests
        assert "Read" in result.sdk_tool_names

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", FAKE_BROWSER_TOOLS)
    def test_resolve_sdk_tools_browser_read_not_duplicated(self):
        """If read_file is already present, Read is not duplicated for browser tools."""
        result = resolve_sdk_tools(["read_file", "browser_navigate"])
        assert result.sdk_tool_names.count("Read") == 1

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", {})
    def test_resolve_sdk_tools_mixed_with_unknown(self):
        """Mix of SDK tools and unknown tools raises ValueError for unknown."""
        with pytest.raises(ValueError, match="Unknown Dokumen tool: 'my_custom_tool'"):
            resolve_sdk_tools(["read_file", "my_custom_tool"])

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", FAKE_BROWSER_TOOLS)
    def test_resolve_sdk_tools_all_browser_tools(self):
        """All 8 browser tools resolve to playwright MCP names."""
        all_browser = list(FAKE_BROWSER_TOOLS.keys())
        result = resolve_sdk_tools(all_browser)
        assert len(result.playwright_tool_names) == 8
        for name in all_browser:
            assert f"{PLAYWRIGHT_MCP_PREFIX}{name}" in result.playwright_tool_names


# --- resolve_dokumen_tool ---


class TestResolveDokumenTool:
    """Tests for resolve_dokumen_tool stub."""

    def test_resolve_dokumen_tool_unknown_raises(self):
        """Unknown tool name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown Dokumen tool: 'nonexistent'"):
            resolve_dokumen_tool("nonexistent")

    def test_resolve_dokumen_tool_error_lists_known(self):
        """Error message includes list of known SDK tools."""
        with pytest.raises(ValueError, match="Known SDK tools"):
            resolve_dokumen_tool("some_tool")


# --- create_dokumen_mcp_server ---


class TestCreateDokumenMcpServer:
    """Tests for MCP server creation."""

    def test_create_dokumen_mcp_server_returns_config(self):
        """Creates server config with correct name."""
        tool_def = _make_tool_def("my_tool")
        config = create_dokumen_mcp_server([tool_def])
        # McpSdkServerConfig is a TypedDict with 'name' and 'type' keys
        assert config["name"] == "dokumen-tools"
        assert config["type"] == "sdk"

    def test_create_dokumen_mcp_server_empty_tools(self):
        """Creating server with no tools still returns valid config."""
        config = create_dokumen_mcp_server([])
        assert config["name"] == "dokumen-tools"
        assert config["type"] == "sdk"

    def test_create_dokumen_mcp_server_multiple_tools(self):
        """Creating server with multiple tools succeeds."""
        tools = [_make_tool_def(f"tool_{i}") for i in range(3)]
        config = create_dokumen_mcp_server(tools)
        assert config["name"] == "dokumen-tools"

    @pytest.mark.asyncio
    async def test_create_dokumen_mcp_server_on_tool_call_callback(self):
        """on_tool_call callback is invoked when wrapping tools."""
        callback = MagicMock()
        tool_def = _make_tool_def("callback_tool")

        # Test via _wrap_tool_definition directly
        wrapped = _wrap_tool_definition(tool_def, on_tool_call=callback)
        result = await wrapped.handler({"key": "value"})

        callback.assert_called_once()
        call_args = callback.call_args
        assert call_args[0][0] == "callback_tool"
        assert call_args[0][1] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_wrap_tool_definition_handler_returns_dict(self):
        """Wrapped handler returns a dict with success/output/error keys."""
        tool_def = _make_tool_def("wrap_test")
        wrapped = _wrap_tool_definition(tool_def)
        result = await wrapped.handler({})
        assert result["success"] is True
        assert result["output"] == "ran wrap_test"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_wrap_tool_definition_callback_failure_does_not_raise(self):
        """If on_tool_call callback raises, it is caught and logged."""
        def bad_callback(*args):
            raise RuntimeError("callback failed")

        tool_def = _make_tool_def("safe_tool")
        wrapped = _wrap_tool_definition(tool_def, on_tool_call=bad_callback)
        # Should not raise despite callback failure
        result = await wrapped.handler({})
        assert result["success"] is True


# --- get_playwright_mcp_config ---


class TestGetPlaywrightMcpConfig:
    """Tests for the Playwright MCP stdio config builder."""

    def test_get_playwright_mcp_config_returns_stdio_config(self):
        """Returns a valid McpStdioServerConfig dict."""
        config = get_playwright_mcp_config(test_name="test-browser")
        assert config is not None
        assert config.get("type", "stdio") == "stdio"
        assert "command" in config
        assert "--isolated" in config["args"]
