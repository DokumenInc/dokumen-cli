"""Tests for SDK browser/Playwright tool resolution.

Validates that browser tools in the SDK path correctly resolve to
Playwright MCP tool names and auto-inject Read.
"""

import pytest

from dokumen.sdk.tools import (
    SDK_MAPPING,
    UNSUPPORTED_SDK_TOOLS,
    ResolvedTools,
    get_playwright_mcp_config,
    resolve_sdk_tools,
)


class TestBrowserToolResolutionInSdk:
    def test_single_browser_tool_resolves_to_playwright(self):
        """A single browser tool resolves to mcp__playwright__ prefix."""
        result = resolve_sdk_tools(["browser_navigate"])

        assert "mcp__playwright__browser_navigate" in result.playwright_tool_names

    def test_all_browser_tools_resolve(self):
        """All 8 browser tools resolve to Playwright MCP names."""
        browser_tools = [
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_type",
            "browser_wait",
            "browser_screenshot",
            "browser_take_screenshot",
            "browser_close",
        ]
        result = resolve_sdk_tools(browser_tools)

        for tool in browser_tools:
            assert f"mcp__playwright__{tool}" in result.playwright_tool_names

    def test_browser_tools_auto_inject_read(self):
        """Browser tools auto-inject Read into SDK tools."""
        result = resolve_sdk_tools(["browser_navigate"])

        assert "Read" in result.sdk_tool_names

    def test_browser_tools_no_duplicate_read(self):
        """Read not duplicated when read_file is also in the list."""
        result = resolve_sdk_tools(["browser_navigate", "read_file"])

        read_count = result.sdk_tool_names.count("Read")
        assert read_count == 1

    def test_browser_tools_mixed_with_standard(self):
        """Browser tools work alongside standard SDK tools."""
        result = resolve_sdk_tools(["read_file", "browser_navigate", "glob"])

        assert "Read" in result.sdk_tool_names
        assert "Glob" in result.sdk_tool_names
        assert "mcp__playwright__browser_navigate" in result.playwright_tool_names

    def test_browser_tools_create_playwright_config(self):
        """Browser tools trigger Playwright MCP config creation."""
        result = resolve_sdk_tools(
            ["browser_navigate", "browser_click"],
            test_name="test-pw-config",
        )

        assert len(result.playwright_tool_names) == 2
        assert result.playwright_mcp_config is not None
        assert result.playwright_mcp_config.get("type", "stdio") == "stdio"

    def test_no_browser_tools_no_playwright(self):
        """No browser tools means no Playwright MCP tools."""
        result = resolve_sdk_tools(["read_file", "glob"])

        assert result.playwright_tool_names == []
        assert result.playwright_mcp_config is None


class TestBrowserWithSdkExecutorPattern:
    def test_browser_test_typical_tools(self):
        """Typical browser test scaffold tool set resolves correctly."""
        # A typical browser test uses these tools
        tools = [
            "read_file",
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_snapshot",
            "browser_take_screenshot",
            "browser_wait",
        ]
        result = resolve_sdk_tools(tools)

        # SDK tools: Read (from read_file + auto-inject)
        assert "Read" in result.sdk_tool_names

        # Playwright tools: all browser_* tools
        for t in tools:
            if t.startswith("browser_"):
                assert f"mcp__playwright__{t}" in result.playwright_tool_names

    def test_browser_test_with_shell_command(self):
        """Browser test with run_shell_command gets both Bash and browser tools."""
        result = resolve_sdk_tools([
            "browser_navigate",
            "browser_snapshot",
            "run_shell_command",
        ])

        assert "Bash" in result.sdk_tool_names
        assert "Read" in result.sdk_tool_names  # Auto-injected
        assert "mcp__playwright__browser_navigate" in result.playwright_tool_names
        assert "mcp__playwright__browser_snapshot" in result.playwright_tool_names


class TestUnsupportedToolsInBrowserContext:
    def test_anthropic_web_search_rejected(self):
        """anthropic_web_search raises ValueError in SDK path."""
        with pytest.raises(ValueError, match="not supported"):
            resolve_sdk_tools(["anthropic_web_search"])

    def test_anthropic_web_search_with_browser_tools_rejected(self):
        """anthropic_web_search + browser tools still raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            resolve_sdk_tools(["browser_navigate", "anthropic_web_search"])


class TestSdkMappingCompleteness:
    def test_all_standard_tools_mapped(self):
        """All 8 standard Dokumen tools have SDK mappings."""
        expected = {
            "read_file": "Read",
            "write_file": "Write",
            "glob": "Glob",
            "search_file_content": "Grep",
            "list_directory": "Glob",
            "run_shell_command": "Bash",
            "web_fetch": "WebFetch",
            "web_search": "WebSearch",
        }
        for dokumen_name, sdk_name in expected.items():
            assert SDK_MAPPING[dokumen_name] == sdk_name, f"{dokumen_name} should map to {sdk_name}"

    def test_unsupported_tools_documented(self):
        """Unsupported tools have error messages."""
        assert "anthropic_web_search" in UNSUPPORTED_SDK_TOOLS
        msg = UNSUPPORTED_SDK_TOOLS["anthropic_web_search"]
        assert "web_search" in msg  # Suggests alternative

    def test_playwright_config_returns_stdio_config(self):
        """get_playwright_mcp_config() returns McpStdioServerConfig dict."""
        config = get_playwright_mcp_config(test_name="test-browser")
        assert config is not None
        assert config.get("type", "stdio") == "stdio"
        assert "command" in config
        assert isinstance(config.get("args", []), list)
        # Should include --isolated flag
        args = config.get("args", [])
        assert "--isolated" in args


class TestPlaywrightMcpConfig:
    def test_config_includes_headless_flag(self):
        """Headless mode adds --headless flag."""
        config = get_playwright_mcp_config(
            test_name="test-headless",
            headless=True,
        )
        assert "--headless" in config["args"]

    def test_config_excludes_headless_when_false(self):
        """Non-headless mode omits --headless flag."""
        config = get_playwright_mcp_config(
            test_name="test-no-headless",
            headless=False,
        )
        assert "--headless" not in config["args"]

    def test_config_includes_save_video(self):
        """Video recording adds --save-video and --output-dir flags."""
        config = get_playwright_mcp_config(
            test_name="test-video",
            save_video="1920x1080",
        )
        args = config["args"]
        assert "--save-video" in args
        idx = args.index("--save-video")
        assert args[idx + 1] == "1920x1080"
        assert "--output-dir" in args

    def test_config_includes_viewport_size(self):
        """Viewport size adds --viewport-size flag."""
        config = get_playwright_mcp_config(
            test_name="test-viewport",
            viewport_size="1280x720",
        )
        args = config["args"]
        assert "--viewport-size" in args
        idx = args.index("--viewport-size")
        assert args[idx + 1] == "1280x720"

    def test_config_no_sandbox_when_root(self):
        """Running as root adds --no-sandbox flag."""
        import os
        from unittest.mock import patch

        with patch.object(os, "geteuid", return_value=0):
            config = get_playwright_mcp_config(test_name="test-root")
        assert "--no-sandbox" in config["args"]

    def test_config_defaults_to_headless_in_ci(self):
        """Config uses headless=True by default (CI environment)."""
        config = get_playwright_mcp_config(test_name="test-ci-default")
        # Default headless is True (CI-safe)
        assert "--headless" in config["args"]


class TestBrowserExecutionWiring:
    def test_browser_tools_use_stdio_mcp_server(self):
        """Browser tools use McpStdioServerConfig, not Dokumen MCP wrappers."""
        from dokumen.test_builder import build_sdk_executor

        data = {
            "name": "browser-test",
            "executor": {"user_prompt": "go browse"},
            "timeout": 60,
        }

        executor = build_sdk_executor(
            data=data,
            executor_system_prompt="sys",
            executor_tool_names=["browser_navigate", "browser_click", "read_file"],
            actual_executor_provider=None,
            executor_max_iterations=5,
            tools_config=None,
        )

        allowed = list(getattr(executor._executor._options, "allowed_tools", []) or [])
        # Read should be in allowed tools (SDK built-in)
        assert "Read" in allowed
        # Playwright tool names should be in allowed tools (prefixed)
        assert "mcp__playwright__browser_navigate" in allowed
        assert "mcp__playwright__browser_click" in allowed

        servers = getattr(executor._executor._options, "mcp_servers", {}) or {}
        # Playwright MCP server should be configured as stdio
        assert "playwright" in servers
        pw_config = servers["playwright"]
        assert pw_config.get("type", "stdio") == "stdio"
        assert "command" in pw_config

    def test_browser_tools_not_wrapped_as_dokumen_mcp(self):
        """Browser tools should NOT be wrapped through Dokumen MCP server."""
        from dokumen.test_builder import build_sdk_executor

        data = {
            "name": "browser-test",
            "executor": {"user_prompt": "go browse"},
            "timeout": 60,
        }

        executor = build_sdk_executor(
            data=data,
            executor_system_prompt="sys",
            executor_tool_names=["browser_navigate", "read_file"],
            actual_executor_provider=None,
            executor_max_iterations=5,
            tools_config=None,
        )

        allowed = list(getattr(executor._executor._options, "allowed_tools", []) or [])
        # No mcp__dokumen__browser_* tools (those were the old wrapping approach)
        assert all(not t.startswith("mcp__dokumen__browser_") for t in allowed)
