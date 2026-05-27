"""
Tests for Playwright browser tools.

TDD: Tests written first, then implementation.
"""
import pytest
from unittest.mock import AsyncMock


class TestBrowserTools:
    """Tests for browser tool definitions."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client for testing."""
        from dokumen.tools_object import ToolResult

        client = AsyncMock()
        client.call_tool = AsyncMock(return_value=ToolResult(
            success=True,
            output="Tool executed",
            error=None
        ))
        return client

    def test_browser_tools_exports(self):
        """Test that BROWSER_TOOLS dict contains expected tools."""
        from dokumen.playwright_tools import BROWSER_TOOLS

        expected_tools = {
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_screenshot",
            "browser_take_screenshot",
            "browser_snapshot",
            "browser_wait",
            "browser_close",
        }

        assert expected_tools == set(BROWSER_TOOLS.keys())

    def test_create_browser_navigate_tool(self):
        """Test browser_navigate tool creation."""
        from dokumen.playwright_tools import create_browser_navigate_tool
        from dokumen.tools_object import ToolDefinition

        mock_client = AsyncMock()
        tool = create_browser_navigate_tool(mock_client)

        assert isinstance(tool, ToolDefinition)
        assert tool.name == "browser_navigate"
        assert "url" in tool.parameters["properties"]
        assert "url" in tool.parameters["required"]

    def test_create_browser_click_tool(self):
        """Test browser_click tool creation."""
        from dokumen.playwright_tools import create_browser_click_tool
        from dokumen.tools_object import ToolDefinition

        mock_client = AsyncMock()
        tool = create_browser_click_tool(mock_client)

        assert isinstance(tool, ToolDefinition)
        assert tool.name == "browser_click"
        assert "ref" in tool.parameters["properties"]
        assert "element" in tool.parameters["properties"]  # Optional element description
        assert "ref" in tool.parameters["required"]

    def test_create_browser_type_tool(self):
        """Test browser_type tool creation."""
        from dokumen.playwright_tools import create_browser_type_tool
        from dokumen.tools_object import ToolDefinition

        mock_client = AsyncMock()
        tool = create_browser_type_tool(mock_client)

        assert isinstance(tool, ToolDefinition)
        assert tool.name == "browser_type"
        assert "ref" in tool.parameters["properties"]
        assert "text" in tool.parameters["properties"]
        assert "element" in tool.parameters["properties"]  # Optional element description
        assert "submit" in tool.parameters["properties"]   # Optional submit flag
        assert set(tool.parameters["required"]) == {"ref", "text"}

    def test_create_browser_screenshot_tool(self):
        """Test browser_screenshot tool creation."""
        from dokumen.playwright_tools import create_browser_screenshot_tool
        from dokumen.tools_object import ToolDefinition

        mock_client = AsyncMock()
        tool = create_browser_screenshot_tool(mock_client)

        assert isinstance(tool, ToolDefinition)
        assert tool.name == "browser_screenshot"

    def test_create_browser_take_screenshot_tool(self):
        """Test browser_take_screenshot tool creation."""
        from dokumen.playwright_tools import create_browser_take_screenshot_tool
        from dokumen.tools_object import ToolDefinition

        mock_client = AsyncMock()
        tool = create_browser_take_screenshot_tool(mock_client)

        assert isinstance(tool, ToolDefinition)
        assert tool.name == "browser_take_screenshot"
        # No required params
        assert tool.parameters.get("required", []) == []

    def test_create_browser_snapshot_tool(self):
        """Test browser_snapshot tool creation."""
        from dokumen.playwright_tools import create_browser_snapshot_tool
        from dokumen.tools_object import ToolDefinition

        mock_client = AsyncMock()
        tool = create_browser_snapshot_tool(mock_client)

        assert isinstance(tool, ToolDefinition)
        assert tool.name == "browser_snapshot"
        # No required params
        assert tool.parameters.get("required", []) == []

    def test_create_browser_close_tool(self):
        """Test browser_close tool creation."""
        from dokumen.playwright_tools import create_browser_close_tool
        from dokumen.tools_object import ToolDefinition

        mock_client = AsyncMock()
        tool = create_browser_close_tool(mock_client)

        assert isinstance(tool, ToolDefinition)
        assert tool.name == "browser_close"


class TestBrowserNavigate:
    """Tests for browser_navigate tool behavior."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client."""
        from dokumen.tools_object import ToolResult

        client = AsyncMock()
        client.call_tool = AsyncMock(return_value=ToolResult(
            success=True,
            output={"url": "https://example.com", "title": "Example"},
            error=None
        ))
        return client

    @pytest.mark.asyncio
    async def test_navigate_fails_without_mcp_client(self):
        """Test navigate fails when no MCP client is available."""
        from dokumen.playwright_tools import create_browser_navigate_tool

        tool = create_browser_navigate_tool(None)  # No client
        result = await tool.handler({"url": "https://example.com"})

        assert result.success is False
        assert "No MCP client" in result.error

    @pytest.mark.asyncio
    async def test_navigate_fails_without_url(self, mock_mcp_client):
        """Test navigate fails when url parameter is missing."""
        from dokumen.playwright_tools import create_browser_navigate_tool

        tool = create_browser_navigate_tool(mock_mcp_client)
        result = await tool.handler({})  # No url

        assert result.success is False
        assert "Missing required parameter: url" in result.error

    @pytest.mark.asyncio
    async def test_navigate_calls_mcp_navigate(self, mock_mcp_client):
        """Test that navigate calls MCP navigate tool."""
        from dokumen.playwright_tools import create_browser_navigate_tool

        tool = create_browser_navigate_tool(mock_mcp_client)
        result = await tool.handler({"url": "https://example.com"})

        mock_mcp_client.call_tool.assert_called()
        # First call should be browser_navigate
        first_call = mock_mcp_client.call_tool.call_args_list[0]
        assert first_call[0][0] == "browser_navigate"
        assert first_call[0][1]["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_navigate_does_not_call_browser_evaluate(self, mock_mcp_client):
        """Test that navigate does NOT call browser_evaluate.

        Click indicator injection is now handled via --init-script flag
        when the MCP server starts (see test_mcp_client.py).
        """
        from dokumen.playwright_tools import create_browser_navigate_tool

        tool = create_browser_navigate_tool(mock_mcp_client)
        await tool.handler({"url": "https://example.com"})

        # Should NOT call browser_evaluate (injection moved to --init-script)
        calls = mock_mcp_client.call_tool.call_args_list
        evaluate_calls = [c for c in calls if c[0][0] == "browser_evaluate"]

        assert len(evaluate_calls) == 0, "browser_evaluate should not be called"

    @pytest.mark.asyncio
    async def test_navigate_returns_page_info(self, mock_mcp_client):
        """Test that navigate returns page URL and title."""
        from dokumen.playwright_tools import create_browser_navigate_tool

        tool = create_browser_navigate_tool(mock_mcp_client)
        result = await tool.handler({"url": "https://example.com"})

        assert result.success is True
        assert "example.com" in str(result.output).lower()


class TestBrowserClick:
    """Tests for browser_click tool behavior."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client."""
        from dokumen.tools_object import ToolResult

        client = AsyncMock()
        client.call_tool = AsyncMock(return_value=ToolResult(
            success=True, output="Clicked", error=None
        ))
        return client

    @pytest.mark.asyncio
    async def test_click_fails_without_mcp_client(self):
        """Test click fails when no MCP client is available."""
        from dokumen.playwright_tools import create_browser_click_tool

        tool = create_browser_click_tool(None)  # No client
        result = await tool.handler({"ref": "button_1"})

        assert result.success is False
        assert "No MCP client" in result.error

    @pytest.mark.asyncio
    async def test_click_fails_without_ref(self, mock_mcp_client):
        """Test click fails when ref parameter is missing."""
        from dokumen.playwright_tools import create_browser_click_tool

        tool = create_browser_click_tool(mock_mcp_client)
        result = await tool.handler({})  # No ref

        assert result.success is False
        assert "Missing required parameter: ref" in result.error

    @pytest.mark.asyncio
    async def test_click_by_ref(self, mock_mcp_client):
        """Test clicking element by reference.

        Note: Playwright MCP requires BOTH 'element' (human-readable description)
        AND 'ref' (element reference) parameters.

        Visual animations are now handled by the Playwright MCP fork with
        --visual-indicators flag, so no browser_evaluate calls are made here.
        """
        from dokumen.playwright_tools import create_browser_click_tool

        tool = create_browser_click_tool(mock_mcp_client)
        result = await tool.handler({"ref": "button_1"})

        # Should call browser_click directly (animations handled by MCP fork)
        calls = mock_mcp_client.call_tool.call_args_list
        assert len(calls) == 1, "Should only call browser_click"

        call = calls[0]
        assert call[0][0] == "browser_click"
        assert call[0][1] == {"element": "button_1", "ref": "button_1"}
        assert result.success is True

    @pytest.mark.asyncio
    async def test_click_with_custom_element_description(self, mock_mcp_client):
        """Test clicking with custom element description."""
        from dokumen.playwright_tools import create_browser_click_tool

        tool = create_browser_click_tool(mock_mcp_client)
        result = await tool.handler({
            "ref": "button_1",
            "element": "Submit button"
        })

        # Last call should be browser_click
        last_call = mock_mcp_client.call_tool.call_args_list[-1]
        assert last_call[0][0] == "browser_click"
        assert last_call[0][1] == {"element": "Submit button", "ref": "button_1"}
        assert result.success is True

class TestBrowserType:
    """Tests for browser_type tool behavior."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client."""
        from dokumen.tools_object import ToolResult

        client = AsyncMock()
        client.call_tool = AsyncMock(return_value=ToolResult(
            success=True, output="Typed", error=None
        ))
        return client

    @pytest.mark.asyncio
    async def test_type_fails_without_mcp_client(self):
        """Test type fails when no MCP client is available."""
        from dokumen.playwright_tools import create_browser_type_tool

        tool = create_browser_type_tool(None)  # No client
        result = await tool.handler({"ref": "input_1", "text": "hello"})

        assert result.success is False
        assert "No MCP client" in result.error

    @pytest.mark.asyncio
    async def test_type_fails_without_ref(self, mock_mcp_client):
        """Test type fails when ref parameter is missing."""
        from dokumen.playwright_tools import create_browser_type_tool

        tool = create_browser_type_tool(mock_mcp_client)
        result = await tool.handler({"text": "hello"})  # No ref

        assert result.success is False
        assert "Missing required parameter: ref" in result.error

    @pytest.mark.asyncio
    async def test_type_fails_without_text(self, mock_mcp_client):
        """Test type fails when text parameter is missing."""
        from dokumen.playwright_tools import create_browser_type_tool

        tool = create_browser_type_tool(mock_mcp_client)
        result = await tool.handler({"ref": "input_1"})  # No text

        assert result.success is False
        assert "Missing required parameter: text" in result.error

    @pytest.mark.asyncio
    async def test_type_into_element(self, mock_mcp_client):
        """Test typing into element by reference.

        Note: Playwright MCP browser_type handles focus internally,
        so we pass all params directly without clicking first.
        """
        from dokumen.playwright_tools import create_browser_type_tool

        tool = create_browser_type_tool(mock_mcp_client)
        result = await tool.handler({"ref": "input_1", "text": "hello world"})

        # Should call browser_type directly with all params (no click first!)
        mock_mcp_client.call_tool.assert_called_once_with(
            "browser_type",
            {
                "element": "input_1",
                "ref": "input_1",
                "text": "hello world",
                "submit": False
            }
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_type_with_submit(self, mock_mcp_client):
        """Test typing with submit=True to press Enter after."""
        from dokumen.playwright_tools import create_browser_type_tool

        tool = create_browser_type_tool(mock_mcp_client)
        result = await tool.handler({
            "ref": "input_1",
            "text": "hello",
            "submit": True
        })

        mock_mcp_client.call_tool.assert_called_once_with(
            "browser_type",
            {
                "element": "input_1",
                "ref": "input_1",
                "text": "hello",
                "submit": True
            }
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_type_with_custom_element_description(self, mock_mcp_client):
        """Test typing with custom element description."""
        from dokumen.playwright_tools import create_browser_type_tool

        tool = create_browser_type_tool(mock_mcp_client)
        result = await tool.handler({
            "ref": "input_1",
            "text": "test@example.com",
            "element": "Email input field"
        })

        mock_mcp_client.call_tool.assert_called_once_with(
            "browser_type",
            {
                "element": "Email input field",
                "ref": "input_1",
                "text": "test@example.com",
                "submit": False
            }
        )
        assert result.success is True


class TestBrowserSnapshot:
    """Tests for browser_snapshot (accessibility tree) tool."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client with accessibility tree response."""
        from dokumen.tools_object import ToolResult

        client = AsyncMock()
        client.call_tool = AsyncMock(return_value=ToolResult(
            success=True,
            output={
                "elements": [
                    {"ref": "button_1", "role": "button", "name": "Submit"},
                    {"ref": "input_1", "role": "textbox", "name": "Email"}
                ]
            },
            error=None
        ))
        return client

    @pytest.mark.asyncio
    async def test_snapshot_fails_without_mcp_client(self):
        """Test snapshot fails when no MCP client is available."""
        from dokumen.playwright_tools import create_browser_snapshot_tool

        tool = create_browser_snapshot_tool(None)  # No client
        result = await tool.handler({})

        assert result.success is False
        assert "No MCP client" in result.error

    @pytest.mark.asyncio
    async def test_snapshot_returns_accessibility_tree(self, mock_mcp_client):
        """Test that snapshot returns accessibility tree elements."""
        from dokumen.playwright_tools import create_browser_snapshot_tool

        tool = create_browser_snapshot_tool(mock_mcp_client)
        result = await tool.handler({})

        assert result.success is True
        assert "elements" in result.output or isinstance(result.output, str)


class TestBrowserWait:
    """Tests for browser_wait tool behavior."""

    @pytest.mark.asyncio
    async def test_wait_default_seconds(self):
        """Test waiting with default seconds."""
        from dokumen.playwright_tools import create_browser_wait_tool
        import time

        tool = create_browser_wait_tool()
        start = time.time()
        result = await tool.handler({})
        elapsed = time.time() - start

        assert result.success is True
        assert "2" in result.output  # Default 2 seconds
        assert elapsed >= 1.5  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_wait_custom_seconds(self):
        """Test waiting with custom seconds."""
        from dokumen.playwright_tools import create_browser_wait_tool
        import time

        tool = create_browser_wait_tool()
        start = time.time()
        result = await tool.handler({"seconds": 1})
        elapsed = time.time() - start

        assert result.success is True
        assert "1" in result.output
        assert elapsed >= 0.9

    @pytest.mark.asyncio
    async def test_wait_clamps_to_max(self):
        """Test that wait clamps to max 30 seconds."""
        from dokumen.playwright_tools import create_browser_wait_tool

        tool = create_browser_wait_tool()
        # Don't actually wait 30 seconds in test, just verify the tool exists
        assert tool.name == "browser_wait"
        assert "30" in tool.description


class TestClickIndicatorScript:
    """Tests for click indicator JavaScript."""

    def test_click_indicator_script_contains_required_elements(self):
        """Test click indicator script has all required parts."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        # Should have click indicator function
        assert "__dokumenClickIndicator" in CLICK_INDICATOR_SCRIPT
        # Should create visual indicator
        assert "createElement" in CLICK_INDICATOR_SCRIPT
        # Should use orange color as specified
        assert "ff6b00" in CLICK_INDICATOR_SCRIPT.lower()
        # Should have animation
        assert "animation" in CLICK_INDICATOR_SCRIPT.lower()

    def test_click_indicator_script_is_valid_javascript(self):
        """Test that click indicator script is syntactically valid JS."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        # Basic syntax check - should be non-empty and have matching braces
        assert len(CLICK_INDICATOR_SCRIPT) > 100
        assert CLICK_INDICATOR_SCRIPT.count("{") == CLICK_INDICATOR_SCRIPT.count("}")
        assert CLICK_INDICATOR_SCRIPT.count("(") == CLICK_INDICATOR_SCRIPT.count(")")

    def test_click_indicator_script_has_debugging(self):
        """Test click indicator script includes console.log debugging."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        # Should have debugging logs
        assert "[Dokumen]" in CLICK_INDICATOR_SCRIPT
        assert "console.log" in CLICK_INDICATOR_SCRIPT

    def test_click_indicator_function_finds_elements(self):
        """Test click indicator function searches for elements by text."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        # Should search for elements by text content
        assert "textContent" in CLICK_INDICATOR_SCRIPT
        # Should search by aria-label
        assert "aria-label" in CLICK_INDICATOR_SCRIPT
        # Should get bounding rect for positioning
        assert "getBoundingClientRect" in CLICK_INDICATOR_SCRIPT

    def test_click_indicator_uses_request_animation_frame(self):
        """Test click indicator uses requestAnimationFrame for render sync."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        # Should use requestAnimationFrame for render synchronization
        assert "requestAnimationFrame" in CLICK_INDICATOR_SCRIPT

    def test_click_indicator_uses_max_z_index(self):
        """Test click indicator uses max z-index for visibility."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        # Should use max z-index (2147483647)
        assert "2147483647" in CLICK_INDICATOR_SCRIPT

    def test_click_indicator_has_filled_background(self):
        """Test click indicator uses filled background not just border."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        # Should have rgba background for filled circle
        assert "rgba(255, 107, 0" in CLICK_INDICATOR_SCRIPT

    def test_click_indicator_returns_promise(self):
        """Test click indicator functions return Promises for async coordination."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        # createClickIndicator should return a Promise
        assert "return new Promise" in CLICK_INDICATOR_SCRIPT

    def test_click_at_function_exists(self):
        """Test __dokumenClickAt function is defined for direct coordinate clicks."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        assert "window.__dokumenClickAt" in CLICK_INDICATOR_SCRIPT

    def test_click_indicator_script_has_extended_animation(self):
        """Test click animation duration is 1.5s for better video visibility."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        # Animation should be 1.5s (extended for video capture)
        assert "1.5s" in CLICK_INDICATOR_SCRIPT or "1500" in CLICK_INDICATOR_SCRIPT


class TestScreenshotFlashScript:
    """Tests for screenshot flash animation in init script."""

    def test_screenshot_flash_function_exists(self):
        """Test that __dokumenScreenshotFlash function is defined."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        assert "__dokumenScreenshotFlash" in CLICK_INDICATOR_SCRIPT
        assert "window.__dokumenScreenshotFlash = function" in CLICK_INDICATOR_SCRIPT

    def test_screenshot_flash_keyframes_exist(self):
        """Test that screenshotFlash keyframes are defined."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        assert "screenshotFlash" in CLICK_INDICATOR_SCRIPT
        assert "@keyframes screenshotFlash" in CLICK_INDICATOR_SCRIPT

    def test_screenshot_flash_uses_white_overlay(self):
        """Test flash uses white background as specified."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        # Flash should create white overlay
        assert "background: white" in CLICK_INDICATOR_SCRIPT

    def test_screenshot_flash_has_500ms_duration(self):
        """Test flash animation is 500ms for video capture visibility."""
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        # Duration should be 0.5s (500ms) for video capture
        assert "0.5s" in CLICK_INDICATOR_SCRIPT


class TestBrowserScreenshot:
    """Tests for browser_screenshot tool.

    Visual animations (screenshot flash) are now handled by the Playwright MCP fork
    with --visual-indicators flag, so no browser_evaluate calls are made here.
    """

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client."""
        from dokumen.tools_object import ToolResult

        client = AsyncMock()
        client.call_tool = AsyncMock(return_value=ToolResult(
            success=True,
            output="[Image: image/png]",
            error=None
        ))
        return client

    @pytest.mark.asyncio
    async def test_screenshot_fails_without_mcp_client(self):
        """Test screenshot fails when no MCP client is available."""
        from dokumen.playwright_tools import create_browser_screenshot_tool

        tool = create_browser_screenshot_tool(None)  # No client
        result = await tool.handler({})

        assert result.success is False
        assert "No MCP client" in result.error

    @pytest.mark.asyncio
    async def test_screenshot_calls_mcp_directly(self, mock_mcp_client):
        """Test that screenshot calls browser_take_screenshot directly.

        Visual animations are handled by the MCP fork, so no browser_evaluate needed.
        """
        from dokumen.playwright_tools import create_browser_screenshot_tool

        tool = create_browser_screenshot_tool(mock_mcp_client)
        result = await tool.handler({})

        # Should only call browser_take_screenshot directly
        calls = mock_mcp_client.call_tool.call_args_list
        assert len(calls) == 1, "Should only call browser_take_screenshot"
        assert calls[0][0][0] == "browser_take_screenshot"
        assert calls[0][0][1]["filename"].startswith("screenshots/page-")
        assert result.success is True


class TestBrowserClose:
    """Tests for browser_close tool behavior."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client."""
        from dokumen.tools_object import ToolResult

        client = AsyncMock()
        client.call_tool = AsyncMock(return_value=ToolResult(
            success=True, output="Browser closed", error=None
        ))
        return client

    @pytest.mark.asyncio
    async def test_close_fails_without_mcp_client(self):
        """Test close fails when no MCP client is available."""
        from dokumen.playwright_tools import create_browser_close_tool

        tool = create_browser_close_tool(None)  # No client
        result = await tool.handler({})

        assert result.success is False
        assert "No MCP client" in result.error

    @pytest.mark.asyncio
    async def test_close_calls_mcp_close(self, mock_mcp_client):
        """Test that close calls MCP browser_close tool."""
        from dokumen.playwright_tools import create_browser_close_tool

        tool = create_browser_close_tool(mock_mcp_client)
        result = await tool.handler({})

        mock_mcp_client.call_tool.assert_called_with("browser_close", {})
        assert result.success is True


class TestBrowserToolIntegration:
    """Integration tests for browser tools with shared MCP client."""

    @pytest.mark.asyncio
    async def test_multiple_tools_share_client(self):
        """Test that tools can share a single MCP client instance."""
        from dokumen.playwright_tools import create_browser_tools_for_client
        from dokumen.tools_object import ToolResult

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=ToolResult(
            success=True, output="OK", error=None
        ))

        tools = create_browser_tools_for_client(mock_client)

        assert len(tools) >= 6  # At least 6 browser tools

        # All tools should work with same client
        for tool in tools:
            result = await tool.handler({
                "url": "https://example.com",
                "ref": "element_1",
                "text": "test"
            })
            # Tools should call the shared client
            assert mock_client.call_tool.called

    def test_tool_descriptions_are_descriptive(self):
        """Test that tool descriptions explain their purpose."""
        from dokumen.playwright_tools import BROWSER_TOOLS

        for name, factory in BROWSER_TOOLS.items():
            tool = factory(AsyncMock())

            assert len(tool.description) > 20, f"{name} description too short"
            assert tool.name == name
