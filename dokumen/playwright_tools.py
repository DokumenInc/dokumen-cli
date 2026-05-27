"""
Playwright browser automation tools for Dokumen CLI.

Provides tool definitions that wrap the Playwright MCP server for browser automation
with video recording and click indicators.
"""
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .tools_object import ToolDefinition, ToolResult
if TYPE_CHECKING:
    from .mcp_client import PlaywrightMCPClient

logger = logging.getLogger(__name__)


# JavaScript to inject click indicators and screenshot flash for video recording
# Uses requestAnimationFrame for render synchronization and longer durations for video capture
CLICK_INDICATOR_SCRIPT = """
(function() {
    console.log('[Dokumen] ========================================');
    console.log('[Dokumen] Click indicator script initializing...');
    console.log('[Dokumen] Document readyState:', document.readyState);
    console.log('[Dokumen] Document head exists:', !!document.head);
    console.log('[Dokumen] Document body exists:', !!document.body);

    // Add animation keyframes with longer durations for video capture
    if (!document.getElementById('dokumen-click-indicator-styles')) {
        console.log('[Dokumen] Injecting styles...');
        const style = document.createElement('style');
        style.id = 'dokumen-click-indicator-styles';
        style.textContent = `
            @keyframes clickPulse {
                0% { transform: scale(1); opacity: 0.8; }
                100% { transform: scale(3); opacity: 0; }
            }
            @keyframes screenshotFlash {
                0% { opacity: 0.8; }
                100% { opacity: 0; }
            }
        `;
        document.head.appendChild(style);
        console.log('[Dokumen] Styles injected successfully');
        console.log('[Dokumen] Style element in DOM:', !!document.getElementById('dokumen-click-indicator-styles'));
    } else {
        console.log('[Dokumen] Styles already exist, skipping injection');
    }

    // Helper to create click animation at specific coordinates
    // Returns a Promise that resolves after the animation is rendered
    function createClickIndicator(x, y) {
        console.log('[Dokumen] ----------------------------------------');
        console.log('[Dokumen] createClickIndicator called');
        console.log('[Dokumen] x:', x, 'y:', y);

        return new Promise(function(resolve) {
            const circle = document.createElement('div');
            console.log('[Dokumen] Circle element created');

            // Larger circle (60px), filled background, max z-index
            circle.style.cssText = `
                position: fixed;
                left: ${x - 30}px;
                top: ${y - 30}px;
                width: 60px;
                height: 60px;
                background: rgba(255, 107, 0, 0.6);
                border: 4px solid #ff6b00;
                border-radius: 50%;
                pointer-events: none;
                z-index: 2147483647;
                animation: clickPulse 1.5s ease-out forwards;
            `;
            console.log('[Dokumen] Circle styles applied');

            document.body.appendChild(circle);
            console.log('[Dokumen] Circle appended to body');
            console.log('[Dokumen] Circle in DOM:', !!circle.parentNode);

            // Wait for browser to render using double requestAnimationFrame
            requestAnimationFrame(function() {
                console.log('[Dokumen] First rAF callback');
                requestAnimationFrame(function() {
                    console.log('[Dokumen] Second rAF callback - animation should be visible now');
                    try {
                        console.log('[Dokumen] Circle bounding rect:', JSON.stringify(circle.getBoundingClientRect()));
                        console.log('[Dokumen] Circle computed style animation:', window.getComputedStyle(circle).animation);
                    } catch(e) {
                        console.log('[Dokumen] Could not get computed style:', e.message);
                    }
                    resolve(true);
                });
            });

            // Remove after animation completes (1.5s)
            setTimeout(function() {
                circle.remove();
                console.log('[Dokumen] Circle removed after timeout');
            }, 1500);
        });
    }

    // Click event listener for fallback (captures actual clicks)
    document.addEventListener('click', function(e) {
        console.log('[Dokumen] ----------------------------------------');
        console.log('[Dokumen] CLICK EVENT FIRED');
        console.log('[Dokumen] clientX:', e.clientX);
        console.log('[Dokumen] clientY:', e.clientY);
        console.log('[Dokumen] target:', e.target.tagName);
        // Only trigger if we have valid coordinates
        if (e.clientX > 0 || e.clientY > 0) {
            console.log('[Dokumen] Creating indicator from click event...');
            createClickIndicator(e.clientX, e.clientY);
        }
    }, true);

    // Also listen on mousedown for earlier visual feedback
    document.addEventListener('mousedown', function(e) {
        if (e.clientX > 0 || e.clientY > 0) {
            console.log('[Dokumen] Mousedown at', e.clientX, e.clientY);
            createClickIndicator(e.clientX, e.clientY);
        }
    }, true);

    // Expose async function that waits for render - called from browser_click
    window.__dokumenClickAt = function(x, y) {
        console.log('[Dokumen] __dokumenClickAt called with x:', x, 'y:', y);
        return createClickIndicator(x, y);
    };

    // Click indicator function - called before clicking with element description
    // Returns a Promise for render synchronization
    window.__dokumenClickIndicator = function(elementDesc) {
        console.log('[Dokumen] ----------------------------------------');
        console.log('[Dokumen] __dokumenClickIndicator called for:', elementDesc);

        // Try multiple strategies to find the element
        let element = null;
        const desc = elementDesc.toLowerCase();

        // Strategy 1: Find by text content (buttons, links, labels)
        console.log('[Dokumen] Strategy 1: Finding by text content...');
        const clickables = document.querySelectorAll('a, button, [role="button"], [role="link"], input[type="submit"], label');
        console.log('[Dokumen] Found', clickables.length, 'clickable elements');
        for (const el of clickables) {
            if (el.textContent && el.textContent.trim().toLowerCase().includes(desc)) {
                element = el;
                console.log('[Dokumen] Found element by text:', el.tagName, el.textContent.trim().substring(0, 50));
                break;
            }
        }

        // Strategy 2: Find input by placeholder
        if (!element) {
            console.log('[Dokumen] Strategy 2: Finding by placeholder...');
            const inputs = document.querySelectorAll('input, textarea');
            for (const el of inputs) {
                if (el.placeholder && el.placeholder.toLowerCase().includes(desc)) {
                    element = el;
                    console.log('[Dokumen] Found element by placeholder:', el.placeholder);
                    break;
                }
            }
        }

        // Strategy 3: Find by aria-label
        if (!element) {
            console.log('[Dokumen] Strategy 3: Finding by aria-label...');
            try {
                element = document.querySelector('[aria-label*="' + elementDesc.replace(/"/g, '\\"') + '" i]');
                if (element) console.log('[Dokumen] Found element by aria-label');
            } catch(e) {
                console.log('[Dokumen] aria-label search failed:', e.message);
            }
        }

        // Strategy 4: Find by title attribute
        if (!element) {
            console.log('[Dokumen] Strategy 4: Finding by title...');
            try {
                element = document.querySelector('[title*="' + elementDesc.replace(/"/g, '\\"') + '" i]');
                if (element) console.log('[Dokumen] Found element by title');
            } catch(e) {}
        }

        // Strategy 5: Find by name attribute
        if (!element) {
            console.log('[Dokumen] Strategy 5: Finding by name...');
            try {
                element = document.querySelector('[name*="' + elementDesc.replace(/"/g, '\\"') + '" i]');
                if (element) console.log('[Dokumen] Found element by name');
            } catch(e) {}
        }

        if (element) {
            const rect = element.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            console.log('[Dokumen] Element found, creating indicator at center:', centerX, centerY);
            console.log('[Dokumen] Element rect:', JSON.stringify(rect));
            return createClickIndicator(centerX, centerY);
        }

        console.log('[Dokumen] Could not find element:', elementDesc);
        return Promise.resolve(false);
    };

    // Screenshot flash function - called before taking screenshots
    // Returns a Promise for render synchronization
    window.__dokumenScreenshotFlash = function() {
        console.log('[Dokumen] ========================================');
        console.log('[Dokumen] SCREENSHOT FLASH TRIGGERED');

        return new Promise(function(resolve) {
            const overlay = document.createElement('div');
            console.log('[Dokumen] Flash overlay created');

            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: white;
                opacity: 0.8;
                pointer-events: none;
                z-index: 2147483647;
                animation: screenshotFlash 0.5s ease-out forwards;
            `;
            console.log('[Dokumen] Flash styles applied');

            document.body.appendChild(overlay);
            console.log('[Dokumen] Flash overlay appended to body');

            // Wait for browser to render
            requestAnimationFrame(function() {
                console.log('[Dokumen] Flash first rAF callback');
                requestAnimationFrame(function() {
                    console.log('[Dokumen] Flash second rAF callback - flash should be visible now');
                    resolve(true);
                });
            });

            // Remove after animation completes (0.5s)
            setTimeout(function() {
                overlay.remove();
                console.log('[Dokumen] Flash overlay removed after timeout');
            }, 500);
        });
    };

    console.log('[Dokumen] Click indicator script fully loaded');
    console.log('[Dokumen] window.__dokumenClickAt exists:', typeof window.__dokumenClickAt);
    console.log('[Dokumen] window.__dokumenClickIndicator exists:', typeof window.__dokumenClickIndicator);
    console.log('[Dokumen] window.__dokumenScreenshotFlash exists:', typeof window.__dokumenScreenshotFlash);
    console.log('[Dokumen] ========================================');
})();
"""


# Module-level shared MCP client (set by test runner)
_shared_mcp_client: Optional["PlaywrightMCPClient"] = None


def set_shared_mcp_client(client: "PlaywrightMCPClient") -> None:
    """Set the shared MCP client for browser tools.

    Called by the test runner to share a single browser instance between
    executor and judge agents.
    """
    global _shared_mcp_client
    _shared_mcp_client = client
    logger.info("Shared MCP client set for browser tools")


def get_shared_mcp_client() -> Optional["PlaywrightMCPClient"]:
    """Get the shared MCP client."""
    return _shared_mcp_client


def clear_shared_mcp_client() -> None:
    """Clear the shared MCP client."""
    global _shared_mcp_client
    _shared_mcp_client = None


def create_browser_navigate_tool(
    mcp_client: Optional["PlaywrightMCPClient"] = None
) -> ToolDefinition:
    """Create browser_navigate tool that navigates to URL and injects click indicators.

    Args:
        mcp_client: Optional MCP client (uses shared client if not provided)
    """
    async def handler(params: Dict[str, Any]) -> ToolResult:
        client = mcp_client or _shared_mcp_client
        if client is None:
            return ToolResult(
                success=False,
                output=None,
                error="No MCP client available - browser not initialized"
            )

        url = params.get("url")
        if not url:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: url"
            )

        logger.info("Navigating browser", extra={"url": url})

        # Navigate to URL
        # Note: Click indicator script is injected automatically via --init-script flag
        # when the MCP server starts (see mcp_client.py start())
        result = await client.call_tool("browser_navigate", {"url": url})
        if not result.success:
            return result

        return ToolResult(
            success=True,
            output=f"Navigated to {url}. Click indicators enabled for video recording.",
            error=None
        )

    return ToolDefinition(
        name="browser_navigate",
        description="Navigate the browser to a URL. Automatically injects click indicators for video recording.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to"
                }
            },
            "required": ["url"]
        },
        handler=handler
    )


def create_browser_click_tool(
    mcp_client: Optional["PlaywrightMCPClient"] = None
) -> ToolDefinition:
    """Create browser_click tool that clicks an element by accessibility ref.

    Note: Playwright MCP requires BOTH 'element' (human-readable description)
    AND 'ref' (element reference) parameters.

    Visual animations (click circles) are now handled by the Playwright MCP fork
    with --visual-indicators flag, so no browser_evaluate calls needed here.
    """
    async def handler(params: Dict[str, Any]) -> ToolResult:
        client = mcp_client or _shared_mcp_client
        if client is None:
            logger.error("browser_click.no_client")
            return ToolResult(
                success=False,
                output=None,
                error="No MCP client available - browser not initialized"
            )

        ref = params.get("ref")
        if not ref:
            logger.error("browser_click.missing_ref")
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: ref"
            )

        # Use provided element description, or default to ref
        element = params.get("element", ref)

        logger.info("browser_click.start", extra={"ref": ref, "element": element})

        # Click the element - visual animations are handled by MCP fork
        result = await client.call_tool("browser_click", {
            "element": element,
            "ref": ref
        })

        logger.info("browser_click.complete", extra={
            "success": result.success,
            "ref": ref,
            "error": result.error
        })

        return result

    return ToolDefinition(
        name="browser_click",
        description="Click an element by its accessibility reference from browser_snapshot.",
        parameters={
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Element reference from browser_snapshot (e.g., 'button_1', 'input_2')"
                },
                "element": {
                    "type": "string",
                    "description": "Human-readable element description (optional, defaults to ref)"
                }
            },
            "required": ["ref"]
        },
        handler=handler
    )


def create_browser_type_tool(
    mcp_client: Optional["PlaywrightMCPClient"] = None
) -> ToolDefinition:
    """Create browser_type tool that types text into an element.

    Note: Playwright MCP browser_type handles element focus internally,
    so we pass all params directly without clicking first.
    """
    async def handler(params: Dict[str, Any]) -> ToolResult:
        client = mcp_client or _shared_mcp_client
        if client is None:
            return ToolResult(
                success=False,
                output=None,
                error="No MCP client available - browser not initialized"
            )

        ref = params.get("ref")
        text = params.get("text")

        if not ref:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: ref"
            )
        if text is None:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: text"
            )

        # Use provided element description, or default to ref
        element = params.get("element", ref)
        submit = params.get("submit", False)

        logger.info("Typing into element", extra={
            "ref": ref,
            "element": element,
            "text_length": len(text),
            "submit": submit
        })

        # Call browser_type directly with all params (no click first!)
        # Playwright MCP handles focus internally
        result = await client.call_tool("browser_type", {
            "element": element,
            "ref": ref,
            "text": text,
            "submit": submit
        })
        return result

    return ToolDefinition(
        name="browser_type",
        description="Type text into an element using its accessibility reference.",
        parameters={
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Element reference from browser_snapshot (e.g., 'input_1')"
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the element"
                },
                "element": {
                    "type": "string",
                    "description": "Human-readable element description (optional, defaults to ref)"
                },
                "submit": {
                    "type": "boolean",
                    "description": "Press Enter after typing (optional, defaults to false)"
                }
            },
            "required": ["ref", "text"]
        },
        handler=handler
    )


def create_browser_screenshot_tool(
    mcp_client: Optional["PlaywrightMCPClient"] = None
) -> ToolDefinition:
    """Create browser_screenshot tool that captures the current page.

    Visual animations (screenshot flash) are now handled by the Playwright MCP fork
    with --visual-indicators flag, so no browser_evaluate calls needed here.
    """
    from datetime import datetime

    async def handler(params: Dict[str, Any]) -> ToolResult:
        client = mcp_client or _shared_mcp_client
        if client is None:
            logger.error("browser_screenshot.no_client")
            return ToolResult(
                success=False,
                output=None,
                error="No MCP client available - browser not initialized"
            )

        logger.info("browser_screenshot.start")

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"screenshots/page-{timestamp}.png"
        payload = {"type": "png", "filename": filename}

        # Take screenshot - visual animations are handled by MCP fork
        result = await client.call_tool("browser_take_screenshot", payload)

        logger.info("browser_screenshot.complete", extra={
            "success": result.success,
            "error": result.error,
            "screenshot_filename": filename
        })

        return result

    return ToolDefinition(
        name="browser_screenshot",
        description="Take a screenshot of the current browser page.",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        handler=handler
    )


def create_browser_take_screenshot_tool(
    mcp_client: Optional["PlaywrightMCPClient"] = None
) -> ToolDefinition:
    """Create browser_take_screenshot tool that saves an image to disk."""
    from datetime import datetime

    async def handler(params: Dict[str, Any]) -> ToolResult:
        client = mcp_client or _shared_mcp_client
        if client is None:
            logger.error("browser_take_screenshot.no_client")
            return ToolResult(
                success=False,
                output=None,
                error="No MCP client available - browser not initialized"
            )

        filename = params.get("filename")
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            filename = f"screenshots/page-{timestamp}.png"

        payload = {
            "type": params.get("type", "png"),
            "filename": filename,
        }

        if params.get("fullPage") is not None:
            payload["fullPage"] = params.get("fullPage")
        if params.get("ref") is not None:
            payload["ref"] = params.get("ref")
        if params.get("element") is not None:
            payload["element"] = params.get("element")

        logger.info("browser_take_screenshot.start", extra={"screenshot_filename": filename})

        result = await client.call_tool("browser_take_screenshot", payload)

        logger.info("browser_take_screenshot.complete", extra={
            "success": result.success,
            "error": result.error,
            "screenshot_filename": filename
        })

        return result

    return ToolDefinition(
        name="browser_take_screenshot",
        description="Take a screenshot and save it to disk. Defaults to screenshots/page-<timestamp>.png",
        parameters={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["png", "jpeg"],
                    "description": "Image format (default: png)"
                },
                "filename": {
                    "type": "string",
                    "description": "Relative file name to save under output dir"
                },
                "element": {
                    "type": "string",
                    "description": "Human-readable element description (optional, used with ref)"
                },
                "ref": {
                    "type": "string",
                    "description": "Element reference from browser_snapshot (optional)"
                },
                "fullPage": {
                    "type": "boolean",
                    "description": "Capture full page instead of viewport (optional)"
                }
            },
            "required": []
        },
        handler=handler
    )


def create_browser_snapshot_tool(
    mcp_client: Optional["PlaywrightMCPClient"] = None
) -> ToolDefinition:
    """Create browser_snapshot tool that returns the accessibility tree."""
    async def handler(params: Dict[str, Any]) -> ToolResult:
        client = mcp_client or _shared_mcp_client
        if client is None:
            return ToolResult(
                success=False,
                output=None,
                error="No MCP client available - browser not initialized"
            )

        logger.info("Getting page accessibility snapshot")

        result = await client.call_tool("browser_snapshot", {})
        return result

    return ToolDefinition(
        name="browser_snapshot",
        description="Get the accessibility tree of the current page. Returns element references that can be used with browser_click and browser_type.",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        handler=handler
    )


def create_browser_wait_tool(
    mcp_client: Optional["PlaywrightMCPClient"] = None
) -> ToolDefinition:
    """Create browser_wait tool that waits for a specified duration.

    This is useful for waiting for page transitions, animations, or async operations.
    """
    import asyncio

    async def handler(params: Dict[str, Any]) -> ToolResult:
        seconds = params.get("seconds", 2)

        # Clamp to reasonable range
        if seconds < 0:
            seconds = 0
        if seconds > 30:
            seconds = 30

        logger.info("Waiting", extra={"seconds": seconds})

        await asyncio.sleep(seconds)

        return ToolResult(
            success=True,
            output=f"Waited {seconds} seconds",
            error=None
        )

    return ToolDefinition(
        name="browser_wait",
        description="Wait for a specified number of seconds. Useful for waiting for page loads, animations, or async operations. Max 30 seconds.",
        parameters={
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Number of seconds to wait (default: 2, max: 30)"
                }
            },
            "required": []
        },
        handler=handler
    )


def create_browser_close_tool(
    mcp_client: Optional["PlaywrightMCPClient"] = None
) -> ToolDefinition:
    """Create browser_close tool that closes the browser."""
    async def handler(params: Dict[str, Any]) -> ToolResult:
        client = mcp_client or _shared_mcp_client
        if client is None:
            return ToolResult(
                success=False,
                output=None,
                error="No MCP client available - browser not initialized"
            )

        logger.info("Closing browser")

        result = await client.call_tool("browser_close", {})
        return result

    return ToolDefinition(
        name="browser_close",
        description="Close the browser. Call this when done with browser automation.",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        handler=handler
    )


# Browser tool registry for easy access
BROWSER_TOOLS = {
    "browser_navigate": create_browser_navigate_tool,
    "browser_click": create_browser_click_tool,
    "browser_type": create_browser_type_tool,
    "browser_screenshot": create_browser_screenshot_tool,
    "browser_take_screenshot": create_browser_take_screenshot_tool,
    "browser_snapshot": create_browser_snapshot_tool,
    "browser_wait": create_browser_wait_tool,
    "browser_close": create_browser_close_tool,
}


def get_browser_tool_names() -> list:
    """Get list of all browser tool names."""
    return list(BROWSER_TOOLS.keys())


def create_browser_tools_for_client(
    mcp_client: "PlaywrightMCPClient"
) -> List[ToolDefinition]:
    """Create all browser tools with a shared MCP client.

    Args:
        mcp_client: The MCP client to use for all browser tools

    Returns:
        List of ToolDefinition objects for all browser tools
    """
    tools = []
    for name, factory in BROWSER_TOOLS.items():
        tools.append(factory(mcp_client))
    return tools
