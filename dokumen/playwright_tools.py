"""Browser tool registry for SDK-managed Playwright MCP integration.

Dokumen no longer wraps Playwright MCP tools with its own client. The CLI only
needs a canonical list of browser tool names so scaffold validation, tool
auto-injection, and SDK allowed-tool construction all agree on the same public
surface.
"""

from dokumen_schema.constants import BROWSER_TOOLS as SCHEMA_BROWSER_TOOLS

BROWSER_TOOL_NAMES = (
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_snapshot",
    "browser_screenshot",
    "browser_take_screenshot",
    "browser_wait",
    "browser_close",
    "browser_evaluate",
)

# Kept as a dict for compatibility with existing membership checks and tests
# that patch the browser registry. Values are intentionally unused because the
# Claude Agent SDK talks to the external Playwright MCP server directly.
BROWSER_TOOLS = {name: None for name in BROWSER_TOOL_NAMES}

if set(BROWSER_TOOL_NAMES) != set(SCHEMA_BROWSER_TOOLS):
    raise RuntimeError(
        "Browser tool registry is out of sync with dokumen_schema.constants.BROWSER_TOOLS"
    )


def get_browser_tool_names() -> list[str]:
    """Return browser tool names in stable auto-injection order."""
    return list(BROWSER_TOOL_NAMES)
