"""
Tools Object module for the Documentation Unit Test Framework.

Provides tool definitions for agents, with support for multiple LLM provider formats.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Awaitable, TYPE_CHECKING
from datetime import datetime
import asyncio
import math
import os

from .logging_config import get_logger

if TYPE_CHECKING:
    from .sandbox import Sandbox

# Module-level logger
logger = get_logger(__name__)

@dataclass
class ToolResult:
    """Result from executing a tool."""

    success: bool
    output: Any
    error: Optional[str] = None


@dataclass
class ToolCall:
    """Record of a tool invocation."""

    tool_name: str
    parameters: Dict[str, Any]
    result: ToolResult
    timestamp: datetime
    duration: float  # seconds


@dataclass
class SubagentResult:
    """Result from a single subagent execution."""

    file_path: str
    start_line: int
    end_line: int
    goal: str
    success: bool
    response: str
    tool_calls: List[Dict[str, Any]]
    covered_lines: List[int] = field(default_factory=list)  # Lines identified as important
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "lines": f"{self.start_line}-{self.end_line}",
            "goal": self.goal,
            "success": self.success,
            "response": self.response,
            "tool_calls": self.tool_calls,
            "covered_lines": self.covered_lines,
            "error": self.error,
        }


# Type alias for tool handlers
ToolHandler = Callable[[Dict[str, Any]], Awaitable[ToolResult]]


@dataclass
class ToolDefinition:
    """Definition of a tool available to agents."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: ToolHandler


class ToolsObject:
    """Manages tool definitions and execution."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool.

        Args:
            tool: The tool definition to register

        Raises:
            ValueError: If a tool with the same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already exists: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> bool:
        """Remove a tool from the registry.

        Args:
            tool_name: Name of the tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name.

        Args:
            tool_name: Name of the tool

        Returns:
            The tool definition or None if not found
        """
        return self._tools.get(tool_name)

    async def execute(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with provided parameters.

        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool handler

        Returns:
            ToolResult with success status and output or error
        """
        logger.debug("tools.execute.start", tool_name=tool_name, params=params)
        if tool_name not in self._tools:
            logger.warning("tools.execute.not_found", tool_name=tool_name)
            return ToolResult(success=False, output=None, error=f"Tool not found: {tool_name}")

        tool = self._tools[tool_name]

        try:
            # Validate parameters against schema
            self._validate_params(tool.parameters, params)

            # Execute the handler
            import time

            start = time.time()
            result = await tool.handler(params)
            duration = time.time() - start
            logger.info(
                "tools.execute.complete",
                tool_name=tool_name,
                success=result.success,
                duration_ms=int(duration * 1000),
            )
            return result
        except Exception as e:
            logger.error("tools.execute.error", tool_name=tool_name, error=str(e))
            return ToolResult(success=False, output=None, error=str(e))

    def get_definitions(self) -> List[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def to_anthropic_format(self) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic tool use format."""
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
            for tool in self._tools.values()
        ]

    def to_fastmcp_format(self) -> List[Dict[str, Any]]:
        """Convert tools to fastMCP-compatible format."""
        return [
            {"name": tool.name, "description": tool.description, "inputSchema": tool.parameters}
            for tool in self._tools.values()
        ]

    @staticmethod
    def _validate_params(schema: Dict[str, Any], params: Dict[str, Any]) -> None:
        """Validate parameters against JSON schema.

        Args:
            schema: JSON schema for parameters
            params: Parameters to validate

        Raises:
            ValueError: If required parameter is missing
        """
        required = schema.get("required", [])
        for req in required:
            if req not in params:
                raise ValueError(f"Missing required parameter: {req}")


def create_bash_tool(
    sandbox: Optional["Sandbox"] = None, timeout: float = 30.0, base_dir: str = "."
) -> ToolDefinition:
    """Create a bash tool that executes shell commands.

    Args:
        sandbox: Optional sandbox to execute commands in. If None, runs directly.
        timeout: Command timeout in seconds
        base_dir: Base directory for tool description (helps AI understand working dir)

    Returns:
        ToolDefinition for the bash tool
    """
    import re

    # Patterns that indicate attempts to escape workspace
    RESTRICTED_PATTERNS = [
        r"\bfind\s+/",  # find / (searching from root)
        r"\bls\s+/",  # ls / (listing from root)
        r"\bcat\s+/",  # cat /etc/passwd etc
        r"\bhead\s+/",  # head /etc/passwd etc
        r"\btail\s+/",  # tail /var/log/syslog etc
        r"\bcd\s+/",  # cd /home etc
        r"\.\./",  # ../../../ path traversal
        r"/etc\b",  # /etc access
        r"/usr\b",  # /usr access
        r"/var\b",  # /var access
        r"/home\b",  # /home access
        r"/root\b",  # /root access
        r"/proc\b",  # /proc access
        r"/sys\b",  # /sys access
    ]

    def _is_restricted_command(cmd: str) -> Optional[str]:
        """Check if command tries to access paths outside /workspace."""
        for pattern in RESTRICTED_PATTERNS:
            if re.search(pattern, cmd):
                return f"Command attempts to access paths outside /workspace. Use relative paths or paths starting with /workspace/. Matched pattern: {pattern}"
        return None

    async def bash_handler(params: Dict[str, Any]) -> ToolResult:
        command = params.get("command")
        # Resolve per-invocation timeout: clamp to [1.0, config_timeout]
        model_timeout = params.get("timeout")
        effective_timeout = timeout  # closure default from config
        if model_timeout is not None:
            try:
                val = float(model_timeout)
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite timeout: {model_timeout}")
                effective_timeout = max(1.0, min(val, timeout))
                logger.debug(
                    "Per-invocation timeout resolved",
                    extra={
                        "model_requested": model_timeout,
                        "effective": effective_timeout,
                        "config_max": timeout,
                    },
                )
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid timeout value from model, using config default",
                    extra={"model_timeout": model_timeout, "config_default": timeout},
                )

        if not command:
            return ToolResult(success=False, output=None, error="Missing 'command' parameter")

        # Check for restricted patterns when running in sandbox
        if sandbox:
            restriction_error = _is_restricted_command(command)
            if restriction_error:
                return ToolResult(success=False, output=None, error=restriction_error)

        try:
            if sandbox:
                # Execute in sandbox (cwd=/workspace)
                result = await sandbox.execute(command, cwd="/workspace", timeout=effective_timeout)
                output = result.stdout
                if result.stderr:
                    output += f"\n\nSTDERR:\n{result.stderr}"
                return ToolResult(success=result.success, output=output, error=result.error)
            else:
                # Direct execution (for development/testing only)
                proc = await asyncio.create_subprocess_shell(
                    command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=effective_timeout
                    )
                    output = stdout.decode("utf-8", errors="replace")
                    if stderr:
                        output += f"\n\nSTDERR:\n{stderr.decode('utf-8', errors='replace')}"
                    return ToolResult(
                        success=proc.returncode == 0,
                        output=output,
                        error=None if proc.returncode == 0 else f"Exit code: {proc.returncode}",
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    return ToolResult(
                        success=False,
                        output=f"Error: Command timed out after {effective_timeout} seconds",
                        error=f"Command timed out after {effective_timeout} seconds",
                    )
        except Exception as e:
            return ToolResult(success=False, output=f"Error: {str(e)}", error=str(e))

    # Build description with actual working directory
    working_dir_desc = base_dir if base_dir != "." else "the current directory"
    return ToolDefinition(
        name="run_shell_command",
        description=(
            f"Execute a shell command in {working_dir_desc}. "
            "Use relative paths (e.g., 'cat docs/file.md', 'find . -name *.md'). "
            "Use for: reading files (cat), listing (ls), searching (find ., grep), running scripts. "
            f"Optional timeout parameter (1-{timeout:.0f}s) for slow commands."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "timeout": {
                    "type": "number",
                    "description": (
                        f"Optional timeout in seconds for this command (1 to "
                        f"{timeout:.0f}s). If omitted, uses the configured default "
                        f"of {timeout:.0f}s. Use higher values for slow API calls or scripts."
                    ),
                },
            },
            "required": ["command"],
        },
        handler=bash_handler,
    )


def create_grep_tool(sandbox: Optional["Sandbox"] = None, base_dir: str = ".") -> ToolDefinition:
    """Create a grep tool for searching file contents with regex patterns.

    Args:
        sandbox: Optional sandbox for executing grep command. If None, runs directly.
        base_dir: Base directory for direct execution (used when sandbox is None)

    Returns:
        ToolDefinition for the grep tool
    """
    import shlex

    async def grep_handler(params: Dict[str, Any]) -> ToolResult:
        pattern = params.get("pattern")
        path = params.get("path", ".")
        case_insensitive = params.get("case_insensitive", False)

        if not pattern:
            return ToolResult(
                success=False, output=None, error="Missing required parameter: pattern"
            )

        try:
            # Build grep command with options
            flags = "-rn"  # recursive, line numbers
            if case_insensitive:
                flags += "i"

            # Security: Use shlex.quote for proper shell escaping
            # This prevents shell injection attacks like pattern: '" -r /etc #'
            escaped_pattern = shlex.quote(pattern)

            # Also escape path to prevent injection via path parameter
            escaped_path = shlex.quote(path)

            command = f"grep {flags} -- {escaped_pattern} {escaped_path}"

            if sandbox:
                result = await sandbox.execute(command)

                # grep returns 0 for matches, 1 for no matches, >1 for errors
                if result.returncode == 0:
                    return ToolResult(success=True, output=result.stdout)
                elif result.returncode == 1:
                    return ToolResult(success=True, output="No matches found")
                else:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=result.stderr or f"grep failed with code {result.returncode}",
                    )
            else:
                # Direct execution fallback (matches create_bash_tool pattern)
                logger.debug("grep.direct_execution", command=command, cwd=base_dir)
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=base_dir,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
                output = stdout.decode("utf-8", errors="replace")

                # grep returns 0 for matches, 1 for no matches, >1 for errors
                if proc.returncode == 0:
                    return ToolResult(success=True, output=output)
                elif proc.returncode == 1:
                    return ToolResult(success=True, output="No matches found")
                else:
                    error_msg = stderr.decode("utf-8", errors="replace")
                    return ToolResult(
                        success=False,
                        output=None,
                        error=error_msg or f"grep failed with code {proc.returncode}",
                    )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    return ToolDefinition(
        name="search_file_content",
        description="Search file contents for a regex pattern. Returns matching lines with file paths and line numbers.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regex pattern to search for"},
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (default: current directory)",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Whether to ignore case (default: false)",
                },
            },
            "required": ["pattern"],
        },
        handler=grep_handler,
    )


def create_web_search_tool(sandbox: "Sandbox") -> ToolDefinition:
    """Create a web search tool using DuckDuckGo.

    Args:
        sandbox: Sandbox for executing search (requires network access)

    Returns:
        ToolDefinition for the web_search tool
    """

    async def web_search_handler(params: Dict[str, Any]) -> ToolResult:
        query = params.get("query")
        max_results = params.get("max_results", 5)

        if not query:
            return ToolResult(success=False, output=None, error="Missing required parameter: query")

        try:
            # Use repr() to safely escape query for embedding in Python source
            safe_query = repr(query)
            safe_max_results = int(max_results)

            # Python script to run in sandbox
            script = f"""
import json
import sys

try:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text({safe_query}, max_results={safe_max_results}))
        print(json.dumps(results, indent=2))
except ImportError:
    print(json.dumps({{"error": "duckduckgo-search not installed. Run: pip install duckduckgo-search"}}))
    sys.exit(1)
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
"""
            # Write script to temp file to avoid shell metacharacter issues with -c
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script)
                script_path = f.name
            try:
                result = await sandbox.execute(f"python3 {script_path}")
            finally:
                os.unlink(script_path)

            if result.returncode == 0:
                return ToolResult(success=True, output=result.stdout)
            else:
                error_msg = result.stderr or result.stdout or "Search failed"
                return ToolResult(success=False, output=None, error=error_msg)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    return ToolDefinition(
        name="google_web_search",
        description="Search the web for information using DuckDuckGo. Returns titles, URLs, and snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                },
            },
            "required": ["query"],
        },
        handler=web_search_handler,
    )


def create_perplexity_web_search_tool(
    api_key: Optional[str] = None,
    model: str = "sonar",
    max_searches: int = 5,
) -> ToolDefinition:
    """Create a web_search tool powered by Perplexity API.

    Args:
        api_key: Perplexity API key (falls back to PERPLEXITY_API_KEY env var)
        model: Perplexity model to use
        max_searches: Maximum searches per test execution

    Returns:
        ToolDefinition for the web_search tool
    """
    import os
    import json
    import hashlib
    import urllib.request
    import urllib.error

    resolved_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
    search_count = 0

    async def perplexity_search_handler(params: Dict[str, Any]) -> ToolResult:
        nonlocal search_count

        query = params.get("query")
        if not query:
            return ToolResult(success=False, output="", error="Missing required parameter: query")

        if not resolved_key:
            return ToolResult(
                success=False,
                output="",
                error="Perplexity API key not configured. Set perplexity.api_key in dokumen.yaml or PERPLEXITY_API_KEY env var.",
            )

        search_count += 1
        if search_count > max_searches:
            return ToolResult(
                success=False,
                output="",
                error=f"Rate limit: maximum {max_searches} web searches per test execution exceeded.",
            )

        query_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
        truncated_query = query[:1000]
        logger.info(
            "tools.web_search.start",
            query_length=len(query),
            query_hash=query_hash,
            model=model,
            search_number=search_count,
            max_searches=max_searches,
        )

        try:
            request_body = json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": truncated_query}],
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                "https://api.perplexity.ai/chat/completions",
                data=request_body,
                headers={
                    "Authorization": f"Bearer {resolved_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=15),
            )
            response_data = json.loads(response.read().decode("utf-8"))

            answer = ""
            citations = []
            if response_data.get("choices"):
                answer = response_data["choices"][0].get("message", {}).get("content", "")
            if response_data.get("citations"):
                citations = response_data["citations"]

            # Format output
            output_parts = [f"[Web Search Results for: {truncated_query[:100]}]\n"]
            if answer:
                output_parts.append(answer)
            if citations:
                output_parts.append("\n\nSources:")
                for i, url in enumerate(citations[:10], 1):
                    output_parts.append(f"  [{i}] {url}")

            result_text = "\n".join(output_parts)
            # Truncate to ~16000 chars to avoid token limits
            if len(result_text) > 16000:
                result_text = result_text[:16000] + "\n[... truncated]"

            logger.info(
                "tools.web_search.complete",
                query_hash=query_hash,
                answer_length=len(answer),
                citation_count=len(citations),
                search_number=search_count,
            )

            return ToolResult(success=True, output=result_text)

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            logger.error(
                "tools.web_search.http_error",
                status=e.code,
                query_hash=query_hash,
                error=error_body[:200],
            )
            return ToolResult(
                success=False, output="", error=f"Perplexity API error (HTTP {e.code})"
            )
        except Exception as e:
            logger.error("tools.web_search.error", query_hash=query_hash, error=str(e))
            return ToolResult(success=False, output="", error=f"Web search failed: {str(e)}")

    return ToolDefinition(
        name="web_search",
        description="Search the web for information using Perplexity AI. Returns relevant results with source citations. Use this when you need current information or facts that may not be in the documentation.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find information about",
                },
            },
            "required": ["query"],
        },
        handler=perplexity_search_handler,
    )


def create_anthropic_web_search_tool(
    max_uses: Optional[int] = None,
    allowed_domains: Optional[list[str]] = None,
    blocked_domains: Optional[list[str]] = None,
) -> ToolDefinition:
    """Create an anthropic_web_search tool (Anthropic server-side web search).

    This is a sentinel tool — Anthropic executes searches server-side during
    generation. The handler should never be called; if it is, it raises
    RuntimeError. The provider reads _server_tool_config from the handler to
    build the correct API payload.

    Args:
        max_uses: Maximum web searches per API call (default 20)
        allowed_domains: Restrict searches to these domains
        blocked_domains: Block searches from these domains

    Returns:
        ToolDefinition for the anthropic_web_search tool
    """
    effective_max_uses = max_uses if max_uses is not None else 20

    logger.info(
        "tools.anthropic_web_search.create",
        max_uses=effective_max_uses,
        has_allowed_domains=allowed_domains is not None,
        has_blocked_domains=blocked_domains is not None,
    )

    async def sentinel_handler(params: Dict[str, Any]) -> ToolResult:
        raise RuntimeError(
            "anthropic_web_search is a server-side tool — Anthropic executes "
            "searches during generation. This handler should never be called."
        )

    # Attach server tool config for the provider to read
    server_config: Dict[str, Any] = {
        "type": "web_search_20250305",
        "max_uses": effective_max_uses,
    }
    if allowed_domains is not None:
        server_config["allowed_domains"] = allowed_domains
    if blocked_domains is not None:
        server_config["blocked_domains"] = blocked_domains

    sentinel_handler._server_tool_config = server_config  # type: ignore[attr-defined]

    return ToolDefinition(
        name="anthropic_web_search",
        description=(
            "Search the web using Anthropic's built-in web search. "
            "Returns relevant results with source citations. This tool is "
            "executed server-side by Anthropic during generation — results "
            "are automatically included in the model's context."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
            "_server_side": True,
        },
        handler=sentinel_handler,
    )


def create_http_request_tool(sandbox: "Sandbox" = None, timeout: float = 30.0) -> ToolDefinition:
    """Create an http_request tool for making HTTP requests.

    Args:
        sandbox: Optional sandbox for executing requests inside container
        timeout: Request timeout in seconds

    Returns:
        ToolDefinition for the http_request tool
    """
    from .debug import debug
    from urllib.parse import urlparse
    import ipaddress
    import socket

    debug(
        f"[DEBUG TOOLS] create_http_request_tool called with sandbox={sandbox} (id={id(sandbox) if sandbox else None})"
    )

    # SSRF Protection: Block internal/private hosts
    # These hosts could expose internal services, cloud metadata, or sensitive data
    BLOCKED_HOSTNAMES = {
        "localhost",
        "localhost.localdomain",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "169.254.169.254",  # AWS/GCP/Azure metadata endpoint
        "metadata.google.internal",  # GCP metadata
        "metadata.google",
        "169.254.170.2",  # AWS ECS metadata
    }

    def _is_private_ip(ip_str: str) -> bool:
        """Check if an IP address is private/internal."""
        try:
            ip = ipaddress.ip_address(ip_str)
            return (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            )
        except ValueError:
            return False

    def _is_blocked_host(hostname: str) -> tuple[bool, str]:
        """Check if a hostname should be blocked for SSRF protection.

        Returns:
            Tuple of (is_blocked, reason)
        """
        if not hostname:
            return True, "Empty hostname"

        hostname_lower = hostname.lower()

        # Check against blocklist
        if hostname_lower in BLOCKED_HOSTNAMES:
            return True, f"Blocked hostname: {hostname}"

        # Check if it's a blocked IP address
        if _is_private_ip(hostname):
            return True, f"Private/internal IP address not allowed: {hostname}"

        # Try to resolve hostname and check if it resolves to a private IP
        # This catches DNS rebinding attacks and internal hostnames
        try:
            resolved_ips = socket.gethostbyname_ex(hostname)[2]
            for ip in resolved_ips:
                if _is_private_ip(ip):
                    return True, f"Hostname resolves to private IP: {hostname} -> {ip}"
        except socket.gaierror:
            # DNS resolution failed - could be a valid external host that's down
            # or an invalid hostname. Allow it and let the HTTP request fail naturally.
            pass

        return False, ""

    async def http_request_handler(params: Dict[str, Any]) -> ToolResult:
        from .debug import debug

        url = params.get("url")
        method = params.get("method", "GET").upper()
        headers = params.get("headers", {})
        body = params.get("body")

        debug("[DEBUG TOOLS] http_request_handler called:")
        debug(f"[DEBUG TOOLS]   url: {url}")
        debug(f"[DEBUG TOOLS]   method: {method}")
        debug(f"[DEBUG TOOLS]   sandbox available: {sandbox is not None}")
        if sandbox:
            debug(f"[DEBUG TOOLS]   sandbox id: {id(sandbox)}")
            debug(f"[DEBUG TOOLS]   sandbox type: {type(sandbox).__name__}")

        if not url:
            return ToolResult(success=False, output=None, error="Missing 'url' parameter")

        # SSRF Protection: Validate URL and block internal hosts
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname

            if not hostname:
                return ToolResult(
                    success=False, output=None, error="Invalid URL: could not extract hostname"
                )

            is_blocked, reason = _is_blocked_host(hostname)
            if is_blocked:
                logger.warning("http_request.ssrf_blocked", url=url, reason=reason)
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Request blocked for security: {reason}. Only external URLs are allowed.",
                )

            # Block non-HTTP(S) schemes
            if parsed.scheme not in ("http", "https"):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid URL scheme: {parsed.scheme}. Only http and https are allowed.",
                )

        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Invalid URL: {str(e)}")

        # If sandbox is active, use curl inside the container
        if sandbox:
            debug("[DEBUG TOOLS] Using sandbox for HTTP request")
            return await _http_request_via_sandbox(sandbox, url, method, headers, body, timeout)

        debug("[DEBUG TOOLS] NO sandbox - making HTTP request directly from host")
        try:
            # Use aiohttp if available, fall back to urllib
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method,
                        url,
                        headers=headers,
                        data=body,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as response:
                        content = await response.text()
                        return ToolResult(
                            success=200 <= response.status < 400,
                            output={
                                "status": response.status,
                                "headers": dict(response.headers),
                                "body": content,
                            },
                        )
            except ImportError:
                # Fall back to urllib (synchronous)
                import urllib.request
                import urllib.error

                req = urllib.request.Request(url, method=method)
                for key, value in headers.items():
                    req.add_header(key, value)

                data = body.encode("utf-8") if body else None

                try:
                    with urllib.request.urlopen(req, data=data, timeout=timeout) as response:
                        content = response.read().decode("utf-8", errors="replace")
                        return ToolResult(
                            success=True,
                            output={
                                "status": response.status,
                                "headers": dict(response.headers),
                                "body": content,
                            },
                        )
                except urllib.error.HTTPError as e:
                    content = e.read().decode("utf-8", errors="replace") if e.fp else ""
                    return ToolResult(
                        success=False,
                        output={"status": e.code, "headers": dict(e.headers), "body": content},
                        error=f"HTTP {e.code}: {e.reason}",
                    )
        except Exception as e:
            return ToolResult(success=False, output=f"Error: {str(e)}", error=str(e))

    return ToolDefinition(
        name="web_fetch",
        description=(
            "Make an HTTP request to a URL. "
            "Use this to test APIs, ping servers, or fetch remote resources."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to request"},
                "method": {
                    "type": "string",
                    "description": "HTTP method (GET, POST, PUT, DELETE, etc.)",
                    "default": "GET",
                },
                "headers": {
                    "type": "object",
                    "description": "HTTP headers as key-value pairs",
                    "additionalProperties": {"type": "string"},
                },
                "body": {"type": "string", "description": "Request body (for POST, PUT, etc.)"},
            },
            "required": ["url"],
        },
        handler=http_request_handler,
    )


async def _http_request_via_sandbox(
    sandbox: "Sandbox", url: str, method: str, headers: Dict[str, str], body: str, timeout: float
) -> ToolResult:
    """Execute HTTP request inside sandbox using Python (guaranteed available).

    Args:
        sandbox: Sandbox to execute in
        url: URL to request
        method: HTTP method
        headers: Request headers
        body: Request body
        timeout: Timeout in seconds

    Returns:
        ToolResult with response data
    """
    from .debug import debug

    debug("[DEBUG TOOLS] _http_request_via_sandbox called:")
    debug(f"[DEBUG TOOLS]   sandbox type: {type(sandbox).__name__}")
    debug(f"[DEBUG TOOLS]   sandbox id: {id(sandbox)}")
    debug(f"[DEBUG TOOLS]   url: {url}")
    debug(f"[DEBUG TOOLS]   method: {method}")
    debug(f"[DEBUG TOOLS]   timeout: {timeout}")
    import json as json_module

    # Build Python one-liner for HTTP request
    headers_json = json_module.dumps(headers)

    python_script = f"""
import urllib.request
import urllib.error
import json
import sys

url = {repr(url)}
method = {repr(method)}
headers = {headers_json}
body = {repr(body) if body else 'None'}
timeout = {timeout}

try:
    req = urllib.request.Request(url, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    data = body.encode() if body else None
    with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
        result = {{
            "status": resp.status,
            "headers": dict(resp.headers),
            "body": resp.read().decode("utf-8", errors="replace")
        }}
        print(json.dumps(result))
except urllib.error.HTTPError as e:
    result = {{
        "status": e.code,
        "headers": dict(e.headers) if e.headers else {{}},
        "body": e.read().decode("utf-8", errors="replace") if e.fp else "",
        "error": str(e)
    }}
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
"""

    # Write script to temp file and execute (avoids shell escaping issues)
    # Use heredoc approach similar to write_file
    cmd = f"""cat > /tmp/_http_request.py << 'DOKUMEN_HTTP_EOF'
{python_script}
DOKUMEN_HTTP_EOF
python3 /tmp/_http_request.py"""

    try:
        debug("[DEBUG TOOLS] Executing HTTP request in sandbox...")
        result = await sandbox.execute(cmd)
        debug("[DEBUG TOOLS] Sandbox execute result:")
        debug(f"[DEBUG TOOLS]   success: {result.success}")
        debug(f"[DEBUG TOOLS]   stdout: {result.stdout[:200] if result.stdout else 'None'}...")
        debug(f"[DEBUG TOOLS]   stderr: {result.stderr[:200] if result.stderr else 'None'}...")
        debug(f"[DEBUG TOOLS]   error: {result.error}")

        if not result.stdout:
            error_msg = result.error or result.stderr or "No response from server"
            return ToolResult(success=False, output=f"Error: {error_msg}", error=error_msg)

        # Parse JSON output
        try:
            response_data = json_module.loads(result.stdout.strip())
        except json_module.JSONDecodeError:
            return ToolResult(
                success=False,
                output=f"Error: Failed to parse response: {result.stdout}",
                error=f"Failed to parse response: {result.stdout}",
            )

        if "error" in response_data and "status" not in response_data:
            return ToolResult(
                success=False,
                output=f"Error: {response_data['error']}",
                error=response_data["error"],
            )

        status_code = response_data.get("status", 500)
        return ToolResult(
            success=200 <= status_code < 400,
            output={
                "status": status_code,
                "headers": response_data.get("headers", {}),
                "body": response_data.get("body", ""),
            },
            error=response_data.get("error") if not (200 <= status_code < 400) else None,
        )
    except Exception as e:
        return ToolResult(success=False, output=f"Error: {str(e)}", error=str(e))


# ============================================================================
# Agent Delegation Tool
# ============================================================================


def create_delegate_to_agent_tool(
    registry: Any,
    provider: Any,
    sandbox: Optional["Sandbox"] = None,
    timeout: float = 60.0,
    parent_tools: Optional[List[ToolDefinition]] = None,
) -> ToolDefinition:
    """
    Create a delegate_to_agent tool for agent-to-agent delegation.

    This tool allows agents to delegate tasks to other registered agents,
    enabling composition of specialized agents for complex workflows.

    Args:
        registry: Agent registry to look up agent definitions
        provider: LLM provider for running delegated agents
        sandbox: Optional sandbox for tool execution
        timeout: Timeout for delegated agent execution in seconds
        parent_tools: Optional list of parent agent tools. When provided,
            the subagent's tools are restricted to this set (used by judges
            to ensure subagents only get read-only tools).

    Returns:
        ToolDefinition for the delegate_to_agent tool
    """

    async def delegate_to_agent_handler(params: Dict[str, Any]) -> ToolResult:
        """Handler for delegate_to_agent tool."""
        agent_name = params.get("agent")
        input_text = params.get("input")
        thoroughness = params.get("thoroughness", "medium")

        # Validate thoroughness
        valid_thoroughness = ["quick", "medium", "very-thorough"]
        if thoroughness not in valid_thoroughness:
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid thoroughness: {thoroughness}. Must be one of: {', '.join(valid_thoroughness)}",
            )

        if not agent_name:
            return ToolResult(success=False, output=None, error="Missing required parameter: agent")

        if input_text is None:
            input_text = ""

        # Look up agent in registry
        definition = registry.get(agent_name)
        if not definition:
            available = [a.name for a in registry.list_all()]
            return ToolResult(
                success=False,
                output=None,
                error=f"Agent not found: {agent_name}. Available agents: {', '.join(available)}",
            )

        # Log subagent spawning
        parent_tool_names = [t.name for t in parent_tools] if parent_tools else None
        logger.info(
            "delegate_to_agent.spawning",
            agent=agent_name,
            input_length=len(input_text),
            thoroughness=thoroughness,
            has_parent_tools=parent_tools is not None,
            parent_tool_names=parent_tool_names,
        )

        # Run the agent as a sub-agent (prevents recursive delegation)
        try:
            from .agents import run_agent

            result = await asyncio.wait_for(
                run_agent(
                    definition=definition,
                    input_data=input_text,
                    provider=provider,
                    sandbox=sandbox,
                    timeout=timeout,
                    is_subagent=True,  # Prevents recursive delegation
                    thoroughness=thoroughness,
                ),
                timeout=timeout,
            )

            if result.success:
                return ToolResult(
                    success=True,
                    output={
                        "agent": agent_name,
                        "output": result.output,
                        "duration": result.duration,
                        "tool_calls": len(result.tool_calls),
                    },
                )
            else:
                return ToolResult(
                    success=False, output=None, error=f"Agent {agent_name} failed: {result.error}"
                )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False, output=None, error=f"Agent {agent_name} timed out after {timeout}s"
            )
        except Exception as e:
            return ToolResult(
                success=False, output=None, error=f"Error delegating to {agent_name}: {str(e)}"
            )

    # Build description based on whether parent tools restrict the subagent
    if parent_tools:
        parent_names = ", ".join(t.name for t in parent_tools)
        description = (
            "Delegate a task to another registered agent. "
            "Use this to invoke specialized agents for specific tasks. "
            f"The delegated agent's tools are restricted to the parent's tool set: {parent_names}. "
            "For the 'explore' agent, use thoroughness to control search depth."
        )
    else:
        description = (
            "Delegate a task to another registered agent. "
            "Use this to invoke specialized agents for specific tasks. "
            "The delegated agent runs with its own tools and configuration. "
            "For the 'explore' agent, use thoroughness to control search depth."
        )

    return ToolDefinition(
        name="delegate_to_agent",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Name of the agent to delegate to (e.g., 'explore')",
                },
                "input": {"type": "string", "description": "The task or question for the agent"},
                "thoroughness": {
                    "type": "string",
                    "enum": ["quick", "medium", "very-thorough"],
                    "description": "Search depth (explore agent only). quick=fast surface search, medium=balanced, very-thorough=comprehensive. Default: medium",
                },
            },
            "required": ["agent", "input"],
        },
        handler=delegate_to_agent_handler,
    )


# ============================================================================
# Gemini-CLI Compatible Tools
# ============================================================================


def create_read_file_tool(base_dir: str = ".") -> ToolDefinition:
    """Create a read_file tool for reading any file with line range support.

    Matches gemini-cli's read_file tool for text files with line numbers.
    Image and PDF handling is intentionally left to the SDK/native model path.

    Args:
        base_dir: Base directory for resolving relative paths

    Returns:
        ToolDefinition for the read_file tool
    """
    import os

    # Text file size limit (matches backend MAX_FILE_SIZE)
    MAX_TEXT_FILE_SIZE = 1024 * 1024  # 1MB

    async def read_file_handler(params: Dict[str, Any]) -> ToolResult:
        file_path = params.get("file_path")
        offset = params.get("offset", 1)  # 1-indexed, default to start
        limit = params.get("limit")  # None means read all

        if not file_path:
            return ToolResult(success=False, output=None, error="Missing 'file_path' parameter")

        # Resolve path relative to base_dir
        if not os.path.isabs(file_path):
            full_path = os.path.join(base_dir, file_path)
        else:
            full_path = file_path

        # Normalize path
        full_path = os.path.normpath(full_path)

        # Path traversal protection: resolve symlinks and ensure path is within base_dir
        normalized_base = os.path.realpath(base_dir)
        normalized_full = os.path.realpath(full_path)
        if (
            not normalized_full.startswith(normalized_base + os.sep)
            and normalized_full != normalized_base
        ):
            return ToolResult(
                success=False,
                output=None,
                error="Access denied: path traversal not allowed. Path must be within base directory.",
            )

        # Check file exists
        if not os.path.exists(full_path):
            return ToolResult(success=False, output=None, error=f"File not found: {file_path}")

        if os.path.isdir(full_path):
            return ToolResult(
                success=False,
                output=None,
                error=f"Path is a directory, not a file: {file_path}. Use list_directory instead.",
            )

        # Check text file size
        file_size = os.path.getsize(full_path)
        if file_size > MAX_TEXT_FILE_SIZE:
            return ToolResult(
                success=False,
                output=None,
                error=f"File too large ({file_size:,} bytes, max {MAX_TEXT_FILE_SIZE:,} bytes). Use offset and limit parameters to read specific sections.",
            )

        # Read text file
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)

            # Apply offset (1-indexed)
            start_idx = max(0, offset - 1)

            # Apply limit
            if limit is not None:
                end_idx = min(start_idx + limit, total_lines)
            else:
                end_idx = total_lines

            # Format with line numbers
            output_lines = []
            for i in range(start_idx, end_idx):
                line_num = i + 1
                line_content = lines[i].rstrip("\n\r")
                output_lines.append(f"{line_num:6d}\t{line_content}")

            result_text = "\n".join(output_lines)

            # Add metadata header
            header = f"File: {file_path}\nLines: {start_idx + 1}-{end_idx} of {total_lines}\n\n"

            return ToolResult(success=True, output=header + result_text)

        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                output=None,
                error=f"File appears to be binary, not text: {file_path}",
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Failed to read file: {str(e)}")

    return ToolDefinition(
        name="read_file",
        description=(
            "Read a text file's contents with line numbers. "
            "Use offset and limit to read specific line ranges for large text files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to project root or absolute)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed). Default: 1",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Default: all lines",
                },
            },
            "required": ["file_path"],
        },
        handler=read_file_handler,
    )


def create_glob_tool(base_dir: str = ".") -> ToolDefinition:
    """Create a glob tool for finding files by pattern.

    Matches gemini-cli's glob/FindFiles tool. Returns file paths sorted by
    modification time (newest first).

    Args:
        base_dir: Base directory for resolving patterns

    Returns:
        ToolDefinition for the glob tool
    """
    import os
    import glob as glob_module

    async def glob_handler(params: Dict[str, Any]) -> ToolResult:
        pattern = params.get("pattern")
        path = params.get("path", ".")
        respect_gitignore = params.get("respect_gitignore", True)

        if not pattern:
            return ToolResult(success=False, output=None, error="Missing 'pattern' parameter")

        # Resolve base path
        if not os.path.isabs(path):
            search_path = os.path.join(base_dir, path)
        else:
            search_path = path

        search_path = os.path.realpath(search_path)

        # Path traversal protection: ensure resolved path is within base_dir
        real_base = os.path.realpath(base_dir)
        if not search_path.startswith(real_base + os.sep) and search_path != real_base:
            return ToolResult(
                success=False,
                output=None,
                error="Access denied: path traversal not allowed. Path must be within base directory.",
            )

        if not os.path.exists(search_path):
            return ToolResult(success=False, output=None, error=f"Path not found: {path}")

        # Build full glob pattern
        full_pattern = os.path.join(search_path, pattern)

        try:
            # Find matching files
            matches = glob_module.glob(full_pattern, recursive=True)

            # Filter out directories, keep only files
            # Also filter out files that resolve (via symlink) outside base_dir
            real_base = os.path.realpath(base_dir)
            files = []
            for m in matches:
                if not os.path.isfile(m):
                    continue
                real_m = os.path.realpath(m)
                if real_m.startswith(real_base + os.sep) or real_m == real_base:
                    files.append(m)

            # Load gitignore patterns if requested
            ignored_patterns = set()
            if respect_gitignore:
                gitignore_path = os.path.join(base_dir, ".gitignore")
                if os.path.exists(gitignore_path):
                    try:
                        with open(gitignore_path, "r") as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith("#"):
                                    ignored_patterns.add(line)
                    except Exception:
                        pass

            # Filter by gitignore patterns (simple matching)
            if ignored_patterns:
                filtered_files = []
                for f in files:
                    rel_path = os.path.relpath(f, base_dir)
                    ignored = False
                    for pattern in ignored_patterns:
                        if pattern in rel_path or rel_path.startswith(pattern):
                            ignored = True
                            break
                    if not ignored:
                        filtered_files.append(f)
                files = filtered_files

            # Sort by modification time (newest first)
            files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

            # Convert to relative paths for cleaner output
            rel_files = [os.path.relpath(f, base_dir) for f in files]

            if not rel_files:
                return ToolResult(
                    success=True, output=f"No files found matching pattern: {pattern}"
                )

            output = f"Found {len(rel_files)} files matching '{pattern}':\n\n"
            output += "\n".join(rel_files)

            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Glob search failed: {str(e)}")

    return ToolDefinition(
        name="glob",
        description=(
            "Find files matching a glob pattern. Returns file paths sorted by "
            "modification time (newest first). Supports ** for recursive matching."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '**/*.py', 'src/*.ts', '*.md')",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in. Default: project root",
                },
                "respect_gitignore": {
                    "type": "boolean",
                    "description": "Whether to respect .gitignore patterns. Default: true",
                },
            },
            "required": ["pattern"],
        },
        handler=glob_handler,
    )


def create_list_directory_tool(base_dir: str = ".") -> ToolDefinition:
    """Create a list_directory tool for listing directory contents.

    Matches gemini-cli's list_directory/LS tool. Returns files and directories
    with metadata (size, type, modification time).

    Args:
        base_dir: Base directory for resolving paths

    Returns:
        ToolDefinition for the list_directory tool
    """
    import os
    from datetime import datetime

    async def list_directory_handler(params: Dict[str, Any]) -> ToolResult:
        path = params.get("path", ".")
        recursive = params.get("recursive", False)
        include_hidden = params.get("include_hidden", False)

        # Resolve path (realpath resolves symlinks for traversal protection)
        if not os.path.isabs(path):
            full_path = os.path.join(base_dir, path)
        else:
            full_path = path

        full_path = os.path.realpath(full_path)

        # Path traversal protection: ensure resolved path is within base_dir
        real_base = os.path.realpath(base_dir)
        if not full_path.startswith(real_base + os.sep) and full_path != real_base:
            return ToolResult(
                success=False,
                output=None,
                error="Access denied: path traversal not allowed. Path must be within base directory.",
            )

        if not os.path.exists(full_path):
            return ToolResult(success=False, output=None, error=f"Directory not found: {path}")

        if not os.path.isdir(full_path):
            return ToolResult(
                success=False,
                output=None,
                error=f"Path is not a directory: {path}. Use read_file instead.",
            )

        try:
            entries = []

            if recursive:
                for root, dirs, files in os.walk(full_path):
                    # Filter hidden if needed
                    if not include_hidden:
                        dirs[:] = [d for d in dirs if not d.startswith(".")]
                        files = [f for f in files if not f.startswith(".")]

                    rel_root = os.path.relpath(root, full_path)
                    if rel_root == ".":
                        rel_root = ""

                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        rel_path = os.path.join(rel_root, d) if rel_root else d
                        mtime = datetime.fromtimestamp(os.path.getmtime(dir_path))
                        entries.append(
                            {
                                "name": rel_path + "/",
                                "type": "dir",
                                "size": "-",
                                "modified": mtime.strftime("%Y-%m-%d %H:%M"),
                            }
                        )

                    for f in files:
                        file_path = os.path.join(root, f)
                        rel_path = os.path.join(rel_root, f) if rel_root else f
                        size = os.path.getsize(file_path)
                        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        entries.append(
                            {
                                "name": rel_path,
                                "type": "file",
                                "size": _format_size(size),
                                "modified": mtime.strftime("%Y-%m-%d %H:%M"),
                            }
                        )
            else:
                for name in os.listdir(full_path):
                    if not include_hidden and name.startswith("."):
                        continue

                    item_path = os.path.join(full_path, name)
                    mtime = datetime.fromtimestamp(os.path.getmtime(item_path))

                    if os.path.isdir(item_path):
                        entries.append(
                            {
                                "name": name + "/",
                                "type": "dir",
                                "size": "-",
                                "modified": mtime.strftime("%Y-%m-%d %H:%M"),
                            }
                        )
                    else:
                        size = os.path.getsize(item_path)
                        entries.append(
                            {
                                "name": name,
                                "type": "file",
                                "size": _format_size(size),
                                "modified": mtime.strftime("%Y-%m-%d %H:%M"),
                            }
                        )

            # Sort: directories first, then files, alphabetically
            entries.sort(key=lambda e: (0 if e["type"] == "dir" else 1, e["name"].lower()))

            if not entries:
                return ToolResult(success=True, output=f"Directory is empty: {path}")

            # Format output
            output = f"Contents of {path}:\n\n"
            output += f"{'Name':<50} {'Type':<6} {'Size':<10} {'Modified'}\n"
            output += "-" * 85 + "\n"

            for e in entries:
                output += f"{e['name']:<50} {e['type']:<6} {e['size']:<10} {e['modified']}\n"

            output += f"\nTotal: {len(entries)} items"

            return ToolResult(success=True, output=output)

        except PermissionError:
            return ToolResult(success=False, output=None, error=f"Permission denied: {path}")
        except Exception as e:
            return ToolResult(
                success=False, output=None, error=f"Failed to list directory: {str(e)}"
            )

    return ToolDefinition(
        name="list_directory",
        description=(
            "List contents of a directory with file/directory names, types, sizes, "
            "and modification times. Use recursive=true to list subdirectories."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Default: current directory",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List subdirectories recursively. Default: false",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (starting with '.'). Default: false",
                },
            },
            "required": [],
        },
        handler=list_directory_handler,
    )


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}" if unit != "B" else f"{size_bytes}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def create_read_many_files_tool(base_dir: str = ".") -> ToolDefinition:
    """Create a read_many_files tool for batch file reading.

    Matches gemini-cli's read_many_files tool. Reads multiple files matching
    glob patterns in a single call.

    Args:
        base_dir: Base directory for resolving paths

    Returns:
        ToolDefinition for the read_many_files tool
    """
    import os
    import glob as glob_module

    async def read_many_files_handler(params: Dict[str, Any]) -> ToolResult:
        patterns = params.get("patterns", [])
        exclude = params.get("exclude", [])
        max_files = params.get("max_files", 50)
        max_lines_per_file = params.get("max_lines_per_file", 500)

        if not patterns:
            return ToolResult(
                success=False,
                output=None,
                error="Missing 'patterns' parameter - provide list of glob patterns",
            )

        if isinstance(patterns, str):
            patterns = [patterns]

        try:
            # Collect all matching files
            all_files = set()
            for pattern in patterns:
                full_pattern = os.path.join(base_dir, pattern)
                matches = glob_module.glob(full_pattern, recursive=True)
                for m in matches:
                    if os.path.isfile(m):
                        all_files.add(os.path.normpath(m))

            # Apply exclusions
            if exclude:
                excluded = set()
                for exc_pattern in exclude:
                    exc_full = os.path.join(base_dir, exc_pattern)
                    exc_matches = glob_module.glob(exc_full, recursive=True)
                    excluded.update(os.path.normpath(m) for m in exc_matches)
                all_files -= excluded

            # Limit number of files
            files_list = sorted(all_files)[:max_files]

            if not files_list:
                return ToolResult(
                    success=True, output="No files found matching the provided patterns."
                )

            # Read each file
            results = []
            for file_path in files_list:
                rel_path = os.path.relpath(file_path, base_dir)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()

                    # Apply line limit
                    if len(lines) > max_lines_per_file:
                        content = "".join(lines[:max_lines_per_file])
                        content += (
                            f"\n... (truncated, {len(lines) - max_lines_per_file} more lines)"
                        )
                    else:
                        content = "".join(lines)

                    results.append(f"=== {rel_path} ===\n{content}")

                except Exception as e:
                    results.append(f"=== {rel_path} ===\n[Error reading file: {str(e)}]")

            output = f"Read {len(files_list)} files:\n\n"
            output += "\n\n".join(results)

            if len(all_files) > max_files:
                output += f"\n\n[Note: {len(all_files) - max_files} additional files not shown due to max_files limit]"

            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Failed to read files: {str(e)}")

    return ToolDefinition(
        name="read_many_files",
        description=(
            "Read multiple files matching glob patterns in a single call. "
            "Efficient for batch reading. Files are returned concatenated with headers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of glob patterns (e.g., ['src/**/*.py', 'tests/*.py'])",
                },
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Patterns to exclude from results",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of files to read. Default: 50",
                },
                "max_lines_per_file": {
                    "type": "integer",
                    "description": "Maximum lines per file. Default: 500",
                },
            },
            "required": ["patterns"],
        },
        handler=read_many_files_handler,
    )


def create_edit_tool(sandbox: "Sandbox") -> ToolDefinition:
    """Create an edit tool for find/replace in files.

    Matches gemini-cli's replace/edit tool. Finds content in a file and
    replaces it with new content. Requires sandbox for safety.

    Args:
        sandbox: Sandbox for safe file operations

    Returns:
        ToolDefinition for the edit tool
    """

    async def edit_handler(params: Dict[str, Any]) -> ToolResult:
        file_path = params.get("file_path")
        old_content = params.get("old_content")
        new_content = params.get("new_content")

        if not file_path:
            return ToolResult(success=False, output=None, error="Missing 'file_path' parameter")

        if old_content is None:
            return ToolResult(success=False, output=None, error="Missing 'old_content' parameter")

        if new_content is None:
            return ToolResult(success=False, output=None, error="Missing 'new_content' parameter")

        try:
            # Read through sandbox
            read_result = await sandbox.read_file(file_path)
            if not read_result.get("success", False):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Failed to read file: {read_result.get('error', 'Unknown error')}",
                )

            content = read_result.get("content", "")

            # Check if old_content exists
            if old_content not in content:
                # Show context to help user find the right content
                preview = content[:500] + "..." if len(content) > 500 else content
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Could not find the specified content to replace in {file_path}.\n\nFile preview:\n{preview}",
                )

            # Count occurrences
            occurrences = content.count(old_content)
            if occurrences > 1:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Found {occurrences} occurrences of the content. Please provide more context to make the match unique.",
                )

            # Perform replacement
            new_file_content = content.replace(old_content, new_content, 1)

            # Write through sandbox
            write_result = await sandbox.write_file(file_path, new_file_content)
            if not write_result.get("success", False):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Failed to write file: {write_result.get('error', 'Unknown error')}",
                )

            # Generate simple diff
            old_lines = old_content.split("\n")
            new_lines = new_content.split("\n")

            diff_output = f"Edited {file_path}:\n\n"
            diff_output += "--- Before:\n"
            for line in old_lines[:10]:
                diff_output += f"- {line}\n"
            if len(old_lines) > 10:
                diff_output += f"  ... ({len(old_lines) - 10} more lines)\n"

            diff_output += "\n+++ After:\n"
            for line in new_lines[:10]:
                diff_output += f"+ {line}\n"
            if len(new_lines) > 10:
                diff_output += f"  ... ({len(new_lines) - 10} more lines)\n"

            return ToolResult(success=True, output=diff_output)

        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Edit failed: {str(e)}")

    return ToolDefinition(
        name="edit",
        description=(
            "Edit a file by finding and replacing content. The old_content must match "
            "exactly and uniquely in the file. Provide enough context to ensure a unique match."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to edit"},
                "old_content": {
                    "type": "string",
                    "description": "The exact content to find and replace (must be unique in file)",
                },
                "new_content": {"type": "string", "description": "The new content to replace with"},
            },
            "required": ["file_path", "old_content", "new_content"],
        },
        handler=edit_handler,
    )


# ============================================================================
# Code Repository Tools (read-only, for cross-referencing docs with code)
# ============================================================================


def _matches_pattern(path: str, patterns: List[str]) -> bool:
    """Check if a path matches any of the glob patterns.

    Args:
        path: Relative file path to check
        patterns: List of glob patterns (e.g., ["src/*.py", "**/*.ts"])

    Returns:
        True if path matches any pattern, or if patterns list is empty
    """
    import fnmatch

    if not patterns:
        return True  # No patterns = match everything
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def _validate_code_path(
    path: str,
    base_dir: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
) -> tuple:
    """Validate a code file path against include/exclude patterns and traversal.

    Args:
        path: Relative file path to validate
        base_dir: Base directory of the code repository
        include_patterns: Glob patterns for allowed files (empty = all)
        exclude_patterns: Glob patterns for denied files (empty = none)

    Returns:
        Tuple of (valid: bool, error_message: str)
    """
    logger.debug(
        "code_tools.validate_path",
        path=path,
        base_dir=base_dir,
        include_count=len(include_patterns),
        exclude_count=len(exclude_patterns),
    )

    # Normalize path
    normalized = os.path.normpath(path).lstrip("/")

    # Check traversal
    if ".." in normalized or os.path.isabs(path):
        logger.warning("code_tools.path_traversal_blocked", path=path)
        return False, "Path traversal not allowed"

    # Resolve and check within base_dir
    resolved = os.path.realpath(os.path.join(base_dir, normalized))
    real_base = os.path.realpath(base_dir)
    if not resolved.startswith(real_base + os.sep) and resolved != real_base:
        logger.warning("code_tools.path_outside_repo", path=path, resolved=resolved)
        return False, "Path outside code repository"

    # Check include patterns (empty = include all)
    if include_patterns and not _matches_pattern(normalized, include_patterns):
        logger.debug("code_tools.path_not_included", path=normalized, patterns=include_patterns)
        return False, f"Path not in include patterns: {include_patterns}"

    # Check exclude patterns
    if exclude_patterns and _matches_pattern(normalized, exclude_patterns):
        logger.debug("code_tools.path_excluded", path=normalized, patterns=exclude_patterns)
        return False, "Path matches exclude pattern"

    return True, ""


def create_code_read_file_tool(
    base_dir: str,
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
) -> ToolDefinition:
    """Create a read-only file reader for a linked code repository.

    Args:
        base_dir: Root directory of the code repository
        include_patterns: Glob patterns for allowed files (empty/None = all)
        exclude_patterns: Glob patterns for denied files (empty/None = none)

    Returns:
        ToolDefinition for the code_read_file tool
    """
    include = include_patterns or []
    exclude = exclude_patterns or []

    logger.info(
        "code_tools.create_read_file",
        base_dir=base_dir,
        include_patterns=include,
        exclude_patterns=exclude,
    )

    async def handler(params: Dict[str, Any]) -> ToolResult:
        file_path = params.get("file_path", params.get("path", ""))

        if not file_path:
            return ToolResult(
                success=False,
                output=None,
                error="Missing 'file_path' parameter",
            )

        logger.debug("code_tools.read_file.start", file_path=file_path)

        valid, err = _validate_code_path(file_path, base_dir, include, exclude)
        if not valid:
            logger.warning("code_tools.read_file.rejected", file_path=file_path, reason=err)
            return ToolResult(success=False, output=None, error=err)

        full_path = os.path.join(base_dir, os.path.normpath(file_path).lstrip("/"))
        if not os.path.exists(full_path):
            return ToolResult(
                success=False,
                output=None,
                error=f"File not found: {file_path}",
            )

        if os.path.isdir(full_path):
            return ToolResult(
                success=False,
                output=None,
                error=f"Path is a directory, not a file: {file_path}. Use code_list_directory instead.",
            )

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            logger.info(
                "code_tools.read_file.success",
                file_path=file_path,
                size_bytes=len(content),
            )
            return ToolResult(success=True, output=content)
        except Exception as e:
            logger.error("code_tools.read_file.error", file_path=file_path, error=str(e))
            return ToolResult(success=False, output=None, error=f"Failed to read file: {e}")

    return ToolDefinition(
        name="code_read_file",
        description=(
            "Read a file from a linked code repository. Use this to cross-reference "
            "documentation with actual implementation code. Only reads text files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file in the code repository",
                }
            },
            "required": ["file_path"],
        },
        handler=handler,
    )


def create_code_glob_tool(
    base_dir: str,
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
) -> ToolDefinition:
    """Create a glob tool scoped to a linked code repository.

    Args:
        base_dir: Root directory of the code repository
        include_patterns: Glob patterns for allowed files (empty/None = all)
        exclude_patterns: Glob patterns for denied files (empty/None = none)

    Returns:
        ToolDefinition for the code_glob tool
    """
    import glob as glob_module

    include = include_patterns or []
    exclude = exclude_patterns or []

    logger.info("code_tools.create_glob", base_dir=base_dir)

    async def handler(params: Dict[str, Any]) -> ToolResult:
        pattern = params.get("pattern")

        if not pattern:
            return ToolResult(
                success=False,
                output=None,
                error="Missing 'pattern' parameter",
            )

        logger.debug("code_tools.glob.start", pattern=pattern)

        try:
            full_pattern = os.path.join(base_dir, pattern)
            matches = glob_module.glob(full_pattern, recursive=True)

            # Filter to files only
            files = [m for m in matches if os.path.isfile(m)]

            # Convert to relative paths
            rel_files = [os.path.relpath(f, base_dir) for f in files]

            # Apply include/exclude filters
            if include:
                rel_files = [f for f in rel_files if _matches_pattern(f, include)]
            if exclude:
                rel_files = [f for f in rel_files if not _matches_pattern(f, exclude)]

            # Sort by modification time (newest first)
            rel_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(base_dir, f)),
                reverse=True,
            )

            if not rel_files:
                return ToolResult(
                    success=True,
                    output=f"No files found matching pattern: {pattern}",
                )

            output = f"Found {len(rel_files)} files matching '{pattern}':\n\n"
            output += "\n".join(rel_files)

            logger.info("code_tools.glob.success", pattern=pattern, count=len(rel_files))
            return ToolResult(success=True, output=output)
        except Exception as e:
            logger.error("code_tools.glob.error", pattern=pattern, error=str(e))
            return ToolResult(success=False, output=None, error=f"Glob search failed: {e}")

    return ToolDefinition(
        name="code_glob",
        description=(
            "Find files matching a glob pattern in a linked code repository. "
            "Returns file paths sorted by modification time (newest first). "
            "Supports ** for recursive matching."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '**/*.py', 'src/*.ts')",
                }
            },
            "required": ["pattern"],
        },
        handler=handler,
    )


def create_code_search_tool(
    base_dir: str,
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
) -> ToolDefinition:
    """Create a search tool for finding content in a linked code repository.

    Args:
        base_dir: Root directory of the code repository
        include_patterns: Glob patterns for allowed files (empty/None = all)
        exclude_patterns: Glob patterns for denied files (empty/None = none)

    Returns:
        ToolDefinition for the code_search tool
    """
    import shlex

    logger.info("code_tools.create_search", base_dir=base_dir)

    async def handler(params: Dict[str, Any]) -> ToolResult:
        pattern = params.get("pattern")
        path = params.get("path", ".")
        case_insensitive = params.get("case_insensitive", False)

        if not pattern:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: pattern",
            )

        logger.debug("code_tools.search.start", pattern=pattern, path=path)

        try:
            # Build grep command
            flags = "-rn"
            if case_insensitive:
                flags += "i"

            search_path = os.path.join(base_dir, path) if path != "." else base_dir
            cmd = f"grep {flags} {shlex.quote(pattern)} {shlex.quote(search_path)}"

            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            output = stdout.decode("utf-8", errors="replace")

            if not output.strip():
                return ToolResult(
                    success=True,
                    output=f"No matches found for pattern: {pattern}",
                )

            # Convert absolute paths to relative paths in output
            output = output.replace(base_dir + os.sep, "")

            logger.info(
                "code_tools.search.success",
                pattern=pattern,
                match_lines=output.count("\n"),
            )
            return ToolResult(success=True, output=output)
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error="Search timed out after 30 seconds",
            )
        except Exception as e:
            logger.error("code_tools.search.error", pattern=pattern, error=str(e))
            return ToolResult(success=False, output=None, error=f"Search failed: {e}")

    return ToolDefinition(
        name="code_search",
        description=(
            "Search for a pattern in files within a linked code repository. "
            "Uses grep-style regex pattern matching. Returns matching lines with "
            "file paths and line numbers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory to search in (default: repo root)",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)",
                },
            },
            "required": ["pattern"],
        },
        handler=handler,
    )


def create_code_list_directory_tool(
    base_dir: str,
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
) -> ToolDefinition:
    """Create a directory listing tool scoped to a linked code repository.

    Args:
        base_dir: Root directory of the code repository
        include_patterns: Glob patterns for allowed files (empty/None = all)
        exclude_patterns: Glob patterns for denied files (empty/None = none)

    Returns:
        ToolDefinition for the code_list_directory tool
    """
    logger.info("code_tools.create_list_directory", base_dir=base_dir)

    async def handler(params: Dict[str, Any]) -> ToolResult:
        path = params.get("path", ".")

        # Resolve path within base_dir
        if path == "." or not path:
            full_path = base_dir
        else:
            normalized = os.path.normpath(path).lstrip("/")
            if ".." in normalized:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Path traversal not allowed",
                )
            full_path = os.path.join(base_dir, normalized)

        full_path = os.path.normpath(full_path)

        logger.debug("code_tools.list_directory.start", path=path, full_path=full_path)

        if not os.path.exists(full_path):
            return ToolResult(
                success=False,
                output=None,
                error=f"Directory not found: {path}",
            )

        if not os.path.isdir(full_path):
            return ToolResult(
                success=False,
                output=None,
                error=f"Path is not a directory: {path}. Use code_read_file instead.",
            )

        try:
            entries = []
            for name in os.listdir(full_path):
                if name.startswith("."):
                    continue  # Skip hidden files

                item_path = os.path.join(full_path, name)
                mtime = datetime.fromtimestamp(os.path.getmtime(item_path))

                if os.path.isdir(item_path):
                    entries.append(
                        {
                            "name": name + "/",
                            "type": "dir",
                            "size": "-",
                            "modified": mtime.strftime("%Y-%m-%d %H:%M"),
                        }
                    )
                else:
                    size = os.path.getsize(item_path)
                    entries.append(
                        {
                            "name": name,
                            "type": "file",
                            "size": _format_size(size),
                            "modified": mtime.strftime("%Y-%m-%d %H:%M"),
                        }
                    )

            # Sort: directories first, then files, alphabetically
            entries.sort(key=lambda e: (0 if e["type"] == "dir" else 1, e["name"].lower()))

            if not entries:
                return ToolResult(
                    success=True,
                    output=f"Directory is empty: {path}",
                )

            display_path = path if path != "." else "(repo root)"
            output = f"Contents of {display_path}:\n\n"
            output += f"{'Name':<50} {'Type':<6} {'Size':<10} {'Modified'}\n"
            output += "-" * 85 + "\n"
            for e in entries:
                output += f"{e['name']:<50} {e['type']:<6} {e['size']:<10} {e['modified']}\n"
            output += f"\nTotal: {len(entries)} items"

            logger.info(
                "code_tools.list_directory.success",
                path=path,
                item_count=len(entries),
            )
            return ToolResult(success=True, output=output)
        except PermissionError:
            return ToolResult(success=False, output=None, error=f"Permission denied: {path}")
        except Exception as e:
            logger.error("code_tools.list_directory.error", path=path, error=str(e))
            return ToolResult(success=False, output=None, error=f"Failed to list directory: {e}")

    return ToolDefinition(
        name="code_list_directory",
        description=(
            "List contents of a directory in a linked code repository. "
            "Returns files and directories with metadata (size, type, modification time)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to code repo root (default: repo root)",
                }
            },
            "required": [],
        },
        handler=handler,
    )


def create_write_file_tool(base_dir: str = ".") -> ToolDefinition:
    """Create a write_file tool for writing content to files.

    Supports creating new files, overwriting existing files, and appending.
    Automatically creates parent directories as needed.

    Args:
        base_dir: Base directory for resolving relative paths

    Returns:
        ToolDefinition for the write_file tool
    """
    import os

    logger = get_logger(__name__)

    async def write_file_handler(params: Dict[str, Any]) -> ToolResult:
        file_path = params.get("file_path")
        content = params.get("content")
        append = params.get("append", False)

        if not file_path:
            return ToolResult(success=False, output=None, error="Missing 'file_path' parameter")

        if content is None:
            return ToolResult(success=False, output=None, error="Missing 'content' parameter")

        # Resolve path relative to base_dir
        if not os.path.isabs(file_path):
            full_path = os.path.join(base_dir, file_path)
        else:
            full_path = file_path

        # Normalize path
        full_path = os.path.normpath(full_path)

        # Path traversal protection (matches read_file pattern)
        normalized_base = os.path.realpath(base_dir)
        abs_target = os.path.abspath(full_path)
        if not abs_target.startswith(normalized_base + os.sep) and abs_target != normalized_base:
            return ToolResult(
                success=False,
                output=None,
                error="Access denied: path traversal not allowed. Path must be within base directory.",
            )

        # Block symlinks pointing outside base_dir
        if os.path.islink(full_path):
            resolved = os.path.realpath(full_path)
            if not resolved.startswith(normalized_base + os.sep) and resolved != normalized_base:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Access denied: symlink target is outside base directory.",
                )

        try:
            # Create parent directories
            os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)

            mode = "a" if append else "w"
            with open(full_path, mode, encoding="utf-8") as f:
                f.write(content)

            file_size = os.path.getsize(full_path)
            action = "Appended to" if append else "Wrote"
            logger.info(
                "tool.write_file.complete",
                file_path=file_path,
                bytes_written=len(content),
                total_size=file_size,
                append=append,
            )
            return ToolResult(
                success=True,
                output=f"{action} {file_path} ({len(content)} bytes written, total size: {file_size} bytes)",
                error=None,
            )
        except (OSError, ValueError) as e:
            logger.error("tool.write_file.error", file_path=file_path, error=str(e))
            return ToolResult(success=False, output=None, error=f"Failed to write file: {e}")

    return ToolDefinition(
        name="write_file",
        description="Write content to a file. Creates parent directories automatically. Use append=true to add to an existing file instead of overwriting.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write (relative to workspace root)",
                },
                "content": {"type": "string", "description": "Content to write to the file"},
                "append": {
                    "type": "boolean",
                    "description": "If true, append to the file instead of overwriting. Default: false",
                },
            },
            "required": ["file_path", "content"],
        },
        handler=write_file_handler,
    )


# ============================================================================
# Tool Registries (Gemini-CLI Compatible)
# ============================================================================

# Phase 0: Built-in tools registry (file tools that don't require sandbox)
BUILTIN_TOOLS = {
    "read_file": create_read_file_tool,
    "write_file": create_write_file_tool,
    "glob": create_glob_tool,
    "list_directory": create_list_directory_tool,
    "read_many_files": create_read_many_files_tool,
}

# Phase 0: Tools that require a sandbox (read-write workspace)
SANDBOX_TOOLS = {
    "run_shell_command": create_bash_tool,
    "search_file_content": create_grep_tool,
}


# Phase 0: Standalone tools (don't require sandbox)
def _create_new_web_fetch_tool(sandbox=None, **kwargs):
    """default web_fetch — uses the improved WebFetcher with ProviderSummarizer (haiku)."""
    try:
        from .tools.web_fetch import WebFetcher, ProviderSummarizer, create_web_fetch_tool

        # wire up haiku as the default summarizer for prompt= queries
        summarizer = None
        try:
            from .test_builder import create_provider

            provider = create_provider("anthropic", model="claude-haiku-4-5-20251001")
            if provider:
                summarizer = ProviderSummarizer(provider)
        except Exception:
            pass  # no provider available — web_fetch works fine without summarizer

        fetcher = WebFetcher(summarizer=summarizer)
        return create_web_fetch_tool(fetcher, sandbox=sandbox)
    except Exception:
        # fallback to legacy if anything goes wrong
        return create_http_request_tool(sandbox=sandbox)


# legacy_web_fetch: the old raw HTTP tool, kept as fallback
legacy_web_fetch = create_http_request_tool

STANDALONE_TOOLS = {
    "web_fetch": _create_new_web_fetch_tool,
    "legacy_web_fetch": legacy_web_fetch,
    "web_search": create_perplexity_web_search_tool,
    "anthropic_web_search": create_anthropic_web_search_tool,
}

# Phase 0: No context tools (delegate_to_agent is Phase 1+)
CONTEXT_TOOLS = {}

# task tools — registered when tasks.enabled=true in config
# these let the executor create/track subtasks mid-run
TASK_TOOLS = {}


def _register_task_tools():
    """lazily register task tools from dokumen.tasks.tools."""
    global TASK_TOOLS
    if TASK_TOOLS:
        return
    try:
        from .tasks.tools import TASK_TOOL_DEFINITIONS

        for defn in TASK_TOOL_DEFINITIONS:
            name = defn["name"]
            handler = defn["handler"]
            TASK_TOOLS[name] = lambda _sandbox=None, _defn=defn, _handler=handler: ToolDefinition(
                name=_defn["name"],
                description=_defn["description"],
                parameters=_defn["parameters"],
                handler=_handler,
            )
        logger.info("tools.task_tools_registered", extra={"count": len(TASK_TOOLS)})
    except ImportError as e:
        logger.debug("tools.task_tools_unavailable", extra={"error": str(e)})


# Code repository tools (read-only, for cross-referencing docs with code)
CODE_TOOLS = {
    "code_read_file": create_code_read_file_tool,
    "code_glob": create_code_glob_tool,
    "code_search": create_code_search_tool,
    "code_list_directory": create_code_list_directory_tool,
}


def resolve_builtin_tool(
    tool_name: str,
    base_dir: str = ".",
    sandbox: Optional["Sandbox"] = None,
    perplexity_config: Optional[dict] = None,
) -> Optional[ToolDefinition]:
    """Resolve a built-in tool by name.

    Args:
        tool_name: Name of the built-in tool
        base_dir: Base directory for file-based tools
        sandbox: Optional sandbox for sandbox-based tools
        perplexity_config: Optional dict with api_key, model, max_searches for web_search

    Returns:
        ToolDefinition if found, None otherwise
    """
    logger.debug("tools.resolve.start", tool_name=tool_name, has_sandbox=sandbox is not None)

    # File tools (read-only, can access any project file)
    if tool_name in BUILTIN_TOOLS:
        logger.debug("tools.resolve.builtin", tool_name=tool_name)
        return BUILTIN_TOOLS[tool_name](base_dir)

    # Sandbox tools (require sandbox)
    if tool_name in SANDBOX_TOOLS:
        if sandbox is not None:
            logger.debug("tools.resolve.sandbox", tool_name=tool_name)
            return SANDBOX_TOOLS[tool_name](sandbox)
        else:
            # For run_shell_command and search_file_content, allow without sandbox
            if tool_name == "run_shell_command":
                logger.debug("tools.resolve.sandbox_fallback", tool_name=tool_name)
                return create_bash_tool(sandbox=None, base_dir=base_dir)
            if tool_name == "search_file_content":
                logger.debug("tools.resolve.sandbox_fallback", tool_name=tool_name)
                return create_grep_tool(sandbox=None, base_dir=base_dir)
            logger.warning("tools.resolve.sandbox_required", tool_name=tool_name)
            return None

    # Standalone tools (can optionally use sandbox)
    if tool_name in STANDALONE_TOOLS:
        logger.debug("tools.resolve.standalone", tool_name=tool_name)
        if tool_name in ("web_fetch", "legacy_web_fetch"):
            return STANDALONE_TOOLS[tool_name](sandbox=sandbox)
        if tool_name == "web_search":
            cfg = perplexity_config or {}
            return STANDALONE_TOOLS[tool_name](
                api_key=cfg.get("api_key"),
                model=cfg.get("model", "sonar"),
                max_searches=cfg.get("max_searches", 5),
            )
        return STANDALONE_TOOLS[tool_name]()

    # task tools (registered lazily when first requested)
    _register_task_tools()
    if tool_name in TASK_TOOLS:
        logger.debug("tools.resolve.task_tool", tool_name=tool_name)
        return TASK_TOOLS[tool_name]()

    logger.warning("tools.resolve.not_found", tool_name=tool_name)
    return None


def get_all_tool_names() -> List[str]:
    """Get names of all available tools.

    Returns:
        List of all tool names
    """
    _register_task_tools()
    return (
        list(BUILTIN_TOOLS.keys())
        + list(SANDBOX_TOOLS.keys())
        + list(STANDALONE_TOOLS.keys())
        + list(CONTEXT_TOOLS.keys())
        + list(CODE_TOOLS.keys())
        + list(TASK_TOOLS.keys())
    )


def create_create_test_tool(base_dir: str = ".") -> ToolDefinition:
    """Create a tool for generating test scaffolds.

    This tool uses the CreateAgent to generate test scaffolds from
    natural language goals. The scaffold is returned for review before
    being written to a file.

    Args:
        base_dir: Base directory for the project

    Returns:
        ToolDefinition for the create_test tool
    """
    from pathlib import Path

    async def create_test_handler(params: Dict[str, Any]) -> ToolResult:
        goal = params.get("goal")
        files = params.get("files", [])
        test_type = params.get("type", "standard")

        if not goal:
            return ToolResult(success=False, output=None, error="Missing required parameter: goal")

        # Validate test_type
        if test_type not in ("standard", "browser"):
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid type: '{test_type}'. Must be 'standard' or 'browser'",
            )

        try:
            # Import CreateAgent here to avoid circular imports
            from .create_agent import CreateAgent
            from .loader import get_configured_provider

            # Get existing test names to avoid conflicts
            tests_dir = Path(base_dir) / "tests"
            existing_tests = []
            if tests_dir.exists():
                for test_file in tests_dir.glob("*.test.yaml"):
                    # Extract test name from filename
                    existing_tests.append(test_file.stem.replace(".test", ""))

            # Create agent and generate scaffold
            provider = get_configured_provider()
            agent = CreateAgent(
                provider=provider,
                base_dir=base_dir,
            )

            result = await agent.create(
                goal=goal,
                files=files if files else None,
                existing_tests=existing_tests if existing_tests else None,
                test_type=test_type,
            )

            if result.success:
                return ToolResult(
                    success=True,
                    output={
                        "test_name": result.name,
                        "scaffold_yaml": result.scaffold_yaml,
                        "discovered_files": result.discovered_files,
                        "suggested_path": f"tests/{result.name}.test.yaml",
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=result.error or "Failed to generate test scaffold",
                )

        except Exception as e:
            logger.error(f"create_test failed: {e}", exc_info=True)
            return ToolResult(success=False, output=None, error=str(e))

    return ToolDefinition(
        name="create_test",
        description=(
            "Generate a test scaffold for validating documentation. "
            "Takes a goal describing what to validate and optionally specific files to test. "
            "Returns the generated YAML scaffold for review before saving. "
            "After reviewing, use write_file to save to tests/{name}.test.yaml."
        ),
        parameters={
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": (
                        "What the test should validate "
                        "(e.g., 'Verify refund policy handles 30-day returns')"
                    ),
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Test name in kebab-case (optional, auto-generated from goal if not provided)"
                    ),
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Documentation files to test (optional, auto-discovered if not provided)"
                    ),
                },
                "type": {
                    "type": "string",
                    "enum": ["standard", "browser"],
                    "description": (
                        "Test type: 'standard' for documentation validation (default), "
                        "'browser' for browser automation tests"
                    ),
                },
            },
            "required": ["goal"],
        },
        handler=create_test_handler,
    )


def create_chat_write_file_tool(base_dir: str = ".") -> ToolDefinition:
    """Create a write_file tool for chat sessions.

    This tool writes files to the local clone filesystem. Changes are
    batch-committed to GitLab at the end of the conversation turn by
    the backend AskHandler.

    Args:
        base_dir: Base directory for file operations (local clone path)

    Returns:
        ToolDefinition for the write_file tool
    """
    from pathlib import Path

    # Allowed directory prefixes for writes
    ALLOWED_WRITE_PREFIXES = ["docs/", "tests/"]

    async def write_file_handler(params: Dict[str, Any]) -> ToolResult:
        path = params.get("path")
        content = params.get("content")

        if not path:
            return ToolResult(success=False, output=None, error="Missing required parameter: path")

        if content is None:
            return ToolResult(
                success=False, output=None, error="Missing required parameter: content"
            )

        # Validate path: no traversal
        if ".." in path:
            return ToolResult(
                success=False, output=None, error=f"Path traversal not allowed: {path}"
            )

        # Validate path: must start with allowed prefix
        normalized = path.lstrip("/")
        if not any(normalized.startswith(prefix) for prefix in ALLOWED_WRITE_PREFIXES):
            return ToolResult(
                success=False,
                output=None,
                error=f"Write not allowed outside docs/ and tests/: {path}",
            )

        # Write to local filesystem
        try:
            full_path = Path(base_dir) / normalized
            existed = full_path.exists()
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")

            action = "updated" if existed else "created"
            return ToolResult(
                success=True,
                output={
                    "action": action,
                    "path": normalized,
                    "bytes": len(content.encode("utf-8")),
                    "message": f"File {action}: {normalized}",
                },
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Failed to write file: {str(e)}")

    return ToolDefinition(
        name="write_file",
        description=(
            "Write a file to the local clone. "
            "Use this to save test scaffolds, documentation updates, or any file changes. "
            "Changes are batch-committed to your user branch at the end of the turn. "
            "Only files in docs/ and tests/ can be written."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File path relative to project root "
                        "(e.g., 'tests/my-test.test.yaml' or 'docs/policy.md')"
                    ),
                },
                "content": {"type": "string", "description": "The file content to write"},
            },
            "required": ["path", "content"],
        },
        handler=write_file_handler,
    )


def create_chat_delete_file_tool(base_dir: str = ".") -> ToolDefinition:
    """Create a delete_file tool for chat sessions.

    This tool deletes files from the local clone filesystem. Deletions are
    batch-committed to GitLab at the end of the conversation turn by
    the backend AskHandler.

    Args:
        base_dir: Base directory for file operations (local clone path)

    Returns:
        ToolDefinition for the delete_file tool
    """
    from pathlib import Path

    # Allowed directory prefixes for deletions (same as writes)
    ALLOWED_WRITE_PREFIXES = ["docs/", "tests/"]

    async def delete_file_handler(params: Dict[str, Any]) -> ToolResult:
        path = params.get("path")

        if not path:
            return ToolResult(success=False, output=None, error="Missing required parameter: path")

        # Validate path: no traversal
        if ".." in path:
            return ToolResult(
                success=False, output=None, error=f"Path traversal not allowed: {path}"
            )

        # Validate path: must start with allowed prefix
        normalized = path.lstrip("/")
        if not any(normalized.startswith(prefix) for prefix in ALLOWED_WRITE_PREFIXES):
            return ToolResult(
                success=False,
                output=None,
                error=f"Delete not allowed outside docs/ and tests/: {path}",
            )

        # Check file exists
        full_path = Path(base_dir) / normalized
        if not full_path.exists():
            return ToolResult(success=False, output=None, error=f"File not found: {normalized}")

        # Delete the file
        try:
            full_path.unlink()
            logger.info(
                "chat_delete_file.success",
                path=normalized,
                base_dir=base_dir,
            )
            return ToolResult(
                success=True,
                output={
                    "action": "deleted",
                    "path": normalized,
                    "message": f"File deleted: {normalized}",
                },
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Failed to delete file: {str(e)}")

    return ToolDefinition(
        name="delete_file",
        description=(
            "Delete a file from docs/ or tests/. "
            "The deletion is committed to your user branch at the end of the turn. "
            "Only files in docs/ and tests/ can be deleted."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File path relative to project root "
                        "(e.g., 'tests/old-test.test.yaml' or 'docs/outdated.md')"
                    ),
                }
            },
            "required": ["path"],
        },
        handler=delete_file_handler,
    )


def create_re_explore_tool() -> ToolDefinition:
    """Create a re_explore tool for chat sessions.

    This tool allows the agent to re-explore the codebase with a new topic
    when the initial exploration found the wrong files or the user indicates
    a misunderstanding.

    The tool is handled specially by the AskAgent - when called, it resets
    the session and runs a fresh explore phase before continuing.

    Returns:
        ToolDefinition for the re_explore tool
    """

    async def re_explore_handler(params: Dict[str, Any]) -> ToolResult:
        topic = params.get("topic", "")

        if not topic:
            return ToolResult(success=False, output=None, error="Missing required parameter: topic")

        # The actual re-explore is handled by AskAgent when it sees this tool
        # This handler just validates the params
        return ToolResult(
            success=True,
            output={
                "action": "re_explore_requested",
                "topic": topic,
                "message": f"Re-exploring codebase for: {topic}",
            },
        )

    return ToolDefinition(
        name="re_explore",
        description=(
            "Re-explore the codebase to find different or additional relevant files. "
            "Use this when you realize the initial exploration found the wrong files, "
            "or the user indicates you're looking at incorrect documentation. "
            "Provide a topic describing what you're now looking for. "
            "Examples: 'authentication API endpoints', 'refund policy documentation', "
            "'margin trading rules'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "What to search for in the re-exploration "
                        "(e.g., 'authentication API endpoints' or 'refund policy documentation')"
                    ),
                }
            },
            "required": ["topic"],
        },
        handler=re_explore_handler,
    )


# ============================================================================
# Agent Tools (explore/ask — run sub-agents as executor tools)
# ============================================================================


def _create_explore_provider(config):
    """Create an LLM provider for explore/ask agent tools.

    Uses the explore model from config if available, falling back to the
    main provider model. Requires ANTHROPIC_API_KEY in the environment.

    Args:
        config: DokumenConfig instance

    Returns:
        Provider instance for explore/ask operations
    """
    from .providers.anthropic import AnthropicProvider

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — cannot create provider for agent tools")
    model = getattr(config.explore, "model", None) or config.provider.model
    logger.info(
        "agent_tools.create_provider",
        model=model,
        has_api_key=True,
    )
    return AnthropicProvider(api_key=api_key, model=model)


def create_explore_tool(config, project_root: str) -> ToolDefinition:
    """Create an explore tool that runs ExploreAgent as an executor tool.

    Args:
        config: DokumenConfig instance
        project_root: Root directory of the project

    Returns:
        ToolDefinition for the explore tool
    """
    logger.info(
        "agent_tools.create_explore_tool",
        project_root=project_root,
    )

    async def explore_handler(params: Dict[str, Any]) -> ToolResult:
        query = params.get("query")
        if not query:
            logger.warning("agent_tools.explore.missing_query")
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: query",
            )

        explore_type = params.get("explore_type", "both")
        if explore_type not in ("docs", "code", "both"):
            logger.warning("agent_tools.explore.invalid_explore_type", explore_type=explore_type)
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid explore_type '{explore_type}'. Must be 'docs', 'code', or 'both'.",
            )
        logger.info(
            "agent_tools.explore.start",
            query=query[:100],
            explore_type=explore_type,
        )

        try:
            from .explore_agent import ExploreAgent

            provider = _create_explore_provider(config)
            max_files = getattr(config.explore, "max_files", 20)

            agent = ExploreAgent(
                provider=provider,
                base_dir=project_root,
                max_files=max_files,
                explore_type=explore_type,
            )

            result = await agent.explore(query)

            if not result.success:
                logger.warning(
                    "agent_tools.explore.failed",
                    error=result.error,
                )
                return ToolResult(
                    success=False,
                    output=None,
                    error=result.error or "Exploration failed",
                )

            # Build output summary
            file_list = "\n".join(f"- {f.path}: {f.summary}" for f in result.files)
            output = (
                f"Exploration complete. Found {len(result.files)} relevant files "
                f"in {result.duration:.1f}s.\n\n"
                f"{result.summary or ''}\n\n"
                f"Files:\n{file_list}"
            )

            logger.info(
                "agent_tools.explore.complete",
                files_found=len(result.files),
                duration_ms=int(result.duration * 1000),
            )
            return ToolResult(success=True, output=output)

        except Exception as e:
            logger.error(
                "agent_tools.explore.error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return ToolResult(
                success=False,
                output=None,
                error=f"Explore failed: {e}",
            )

    return ToolDefinition(
        name="explore",
        description=(
            "Explore the codebase to discover relevant files for a given "
            "topic or question. Returns a summary and list of relevant file "
            "paths with descriptions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to explore — a topic, question, or area of "
                        "interest (e.g., 'authentication flow', 'API endpoints')"
                    ),
                },
                "explore_type": {
                    "type": "string",
                    "description": (
                        "Type of exploration: 'docs' for documentation only, "
                        "'code' for source code only, 'both' for both"
                    ),
                    "enum": ["docs", "code", "both"],
                    "default": "both",
                },
            },
            "required": ["query"],
        },
        handler=explore_handler,
    )


def create_ask_tool(config, project_root: str) -> ToolDefinition:
    """Create an ask tool that runs AskAgent as an executor tool.

    Args:
        config: DokumenConfig instance
        project_root: Root directory of the project

    Returns:
        ToolDefinition for the ask tool
    """
    logger.info(
        "agent_tools.create_ask_tool",
        project_root=project_root,
    )

    async def ask_handler(params: Dict[str, Any]) -> ToolResult:
        question = params.get("question")
        if not question:
            logger.warning("agent_tools.ask.missing_question")
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: question",
            )

        logger.info(
            "agent_tools.ask.start",
            question=question[:100],
        )

        try:
            from .ask_agent import AskAgent

            provider = _create_explore_provider(config)

            agent = AskAgent(
                provider=provider,
                base_dir=project_root,
            )

            result = await agent.ask(question)

            if not result.success:
                logger.warning(
                    "agent_tools.ask.failed",
                    error=result.error,
                )
                return ToolResult(
                    success=False,
                    output=None,
                    error=result.error or "Ask failed",
                )

            # Build output with answer and sources
            sources_str = ""
            if result.sources:
                sources_list = [str(s) for s in result.sources]
                sources_str = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources_list)

            output = f"{result.answer}{sources_str}"

            logger.info(
                "agent_tools.ask.complete",
                sources_count=len(result.sources),
                duration_ms=int(result.duration * 1000),
            )
            return ToolResult(success=True, output=output)

        except Exception as e:
            logger.error(
                "agent_tools.ask.error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return ToolResult(
                success=False,
                output=None,
                error=f"Ask failed: {e}",
            )

    return ToolDefinition(
        name="ask",
        description=(
            "Ask a question about the documentation and get an AI-generated "
            "answer grounded in the actual docs. Returns the answer with "
            "source references."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The question to ask about the documentation "
                        "(e.g., 'How does authentication work?', "
                        "'What is the refund policy?')"
                    ),
                },
            },
            "required": ["question"],
        },
        handler=ask_handler,
    )


# Agent tools dict — maps tool name to factory function
# Factories take (config, project_root) and return ToolDefinition
AGENT_TOOLS = {
    "explore": create_explore_tool,
    "ask": create_ask_tool,
}
