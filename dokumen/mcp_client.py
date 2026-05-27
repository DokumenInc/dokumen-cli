"""
Playwright MCP Client for browser automation.

Manages subprocess communication with @playwright/mcp server via JSON-RPC over stdio.
"""
import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

from .tools_object import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class MCPResponse:
    """Response from MCP server."""
    success: bool
    result: Any
    error: Optional[str] = None


class PlaywrightMCPClient:
    """MCP client using stdio subprocess communication with Playwright server.

    Usage:
        async with PlaywrightMCPClient(headless=False, save_video="1920x1080") as client:
            result = await client.call_tool("browser_navigate", {"url": "https://example.com"})
    """

    def __init__(
        self,
        headless: bool = True,
        save_video: Optional[str] = None,
        viewport_size: Optional[str] = None,
        output_dir: str = ".dokumen-cache/recordings",
        timeout: float = 30.0,
        visual_indicators: bool = True,
        use_fork: bool = True
    ):
        """Initialize Playwright MCP client.

        Args:
            headless: Run browser in headless mode (default True)
            save_video: Video size to record, e.g. "1920x1080" (None to disable)
            viewport_size: Browser viewport size, e.g. "1512x982" (None for default)
            output_dir: Directory for video/screenshot output
            timeout: Timeout for tool calls in seconds
            visual_indicators: Show click circles and screenshot flashes (default True)
            use_fork: Use local Playwright MCP fork with visual indicators (default True)
        """
        self.headless = headless
        self.save_video = save_video
        self.viewport_size = viewport_size
        self.output_dir = output_dir
        self.timeout = timeout
        self.visual_indicators = visual_indicators
        self.use_fork = use_fork
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._exit_task: Optional[asyncio.Task[None]] = None
        self._stderr_lines: Deque[str] = deque(maxlen=200)
        self._stopping = False

        logger.info(
            "PlaywrightMCPClient initialized",
            extra={
                "headless": headless,
                "save_video": save_video,
                "viewport_size": viewport_size,
                "output_dir": output_dir,
                "visual_indicators": visual_indicators,
                "use_fork": use_fork
            }
        )

    async def start(self) -> None:
        """Start the Playwright MCP subprocess.

        Uses local Playwright fork with built-in visual indicators when use_fork=True.
        Falls back to npx @playwright/mcp@latest with init-script injection otherwise.
        """
        import os

        if self._process is not None:
            logger.warning("MCP client already started")
            return

        # Build command arguments
        if self.use_fork:
            # Use local Playwright fork with built-in visual indicators when the build exists.
            fork_path = os.environ.get("PLAYWRIGHT_MCP_PATH")
            if not fork_path:
                fork_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "playwright-mcp-fork"
                )
            cli_path = os.path.join(fork_path, "packages", "playwright", "cli.js")
            built_program = os.path.join(fork_path, "packages", "playwright", "lib", "program.js")
            if os.path.exists(cli_path) and os.path.exists(built_program):
                cmd = ["node", cli_path, "run-mcp-server"]
                self._fork_cwd = None
                logger.info(
                    "Using Playwright MCP fork",
                    extra={"fork_path": fork_path, "cli_path": cli_path, "env_path": os.environ.get("PLAYWRIGHT_MCP_PATH")}
                )
            else:
                logger.warning(
                    "Playwright MCP fork unavailable or unbuilt; falling back to upstream package",
                    extra={"fork_path": fork_path, "cli_path": cli_path, "built_program": built_program}
                )
                cmd = ["npx", "@playwright/mcp@latest"]
                self._fork_cwd = None
        else:
            cmd = ["npx", "@playwright/mcp@latest"]
            self._fork_cwd = None

        # Always start with isolated profile (no saved cookies/session)
        cmd.append("--isolated")

        # Disable Chromium sandbox when requested or when running as root.
        sandbox_env = (os.environ.get("PLAYWRIGHT_MCP_SANDBOX") or "").lower()
        sandbox_disabled = sandbox_env in ("0", "false", "no", "off")
        sandbox_enabled = sandbox_env in ("1", "true", "yes", "on")
        if sandbox_disabled:
            cmd.append("--no-sandbox")
        elif hasattr(os, "geteuid") and os.geteuid() == 0 and not sandbox_enabled:
            cmd.append("--no-sandbox")

        if self.headless:
            cmd.append("--headless")

        if self.save_video:
            cmd.extend(["--save-video", self.save_video])
            os.makedirs(self.output_dir, exist_ok=True)
            cmd.extend(["--output-dir", self.output_dir])

        if self.viewport_size:
            cmd.extend(["--viewport-size", self.viewport_size])

        # Add visual indicators flag (only works with fork)
        if self.visual_indicators and self.use_fork and cmd[:2] != ["npx", "@playwright/mcp@latest"]:
            cmd.append("--visual-indicators")
        elif self.visual_indicators and (not self.use_fork or cmd[:2] == ["npx", "@playwright/mcp@latest"]):
            # Fallback: inject init-script for upstream package
            from .playwright_tools import CLICK_INDICATOR_SCRIPT
            os.makedirs(self.output_dir, exist_ok=True)
            init_script_path = os.path.join(self.output_dir, "click-indicator.js")
            with open(init_script_path, "w") as f:
                f.write(CLICK_INDICATOR_SCRIPT)
            cmd.extend(["--init-script", init_script_path])
            logger.debug("Wrote click indicator script", extra={"path": init_script_path})

        logger.info(
            "Playwright MCP environment",
            extra={
                "user": os.environ.get("USER"),
                "home": os.environ.get("HOME"),
                "playwright_browsers_path": os.environ.get("PLAYWRIGHT_BROWSERS_PATH"),
                "cwd": os.getcwd(),
                "uid": os.geteuid() if hasattr(os, "geteuid") else None,
            },
        )
        logger.info(
            "Starting Playwright MCP server",
            extra={
                "command": " ".join(cmd),
                "use_fork": self.use_fork,
                "visual_indicators": self.visual_indicators,
            },
        )

        # Start in new process group so we can kill Chrome and all children
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # Creates new process group
            cwd=self._fork_cwd,  # Use fork directory if available
            limit=10 * 1024 * 1024  # Allow large MCP responses (e.g., screenshots)
        )

        self._stopping = False
        self._stderr_task = asyncio.create_task(self._log_stderr())
        self._exit_task = asyncio.create_task(self._watch_process())
        logger.info("Playwright MCP server started", extra={
            "pid": self._process.pid,
            "cwd": self._fork_cwd
        })

        # Wait for MCP server to be ready before returning
        await self._wait_for_ready()

    async def _log_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                if not isinstance(line, (bytes, bytearray)):
                    line_text = str(line).rstrip()
                    self._stderr_lines.append(line_text)
                    if "[Dokumen]" in line_text:
                        logger.debug("Playwright MCP stderr: %s", line_text)
                    else:
                        logger.warning("Playwright MCP stderr: %s", line_text)
                    break
                line_text = line.decode(errors="replace").rstrip()
                self._stderr_lines.append(line_text)
                if "[Dokumen]" in line_text:
                    logger.debug("Playwright MCP stderr: %s", line_text)
                else:
                    logger.warning("Playwright MCP stderr: %s", line_text)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("Playwright MCP stderr reader stopped", extra={"error": str(exc)})

    async def _watch_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            returncode = await process.wait()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("MCP process watch error: %s", str(exc))
            return

        stderr_tail = self._format_stderr_tail()
        if self._stopping:
            logger.info("Playwright MCP server exited", extra={"returncode": returncode})
            return
        if stderr_tail:
            logger.error(
                "Playwright MCP server exited unexpectedly (code %s). Stderr tail:\n%s",
                returncode,
                stderr_tail
            )
        else:
            logger.error("Playwright MCP server exited unexpectedly (code %s).", returncode)

    def _format_stderr_tail(self) -> str:
        if not self._stderr_lines:
            return ""
        return "\n".join(self._stderr_lines)

    async def _wait_for_ready(self, max_retries: int = 10, delay: float = 0.5) -> None:
        """Wait for MCP server to be ready to accept requests.

        Sends a browser_snapshot probe to verify the server is responsive.
        The probe may fail with "no page" error, which indicates readiness.

        Args:
            max_retries: Maximum number of attempts (default 10)
            delay: Seconds to wait between attempts (default 0.5)
        """
        for attempt in range(max_retries):
            try:
                result = await self.call_tool("browser_snapshot", {}, timeout=5.0)
                # Success or "no page" error both indicate server is ready
                if result.success or (result.error and "no page" in result.error.lower()):
                    logger.info("MCP server ready", extra={"attempts": attempt + 1})
                    return
                logger.debug(
                    "MCP readiness probe returned unexpected result",
                    extra={"attempt": attempt, "result": result.output, "error": result.error}
                )
            except Exception as e:
                logger.debug(
                    "MCP not ready yet",
                    extra={"attempt": attempt + 1, "error": str(e)}
                )
            await asyncio.sleep(delay)
        logger.warning(
            "MCP server may not be fully ready after all retries",
            extra={"max_retries": max_retries}
        )

    async def stop(self) -> None:
        """Stop the Playwright MCP subprocess and all children (including Chrome).

        Uses graceful SIGTERM to process group with 5-second timeout, then SIGKILL.
        """
        import os
        import signal

        if self._process is None:
            return

        self._stopping = True

        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("Playwright MCP stderr task error", extra={"error": str(exc)})
        self._stderr_task = None

        pid = self._process.pid
        logger.info("Stopping Playwright MCP server", extra={"pid": pid})

        # Try to kill entire process group, fall back to just the process
        def kill_process_group(sig: int) -> bool:
            """Kill process group, return True if successful."""
            try:
                os.killpg(pid, sig)
                return True
            except (ProcessLookupError, PermissionError, OSError):
                # Process already dead, or not a real process group (e.g., in tests)
                return False

        # Send SIGTERM - try process group first, fall back to process
        if not kill_process_group(signal.SIGTERM):
            self._process.terminate()

        try:
            # Wait up to 5 seconds for graceful shutdown
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
            logger.info("Playwright MCP server stopped gracefully")
        except asyncio.TimeoutError:
            # Force kill - try process group first, fall back to process
            logger.warning(
                "MCP server didn't respond to SIGTERM, sending SIGKILL",
                extra={"pid": pid}
            )
            if not kill_process_group(signal.SIGKILL):
                self._process.kill()
            await self._process.wait()
            logger.info("Playwright MCP server force-killed")

        self._process = None
        if self._exit_task and not self._exit_task.done():
            try:
                await self._exit_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("Playwright MCP exit task error", extra={"error": str(exc)})
        self._exit_task = None

    async def call_tool(
        self, name: str, arguments: Dict[str, Any], timeout: Optional[float] = None
    ) -> ToolResult:
        """Call an MCP tool via JSON-RPC.

        Args:
            name: Tool name (e.g., "browser_navigate")
            arguments: Tool arguments
            timeout: Override timeout in seconds (uses self.timeout if None)

        Returns:
            ToolResult with success status and output/error
        """
        if self._process is None:
            raise RuntimeError("MCP client not started - call start() first")
        if self._process.returncode is not None:
            stderr_tail = self._format_stderr_tail()
            logger.error(
                "MCP server is not running (exit code %s). Stderr tail:\n%s",
                self._process.returncode,
                stderr_tail
            )
            return ToolResult(
                success=False,
                output=None,
                error=f"MCP server exited with code {self._process.returncode}"
            )

        effective_timeout = timeout if timeout is not None else self.timeout

        async with self._lock:
            self._request_id += 1
            request_id = self._request_id

        # Build JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }

        logger.debug(
            "Sending MCP request",
            extra={"tool": name, "request_id": request_id, "arguments": arguments}
        )

        try:
            # Send request
            request_data = json.dumps(request).encode() + b"\n"
            self._process.stdin.write(request_data)
            await self._process.stdin.drain()

            # Wait for response with timeout
            response_data = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=effective_timeout
            )

            if not response_data:
                return ToolResult(
                    success=False,
                    output=None,
                    error="No response from MCP server"
                )

            response = json.loads(response_data.decode())

            logger.debug(
                "Received MCP response",
                extra={"request_id": request_id, "response": response}
            )

            # Handle JSON-RPC error
            if "error" in response:
                error_msg = response["error"].get("message", "Unknown error")
                return ToolResult(
                    success=False,
                    output=None,
                    error=error_msg
                )

            # Parse result
            result = response.get("result", {})
            content = result.get("content", [])

            # Extract text content from MCP response
            output_text = ""
            for item in content:
                if item.get("type") == "text":
                    output_text += item.get("text", "")
                elif item.get("type") == "image":
                    output_text += f"[Image: {item.get('mimeType', 'image')}]"

            return ToolResult(
                success=True,
                output=output_text if output_text else result,
                error=None
            )

        except asyncio.TimeoutError:
            logger.error(
                "MCP tool call timeout",
                extra={"tool": name, "timeout": effective_timeout}
            )
            return ToolResult(
                success=False,
                output=None,
                error=f"Timeout after {effective_timeout}s waiting for MCP response"
            )
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON response from MCP", extra={"error": str(e)})
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid JSON from MCP server: {e}"
            )
        except Exception as e:
            logger.error(
                "MCP call failed: %s\nStderr tail:\n%s",
                str(e),
                self._format_stderr_tail(),
                exc_info=True
            )
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )

    async def __aenter__(self) -> "PlaywrightMCPClient":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
