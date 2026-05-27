"""
Tests for Playwright MCP client.

TDD: Tests written first, then implementation.
"""
import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestPlaywrightMCPClient:
    """Tests for PlaywrightMCPClient subprocess communication."""

    @pytest.fixture
    def mock_process(self):
        """Create mock subprocess for testing."""
        process = AsyncMock()
        process.stdin = AsyncMock()
        process.stdout = AsyncMock()
        process.stderr = AsyncMock()
        process.returncode = None
        process.pid = 12345  # Required for os.killpg in stop()
        process.terminate = MagicMock()
        process.wait = AsyncMock()
        return process

    @pytest.mark.asyncio
    async def test_start_spawns_npx_process(self, mock_process):
        """Test that start() spawns the correct npx command when use_fork=False."""
        from dokumen.mcp_client import PlaywrightMCPClient

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_process

            client = PlaywrightMCPClient(headless=True, use_fork=False)
            await client.start()

            # Should call npx with @playwright/mcp
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            assert "npx" in call_args[0]
            assert "@playwright/mcp@latest" in call_args[0]

            await client.stop()

    @pytest.mark.asyncio
    async def test_start_with_video_recording(self, mock_process, tmp_path):
        """Test start() with video recording enabled."""
        from dokumen.mcp_client import PlaywrightMCPClient

        output_dir = str(tmp_path / "recordings")

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_process

            client = PlaywrightMCPClient(
                headless=False,
                save_video="1920x1080",
                output_dir=output_dir
            )
            await client.start()

            call_args = mock_exec.call_args[0]
            assert "--save-video" in call_args
            assert "1920x1080" in call_args
            assert "--output-dir" in call_args
            assert output_dir in call_args

            await client.stop()

    @pytest.mark.asyncio
    async def test_start_writes_click_indicator_init_script(self, mock_process, tmp_path):
        """Test that start() writes click-indicator.js and passes --init-script when use_fork=False."""
        from dokumen.mcp_client import PlaywrightMCPClient
        from dokumen.playwright_tools import CLICK_INDICATOR_SCRIPT

        output_dir = str(tmp_path / "recordings")

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_process

            # use_fork=False triggers init-script injection for visual indicators
            client = PlaywrightMCPClient(output_dir=output_dir, use_fork=False, visual_indicators=True)
            await client.start()

            # Verify --init-script flag is passed
            call_args = mock_exec.call_args[0]
            assert "--init-script" in call_args

            # Verify click-indicator.js file was created
            init_script_path = os.path.join(output_dir, "click-indicator.js")
            assert os.path.exists(init_script_path)

            # Verify file contains the click indicator script
            with open(init_script_path, "r") as f:
                content = f.read()
            assert "clickPulse" in content
            assert "#ff6b00" in content  # Orange color

            await client.stop()

    @pytest.mark.asyncio
    async def test_call_tool_sends_json_rpc_request(self, mock_process):
        """Test that call_tool sends proper JSON-RPC message."""
        from dokumen.mcp_client import PlaywrightMCPClient

        # Mock response from MCP server
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "Success"}]}
        }
        mock_process.stdout.readline = AsyncMock(
            return_value=json.dumps(response).encode() + b"\n"
        )

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_process

            client = PlaywrightMCPClient()
            await client.start()

            result = await client.call_tool("browser_navigate", {"url": "https://example.com"})

            # Verify JSON-RPC request was written
            mock_process.stdin.write.assert_called()
            written_data = mock_process.stdin.write.call_args[0][0]
            request = json.loads(written_data.decode())

            assert request["jsonrpc"] == "2.0"
            assert request["method"] == "tools/call"
            assert request["params"]["name"] == "browser_navigate"
            assert request["params"]["arguments"]["url"] == "https://example.com"

            # Verify response parsing
            assert result.success is True
            assert "Success" in result.output

            await client.stop()

    @pytest.mark.asyncio
    async def test_call_tool_handles_error_response(self, mock_process):
        """Test call_tool handles JSON-RPC error responses."""
        from dokumen.mcp_client import PlaywrightMCPClient

        # Mock error response
        error_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32600, "message": "Invalid Request"}
        }
        mock_process.stdout.readline = AsyncMock(
            return_value=json.dumps(error_response).encode() + b"\n"
        )

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_process

            client = PlaywrightMCPClient()
            await client.start()

            result = await client.call_tool("invalid_tool", {})

            assert result.success is False
            assert "Invalid Request" in result.error

            await client.stop()

    @pytest.mark.asyncio
    async def test_stop_terminates_subprocess(self, mock_process):
        """Test that stop() properly terminates the subprocess."""
        from dokumen.mcp_client import PlaywrightMCPClient

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("os.killpg", side_effect=ProcessLookupError):  # Mock killpg to fall back to terminate
            mock_exec.return_value = mock_process

            client = PlaywrightMCPClient()
            await client.start()
            await client.stop()

            mock_process.terminate.assert_called_once()
            mock_process.wait.assert_called()

    @pytest.mark.asyncio
    async def test_stop_uses_sigkill_on_timeout(self, mock_process):
        """Test that stop() uses SIGKILL if SIGTERM times out."""
        from dokumen.mcp_client import PlaywrightMCPClient

        # Configure wait() to return immediately (we'll mock wait_for to timeout)
        mock_process.wait = AsyncMock()
        mock_process.kill = MagicMock()

        # Track whether we're in start() or stop()
        in_stop = [False]

        # Mock wait_for to raise TimeoutError only during stop()
        original_wait_for = asyncio.wait_for

        async def mock_wait_for(coro, timeout):
            if in_stop[0]:
                # During stop(), first call should timeout
                coro.close()
                raise asyncio.TimeoutError()
            # During start()/_wait_for_ready(), work normally
            return await original_wait_for(coro, timeout)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("os.killpg", side_effect=ProcessLookupError), \
             patch("asyncio.wait_for", side_effect=mock_wait_for), \
             patch.object(PlaywrightMCPClient, "_wait_for_ready", new_callable=AsyncMock):
            mock_exec.return_value = mock_process

            client = PlaywrightMCPClient()
            await client.start()
            in_stop[0] = True  # Now we're stopping
            await client.stop()

            # Should have called terminate first, then kill
            mock_process.terminate.assert_called_once()
            mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_protocol(self, mock_process):
        """Test that client works as async context manager."""
        from dokumen.mcp_client import PlaywrightMCPClient

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("os.killpg", side_effect=ProcessLookupError):  # Mock killpg for test stability
            mock_exec.return_value = mock_process

            async with PlaywrightMCPClient() as client:
                assert client._process is not None

            mock_process.terminate.assert_called()

    @pytest.mark.asyncio
    async def test_call_tool_increments_request_id(self, mock_process):
        """Test that each call_tool uses incrementing request IDs."""
        from dokumen.mcp_client import PlaywrightMCPClient

        responses = [
            {"jsonrpc": "2.0", "id": i, "result": {"content": []}}
            for i in range(1, 4)
        ]
        call_count = [0]

        async def mock_readline():
            idx = call_count[0]
            call_count[0] += 1
            return json.dumps(responses[idx]).encode() + b"\n"

        mock_process.stdout.readline = mock_readline

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch.object(PlaywrightMCPClient, "_wait_for_ready", new_callable=AsyncMock):
            mock_exec.return_value = mock_process

            client = PlaywrightMCPClient()
            await client.start()

            await client.call_tool("tool1", {})
            await client.call_tool("tool2", {})
            await client.call_tool("tool3", {})

            # Verify incrementing IDs
            calls = mock_process.stdin.write.call_args_list
            ids = []
            for call in calls:
                data = json.loads(call[0][0].decode())
                ids.append(data["id"])

            assert ids == [1, 2, 3]

            await client.stop()

    @pytest.mark.asyncio
    async def test_call_tool_timeout(self, mock_process):
        """Test call_tool raises timeout error on slow response."""
        from dokumen.mcp_client import PlaywrightMCPClient

        # Simulate slow response
        async def slow_readline():
            await asyncio.sleep(5)  # Longer than timeout
            return b"{}\n"

        mock_process.stdout.readline = slow_readline

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_process

            client = PlaywrightMCPClient(timeout=0.1)  # Very short timeout
            await client.start()

            result = await client.call_tool("browser_navigate", {"url": "https://example.com"})

            assert result.success is False
            assert "timeout" in result.error.lower()

            await client.stop()


class TestMCPServerReadiness:
    """Tests for MCP server readiness checking."""

    @pytest.fixture
    def mock_process(self):
        """Create mock subprocess for testing."""
        process = AsyncMock()
        process.stdin = AsyncMock()
        process.stdout = AsyncMock()
        process.stderr = AsyncMock()
        process.returncode = None
        process.pid = 12345
        process.terminate = MagicMock()
        process.wait = AsyncMock()
        return process

    @pytest.mark.asyncio
    async def test_wait_for_ready_succeeds_on_no_page_error(self, mock_process):
        """Test _wait_for_ready succeeds when browser_snapshot returns 'no page' error."""
        from dokumen.mcp_client import PlaywrightMCPClient

        # Mock response indicating no page open (server is ready)
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32000, "message": "No page open - call browser_navigate first"}
        }
        mock_process.stdout.readline = AsyncMock(
            return_value=json.dumps(response).encode() + b"\n"
        )

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_process

            client = PlaywrightMCPClient()
            await client.start()

            # _wait_for_ready should have been called during start() and found server ready
            # Verify at least one call_tool was made for readiness check
            assert mock_process.stdin.write.called

            await client.stop()

    @pytest.mark.asyncio
    async def test_wait_for_ready_retries_on_exception(self, mock_process):
        """Test _wait_for_ready retries when call_tool raises exception."""
        from dokumen.mcp_client import PlaywrightMCPClient

        call_count = [0]

        async def mock_readline():
            call_count[0] += 1
            if call_count[0] < 3:
                # First 2 calls fail
                raise ConnectionError("Connection reset")
            # Third call succeeds with "no page" error (server ready)
            response = {
                "jsonrpc": "2.0",
                "id": call_count[0],
                "error": {"code": -32000, "message": "No page open"}
            }
            return json.dumps(response).encode() + b"\n"

        mock_process.stdout.readline = mock_readline

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_process

            client = PlaywrightMCPClient()
            await client.start()

            # Should have retried until ready
            assert call_count[0] >= 3

            await client.stop()


class TestMCPClientInitialization:
    """Tests for client initialization and configuration."""

    def test_default_configuration(self):
        """Test default client configuration."""
        from dokumen.mcp_client import PlaywrightMCPClient

        client = PlaywrightMCPClient()

        assert client.headless is True
        assert client.save_video is None
        assert ".dokumen-cache/recordings" in client.output_dir

    def test_custom_configuration(self):
        """Test custom client configuration."""
        from dokumen.mcp_client import PlaywrightMCPClient

        client = PlaywrightMCPClient(
            headless=False,
            save_video="1280x720",
            output_dir="/custom/path",
            timeout=30.0
        )

        assert client.headless is False
        assert client.save_video == "1280x720"
        assert client.output_dir == "/custom/path"
        assert client.timeout == 30.0

    def test_not_started_raises_error_on_call_tool(self):
        """Test that calling tools before start raises error."""
        from dokumen.mcp_client import PlaywrightMCPClient

        client = PlaywrightMCPClient()

        with pytest.raises(RuntimeError, match="not started"):
            asyncio.run(client.call_tool("test", {}))


class TestPlaywrightMCPPathResolution:
    """Tests for fork path resolution using PLAYWRIGHT_MCP_PATH env var."""

    @pytest.fixture
    def mock_process(self):
        """Create mock subprocess for testing."""
        process = AsyncMock()
        process.stdin = AsyncMock()
        process.stdout = AsyncMock()
        process.stderr = AsyncMock()
        process.returncode = None
        process.pid = 12345
        process.terminate = MagicMock()
        process.wait = AsyncMock()
        return process

    @pytest.mark.asyncio
    async def test_start_uses_playwright_mcp_path_env_var(self, mock_process):
        """Test that start() uses PLAYWRIGHT_MCP_PATH env var when set."""
        from dokumen.mcp_client import PlaywrightMCPClient

        with patch.dict(os.environ, {"PLAYWRIGHT_MCP_PATH": "/opt/playwright-mcp-fork"}):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
                 patch("os.path.exists", return_value=True), \
                 patch.object(PlaywrightMCPClient, "_wait_for_ready", new_callable=AsyncMock):
                mock_exec.return_value = mock_process

                client = PlaywrightMCPClient(use_fork=True, headless=True)
                await client.start()

                # Verify the command uses the env var path
                call_args = mock_exec.call_args[0]
                assert call_args[0] == "node"
                assert call_args[1] == "/opt/playwright-mcp-fork/packages/playwright/cli.js"
                assert "run-mcp-server" in call_args

                # Cleanup
                client._process = None

    @pytest.mark.asyncio
    async def test_start_uses_relative_path_without_env_var(self, mock_process):
        """Test that start() uses relative path when PLAYWRIGHT_MCP_PATH not set."""
        from dokumen.mcp_client import PlaywrightMCPClient

        # Create env without PLAYWRIGHT_MCP_PATH
        env = os.environ.copy()
        env.pop("PLAYWRIGHT_MCP_PATH", None)

        with patch.dict(os.environ, env, clear=True):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
                 patch("os.path.exists", return_value=True), \
                 patch.object(PlaywrightMCPClient, "_wait_for_ready", new_callable=AsyncMock):
                mock_exec.return_value = mock_process

                client = PlaywrightMCPClient(use_fork=True, headless=True)
                await client.start()

                # Verify the command uses a relative path containing playwright-mcp-fork
                call_args = mock_exec.call_args[0]
                assert call_args[0] == "node"
                assert "playwright-mcp-fork/packages/playwright/cli.js" in call_args[1]
                assert "run-mcp-server" in call_args

                # Cleanup
                client._process = None

    @pytest.mark.asyncio
    async def test_use_fork_false_uses_npx(self, mock_process):
        """Test that use_fork=False uses npx instead of local fork."""
        from dokumen.mcp_client import PlaywrightMCPClient

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_process

            client = PlaywrightMCPClient(use_fork=False, headless=True)
            await client.start()

            # Should use npx, not node
            call_args = mock_exec.call_args[0]
            assert call_args[0] == "npx"
            assert "@playwright/mcp@latest" in call_args

            # Cleanup
            client._process = None


class TestStderrLogLevelRouting:
    """Tests for _log_stderr routing [Dokumen] messages to DEBUG level."""

    @pytest.fixture
    def mock_process(self):
        """Create mock subprocess for testing."""
        process = AsyncMock()
        process.stdin = AsyncMock()
        process.stdout = AsyncMock()
        process.stderr = AsyncMock()
        process.returncode = None
        process.pid = 12345
        process.terminate = MagicMock()
        process.wait = AsyncMock()
        return process

    @pytest.mark.asyncio
    async def test_dokumen_prefixed_stderr_logged_at_debug(self, mock_process):
        """Test that [Dokumen] prefixed stderr lines are logged at DEBUG, not WARNING."""
        from dokumen.mcp_client import PlaywrightMCPClient
        import logging

        lines = [
            b"[Dokumen] Click animation injected at (100, 200)\n",
            b""  # EOF
        ]
        line_iter = iter(lines)
        mock_process.stderr.readline = AsyncMock(side_effect=lambda: next(line_iter))

        client = PlaywrightMCPClient()
        client._process = mock_process
        client._stderr_lines = []

        with patch("dokumen.mcp_client.logger") as mock_logger:
            await client._log_stderr()

            mock_logger.debug.assert_called_once()
            assert "[Dokumen]" in mock_logger.debug.call_args[0][1]
            mock_logger.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_dokumen_stderr_logged_at_warning(self, mock_process):
        """Test that non-[Dokumen] stderr lines are still logged at WARNING."""
        from dokumen.mcp_client import PlaywrightMCPClient

        lines = [
            b"Some unexpected error from MCP\n",
            b""  # EOF
        ]
        line_iter = iter(lines)
        mock_process.stderr.readline = AsyncMock(side_effect=lambda: next(line_iter))

        client = PlaywrightMCPClient()
        client._process = mock_process
        client._stderr_lines = []

        with patch("dokumen.mcp_client.logger") as mock_logger:
            await client._log_stderr()

            mock_logger.warning.assert_called_once()
            assert "Some unexpected error" in mock_logger.warning.call_args[0][1]
            mock_logger.debug.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_stderr_routes_correctly(self, mock_process):
        """Test that mixed stderr output routes each line to the correct level."""
        from dokumen.mcp_client import PlaywrightMCPClient

        lines = [
            b"[Dokumen] Screenshot flash enabled\n",
            b"Warning: deprecated API usage\n",
            b"[Dokumen] Config: visual_indicators=true\n",
            b""  # EOF
        ]
        line_iter = iter(lines)
        mock_process.stderr.readline = AsyncMock(side_effect=lambda: next(line_iter))

        client = PlaywrightMCPClient()
        client._process = mock_process
        client._stderr_lines = []

        with patch("dokumen.mcp_client.logger") as mock_logger:
            await client._log_stderr()

            assert mock_logger.debug.call_count == 2
            assert mock_logger.warning.call_count == 1
