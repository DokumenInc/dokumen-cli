"""Tests for SetupRunner — deterministic pre-test setup commands."""

import asyncio
import os
import signal

import pytest

from dokumen.setup_runner import SetupRunner, SetupError


class FakeStep:
    """Minimal step object matching SetupStep interface."""

    def __init__(
        self,
        name="step",
        command="echo ok",
        working_dir=None,
        timeout=60,
        background=False,
        ready_url=None,
        ready_timeout=30,
    ):
        self.name = name
        self.command = command
        self.working_dir = working_dir
        self.timeout = timeout
        self.background = background
        self.ready_url = ready_url
        self.ready_timeout = ready_timeout


class TestForegroundSteps:
    """Tests for foreground (blocking) setup steps."""

    @pytest.mark.asyncio
    async def test_successful_foreground_step(self):
        runner = SetupRunner()
        step = FakeStep(name="echo", command="echo hello")
        await runner.run_steps([step])
        # No exception = success

    @pytest.mark.asyncio
    async def test_foreground_step_failure(self):
        runner = SetupRunner()
        step = FakeStep(name="fail", command="exit 1")
        with pytest.raises(SetupError, match="fail"):
            await runner.run_steps([step])

    @pytest.mark.asyncio
    async def test_foreground_step_nonzero_exit(self):
        runner = SetupRunner()
        step = FakeStep(name="bad", command="exit 42")
        with pytest.raises(SetupError, match="Exit code 42"):
            await runner.run_steps([step])

    @pytest.mark.asyncio
    async def test_foreground_step_timeout(self):
        runner = SetupRunner()
        step = FakeStep(name="slow", command="sleep 60", timeout=1)
        with pytest.raises(SetupError, match="timed out"):
            await runner.run_steps([step])

    @pytest.mark.asyncio
    async def test_foreground_step_with_working_dir(self, tmp_path):
        runner = SetupRunner()
        step = FakeStep(
            name="pwd",
            command=f"test -d {tmp_path}",
            working_dir=str(tmp_path),
        )
        await runner.run_steps([step])

    @pytest.mark.asyncio
    async def test_multiple_foreground_steps_run_in_order(self, tmp_path):
        marker = tmp_path / "order.txt"
        runner = SetupRunner()
        steps = [
            FakeStep(name="first", command=f"echo first > {marker}"),
            FakeStep(name="second", command=f"echo second >> {marker}"),
        ]
        await runner.run_steps(steps)
        content = marker.read_text().strip().split("\n")
        assert content == ["first", "second"]

    @pytest.mark.asyncio
    async def test_foreground_step_stops_on_failure(self, tmp_path):
        marker = tmp_path / "reached.txt"
        runner = SetupRunner()
        steps = [
            FakeStep(name="fail", command="exit 1"),
            FakeStep(name="never", command=f"touch {marker}"),
        ]
        with pytest.raises(SetupError):
            await runner.run_steps(steps)
        assert not marker.exists()

    @pytest.mark.asyncio
    async def test_env_vars_passed_to_command(self):
        runner = SetupRunner(env={"MY_TEST_VAR": "hello123"})
        step = FakeStep(
            name="env",
            command='test "$MY_TEST_VAR" = "hello123"',
        )
        await runner.run_steps([step])

    @pytest.mark.asyncio
    async def test_bad_working_dir_raises_setup_error(self):
        """Invalid working_dir should raise SetupError, not FileNotFoundError."""
        runner = SetupRunner()
        step = FakeStep(
            name="bad-cwd",
            command="echo hello",
            working_dir="/definitely/missing/path",
        )
        with pytest.raises(SetupError, match="Failed to spawn"):
            await runner.run_steps([step])


class TestBackgroundSteps:
    """Tests for background (non-blocking) setup steps."""

    @pytest.mark.asyncio
    async def test_background_step_starts_process(self):
        runner = SetupRunner()
        step = FakeStep(
            name="bg", command="sleep 60", background=True
        )
        await runner.run_steps([step])
        assert len(runner._background_procs) == 1
        assert runner._background_procs[0].returncode is None
        await runner.cleanup()

    @pytest.mark.asyncio
    async def test_background_step_with_ready_url(self, tmp_path):
        """Background step with ready_url polls until HTTP 200."""
        # Start a simple HTTP server
        runner = SetupRunner()
        step = FakeStep(
            name="server",
            command=f"python3 -m http.server 0 --directory {tmp_path}",
            background=True,
            # Don't use ready_url here because port 0 is random
        )
        await runner.run_steps([step])
        assert len(runner._background_procs) == 1
        await runner.cleanup()

    @pytest.mark.asyncio
    async def test_background_bad_working_dir_raises_setup_error(self):
        """Invalid working_dir on background step should raise SetupError."""
        runner = SetupRunner()
        step = FakeStep(
            name="bad-bg-cwd",
            command="sleep 60",
            background=True,
            working_dir="/definitely/missing/path",
        )
        with pytest.raises(SetupError, match="Failed to spawn"):
            await runner.run_steps([step])

    @pytest.mark.asyncio
    async def test_ready_url_timeout_raises(self):
        """ready_url that never responds should raise SetupError."""
        runner = SetupRunner()
        step = FakeStep(
            name="unreachable",
            command="sleep 60",
            background=True,
            ready_url="http://127.0.0.1:19999",
            ready_timeout=2,
        )
        with pytest.raises(SetupError, match="not reachable"):
            await runner.run_steps([step])
        await runner.cleanup()


class TestCleanup:
    """Tests for cleanup of background processes."""

    @pytest.mark.asyncio
    async def test_cleanup_terminates_background_procs(self):
        runner = SetupRunner()
        step = FakeStep(
            name="long", command="sleep 600", background=True
        )
        await runner.run_steps([step])
        proc = runner._background_procs[0]
        assert proc.returncode is None

        await runner.cleanup()
        # After cleanup, proc list is cleared
        assert len(runner._background_procs) == 0

    @pytest.mark.asyncio
    async def test_cleanup_with_no_procs(self):
        """Cleanup with no background procs should be a no-op."""
        runner = SetupRunner()
        await runner.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_handles_already_exited_proc(self):
        runner = SetupRunner()
        step = FakeStep(
            name="quick", command="echo done", background=True
        )
        await runner.run_steps([step])
        # Wait for it to exit naturally
        await asyncio.sleep(0.5)
        await runner.cleanup()
        assert len(runner._background_procs) == 0

    @pytest.mark.asyncio
    async def test_cleanup_idempotent(self):
        """Calling cleanup twice should not raise."""
        runner = SetupRunner()
        step = FakeStep(
            name="bg", command="sleep 600", background=True
        )
        await runner.run_steps([step])
        await runner.cleanup()
        await runner.cleanup()  # Should be safe
