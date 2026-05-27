"""Setup runner for deterministic pre-test commands.

Runs shell commands (clone, install, start server) before the executor agent
begins. This avoids wasting LLM tokens on deterministic build steps.
"""

import asyncio
import os
import signal
from typing import List, Optional

import httpx

from .logging_config import get_logger

logger = get_logger(__name__)


class SetupError(Exception):
    """Raised when a setup step fails."""

    def __init__(self, step_name: str, message: str):
        self.step_name = step_name
        super().__init__(f"Setup step '{step_name}' failed: {message}")


class SetupRunner:
    """Runs deterministic setup steps before test execution."""

    def __init__(self, env: Optional[dict[str, str]] = None):
        self._env = env or {}
        self._background_procs: list[asyncio.subprocess.Process] = []

    async def run_steps(self, steps: list) -> None:
        """Run all steps in order. Raises SetupError on failure."""
        for step in steps:
            logger.info(
                "setup.step.start",
                step_name=step.name,
                command=step.command,
                background=step.background,
                timeout=step.timeout,
            )
            if step.background:
                await self._run_background(step)
            else:
                await self._run_foreground(step)

    async def _run_foreground(self, step) -> None:
        """Run command, wait for completion, raise if non-zero exit."""
        env = {**os.environ, **self._env}
        cwd = step.working_dir

        try:
            proc = await asyncio.create_subprocess_shell(
                step.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=step.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise SetupError(
                step.name,
                f"Command timed out after {step.timeout}s",
            )
        except OSError as e:
            raise SetupError(
                step.name,
                f"Failed to spawn command: {e}",
            )

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()
            raise SetupError(
                step.name,
                f"Exit code {proc.returncode}: {stderr_text[:500]}",
            )

        logger.info(
            "setup.step.complete",
            step_name=step.name,
            exit_code=proc.returncode,
        )

    async def _run_background(self, step) -> None:
        """Spawn process in background, poll ready_url if set."""
        env = {**os.environ, **self._env}
        cwd = step.working_dir

        try:
            proc = await asyncio.create_subprocess_shell(
                step.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
                preexec_fn=os.setsid,
            )
        except OSError as e:
            raise SetupError(
                step.name,
                f"Failed to spawn background command: {e}",
            )
        self._background_procs.append(proc)

        logger.info(
            "setup.step.background_started",
            step_name=step.name,
            pid=proc.pid,
        )

        if step.ready_url:
            await self._wait_for_ready(
                step.name, step.ready_url, step.ready_timeout
            )

    async def _wait_for_ready(
        self, step_name: str, url: str, timeout: float
    ) -> None:
        """Poll URL every 1s until HTTP 200 or timeout."""
        logger.info(
            "setup.ready_poll.start",
            step_name=step_name,
            url=url,
            timeout=timeout,
        )

        async with httpx.AsyncClient() as client:
            elapsed = 0.0
            while elapsed < timeout:
                try:
                    resp = await client.get(url, timeout=5.0)
                    if resp.status_code == 200:
                        logger.info(
                            "setup.ready_poll.success",
                            step_name=step_name,
                            url=url,
                            elapsed_s=round(elapsed, 1),
                        )
                        return
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(1.0)
                elapsed += 1.0

        raise SetupError(
            step_name,
            f"ready_url {url} not reachable after {timeout}s",
        )

    async def cleanup(self) -> None:
        """SIGTERM all background procs, wait 5s, SIGKILL survivors."""
        if not self._background_procs:
            return

        logger.info(
            "setup.cleanup.start",
            proc_count=len(self._background_procs),
        )

        for proc in self._background_procs:
            if proc.returncode is not None:
                continue
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                logger.info("setup.cleanup.sigterm", pid=proc.pid)
            except (ProcessLookupError, OSError):
                pass

        # Wait up to 5s for graceful shutdown
        for proc in self._background_procs:
            if proc.returncode is not None:
                continue
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    logger.warning("setup.cleanup.sigkill", pid=proc.pid)
                except (ProcessLookupError, OSError):
                    pass

        self._background_procs.clear()
        logger.info("setup.cleanup.complete")
