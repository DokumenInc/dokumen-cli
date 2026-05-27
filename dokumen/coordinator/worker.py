"""
worker agent — executes a scoped task for the coordinator.

workers run with isolated tool contexts and a focused goal.
they report findings back as structured WorkerResult.
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .types import WorkerTask, WorkerResult, WorkerStatus

logger = logging.getLogger(__name__)


class WorkerAgent:
    """executes a single worker task.

    supports two execution modes:
    - "api": direct provider.complete() loop (default, reliable)
    - "sdk": bundled claude CLI via agent SDK (legacy, has exit code 1 issues)

    usage:
        worker = WorkerAgent(provider=provider, executor_mode="api")
        result = await worker.run(task)
    """

    def __init__(
        self,
        provider: Optional[Any] = None,
        tools_config: Optional[Any] = None,
        executor_mode: str = "api",
        base_dir: str = ".",
    ):
        self._provider = provider
        self._tools_config = tools_config
        self._executor_mode = executor_mode
        self._base_dir = base_dir

    async def run(self, task: WorkerTask) -> WorkerResult:
        """execute a worker task and return the result.

        args:
            task: the WorkerTask to execute

        returns:
            WorkerResult with output and findings
        """
        start = time.time()
        logger.info(
            "worker starting",
            extra={"task_id": task.id, "worker_name": task.name, "goal": task.goal[:100]},
        )

        try:
            result = await asyncio.wait_for(
                self._execute(task),
                timeout=task.timeout,
            )
            result.duration = time.time() - start

            logger.info(
                "worker completed",
                extra={
                    "task_id": task.id,
                    "status": result.status.value,
                    "duration": round(result.duration, 2),
                },
            )
            return result

        except asyncio.TimeoutError:
            elapsed = time.time() - start
            logger.warning(
                "worker timed out",
                extra={"task_id": task.id, "timeout": task.timeout},
            )
            return WorkerResult(
                task_id=task.id,
                status=WorkerStatus.TIMEOUT,
                error=f"timed out after {task.timeout}s",
                duration=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                "worker failed",
                extra={"task_id": task.id, "error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )
            return WorkerResult(
                task_id=task.id,
                status=WorkerStatus.FAILED,
                error=str(e),
                duration=elapsed,
            )

    async def _execute(self, task: WorkerTask) -> WorkerResult:
        """internal execution — routes to api or sdk executor.

        if no provider is configured, returns a stub result.
        """
        if self._provider is None:
            # stub mode — return the goal as output
            return WorkerResult(
                task_id=task.id,
                status=WorkerStatus.COMPLETED,
                output=f"[stub] would execute: {task.goal}",
                findings=[f"stub finding for: {task.name}"],
            )

        if self._executor_mode == "api":
            return await self._execute_api(task)
        else:
            return await self._execute_sdk(task)

    async def _execute_api(self, task: WorkerTask) -> WorkerResult:
        """execute via direct provider.complete() loop — no bundled CLI."""
        from .api_executor import run_api_executor

        tool_names = task.tools or ["read_file", "list_directory", "glob", "search_file_content", "write_file"]
        system_prompt = (
            "you are a worker agent with access to filesystem tools. "
            "use the provided tools to explore the codebase and complete your task. "
            "if your task involves writing files, use write_file. "
            "when you have gathered enough information or completed your task, "
            "STOP calling tools and return your findings as a final text response. "
            "do not keep reading files endlessly — once you have what you need, summarize and respond."
        )
        user_prompt = f"{task.context}\n\ngoal: {task.goal}"

        result = await run_api_executor(
            provider=self._provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tool_names=tool_names,
            max_turns=20,
            timeout=task.timeout,
            base_dir=self._base_dir,
        )

        return WorkerResult(
            task_id=task.id,
            status=WorkerStatus.COMPLETED if result["success"] else WorkerStatus.FAILED,
            output=result.get("output", ""),
            error=result.get("error"),
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
        )

    async def _execute_sdk(self, task: WorkerTask) -> WorkerResult:
        """execute via SDK bundled CLI (legacy, has exit code 1 issues)."""
        from ..sdk.executor import ExecutorAgent
        from ..sdk.tools import resolve_sdk_tools

        tool_names = task.tools or ["read_file", "list_directory", "glob", "search_file_content"]
        resolved = resolve_sdk_tools(tool_names, self._tools_config)

        executor = ExecutorAgent(
            id=task.id,
            system_prompt=f"you are a worker agent. your goal: {task.goal}",
            user_prompt=f"{task.context}\n\ngoal: {task.goal}",
            sdk_tools=list(resolved.sdk_tool_names),
            max_turns=20,
            timeout=task.timeout,
            tools_config=self._tools_config,
            model=getattr(self._provider, "model", None),
        )

        from ..sdk.agent_wrapper import SdkExecutorWrapper
        wrapper = SdkExecutorWrapper(
            executor,
            system_prompt=executor.system_prompt,
            user_prompt=executor.user_prompt,
        )

        exec_result = await wrapper.run()

        has_output = bool(exec_result.final_response and exec_result.final_response.strip())
        effective_success = exec_result.success or has_output

        return WorkerResult(
            task_id=task.id,
            status=WorkerStatus.COMPLETED if effective_success else WorkerStatus.FAILED,
            output=exec_result.final_response or "",
            error=exec_result.error if not effective_success else None,
            input_tokens=exec_result.input_tokens,
            output_tokens=exec_result.output_tokens,
        )
