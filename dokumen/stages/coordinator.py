"""Coordinator stage, used only when coordinator mode is explicitly enabled."""

import logging
import time

from ..pipeline import PipelineContext, PipelineStage
from ..sdk.types import ExecutorResult
from .executor import _prompt_hash

logger = logging.getLogger(__name__)


class CoordinatorStage(PipelineStage):
    """Run executor work through the coordinator.

    Coordinator mode is intentionally opt-in. When enabled, this stage replaces
    ExecutorStage but still has to honor the same prompt metadata, output-folder
    injection, callback, and result-type contract that judges expect.
    """

    def __init__(self, coordinator_config=None):
        self._config = coordinator_config

    @property
    def name(self) -> str:
        return "coordinator"

    def _prepare_prompts(self, ctx: PipelineContext) -> None:
        """Mirror ExecutorStage prompt preparation for coordinator runs."""
        if not ctx.original_user_prompt:
            ctx.original_user_prompt = ctx.executor.user_prompt

        output_folder_instruction = (
            f"\n\n---\nOUTPUT FOLDER: {ctx.output_dir}\n"
            "Write any deliverables, calculations, scripts, or evidence "
            "files to this folder.\n"
            "Files here will be visible to the user after the test completes."
        )
        ctx.executor.user_prompt = ctx.executor.user_prompt + output_folder_instruction

        judge_output_instruction = (
            f"\n\nOUTPUT FOLDER: {ctx.output_dir}\n"
            "You may write analysis or evidence files here. "
            "Files will be visible to the user."
        )
        for judge in ctx.judges:
            ctx.original_judge_prompts[judge.id] = judge.system_prompt
            judge.system_prompt = judge.system_prompt + judge_output_instruction

        exec_sys = (
            ctx.executor.system_prompt if isinstance(ctx.executor.system_prompt, str) else None
        )
        exec_usr = ctx.executor.user_prompt if isinstance(ctx.executor.user_prompt, str) else None
        logger.info(
            "agent.prompt_applied",
            test_id=ctx.test_id,
            role="coordinator",
            system_prompt_hash=_prompt_hash(exec_sys),
            user_prompt_hash=_prompt_hash(exec_usr),
            system_prompt_len=len(exec_sys) if exec_sys else 0,
            user_prompt_len=len(exec_usr) if exec_usr else 0,
        )
        for judge in ctx.judges:
            judge_sys = judge.system_prompt if isinstance(judge.system_prompt, str) else None
            logger.info(
                "agent.prompt_applied",
                test_id=ctx.test_id,
                role="judge",
                judge_name=judge.id,
                system_prompt_hash=_prompt_hash(judge_sys),
                system_prompt_len=len(judge_sys) if judge_sys else 0,
            )

    def _build_executor_result(
        self,
        ctx: PipelineContext,
        coordinator_result: dict,
        duration_ms: int,
    ) -> ExecutorResult:
        worker_results = coordinator_result.get("results", []) or []
        input_tokens = sum(int(r.get("input_tokens", 0) or 0) for r in worker_results)
        output_tokens = sum(int(r.get("output_tokens", 0) or 0) for r in worker_results)
        tool_calls = [
            call
            for worker in worker_results
            for call in (worker.get("tool_calls", []) or [])
            if isinstance(call, dict)
        ]

        return ExecutorResult(
            success=bool(coordinator_result.get("success")),
            final_response=coordinator_result.get("synthesis", "") or "",
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            conversation_log=[
                {
                    "role": "assistant",
                    "content": coordinator_result.get("synthesis", "") or "",
                }
            ],
            system_prompt=ctx.executor.system_prompt,
            user_prompt=ctx.executor.user_prompt,
            original_user_prompt=ctx.original_user_prompt or ctx.executor.user_prompt,
            duration_ms=duration_ms,
            error=coordinator_result.get("error"),
        )

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if self._config is None or not self._config.enabled:
            logger.debug(
                "stage.coordinator.skipped", extra={"test_id": ctx.test_id, "reason": "disabled"}
            )
            return ctx

        start = time.time()
        logger.info(
            "stage.coordinator.start",
            extra={
                "test_id": ctx.test_id,
                "max_workers": self._config.max_workers,
                "strategy": self._config.synthesis_strategy,
            },
        )

        try:
            from ..coordinator.coordinator import CoordinatorAgent

            system_prompt = getattr(ctx.executor, "system_prompt", "") or ""
            self._prepare_prompts(ctx)
            user_prompt = getattr(ctx.executor, "user_prompt", "") or ""

            coordinator = CoordinatorAgent(
                provider=getattr(ctx.executor, "provider", None),
                max_workers=self._config.max_workers,
                synthesis_strategy=self._config.synthesis_strategy,
                default_timeout=self._config.worker_timeout,
                executor_mode=self._config.executor_mode,
                worker_model=self._config.worker_model,
            )

            tool_names = []
            executor_tools = getattr(ctx.executor, "tools", None) or []
            if executor_tools:
                tool_names = [getattr(t, "name", str(t)) for t in executor_tools]

            worker_count = max(1, min(int(self._config.max_workers), 3))
            worker_configs = []
            for i in range(worker_count):
                worker_configs.append(
                    {
                        "id": f"worker-{ctx.test_id}-{i}",
                        "name": f"worker-{i + 1}",
                        "goal": user_prompt,
                        "context": system_prompt,
                        "tools": tool_names,
                        "timeout": self._config.worker_timeout,
                        "model": self._config.worker_model,
                    }
                )

            result = await coordinator.run(
                goal=user_prompt,
                system_prompt=system_prompt,
                worker_configs=worker_configs,
            )

            duration_ms = int((time.time() - start) * 1000)
            ctx.executor_output = self._build_executor_result(ctx, result, duration_ms)

            if ctx.on_executor_complete:
                ctx.on_executor_complete(ctx.executor_output)

            if result.get("success"):
                logger.info(
                    "stage.coordinator.complete",
                    extra={
                        "test_id": ctx.test_id,
                        "workers_succeeded": result.get("workers_succeeded", 0),
                        "duration_ms": duration_ms,
                    },
                )
            else:
                ctx.fail(f"coordinator failed: {result.get('error', 'unknown')}")

        except Exception as e:
            logger.error("stage.coordinator.error", extra={"test_id": ctx.test_id, "error": str(e)})
            ctx.fail(f"coordinator error: {e}")

        return ctx
