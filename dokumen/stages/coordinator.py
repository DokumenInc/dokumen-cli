"""coordinator stage — replaces standard executor with multi-agent coordination."""

import logging
import time

from ..pipeline import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class CoordinatorStage(PipelineStage):
    """run executor via coordinator mode with parallel workers.

    when enabled, replaces the standard ExecutorStage. the coordinator
    splits the task into subtasks, spawns worker agents in parallel,
    and synthesizes their results.
    """

    def __init__(self, coordinator_config=None):
        self._config = coordinator_config

    @property
    def name(self) -> str:
        return "coordinator"

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
            from ..coordinator.types import WorkerTask, CoordinatorPlan

            # build worker tasks from the executor prompt
            system_prompt = getattr(ctx.executor, "system_prompt", "") or ""
            user_prompt = getattr(ctx.executor, "user_prompt", "") or ""

            # the coordinator creates its own plan from the user prompt
            coordinator = CoordinatorAgent(
                max_workers=self._config.max_workers,
                synthesis_strategy=self._config.synthesis_strategy,
            )

            # build worker configs from executor tools
            tool_names = []
            executor_tools = getattr(ctx.executor, "tools", None) or []
            if executor_tools:
                tool_names = [getattr(t, "name", str(t)) for t in executor_tools]

            worker_configs = []
            for i in range(min(self._config.max_workers, 3)):
                worker_configs.append(
                    {
                        "id": f"worker-{ctx.test_id}-{i}",
                        "tools": tool_names,
                        "timeout": self._config.worker_timeout,
                        "model": self._config.worker_model,
                    }
                )

            # create a plan with the user prompt as the main task
            plan = CoordinatorPlan(
                tasks=[
                    WorkerTask(
                        id=f"task-{ctx.test_id}-{i}",
                        description=f"part {i+1} of: {user_prompt[:200]}",
                        tools=tool_names,
                        timeout=self._config.worker_timeout,
                    )
                    for i in range(min(self._config.max_workers, 3))
                ],
                synthesis_strategy=self._config.synthesis_strategy,
            )

            result = await coordinator.run(
                plan=plan,
                system_prompt=system_prompt,
                worker_configs=worker_configs,
            )

            # convert coordinator result to executor output format
            if result.get("success"):
                # store synthesized result as executor output
                from ..sdk.executor import ExecutorResult

                ctx.executor_output = ExecutorResult(
                    success=True,
                    response=result.get("synthesis", ""),
                    tool_calls=[],
                    total_tokens=0,
                    conversation_log=[],
                )
                logger.info(
                    "stage.coordinator.complete",
                    extra={
                        "test_id": ctx.test_id,
                        "workers_succeeded": result.get("workers_succeeded", 0),
                        "duration_ms": int((time.time() - start) * 1000),
                    },
                )
            else:
                ctx.fail(f"coordinator failed: {result.get('error', 'unknown')}")

        except Exception as e:
            logger.error("stage.coordinator.error", extra={"test_id": ctx.test_id, "error": str(e)})
            ctx.fail(f"coordinator error: {e}")

        return ctx
