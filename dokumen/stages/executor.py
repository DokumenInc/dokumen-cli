"""Executor stage — runs the executor agent with retries."""

import time

from ..logging_config import get_logger
from ..pipeline import PipelineContext, PipelineStage
from .prompting import ensure_final_response_from_conversation, prepare_agent_prompts

logger = get_logger(__name__)


class ExecutorStage(PipelineStage):
    """Run the executor agent with retry support.

    This stage:
    1. Injects output folder path into executor and judge prompts
    2. Logs prompt observability hashes
    3. Runs the executor with configurable retries
    4. Stores token usage and fires callbacks
    """

    @property
    def name(self) -> str:
        return "executor"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute the executor agent.

        Args:
            ctx: The pipeline context.

        Returns:
            Updated context with executor_output set.
        """
        logger.info("stage.executor.start", test_id=ctx.test_id, max_retries=ctx.retries)
        prepare_agent_prompts(ctx, executor_role="executor", logger=logger)

        # Run executor with retries
        executor_output = None
        attempts = 0
        max_attempts = ctx.retries + 1

        while attempts < max_attempts:
            attempts += 1
            attempt_start = time.time()
            logger.debug(
                "stage.executor.attempt",
                test_id=ctx.test_id,
                attempt=attempts,
                max_attempts=max_attempts,
            )

            try:
                executor_output = await ctx.executor.run(
                    timeout=ctx.timeout,
                    on_tool_call=ctx.on_tool_call,
                    on_conversation_message=ctx.on_conversation_message,
                    original_user_prompt=ctx.original_user_prompt,
                )

                attempt_duration_ms = int((time.time() - attempt_start) * 1000)
                if executor_output.success:
                    logger.info(
                        "stage.executor.attempt.complete",
                        test_id=ctx.test_id,
                        attempt=attempts,
                        duration_ms=attempt_duration_ms,
                        success=True,
                    )
                    break
                elif attempts < max_attempts:
                    logger.info(
                        "stage.executor.attempt.complete",
                        test_id=ctx.test_id,
                        attempt=attempts,
                        duration_ms=attempt_duration_ms,
                        success=False,
                    )
                    logger.debug(
                        "stage.executor.retry",
                        test_id=ctx.test_id,
                        attempt=attempts,
                        error=executor_output.error,
                    )
                    continue
                else:
                    logger.info(
                        "stage.executor.attempt.complete",
                        test_id=ctx.test_id,
                        attempt=attempts,
                        duration_ms=attempt_duration_ms,
                        success=False,
                    )
                    logger.warning(
                        "stage.executor.failed",
                        test_id=ctx.test_id,
                        attempts=attempts,
                        error=executor_output.error,
                    )
                    ensure_final_response_from_conversation(executor_output, logger, ctx.test_id)

                    ctx.executor_output = executor_output
                    ctx.fail(
                        f"Executor failed after {attempts} attempts: " f"{executor_output.error}"
                    )
                    return ctx

            except Exception as e:
                attempt_duration_ms = int((time.time() - attempt_start) * 1000)
                logger.info(
                    "stage.executor.attempt.complete",
                    test_id=ctx.test_id,
                    attempt=attempts,
                    duration_ms=attempt_duration_ms,
                    success=False,
                )
                logger.error(
                    "stage.executor.error",
                    test_id=ctx.test_id,
                    attempt=attempts,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                if attempts >= max_attempts:
                    ctx.fail(f"Executor error: {str(e)}")
                    return ctx

        ensure_final_response_from_conversation(executor_output, logger, ctx.test_id)

        ctx.executor_output = executor_output

        # Fire executor complete callback
        if ctx.on_executor_complete and executor_output:
            ctx.on_executor_complete(executor_output)

        logger.info(
            "stage.executor.complete",
            test_id=ctx.test_id,
            success=bool(executor_output and executor_output.success),
        )
        return ctx
