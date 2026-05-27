"""Executor stage — runs the executor agent with retries."""

import hashlib
import time
from typing import Optional

from ..debug import is_debug, debug
from ..logging_config import get_logger
from ..pipeline import PipelineContext, PipelineStage

logger = get_logger(__name__)


def _prompt_hash(prompt: Optional[str]) -> str:
    """Compute a short SHA256 hash of a prompt for observability logging."""
    if prompt is None or not isinstance(prompt, str):
        return "none"
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


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
        logger.info("stage.executor.start", test_id=ctx.test_id,
                     max_retries=ctx.retries)

        # Capture original user prompt if not already set by explore stage
        if not ctx.original_user_prompt:
            ctx.original_user_prompt = ctx.executor.user_prompt

        # Inject output folder path into executor user prompt
        output_folder_instruction = (
            f"\n\n---\nOUTPUT FOLDER: {ctx.output_dir}\n"
            "Write any deliverables, calculations, scripts, or evidence "
            "files to this folder.\n"
            "Files here will be visible to the user after the test completes."
        )
        ctx.executor.user_prompt = ctx.executor.user_prompt + output_folder_instruction

        # Inject output folder path into judge system prompts
        judge_output_instruction = (
            f"\n\nOUTPUT FOLDER: {ctx.output_dir}\n"
            "You may write analysis or evidence files here. "
            "Files will be visible to the user."
        )
        for judge in ctx.judges:
            ctx.original_judge_prompts[judge.id] = judge.system_prompt
            judge.system_prompt = judge.system_prompt + judge_output_instruction

        # Prompt observability logging
        exec_sys = (
            ctx.executor.system_prompt
            if isinstance(ctx.executor.system_prompt, str)
            else None
        )
        exec_usr = (
            ctx.executor.user_prompt
            if isinstance(ctx.executor.user_prompt, str)
            else None
        )
        logger.info(
            "agent.prompt_applied",
            test_id=ctx.test_id,
            role="executor",
            system_prompt_hash=_prompt_hash(exec_sys),
            user_prompt_hash=_prompt_hash(exec_usr),
            system_prompt_len=len(exec_sys) if exec_sys else 0,
            user_prompt_len=len(exec_usr) if exec_usr else 0,
        )
        for judge in ctx.judges:
            judge_sys = (
                judge.system_prompt
                if isinstance(judge.system_prompt, str)
                else None
            )
            logger.info(
                "agent.prompt_applied",
                test_id=ctx.test_id,
                role="judge",
                judge_name=judge.id,
                system_prompt_hash=_prompt_hash(judge_sys),
                system_prompt_len=len(judge_sys) if judge_sys else 0,
            )

        # Run executor with retries
        executor_output = None
        attempts = 0
        max_attempts = ctx.retries + 1

        while attempts < max_attempts:
            attempts += 1
            attempt_start = time.time()
            logger.debug("stage.executor.attempt", test_id=ctx.test_id,
                         attempt=attempts, max_attempts=max_attempts)

            try:
                executor_output = await ctx.executor.run(
                    timeout=ctx.timeout,
                    on_tool_call=ctx.on_tool_call,
                    on_conversation_message=ctx.on_conversation_message,
                    original_user_prompt=ctx.original_user_prompt,
                )

                attempt_duration_ms = int((time.time() - attempt_start) * 1000)
                if executor_output.success:
                    logger.info("stage.executor.attempt.complete",
                                test_id=ctx.test_id,
                                attempt=attempts,
                                duration_ms=attempt_duration_ms,
                                success=True)
                    break
                elif attempts < max_attempts:
                    logger.info("stage.executor.attempt.complete",
                                test_id=ctx.test_id,
                                attempt=attempts,
                                duration_ms=attempt_duration_ms,
                                success=False)
                    logger.debug("stage.executor.retry", test_id=ctx.test_id,
                                 attempt=attempts, error=executor_output.error)
                    continue
                else:
                    logger.info("stage.executor.attempt.complete",
                                test_id=ctx.test_id,
                                attempt=attempts,
                                duration_ms=attempt_duration_ms,
                                success=False)
                    logger.warning("stage.executor.failed", test_id=ctx.test_id,
                                   attempts=attempts, error=executor_output.error)
                    if executor_output and not getattr(executor_output, "final_response", None):
                        conversation_log = getattr(executor_output, "conversation_log", None) or []
                        text_chunks = []
                        for item in conversation_log:
                            if not isinstance(item, dict):
                                continue
                            if item.get("role") != "assistant":
                                continue
                            content = item.get("content")
                            if isinstance(content, str) and content.strip():
                                text_chunks.append(content.strip())
                        if text_chunks:
                            executor_output.final_response = "\n\n".join(text_chunks)
                            logger.warning(
                                "stage.executor.reconstructed_response",
                                test_id=ctx.test_id,
                                chunk_count=len(text_chunks),
                            )

                    ctx.executor_output = executor_output
                    ctx.fail(
                        f"Executor failed after {attempts} attempts: "
                        f"{executor_output.error}"
                    )
                    return ctx

            except Exception as e:
                attempt_duration_ms = int((time.time() - attempt_start) * 1000)
                logger.info("stage.executor.attempt.complete",
                            test_id=ctx.test_id,
                            attempt=attempts,
                            duration_ms=attempt_duration_ms,
                            success=False)
                logger.error("stage.executor.error", test_id=ctx.test_id,
                             attempt=attempts, error=str(e),
                             error_type=type(e).__name__, exc_info=True)
                if attempts >= max_attempts:
                    ctx.fail(f"Executor error: {str(e)}")
                    return ctx

        if executor_output and not getattr(executor_output, "final_response", None):
            conversation_log = getattr(executor_output, "conversation_log", None) or []
            text_chunks = []
            for item in conversation_log:
                if not isinstance(item, dict):
                    continue
                if item.get("role") != "assistant":
                    continue
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    text_chunks.append(content.strip())
            if text_chunks:
                executor_output.final_response = "\n\n".join(text_chunks)
                logger.warning(
                    "stage.executor.reconstructed_response",
                    test_id=ctx.test_id,
                    chunk_count=len(text_chunks),
                )

        ctx.executor_output = executor_output

        # Fire executor complete callback
        if ctx.on_executor_complete and executor_output:
            ctx.on_executor_complete(executor_output)

        logger.info("stage.executor.complete", test_id=ctx.test_id,
                     success=bool(executor_output and executor_output.success))
        return ctx
