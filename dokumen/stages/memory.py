"""memory stage — extracts memories from executor conversation after judge completes."""

import logging
from typing import Any, Optional

from ..logging_config import get_logger
from ..pipeline import PipelineContext, PipelineStage

logger = get_logger(__name__)


class MemoryStage(PipelineStage):
    """extract and store memories from the executor conversation.

    runs after judge stage. processes the executor's conversation log
    through the mem0 extraction → update pipeline to build persistent
    per-company memory.

    only runs if memory is enabled in config and a memory store is
    available on the pipeline context.
    """

    @property
    def name(self) -> str:
        return "memory"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """extract memories from executor conversation.

        args:
            ctx: pipeline context with executor_output set.

        returns:
            updated context (memory extraction is fire-and-forget).
        """
        memory_store = getattr(ctx, "memory_store", None)
        embedding_provider = getattr(ctx, "embedding_provider", None)
        memory_config = getattr(ctx, "memory_config", None)

        if not memory_store or not embedding_provider:
            logger.debug(
                "stage.memory.skipped",
                test_id=ctx.test_id,
                reason="memory not enabled",
            )
            return ctx

        if not ctx.executor_output:
            logger.debug(
                "stage.memory.skipped",
                test_id=ctx.test_id,
                reason="no executor output",
            )
            return ctx

        conversation = getattr(ctx.executor_output, "conversation_log", None) or []
        if not conversation:
            logger.debug(
                "stage.memory.skipped",
                test_id=ctx.test_id,
                reason="empty conversation log",
            )
            return ctx

        model = "gemini/gemini-2.0-flash"
        similarity_threshold = 0.7
        top_k = 10

        if memory_config:
            model = getattr(memory_config, "model", model)
            similarity_threshold = getattr(
                memory_config, "similarity_threshold", similarity_threshold
            )
            top_k = getattr(memory_config, "max_memories_per_query", top_k)

        logger.info(
            "stage.memory.start",
            test_id=ctx.test_id,
            conversation_length=len(conversation),
            model=model,
        )

        try:
            changed = await memory_store.process_conversation(
                conversation=conversation,
                embedding_provider=embedding_provider,
                model=model,
                similarity_threshold=similarity_threshold,
                top_k=top_k,
            )

            logger.info(
                "stage.memory.complete",
                test_id=ctx.test_id,
                memories_changed=len(changed),
                total_memories=len(memory_store.get_all()),
            )

        except Exception as e:
            # memory extraction is non-critical — log and continue
            logger.warning(
                "stage.memory.error",
                test_id=ctx.test_id,
                error=str(e),
                error_type=type(e).__name__,
            )

        return ctx
