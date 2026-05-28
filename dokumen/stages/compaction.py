"""compaction stage — applies context compaction after executor runs."""

import logging
import time

from ..pipeline import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class CompactionStage(PipelineStage):
    """compact context between executor and judge stages.

    when enabled, runs micro-compaction on old tool results and
    full compaction if token usage exceeds the configured threshold.
    sits between ExecutorStage and JudgeStage in the pipeline.
    """

    def __init__(self, compaction_config=None):
        self._config = compaction_config

    @property
    def name(self) -> str:
        return "compaction"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if self._config is None or not self._config.enabled:
            logger.debug(
                "stage.compaction.skipped", extra={"test_id": ctx.test_id, "reason": "disabled"}
            )
            return ctx

        start = time.time()
        logger.info("stage.compaction.start", extra={"test_id": ctx.test_id})

        try:
            from ..context.micro_compact import MicroCompactor
            from ..context.compactor import ContextCompactor

            # micro-compact old tool results from executor output
            if self._config.micro_compact_enabled and ctx.executor_output:
                micro = MicroCompactor(
                    age_threshold=self._config.micro_compact_age_seconds,
                    max_chars=self._config.micro_compact_max_chars,
                )

                # extract tool results from executor conversation log
                conv_log = getattr(ctx.executor_output, "conversation_log", None) or []
                for entry in conv_log:
                    if isinstance(entry, dict) and entry.get("role") == "tool":
                        micro.track(
                            tool_name=entry.get("tool_name", "unknown"),
                            result_text=entry.get("content", ""),
                        )

                micro_result = micro.compact()
                if micro_result.compacted_count > 0:
                    logger.info(
                        "stage.compaction.micro",
                        extra={
                            "test_id": ctx.test_id,
                            "compacted": micro_result.compacted_count,
                            "chars_saved": micro_result.chars_saved,
                        },
                    )

            # full compaction check based on token usage
            executor_tokens = getattr(ctx.executor_output, "total_tokens", 0) or 0
            if executor_tokens > self._config.token_budget * self._config.token_threshold:
                compactor = ContextCompactor(
                    token_budget=self._config.token_budget,
                    threshold=self._config.token_threshold,
                    keep_recent=self._config.keep_recent_turns,
                )

                # feed conversation turns
                for entry in getattr(ctx.executor_output, "conversation_log", None) or []:
                    if isinstance(entry, dict):
                        compactor.add_turn(
                            role=entry.get("role", "unknown"),
                            content=entry.get("content", ""),
                        )

                if compactor.needs_compaction:
                    result = await compactor.compact()
                    logger.info(
                        "stage.compaction.full",
                        extra={
                            "test_id": ctx.test_id,
                            "turns_before": result.turns_before,
                            "turns_after": result.turns_after,
                            "tokens_saved": result.tokens_saved,
                        },
                    )

            duration = time.time() - start
            logger.info(
                "stage.compaction.complete",
                extra={"test_id": ctx.test_id, "duration_ms": int(duration * 1000)},
            )

        except Exception as e:
            # compaction is best-effort — don't fail the pipeline
            logger.warning(
                "stage.compaction.error", extra={"test_id": ctx.test_id, "error": str(e)}
            )

        return ctx
