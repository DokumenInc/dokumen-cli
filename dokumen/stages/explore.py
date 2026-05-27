"""Explore stage — discovers relevant files and injects context into executor."""

import os
from typing import List

from ..debug import is_debug, debug
from ..logging_config import get_logger
from ..pipeline import PipelineContext, PipelineStage

logger = get_logger(__name__)


class ExploreStage(PipelineStage):
    """Run the explore phase to discover documentation files.

    This stage:
    1. Runs the explore agent (SDK-based) to discover relevant files
    2. Verifies required files were found (with deterministic fallback)
    3. Injects explore context into the executor prompt
    """

    @property
    def name(self) -> str:
        return "explore"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute the explore phase.

        Args:
            ctx: The pipeline context.

        Returns:
            Updated context with explore_result and modified executor prompt.
        """
        if ctx.explore_config is not None:
            should_explore = bool(ctx.explore_config.enabled)
        else:
            should_explore = bool(ctx.files)

        if not should_explore:
            logger.info("stage.explore.skip", test_id=ctx.test_id,
                        reason="explore disabled")
            return ctx

        logger.info("stage.explore.start", test_id=ctx.test_id,
                     required_files=ctx.files)
        if is_debug():
            debug(f"[EXPLORE STAGE] {ctx.test_id}: Starting explore phase")
            debug(f"[EXPLORE STAGE] Required files: {ctx.files}")

        explore_result = await self._run_explore(ctx)

        # If files are required but explore failed to run, fail the test
        if ctx.files and explore_result is None:
            ctx.fail(
                "EXPLORE PHASE FAILED: Explore agent could not run. "
                f"Required files: {ctx.files}. "
                "Retrieval verification is mandatory for all tests."
            )
            ctx.explore_status = "fail"
            return ctx

        # Store explore results
        if explore_result:
            ctx.explore_result = explore_result
            ctx.explore_status = "pass" if explore_result.success else "fail"
            ctx.explore_input_tokens = explore_result.input_tokens
            ctx.explore_output_tokens = explore_result.output_tokens
            ctx.explore_cache_creation_tokens = getattr(
                explore_result, "cache_creation_tokens", 0
            )
            ctx.explore_cache_read_tokens = getattr(
                explore_result, "cache_read_tokens", 0
            )
            ctx.explore_model = getattr(explore_result, "model", None)

            logger.info("stage.explore.result", test_id=ctx.test_id,
                        files_found=len(explore_result.files),
                        tool_calls=len(explore_result.tool_history),
                        success=explore_result.success,
                        input_tokens=explore_result.input_tokens,
                        output_tokens=explore_result.output_tokens)

        # Verify required files were found
        if ctx.files and explore_result:
            missing_files = self._verify_explore_found_files(ctx, explore_result)

            if missing_files:
                # Deterministic fallback: check filesystem
                if is_debug():
                    debug(f"[EXPLORE STAGE] Running deterministic check for "
                          f"{len(missing_files)} missing file(s)")
                still_missing = self._check_files_on_disk(
                    ctx, missing_files, explore_result
                )

                if still_missing:
                    logger.warning("stage.explore.missing_files",
                                   test_id=ctx.test_id,
                                   missing_files=still_missing)
                    ctx.fail(
                        self._format_missing_files_error(
                            ctx, still_missing, explore_result
                        )
                    )
                    ctx.explore_status = "fail"
                    # Store partial results even on failure
                    ctx.explore_result = explore_result
                    return ctx
                else:
                    logger.info("stage.explore.all_files_recovered",
                                test_id=ctx.test_id,
                                recovered_count=len(missing_files))

        # Store original user prompt BEFORE injection
        ctx.original_user_prompt = ctx.executor.user_prompt

        # Inject explore context into executor prompt
        if explore_result and (
            explore_result.success or explore_result.files or explore_result.summary
        ):
            self._inject_explore_context(ctx, explore_result)
            logger.info("stage.explore.context_injected", test_id=ctx.test_id,
                        success=explore_result.success,
                        files_count=len(explore_result.files),
                        has_summary=bool(explore_result.summary))
        elif explore_result:
            logger.info("stage.explore.context_skipped", test_id=ctx.test_id,
                        reason="No files or summary to inject")

        logger.info("stage.explore.complete", test_id=ctx.test_id)
        return ctx

    async def _run_explore(self, ctx: PipelineContext):
        """Run the explore agent."""
        from ..explore_agent import ExploreAgent

        explore_config = ctx.explore_config
        if not explore_config and ctx.files:
            from ..config import ExploreConfig
            explore_config = ExploreConfig()
            if is_debug():
                debug("[EXPLORE STAGE] Created default ExploreConfig")

        if not explore_config:
            return None

        base_dir = self._get_base_dir(ctx)
        explore_model = explore_config.model

        logger.info("stage.explore.agent_run", test_id=ctx.test_id,
                     explore_model=explore_model,
                     max_files=explore_config.max_files,
                     timeout=explore_config.timeout)

        from ..sdk.query_runner import SDKQueryRunner

        explore_agent = ExploreAgent(
            query_runner=SDKQueryRunner(),
            base_dir=base_dir,
            max_files=explore_config.max_files,
            max_turns=explore_config.max_iterations,
            timeout=float(explore_config.timeout),
            model=explore_model,
        )

        goal = ctx.executor.user_prompt
        return await explore_agent.explore(
            goal=goal, on_progress=ctx.on_explore_event
        )

    def _verify_explore_found_files(self, ctx, explore_result) -> List[str]:
        """Check that required files appear in explore result."""
        if not ctx.files:
            return []

        missing = []
        summary = explore_result.summary or ""

        found_paths = set()
        found_paths_normalized = set()
        if explore_result.files:
            for f in explore_result.files:
                found_paths.add(f.path)
                normalized = os.path.normpath(f.path).lstrip("./")
                found_paths_normalized.add(normalized)

        for file_path in ctx.files:
            normalized_required = os.path.normpath(file_path).lstrip("./")
            if (
                file_path not in summary
                and normalized_required not in summary
                and file_path not in found_paths
                and normalized_required not in found_paths_normalized
            ):
                missing.append(file_path)

        return missing

    def _check_files_on_disk(self, ctx, missing_files, explore_result) -> List[str]:
        """Deterministic fallback: check filesystem for missing files."""
        from ..explore_agent import FileDiscovery

        base_dir = self._get_base_dir(ctx)
        still_missing = []

        for file_path in missing_files:
            full_path = os.path.join(base_dir, file_path)
            normalized = os.path.normpath(full_path)

            if os.path.isfile(normalized):
                logger.info("stage.explore.deterministic_recovery",
                            test_id=ctx.test_id, file_path=file_path)
                explore_result.files.append(
                    FileDiscovery(
                        path=file_path,
                        summary="(recovered by deterministic filesystem check)",
                        relevance=0.5,
                    )
                )
            else:
                still_missing.append(file_path)

        return still_missing

    def _inject_explore_context(self, ctx, explore_result) -> None:
        """Inject explore results into executor's user_prompt."""
        if not explore_result:
            return
        if not explore_result.summary and not explore_result.files:
            return

        context_block = explore_result.to_context_block()
        if context_block:
            ctx.executor.user_prompt = (
                f"{context_block}\n\n---\n\n{ctx.executor.user_prompt}"
            )

    def _format_missing_files_error(self, ctx, missing_files, explore_result) -> str:
        """Format a detailed error message for missing files."""
        error_lines = [
            "EXPLORE PHASE FAILED: Required files not found",
            "",
            "Required files (from test scaffold):",
        ]
        for f in ctx.files:
            status = "FOUND" if f not in missing_files else "MISSING"
            error_lines.append(f"  [{status}] {f}")

        error_lines.extend([
            "",
            "Explore diagnostics:",
            f"  success: {explore_result.success}",
            f"  error: {explore_result.error or '(none)'}",
            f"  tool_calls_count: {explore_result.tool_calls_count}",
            f"  files_found: {len(explore_result.files)}",
        ])

        return "\n".join(error_lines)

    def _get_base_dir(self, ctx) -> str:
        """Get base directory for explore operations."""
        if ctx.source_path:
            from ..loader import find_project_root
            try:
                return find_project_root(ctx.source_path)
            except FileNotFoundError:
                return os.path.dirname(ctx.source_path)
        return "."
