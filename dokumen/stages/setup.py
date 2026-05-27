"""Setup stage — runs clone/install/start-server steps before execution."""

import os

from ..logging_config import get_logger
from ..pipeline import PipelineContext, PipelineStage

logger = get_logger(__name__)


class SetupStage(PipelineStage):
    """Run pre-execution setup steps (clone, install, start server).

    If no setup steps are configured, this stage is a no-op.
    """

    @property
    def name(self) -> str:
        return "setup"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute setup steps.

        Args:
            ctx: The pipeline context.

        Returns:
            Updated context (with setup_runner set for cleanup).
        """
        if not ctx.setup_steps:
            logger.info("stage.setup.skip", test_id=ctx.test_id, reason="no setup steps configured")
            return ctx

        from ..setup_runner import SetupRunner, SetupError

        logger.info("stage.setup.start", test_id=ctx.test_id, step_count=len(ctx.setup_steps))
        env = {**os.environ}
        setup_runner = SetupRunner(env=env)
        ctx.setup_runner = setup_runner

        try:
            await setup_runner.run_steps(ctx.setup_steps)
            logger.info("stage.setup.complete", test_id=ctx.test_id)
        except SetupError as e:
            logger.error(
                "stage.setup.failed", test_id=ctx.test_id, step_name=e.step_name, error=str(e)
            )
            ctx.fail(f"Setup failed: {e}")

        return ctx
