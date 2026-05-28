"""Artifact stage — collects output artifacts and derives legacy fields."""

import os

from ..logging_config import get_logger
from ..pipeline import PipelineContext, PipelineStage
from ..test_object import collect_output_artifacts

logger = get_logger(__name__)


class ArtifactStage(PipelineStage):
    """Collect output artifacts after all stages complete.

    This stage:
    1. Collects all files from the output directory
    2. Tags artifacts by source (browser, report, output)
    3. Derives legacy browser_artifacts and report_artifacts fields
    """

    @property
    def name(self) -> str:
        return "artifact"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Collect and categorize output artifacts.

        Args:
            ctx: The pipeline context (output_dir must be set).

        Returns:
            Updated context (artifacts stored for TestResult assembly).
        """
        output_dir = ctx.output_dir
        if not output_dir:
            logger.info("stage.artifact.skip", test_id=ctx.test_id, reason="no output dir set")
            return ctx

        logger.info("stage.artifact.start", test_id=ctx.test_id, output_dir=output_dir)

        all_output = collect_output_artifacts(output_dir)

        if all_output:
            # Tag source
            for artifact in all_output:
                if artifact["path"].startswith("recordings/"):
                    artifact["source"] = "browser"
                elif (
                    ctx.research_report_rel_path
                    and artifact["path"] == ctx.research_report_rel_path
                ):
                    artifact["source"] = "report"
                else:
                    artifact["source"] = "output"

            # Override report content with full markdown
            if ctx.research_report_rel_path:
                full_report_path = os.path.join(output_dir, ctx.research_report_rel_path)
                if os.path.exists(full_report_path):
                    try:
                        with open(full_report_path, "r", encoding="utf-8") as f:
                            full_content = f.read()
                        for artifact in all_output:
                            if artifact["path"] == ctx.research_report_rel_path:
                                artifact["content"] = full_content
                    except (OSError, UnicodeDecodeError):
                        pass

            # Store for TestResult assembly (done in TestObject._build_result)
            ctx.output_artifacts = all_output
            logger.info(
                "stage.artifact.complete", test_id=ctx.test_id, artifact_count=len(all_output)
            )
        else:
            ctx.output_artifacts = []
            logger.info("stage.artifact.complete", test_id=ctx.test_id, artifact_count=0)

        return ctx
