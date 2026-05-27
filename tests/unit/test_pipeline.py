"""Tests for the pipeline framework (PipelineContext, PipelineStage, TestPipeline)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime


class TestPipelineContext:
    """Tests for PipelineContext dataclass."""

    def test_creation_minimal(self):
        """PipelineContext can be created with minimal required fields."""
        from dokumen.pipeline import PipelineContext

        ctx = PipelineContext(
            test_id="test-1",
            reason="Test reason",
            executor=MagicMock(),
            judges=[MagicMock()],
            files=["docs/api.md"],
            timeout=60.0,
            retries=0,
        )

        assert ctx.test_id == "test-1"
        assert ctx.reason == "Test reason"
        assert ctx.files == ["docs/api.md"]
        assert ctx.failed is False
        assert ctx.failure_reasons == []
        assert ctx.executor_output is None
        assert ctx.judge_results == []

    def test_creation_with_all_fields(self):
        """PipelineContext accepts all optional fields."""
        from dokumen.pipeline import PipelineContext

        ctx = PipelineContext(
            test_id="test-2",
            reason="Full test",
            executor=MagicMock(),
            judges=[],
            files=[],
            timeout=120.0,
            retries=2,
            browser_config=MagicMock(),
            explore_config=MagicMock(),
            sandbox=MagicMock(),
            source_path="/path/to/test.yaml",
            test_type="research",
        )

        assert ctx.timeout == 120.0
        assert ctx.retries == 2
        assert ctx.browser_config is not None
        assert ctx.test_type == "research"

    def test_fail_sets_flag(self):
        """PipelineContext.fail() sets failed flag and adds reason."""
        from dokumen.pipeline import PipelineContext

        ctx = PipelineContext(
            test_id="test-1",
            reason="r",
            executor=MagicMock(),
            judges=[],
            files=[],
            timeout=60.0,
            retries=0,
        )

        ctx.fail("Something went wrong")

        assert ctx.failed is True
        assert "Something went wrong" in ctx.failure_reasons


class TestPipelineStage:
    """Tests for PipelineStage ABC."""

    def test_cannot_instantiate_abstract(self):
        """PipelineStage is abstract and cannot be instantiated directly."""
        from dokumen.pipeline import PipelineStage

        with pytest.raises(TypeError):
            PipelineStage()

    def test_concrete_stage_works(self):
        """A concrete PipelineStage subclass can be instantiated."""
        from dokumen.pipeline import PipelineStage, PipelineContext

        class NoopStage(PipelineStage):
            async def run(self, ctx: PipelineContext) -> PipelineContext:
                return ctx

            @property
            def name(self) -> str:
                return "noop"

        stage = NoopStage()
        assert stage.name == "noop"


class TestTestPipeline:
    """Tests for TestPipeline runner."""

    @pytest.mark.asyncio
    async def test_runs_stages_in_order(self):
        """Pipeline runs stages in order and accumulates results."""
        from dokumen.pipeline import PipelineStage, PipelineContext, TestPipeline

        call_order = []

        class Stage1(PipelineStage):
            async def run(self, ctx):
                call_order.append("stage1")
                return ctx

            @property
            def name(self):
                return "stage1"

        class Stage2(PipelineStage):
            async def run(self, ctx):
                call_order.append("stage2")
                return ctx

            @property
            def name(self):
                return "stage2"

        ctx = PipelineContext(
            test_id="test-1",
            reason="r",
            executor=MagicMock(),
            judges=[],
            files=[],
            timeout=60.0,
            retries=0,
        )

        pipeline = TestPipeline(stages=[Stage1(), Stage2()])
        result_ctx = await pipeline.run(ctx)

        assert call_order == ["stage1", "stage2"]
        assert result_ctx.failed is False

    @pytest.mark.asyncio
    async def test_short_circuits_on_failure(self):
        """Pipeline stops executing stages when context is marked as failed."""
        from dokumen.pipeline import PipelineStage, PipelineContext, TestPipeline

        call_order = []

        class FailStage(PipelineStage):
            async def run(self, ctx):
                call_order.append("fail")
                ctx.fail("Intentional failure")
                return ctx

            @property
            def name(self):
                return "fail"

        class NeverCalled(PipelineStage):
            async def run(self, ctx):
                call_order.append("never")
                return ctx

            @property
            def name(self):
                return "never"

        ctx = PipelineContext(
            test_id="test-1",
            reason="r",
            executor=MagicMock(),
            judges=[],
            files=[],
            timeout=60.0,
            retries=0,
        )

        pipeline = TestPipeline(stages=[FailStage(), NeverCalled()])
        result_ctx = await pipeline.run(ctx)

        assert call_order == ["fail"]
        assert result_ctx.failed is True
        assert "Intentional failure" in result_ctx.failure_reasons

    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        """Pipeline with no stages returns context unchanged."""
        from dokumen.pipeline import PipelineContext, TestPipeline

        ctx = PipelineContext(
            test_id="test-1",
            reason="r",
            executor=MagicMock(),
            judges=[],
            files=[],
            timeout=60.0,
            retries=0,
        )

        pipeline = TestPipeline(stages=[])
        result_ctx = await pipeline.run(ctx)

        assert result_ctx.failed is False
        assert result_ctx is ctx

    @pytest.mark.asyncio
    async def test_stage_exception_marks_failure(self):
        """Pipeline handles stage exceptions gracefully."""
        from dokumen.pipeline import PipelineStage, PipelineContext, TestPipeline

        class BrokenStage(PipelineStage):
            async def run(self, ctx):
                raise RuntimeError("Stage exploded")

            @property
            def name(self):
                return "broken"

        ctx = PipelineContext(
            test_id="test-1",
            reason="r",
            executor=MagicMock(),
            judges=[],
            files=[],
            timeout=60.0,
            retries=0,
        )

        pipeline = TestPipeline(stages=[BrokenStage()])
        result_ctx = await pipeline.run(ctx)

        assert result_ctx.failed is True
        assert any("Stage exploded" in r for r in result_ctx.failure_reasons)

    @pytest.mark.asyncio
    async def test_cleanup_callbacks_always_run(self):
        """Pipeline cleanup callbacks are called even after failures."""
        from dokumen.pipeline import PipelineStage, PipelineContext, TestPipeline

        cleanup_called = []

        class FailStage(PipelineStage):
            async def run(self, ctx):
                ctx.fail("fail")
                return ctx

            @property
            def name(self):
                return "fail"

        ctx = PipelineContext(
            test_id="test-1",
            reason="r",
            executor=MagicMock(),
            judges=[],
            files=[],
            timeout=60.0,
            retries=0,
        )

        async def cleanup(c):
            cleanup_called.append(True)

        pipeline = TestPipeline(stages=[FailStage()], cleanup_callbacks=[cleanup])
        await pipeline.run(ctx)

        assert len(cleanup_called) == 1


class TestPipelineContextOutputArtifacts:
    """Tests for output_artifacts as a declared field on PipelineContext."""

    def test_output_artifacts_defaults_to_empty_list(self):
        """output_artifacts should default to an empty list."""
        from dokumen.pipeline import PipelineContext

        ctx = PipelineContext(
            test_id="test-1",
            reason="r",
            executor=MagicMock(),
            judges=[],
            files=[],
            timeout=60.0,
            retries=0,
        )

        assert ctx.output_artifacts == []
        assert isinstance(ctx.output_artifacts, list)

    def test_output_artifacts_can_be_set_directly(self):
        """output_artifacts can be set as a regular attribute (no getattr needed)."""
        from dokumen.pipeline import PipelineContext

        ctx = PipelineContext(
            test_id="test-1",
            reason="r",
            executor=MagicMock(),
            judges=[],
            files=[],
            timeout=60.0,
            retries=0,
        )

        artifacts = [{"path": "file.txt", "source": "output", "size_bytes": 100}]
        ctx.output_artifacts = artifacts

        assert ctx.output_artifacts == artifacts
        assert len(ctx.output_artifacts) == 1
        assert ctx.output_artifacts[0]["path"] == "file.txt"

    def test_output_artifacts_not_shared_between_instances(self):
        """Each PipelineContext has its own output_artifacts list."""
        from dokumen.pipeline import PipelineContext

        ctx1 = PipelineContext(
            test_id="test-1", reason="r", executor=MagicMock(),
            judges=[], files=[], timeout=60.0, retries=0,
        )
        ctx2 = PipelineContext(
            test_id="test-2", reason="r", executor=MagicMock(),
            judges=[], files=[], timeout=60.0, retries=0,
        )

        ctx1.output_artifacts.append({"path": "a.txt"})

        assert len(ctx1.output_artifacts) == 1
        assert len(ctx2.output_artifacts) == 0
