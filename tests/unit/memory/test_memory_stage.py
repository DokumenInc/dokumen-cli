"""tests for memory pipeline stage."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from dokumen.stages.memory import MemoryStage
from dokumen.memory.schemas import Memory


def _make_ctx(
    memory_store=None,
    embedding_provider=None,
    memory_config=None,
    executor_output=None,
):
    """build a minimal pipeline context mock."""
    ctx = MagicMock()
    ctx.test_id = "test-memory-stage"
    ctx.memory_store = memory_store
    ctx.embedding_provider = embedding_provider
    ctx.memory_config = memory_config
    ctx.executor_output = executor_output
    ctx.failed = False
    return ctx


class TestMemoryStageSkips:
    @pytest.mark.asyncio
    async def test_skips_when_no_store(self):
        ctx = _make_ctx()
        stage = MemoryStage()
        result = await stage.run(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_skips_when_no_executor_output(self):
        store = MagicMock()
        embed = MagicMock()
        ctx = _make_ctx(memory_store=store, embedding_provider=embed)
        ctx.executor_output = None
        stage = MemoryStage()
        result = await stage.run(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_skips_when_empty_conversation(self):
        store = MagicMock()
        embed = MagicMock()
        executor_output = MagicMock()
        executor_output.conversation_log = []
        ctx = _make_ctx(
            memory_store=store,
            embedding_provider=embed,
            executor_output=executor_output,
        )
        stage = MemoryStage()
        result = await stage.run(ctx)
        assert result is ctx


class TestMemoryStageRuns:
    @pytest.mark.asyncio
    async def test_processes_conversation(self):
        store = MagicMock()
        store.process_conversation = AsyncMock(return_value=[
            Memory(id="new-1", content="learned fact"),
        ])
        store.get_all.return_value = [Memory(id="new-1", content="learned fact")]

        embed = MagicMock()

        executor_output = MagicMock()
        executor_output.conversation_log = [
            {"role": "user", "content": "what auth methods?"},
            {"role": "assistant", "content": "OAuth 2.0 and API keys"},
        ]

        config = MagicMock()
        config.model = "gemini/gemini-2.0-flash"
        config.similarity_threshold = 0.7
        config.max_memories_per_query = 10

        ctx = _make_ctx(
            memory_store=store,
            embedding_provider=embed,
            memory_config=config,
            executor_output=executor_output,
        )

        stage = MemoryStage()
        result = await stage.run(ctx)
        assert result is ctx

        store.process_conversation.assert_called_once()
        call_kwargs = store.process_conversation.call_args
        assert len(call_kwargs.kwargs["conversation"]) == 2
        assert call_kwargs.kwargs["model"] == "gemini/gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_handles_errors_gracefully(self):
        """memory extraction errors should not fail the pipeline."""
        store = MagicMock()
        store.process_conversation = AsyncMock(
            side_effect=RuntimeError("embedding api down")
        )

        embed = MagicMock()

        executor_output = MagicMock()
        executor_output.conversation_log = [
            {"role": "user", "content": "hello"},
        ]

        ctx = _make_ctx(
            memory_store=store,
            embedding_provider=embed,
            executor_output=executor_output,
        )

        stage = MemoryStage()
        result = await stage.run(ctx)
        # should NOT fail the pipeline
        assert result is ctx


class TestMemoryStageConfig:
    @pytest.mark.asyncio
    async def test_uses_config_values(self):
        store = MagicMock()
        store.process_conversation = AsyncMock(return_value=[])
        store.get_all.return_value = []

        embed = MagicMock()

        executor_output = MagicMock()
        executor_output.conversation_log = [
            {"role": "user", "content": "test"},
        ]

        config = MagicMock()
        config.model = "gemini/gemini-2.5-pro"
        config.similarity_threshold = 0.8
        config.max_memories_per_query = 5

        ctx = _make_ctx(
            memory_store=store,
            embedding_provider=embed,
            memory_config=config,
            executor_output=executor_output,
        )

        stage = MemoryStage()
        await stage.run(ctx)

        call_kwargs = store.process_conversation.call_args.kwargs
        assert call_kwargs["model"] == "gemini/gemini-2.5-pro"
        assert call_kwargs["similarity_threshold"] == 0.8
        assert call_kwargs["top_k"] == 5
