"""tests for mem0 memory store."""

import json
import os
import tempfile

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dokumen.memory.schemas import Memory, MemoryOperation
from dokumen.memory.mem0_store import Mem0Store
from dokumen.memory.embeddings import EmbeddingProvider


class TestMem0StoreInit:
    def test_creates_store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "memory")
            store = Mem0Store(store_path=store_path)
            assert os.path.isdir(store_path)

    def test_loads_existing_memories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # pre-populate a memories file
            memories = [
                Memory(id="pre-1", content="existing fact", embedding=[0.1, 0.2]).to_dict(),
            ]
            os.makedirs(tmpdir, exist_ok=True)
            with open(os.path.join(tmpdir, "memories.json"), "w") as f:
                json.dump(memories, f)

            store = Mem0Store(store_path=tmpdir)
            assert len(store._memories) == 1
            assert store._memories[0].id == "pre-1"


class TestMem0StoreBasicOps:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = Mem0Store(store_path=self._tmpdir)

    def test_add_memory(self):
        m = Memory(id="add-1", content="new fact", embedding=[0.1])
        self.store.add(m)
        assert len(self.store._memories) == 1
        assert self.store._memories[0].content == "new fact"

    def test_add_saves_to_disk(self):
        m = Memory(id="disk-1", content="persisted", embedding=[0.1])
        self.store.add(m)
        # reload from disk
        store2 = Mem0Store(store_path=self._tmpdir)
        assert len(store2._memories) == 1
        assert store2._memories[0].content == "persisted"

    def test_update_memory(self):
        m = Memory(id="upd-1", content="old fact", embedding=[0.1])
        self.store.add(m)
        self.store.update("upd-1", "new fact", embedding=[0.2])
        assert self.store._memories[0].content == "new fact"
        assert self.store._memories[0].embedding == [0.2]

    def test_update_nonexistent_is_noop(self):
        self.store.update("nope", "whatever")
        assert len(self.store._memories) == 0

    def test_delete_memory(self):
        m = Memory(id="del-1", content="to delete", embedding=[0.1])
        self.store.add(m)
        self.store.delete("del-1")
        assert len(self.store._memories) == 0

    def test_delete_nonexistent_is_noop(self):
        self.store.delete("nope")
        assert len(self.store._memories) == 0

    def test_get_all(self):
        self.store.add(Memory(id="a", content="one", embedding=[0.1]))
        self.store.add(Memory(id="b", content="two", embedding=[0.2]))
        all_mems = self.store.get_all()
        assert len(all_mems) == 2


class TestMem0StoreSimilaritySearch:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = Mem0Store(store_path=self._tmpdir)

    def test_search_returns_similar(self):
        # add memories with known embeddings
        self.store.add(Memory(id="s1", content="oauth info", embedding=[1.0, 0.0, 0.0]))
        self.store.add(Memory(id="s2", content="api keys", embedding=[0.9, 0.1, 0.0]))
        self.store.add(Memory(id="s3", content="unrelated", embedding=[0.0, 0.0, 1.0]))

        results = self.store.search(query_embedding=[1.0, 0.0, 0.0], top_k=2, threshold=0.5)
        assert len(results) == 2
        assert results[0][0].id == "s1"  # most similar
        assert results[1][0].id == "s2"

    def test_search_respects_threshold(self):
        self.store.add(Memory(id="s1", content="close", embedding=[1.0, 0.0]))
        self.store.add(Memory(id="s2", content="far", embedding=[0.0, 1.0]))

        results = self.store.search(query_embedding=[1.0, 0.0], top_k=10, threshold=0.9)
        assert len(results) == 1
        assert results[0][0].id == "s1"

    def test_search_empty_store(self):
        results = self.store.search(query_embedding=[1.0, 0.0], top_k=5)
        assert results == []

    def test_search_skips_memories_without_embeddings(self):
        self.store.add(Memory(id="no-emb", content="no embedding"))
        self.store.add(Memory(id="has-emb", content="has embedding", embedding=[1.0, 0.0]))

        results = self.store.search(query_embedding=[1.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0][0].id == "has-emb"


class TestMem0StoreExtraction:
    """tests for the extraction phase — extracting facts from conversation."""

    @pytest.mark.asyncio
    async def test_extract_facts_from_conversation(self):
        store = Mem0Store(store_path=tempfile.mkdtemp())

        conversation = [
            {"role": "user", "content": "what is the refund policy?"},
            {"role": "assistant", "content": "the refund policy allows 30-day returns for unused items."},
        ]

        # mock the LLM call that extracts facts
        extracted = json.dumps({
            "facts": [
                "refund policy allows 30-day returns for unused items",
            ]
        })

        with patch("dokumen.providers.dokurouter.dokurouter_completion", new=AsyncMock(return_value=extracted)):
            facts = await store.extract_facts(conversation, model="gemini/gemini-2.0-flash")
            assert len(facts) == 1
            assert "30-day" in facts[0]

    @pytest.mark.asyncio
    async def test_extract_facts_empty_conversation(self):
        store = Mem0Store(store_path=tempfile.mkdtemp())
        facts = await store.extract_facts([], model="gemini/gemini-2.0-flash")
        assert facts == []


class TestMem0StoreUpdatePhase:
    """tests for the update phase — deciding ADD/UPDATE/DELETE/NOOP."""

    @pytest.mark.asyncio
    async def test_update_phase_add(self):
        store = Mem0Store(store_path=tempfile.mkdtemp())

        # mock embedding
        mock_embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        # mock LLM deciding ADD
        decision = json.dumps({"operation": "add", "reason": "new fact"})

        with patch("dokumen.providers.dokurouter.dokurouter_completion", new=AsyncMock(return_value=decision)):
            op = await store.decide_operation(
                fact="new important fact",
                fact_embedding=[0.1, 0.2, 0.3],
                similar_memories=[],
                model="gemini/gemini-2.0-flash",
            )
            assert op == MemoryOperation.ADD

    @pytest.mark.asyncio
    async def test_update_phase_noop(self):
        store = Mem0Store(store_path=tempfile.mkdtemp())
        existing = Memory(id="e1", content="same fact already stored", embedding=[0.1, 0.2])

        decision = json.dumps({"operation": "noop", "reason": "already exists"})

        with patch("dokumen.providers.dokurouter.dokurouter_completion", new=AsyncMock(return_value=decision)):
            op = await store.decide_operation(
                fact="same fact already stored",
                fact_embedding=[0.1, 0.2, 0.3],
                similar_memories=[(existing, 0.95)],
                model="gemini/gemini-2.0-flash",
            )
            assert op == MemoryOperation.NOOP

    @pytest.mark.asyncio
    async def test_update_phase_update(self):
        store = Mem0Store(store_path=tempfile.mkdtemp())
        existing = Memory(id="e1", content="refund policy is 14 days", embedding=[0.1])

        decision = json.dumps({"operation": "update", "memory_id": "e1", "reason": "policy changed"})

        with patch("dokumen.providers.dokurouter.dokurouter_completion", new=AsyncMock(return_value=decision)):
            op = await store.decide_operation(
                fact="refund policy is now 30 days",
                fact_embedding=[0.1, 0.2],
                similar_memories=[(existing, 0.85)],
                model="gemini/gemini-2.0-flash",
            )
            assert op == MemoryOperation.UPDATE

    @pytest.mark.asyncio
    async def test_update_phase_delete(self):
        store = Mem0Store(store_path=tempfile.mkdtemp())
        existing = Memory(id="e1", content="deprecated feature X exists", embedding=[0.1])

        decision = json.dumps({"operation": "delete", "memory_id": "e1", "reason": "feature removed"})

        with patch("dokumen.providers.dokurouter.dokurouter_completion", new=AsyncMock(return_value=decision)):
            op = await store.decide_operation(
                fact="feature X has been removed",
                fact_embedding=[0.1, 0.2],
                similar_memories=[(existing, 0.8)],
                model="gemini/gemini-2.0-flash",
            )
            assert op == MemoryOperation.DELETE


class TestMem0StoreProcessConversation:
    """tests for the full extraction → update pipeline."""

    @pytest.mark.asyncio
    async def test_process_adds_new_memories(self):
        store = Mem0Store(store_path=tempfile.mkdtemp())

        conversation = [
            {"role": "user", "content": "tell me about auth"},
            {"role": "assistant", "content": "we use OAuth 2.0 with refresh tokens"},
        ]

        extracted = json.dumps({"facts": ["uses OAuth 2.0 with refresh tokens"]})
        add_decision = json.dumps({"operation": "add", "reason": "new"})

        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return extracted
            return add_decision

        mock_embed_provider = MagicMock()
        mock_embed_provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        with patch("dokumen.providers.dokurouter.dokurouter_completion", side_effect=mock_completion):
            await store.process_conversation(
                conversation=conversation,
                embedding_provider=mock_embed_provider,
                model="gemini/gemini-2.0-flash",
            )

        assert len(store._memories) == 1
        assert "OAuth" in store._memories[0].content
