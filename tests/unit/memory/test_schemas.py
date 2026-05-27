"""tests for memory schemas."""

import time

from dokumen.memory.schemas import Memory, MemoryOperation


class TestMemory:
    def test_memory_creation(self):
        m = Memory(
            id="mem-1",
            content="refund policy allows 30-day returns",
            metadata={"source": "test-refund-policy"},
        )
        assert m.id == "mem-1"
        assert m.content == "refund policy allows 30-day returns"
        assert m.metadata["source"] == "test-refund-policy"
        assert m.embedding is None
        assert m.created_at > 0
        assert m.updated_at > 0

    def test_memory_with_embedding(self):
        m = Memory(
            id="mem-2",
            content="api uses oauth 2.0",
            embedding=[0.1, 0.2, 0.3],
        )
        assert m.embedding == [0.1, 0.2, 0.3]

    def test_memory_to_dict(self):
        m = Memory(id="mem-3", content="test content")
        d = m.to_dict()
        assert d["id"] == "mem-3"
        assert d["content"] == "test content"
        assert "created_at" in d
        assert "updated_at" in d
        assert "embedding" in d

    def test_memory_from_dict(self):
        d = {
            "id": "mem-4",
            "content": "loaded from disk",
            "embedding": [0.5, 0.6],
            "metadata": {"tag": "test"},
            "created_at": 1000.0,
            "updated_at": 2000.0,
        }
        m = Memory.from_dict(d)
        assert m.id == "mem-4"
        assert m.content == "loaded from disk"
        assert m.embedding == [0.5, 0.6]
        assert m.metadata["tag"] == "test"
        assert m.created_at == 1000.0

    def test_memory_to_dict_roundtrip(self):
        m = Memory(
            id="rt-1",
            content="roundtrip test",
            embedding=[1.0, 2.0],
            metadata={"k": "v"},
        )
        d = m.to_dict()
        m2 = Memory.from_dict(d)
        assert m2.id == m.id
        assert m2.content == m.content
        assert m2.embedding == m.embedding
        assert m2.metadata == m.metadata


class TestMemoryOperation:
    def test_operations_exist(self):
        assert MemoryOperation.ADD.value == "add"
        assert MemoryOperation.UPDATE.value == "update"
        assert MemoryOperation.DELETE.value == "delete"
        assert MemoryOperation.NOOP.value == "noop"
