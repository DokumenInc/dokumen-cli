"""tests for memory configuration."""

import pytest
from pydantic import ValidationError

from dokumen.config import MemoryConfig


class TestMemoryConfig:
    def test_defaults(self):
        config = MemoryConfig()
        assert config.enabled is False
        assert config.store == "mem0"
        assert config.embedding_model == "gemini/text-embedding-004"
        assert config.similarity_threshold == 0.7
        assert config.max_memories_per_query == 10

    def test_enabled(self):
        config = MemoryConfig(enabled=True)
        assert config.enabled is True

    def test_custom_embedding_model(self):
        config = MemoryConfig(embedding_model="gemini/gemini-embedding-2-preview")
        assert config.embedding_model == "gemini/gemini-embedding-2-preview"

    def test_invalid_threshold_too_high(self):
        with pytest.raises(ValidationError):
            MemoryConfig(similarity_threshold=1.5)

    def test_invalid_threshold_too_low(self):
        with pytest.raises(ValidationError):
            MemoryConfig(similarity_threshold=-0.1)

    def test_invalid_store(self):
        with pytest.raises(ValidationError):
            MemoryConfig(store="unknown_store")
