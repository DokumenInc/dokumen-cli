"""tests for memory embeddings."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dokumen.memory.embeddings import cosine_similarity, EmbeddingProvider


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        sim = cosine_similarity([1, 1, 0], [1, 0.9, 0.1])
        assert sim > 0.95

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_empty_vectors_returns_zero(self):
        assert cosine_similarity([], []) == 0.0


class TestEmbeddingProvider:
    def test_default_model(self):
        provider = EmbeddingProvider()
        assert provider.model == "gemini/text-embedding-004"

    def test_custom_model(self):
        provider = EmbeddingProvider(model="gemini/gemini-embedding-2-preview")
        assert provider.model == "gemini/gemini-embedding-2-preview"

    @pytest.mark.asyncio
    async def test_embed_single(self):
        provider = EmbeddingProvider()

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        with patch("dokumen.memory.embeddings.aembedding", new=AsyncMock(return_value=mock_response)):
            result = await provider.embed("hello world")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        provider = EmbeddingProvider()

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]

        with patch("dokumen.memory.embeddings.aembedding", new=AsyncMock(return_value=mock_response)):
            result = await provider.embed_batch(["hello", "world"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_embed_passes_api_key(self):
        provider = EmbeddingProvider(api_key="test-key-123")

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]

        mock_aembedding = AsyncMock(return_value=mock_response)
        with patch("dokumen.memory.embeddings.aembedding", new=mock_aembedding):
            await provider.embed("test")
            mock_aembedding.assert_called_once()
            call_kwargs = mock_aembedding.call_args
            assert call_kwargs.kwargs.get("api_key") == "test-key-123"
