"""Embedding provider for optional memory features."""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# default: gemini text-embedding-004 (free tier, solid quality)
# experimental: gemini/gemini-embedding-2-preview (multimodal)
DEFAULT_EMBEDDING_MODEL = "gemini/text-embedding-004"


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """compute cosine similarity between two vectors. numpy-free."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingProvider:
    """Provider-agnostic embedding via direct external provider APIs."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key

    async def embed(self, text: str) -> List[float]:
        """embed a single text string."""
        from dokumen.providers.direct_provider import embed_text

        logger.debug(
            "Embedding text",
            extra={"model": self.model, "text_length": len(text)},
        )
        results = await embed_text(
            texts=[text],
            model=self.model,
            api_key=self.api_key,
        )
        return results[0] if results else []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """embed multiple texts in a single call."""
        from dokumen.providers.direct_provider import embed_text

        if not texts:
            return []
        logger.debug(
            "Batch embedding",
            extra={"model": self.model, "count": len(texts)},
        )
        return await embed_text(
            texts=texts,
            model=self.model,
            api_key=self.api_key,
        )
