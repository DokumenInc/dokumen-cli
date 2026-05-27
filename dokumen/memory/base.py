"""memory store protocol — swappable backend (rule 2.6)."""

from typing import List, Optional, Protocol, Tuple

from .schemas import Memory


class MemoryStore(Protocol):
    """abstract memory store interface.

    implementations can be backed by JSON files, sqlite, vector db, etc.
    """

    def add(self, memory: Memory) -> None:
        """add a new memory."""
        ...

    def update(self, memory_id: str, content: str, embedding: Optional[List[float]] = None) -> None:
        """update an existing memory's content and optionally its embedding."""
        ...

    def delete(self, memory_id: str) -> None:
        """delete a memory by id."""
        ...

    def get_all(self) -> List[Memory]:
        """return all stored memories."""
        ...

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[Memory, float]]:
        """find memories similar to query embedding.

        returns list of (memory, similarity_score) tuples sorted by score descending.
        """
        ...
