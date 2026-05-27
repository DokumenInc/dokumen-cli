"""mem0-style memory store — extraction → update pipeline.

implements the mem0 base architecture:
1. extraction phase: LLM extracts salient facts from conversation
2. update phase: for each fact, retrieve similar memories via embeddings,
   LLM decides ADD/UPDATE/DELETE/NOOP
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import EmbeddingProvider, cosine_similarity
from .schemas import Memory, MemoryOperation

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """extract the key facts and learnings from this conversation.
return a JSON object with a "facts" array containing short, self-contained factual statements.

rules:
- each fact should be a single, clear statement
- include only information worth remembering for future conversations
- skip greetings, meta-discussion, and obvious context
- aim for 1-5 facts per conversation (fewer is fine if conversation is simple)

example output:
{"facts": ["refund policy allows 30-day returns for unused items", "API uses OAuth 2.0 with refresh tokens"]}

if there are no facts worth extracting, return: {"facts": []}"""

UPDATE_DECISION_PROMPT = """you are managing a memory store. a new fact has been extracted from a conversation.

new fact: {fact}

existing similar memories:
{similar_memories}

decide what to do with this fact. return a JSON object with:
- "operation": one of "add", "update", "delete", "noop"
- "reason": brief explanation
- "memory_id": (only for update/delete) the id of the memory to modify

rules:
- ADD: the fact is genuinely new information not covered by existing memories
- UPDATE: the fact contradicts or refines an existing memory (provide memory_id)
- DELETE: the fact indicates an existing memory is no longer true (provide memory_id)
- NOOP: the fact is already captured by an existing memory

example: {{"operation": "add", "reason": "new fact about auth method"}}"""


class Mem0Store:
    """mem0-style memory store backed by JSON files."""

    def __init__(self, store_path: str):
        self._store_path = store_path
        self._memories: List[Memory] = []
        os.makedirs(store_path, exist_ok=True)
        self._load()

    def _load(self) -> None:
        """load memories from disk."""
        path = os.path.join(self._store_path, "memories.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self._memories = [Memory.from_dict(d) for d in data]
                logger.info(
                    "Loaded memories from disk",
                    extra={"count": len(self._memories), "path": path},
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "Failed to load memories, starting fresh",
                    extra={"error": str(e), "path": path},
                )
                self._memories = []

    def _save(self) -> None:
        """persist memories to disk."""
        path = os.path.join(self._store_path, "memories.json")
        with open(path, "w") as f:
            json.dump([m.to_dict() for m in self._memories], f, indent=2)
        logger.debug(
            "Saved memories to disk",
            extra={"count": len(self._memories), "path": path},
        )

    def add(self, memory: Memory) -> None:
        """add a new memory and persist."""
        self._memories.append(memory)
        self._save()
        logger.info(
            "Memory added",
            extra={"memory_id": memory.id, "content_preview": memory.content[:80]},
        )

    def update(
        self, memory_id: str, content: str, embedding: Optional[List[float]] = None
    ) -> None:
        """update an existing memory's content."""
        for m in self._memories:
            if m.id == memory_id:
                m.content = content
                m.updated_at = time.time()
                if embedding is not None:
                    m.embedding = embedding
                self._save()
                logger.info(
                    "Memory updated",
                    extra={"memory_id": memory_id, "content_preview": content[:80]},
                )
                return
        logger.warning(
            "Memory not found for update",
            extra={"memory_id": memory_id},
        )

    def delete(self, memory_id: str) -> None:
        """delete a memory by id."""
        before = len(self._memories)
        self._memories = [m for m in self._memories if m.id != memory_id]
        if len(self._memories) < before:
            self._save()
            logger.info("Memory deleted", extra={"memory_id": memory_id})

    def get_all(self) -> List[Memory]:
        """return all stored memories."""
        return list(self._memories)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[Memory, float]]:
        """find memories similar to query embedding."""
        scored = []
        for m in self._memories:
            if m.embedding is None:
                continue
            sim = cosine_similarity(query_embedding, m.embedding)
            if sim >= threshold:
                scored.append((m, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ── extraction phase ─────────────────────────────────────────────

    async def extract_facts(
        self,
        conversation: List[Dict[str, Any]],
        model: str = "gemini/gemini-2.0-flash",
    ) -> List[str]:
        """extract salient facts from a conversation using an LLM.

        args:
            conversation: list of message dicts (role, content)
            model: LLM model for extraction

        returns:
            list of fact strings
        """
        if not conversation:
            return []

        conv_text = "\n".join(
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in conversation
            if msg.get("content")
        )

        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": conv_text},
        ]

        logger.info(
            "Extracting facts from conversation",
            extra={"model": model, "message_count": len(conversation)},
        )

        from dokumen.providers.dokurouter import dokurouter_completion

        result_text = await dokurouter_completion(messages=messages, model=model, temperature=0.0)

        try:
            data = json.loads(result_text.strip())
            facts = data.get("facts", [])
            logger.info(
                "Facts extracted",
                extra={"count": len(facts)},
            )
            return facts
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "Failed to parse extraction response",
                extra={"response_preview": result_text[:200] if result_text else None},
            )
            return []

    # ── update phase ─────────────────────────────────────────────────

    async def decide_operation(
        self,
        fact: str,
        fact_embedding: List[float],
        similar_memories: List[Tuple[Memory, float]],
        model: str = "gemini/gemini-2.0-flash",
    ) -> MemoryOperation:
        """decide what operation to perform for a new fact.

        args:
            fact: the extracted fact
            fact_embedding: embedding of the fact
            similar_memories: list of (memory, similarity) tuples
            model: LLM model for decision

        returns:
            MemoryOperation (ADD, UPDATE, DELETE, NOOP)
        """
        if not similar_memories:
            similar_text = "(no similar memories found)"
        else:
            lines = []
            for m, score in similar_memories:
                lines.append(f"- [id={m.id}, similarity={score:.2f}] {m.content}")
            similar_text = "\n".join(lines)

        prompt = UPDATE_DECISION_PROMPT.format(
            fact=fact,
            similar_memories=similar_text,
        )

        messages = [
            {"role": "system", "content": "you are a memory management agent."},
            {"role": "user", "content": prompt},
        ]

        logger.info(
            "Deciding memory operation",
            extra={
                "model": model,
                "fact_preview": fact[:80],
                "similar_count": len(similar_memories),
            },
        )

        from dokumen.providers.dokurouter import dokurouter_completion

        result_text = await dokurouter_completion(messages=messages, model=model, temperature=0.0)

        try:
            data = json.loads(result_text.strip())
            op_str = data.get("operation", "noop").lower()
            op = MemoryOperation(op_str)
            logger.info(
                "Memory operation decided",
                extra={
                    "operation": op.value,
                    "reason": data.get("reason", ""),
                    "memory_id": data.get("memory_id"),
                },
            )
            return op
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Failed to parse operation decision, defaulting to NOOP",
                extra={"response_preview": result_text[:200] if result_text else None},
            )
            return MemoryOperation.NOOP

    # ── full pipeline ────────────────────────────────────────────────

    async def process_conversation(
        self,
        conversation: List[Dict[str, Any]],
        embedding_provider: EmbeddingProvider,
        model: str = "gemini/gemini-2.0-flash",
        similarity_threshold: float = 0.7,
        top_k: int = 10,
    ) -> List[Memory]:
        """run the full mem0 extraction → update pipeline.

        args:
            conversation: list of message dicts
            embedding_provider: provider for generating embeddings
            model: LLM model for extraction and decisions
            similarity_threshold: threshold for similar memory retrieval
            top_k: max similar memories to retrieve per fact

        returns:
            list of newly added or updated memories
        """
        logger.info(
            "Processing conversation for memory",
            extra={"message_count": len(conversation)},
        )

        # step 1: extract facts
        facts = await self.extract_facts(conversation, model=model)
        if not facts:
            logger.info("No facts extracted, skipping memory update")
            return []

        changed: List[Memory] = []

        for fact in facts:
            # step 2: embed the fact
            fact_embedding = await embedding_provider.embed(fact)

            # step 3: find similar existing memories
            similar = self.search(
                query_embedding=fact_embedding,
                top_k=top_k,
                threshold=similarity_threshold,
            )

            # step 4: decide operation
            op = await self.decide_operation(
                fact=fact,
                fact_embedding=fact_embedding,
                similar_memories=similar,
                model=model,
            )

            # step 5: execute operation
            if op == MemoryOperation.ADD:
                new_mem = Memory(
                    id=str(uuid.uuid4()),
                    content=fact,
                    embedding=fact_embedding,
                    metadata={"source": "conversation"},
                )
                self.add(new_mem)
                changed.append(new_mem)

            elif op == MemoryOperation.UPDATE and similar:
                target = similar[0][0]
                self.update(target.id, fact, embedding=fact_embedding)
                changed.append(target)

            elif op == MemoryOperation.DELETE and similar:
                self.delete(similar[0][0].id)

            # NOOP: do nothing

        logger.info(
            "Memory processing complete",
            extra={
                "facts_extracted": len(facts),
                "memories_changed": len(changed),
                "total_memories": len(self._memories),
            },
        )
        return changed
