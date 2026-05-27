"""skill store — persists and retrieves learned skills.

follows the same protocol pattern as MemoryStore (rule 2.6)
so we can swap JSON → redis/sqlite later.
"""

import json
import logging
import os
import time
from typing import List, Optional, Tuple

from .types import SkillEntry

logger = logging.getLogger(__name__)


class SkillStore:
    """json-backed skill store with embedding search.

    similar to Mem0Store but specialized for skills:
    - per-client isolation via client_id
    - effectiveness tracking (times_used / times_helpful)
    - category-based filtering
    """

    def __init__(self, store_path: str):
        self._store_path = store_path
        self._skills: List[SkillEntry] = []
        os.makedirs(store_path, exist_ok=True)
        self._load()

    def _load(self) -> None:
        path = os.path.join(self._store_path, "skills.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._skills = [SkillEntry.from_dict(d) for d in data]
            logger.info("loaded skills", extra={"count": len(self._skills), "path": path})
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("failed to load skills, starting fresh", extra={"error": str(e)})
            self._skills = []

    def _save(self) -> None:
        path = os.path.join(self._store_path, "skills.json")
        with open(path, "w") as f:
            json.dump([s.to_dict() for s in self._skills], f, indent=2)

    def add(self, skill: SkillEntry) -> None:
        """add a new skill."""
        self._skills.append(skill)
        self._save()
        logger.info("added skill", extra={"id": skill.id, "category": skill.category.value})

    def get_all(self, client_id: Optional[str] = None) -> List[SkillEntry]:
        """get all skills, optionally filtered by client."""
        if client_id:
            return [s for s in self._skills if s.client_id == client_id]
        return list(self._skills)

    def get_by_category(self, category: str, client_id: Optional[str] = None) -> List[SkillEntry]:
        """get skills by category."""
        skills = self.get_all(client_id)
        return [s for s in skills if s.category.value == category]

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.5,
        client_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Tuple[SkillEntry, float]]:
        """find relevant skills by embedding similarity."""
        from dokumen.memory.embeddings import cosine_similarity

        candidates = self.get_all(client_id)
        if category:
            candidates = [s for s in candidates if s.category.value == category]

        scored = []
        for skill in candidates:
            if skill.embedding is None:
                continue
            sim = cosine_similarity(query_embedding, skill.embedding)
            if sim >= threshold:
                scored.append((skill, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def record_usage(self, skill_id: str, was_helpful: bool) -> None:
        """record that a skill was used and whether it helped."""
        for skill in self._skills:
            if skill.id == skill_id:
                skill.times_used += 1
                if was_helpful:
                    skill.times_helpful += 1
                skill.updated_at = time.time()
                self._save()
                logger.info(
                    "recorded skill usage",
                    extra={
                        "id": skill_id,
                        "helpful": was_helpful,
                        "effectiveness": skill.effectiveness,
                    },
                )
                return

    def prune(self, min_uses: int = 5, max_effectiveness: float = 0.2) -> int:
        """remove skills that have been used enough times but aren't effective."""
        before = len(self._skills)
        self._skills = [
            s
            for s in self._skills
            if s.times_used < min_uses or s.effectiveness > max_effectiveness
        ]
        pruned = before - len(self._skills)
        if pruned:
            self._save()
            logger.info("pruned ineffective skills", extra={"pruned": pruned})
        return pruned

    def delete(self, skill_id: str) -> None:
        """delete a skill by id."""
        self._skills = [s for s in self._skills if s.id != skill_id]
        self._save()
