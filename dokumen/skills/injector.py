"""skill injector — retrieves relevant skills and injects into prompts.

this is the read side of the skill framework. before an executor or judge
runs, we retrieve skills relevant to the test being run and append them
to the system prompt as tips/guidelines.
"""

import logging
from typing import List, Optional, Tuple

from .types import SkillEntry, SkillCategory
from .store import SkillStore

logger = logging.getLogger(__name__)

# injected into system prompts when skills are available
SKILL_INJECTION_HEADER = "\n\n--- learned tips (from previous runs) ---\n"
SKILL_INJECTION_FOOTER = "\n--- end tips ---\n"


class SkillInjector:
    """retrieves and injects relevant skills into agent prompts."""

    def __init__(self, store: SkillStore, max_skills: int = 5):
        self.store = store
        self.max_skills = max_skills

    def get_relevant_skills(
        self,
        query_embedding: Optional[List[float]] = None,
        category: Optional[str] = None,
        client_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[SkillEntry]:
        """retrieve relevant skills for a test run.

        uses embedding similarity if available, falls back to tag matching.
        """
        if query_embedding:
            results = self.store.search(
                query_embedding=query_embedding,
                top_k=self.max_skills,
                threshold=0.5,
                client_id=client_id,
                category=category,
            )
            skills = [skill for skill, _score in results]
        else:
            # fall back to category + tag filtering
            skills = self.store.get_by_category(category, client_id) if category else self.store.get_all(client_id)
            if tags:
                tag_set = set(tags)
                skills = [s for s in skills if tag_set.intersection(s.tags)]
            # sort by effectiveness (most helpful first)
            skills.sort(key=lambda s: s.effectiveness, reverse=True)
            skills = skills[:self.max_skills]

        return skills

    def inject_into_prompt(
        self,
        system_prompt: str,
        skills: List[SkillEntry],
    ) -> str:
        """append relevant skills to a system prompt."""
        if not skills:
            return system_prompt

        lines = [SKILL_INJECTION_HEADER]
        for i, skill in enumerate(skills, 1):
            lines.append(f"{i}. {skill.content}")
        lines.append(SKILL_INJECTION_FOOTER)

        injected = system_prompt + "\n".join(lines)

        logger.info(
            "injected skills into prompt",
            extra={"count": len(skills), "ids": [s.id for s in skills]},
        )

        return injected

    def record_outcome(self, skills: List[SkillEntry], test_passed: bool) -> None:
        """record whether injected skills led to a pass.

        called after a test run completes. updates effectiveness tracking
        so we can prune bad skills over time.
        """
        for skill in skills:
            self.store.record_usage(skill.id, was_helpful=test_passed)
