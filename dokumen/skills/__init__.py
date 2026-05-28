"""self-improving skill framework.

skills are reusable tips/patterns extracted from test run failures
that get injected into future executor and judge prompts to improve
performance over time. per-client, per-domain.
"""

from .types import SkillEntry, SkillCategory
from .store import SkillStore
from .extractor import SkillExtractor
from .injector import SkillInjector

__all__ = [
    "SkillEntry",
    "SkillCategory",
    "SkillStore",
    "SkillExtractor",
    "SkillInjector",
]
