"""skill extractor — analyzes test run failures to extract reusable skills.

the extraction pipeline:
1. look at judge verdicts (especially decomposed sub-assertions)
2. identify what went wrong and why
3. generate a concise, actionable skill/tip
4. embed it for future retrieval
5. store it in the skill store

this uses an LLM call to extract skills from failures, similar to
how mem0 extracts facts from conversations.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .types import SkillEntry, SkillCategory

logger = logging.getLogger(__name__)

# prompt for extracting skills from test failures
SKILL_EXTRACTION_PROMPT = """analyze this test run failure and extract reusable skills/tips.

test: {test_id}
executor prompt: {executor_prompt}
executor response (truncated): {executor_response}

judge verdicts:
{judge_verdicts}

extract 1-3 concise, actionable skills that would help avoid this failure in future runs.
each skill should be a single sentence tip that can be injected into a prompt.

return JSON:
{{"skills": [
  {{"content": "...", "category": "executor|judge|tool_use|domain|error_recovery", "tags": ["...", "..."]}}
]}}"""


class SkillExtractor:
    """extracts skills from test run results.

    can work in two modes:
    1. llm-based: uses an LLM to analyze failures and generate skills (requires API call)
    2. rule-based: uses heuristics to extract skills without LLM calls (free)
    """

    def __init__(self, use_llm: bool = False, model: str = "gemini/gemini-2.0-flash"):
        self.use_llm = use_llm
        self.model = model

    def extract_from_verdicts(
        self,
        test_id: str,
        executor_prompt: str,
        executor_response: str,
        judge_verdicts: List[Dict[str, Any]],
        client_id: str = "",
    ) -> List[SkillEntry]:
        """extract skills from judge verdicts using rule-based heuristics.

        this is the free path — no LLM calls. analyzes verdict structure
        to identify common failure patterns.
        """
        skills = []

        for verdict in judge_verdicts:
            if verdict.get("passed", True):
                continue  # only learn from failures

            judge_id = verdict.get("judge_id", "unknown")
            sub_assertions = verdict.get("sub_assertions", [])
            failure_reason = verdict.get("failure_reason", "")
            confidence = verdict.get("confidence", 0.0)

            # pattern: decomposed sub-assertions with specific failures
            if sub_assertions:
                failed_subs = [sa for sa in sub_assertions if not sa.get("passed", True)]
                for sa in failed_subs:
                    question = sa.get("question", "")
                    reason = sa.get("reason", "")
                    if question and reason:
                        skill = SkillEntry(
                            content=f"when evaluating '{test_id}': ensure {question.lower().rstrip('?')} — previous failure: {reason}",
                            category=SkillCategory.DOMAIN,
                            source_test=test_id,
                            source_judge=judge_id,
                            tags=_extract_tags(question + " " + reason),
                            client_id=client_id,
                        )
                        skills.append(skill)

            # pattern: low confidence suggests ambiguity
            elif confidence and confidence < 0.3:
                if failure_reason:
                    skill = SkillEntry(
                        content=f"common failure in '{test_id}': {failure_reason}",
                        category=SkillCategory.EXECUTOR,
                        source_test=test_id,
                        source_judge=judge_id,
                        tags=_extract_tags(failure_reason),
                        client_id=client_id,
                    )
                    skills.append(skill)

            # pattern: parse error suggests formatting issue
            elif verdict.get("error"):
                skill = SkillEntry(
                    content=f"judge '{judge_id}' had parse errors on test '{test_id}' — may need clearer assertion format",
                    category=SkillCategory.JUDGE,
                    source_test=test_id,
                    source_judge=judge_id,
                    tags=["parse_error", "formatting"],
                    client_id=client_id,
                )
                skills.append(skill)

        logger.info(
            "extracted skills from verdicts",
            extra={"test_id": test_id, "skills_count": len(skills)},
        )
        return skills

    async def extract_from_verdicts_llm(
        self,
        test_id: str,
        executor_prompt: str,
        executor_response: str,
        judge_verdicts: List[Dict[str, Any]],
        client_id: str = "",
    ) -> List[SkillEntry]:
        """extract skills using LLM analysis (richer but costs money).

        TODO: shelved for now — enable when we have budget for extraction calls.
        the rule-based extractor handles 80% of cases.
        """
        # from dokumen.providers.dokurouter import dokurouter_completion
        #
        # verdicts_str = json.dumps(judge_verdicts, indent=2)[:2000]
        # prompt = SKILL_EXTRACTION_PROMPT.format(
        #     test_id=test_id,
        #     executor_prompt=executor_prompt[:500],
        #     executor_response=executor_response[:1000],
        #     judge_verdicts=verdicts_str,
        # )
        #
        # text = await dokurouter_completion(
        #     messages=[{"role": "user", "content": prompt}],
        #     model=self.model,
        #     temperature=0.3,
        # )
        # ... parse JSON, create SkillEntry objects ...

        logger.info("llm skill extraction not yet enabled, falling back to rule-based")
        return self.extract_from_verdicts(
            test_id, executor_prompt, executor_response, judge_verdicts, client_id
        )


def _extract_tags(text: str) -> List[str]:
    """extract simple keyword tags from text."""
    # common domain keywords
    keywords = [
        "oauth", "auth", "api", "refund", "policy", "token", "endpoint",
        "error", "timeout", "rate_limit", "permission", "format", "json",
        "markdown", "code", "test", "validation", "security", "config",
    ]
    text_lower = text.lower()
    return [kw for kw in keywords if kw in text_lower][:5]
