"""
Agent resolution for the Agent SOP Testing Framework.

Handles loading agent definitions from YAML files, merging agent defaults
with scaffold overrides, skills collection and formatting, and
auto-injection of research judges.
"""

from typing import Any, List, Optional, Tuple
from pathlib import Path
import hashlib

from dokumen_schema.agent_defs import load_agent as _load_agent_def
from dokumen_schema.skills import SkillLoader

from .agent_loader import get_agent_skills
from .logging_config import get_logger

logger = get_logger(__name__)

# Inline research judge prompts
RESEARCH_SOURCES_JUDGE_PROMPT = """\
You are a research sources quality judge. Evaluate the quality and reliability of sources cited in the executor's research output.

## Evaluation Criteria

Score each criterion from 1 (poor) to 5 (excellent):

| Criterion | Description |
|-----------|-------------|
| Relevance | Sources directly address the research question |
| Citation Coverage | Key claims are backed by cited sources |
| Diversity | Multiple independent sources used, not just one |
| Recency | Sources are current enough for the research topic |
| Credibility | Sources come from authoritative or reputable origins |

## Scoring Rules

- PASS if the average score across all criteria is >= 3.0 AND no single criterion scores below 2
- FAIL otherwise
## Output Format

Return a JSON object:

```json
{
  "verdict": "PASS",
  "reason": "Brief explanation of source quality assessment",
  "scores": {
    "relevance": 4,
    "citation_coverage": 3,
    "diversity": 4,
    "recency": 5,
    "credibility": 4
  },
  "average_score": 4.0
}
```

## Important

- Focus only on source quality, not on answer correctness (that is the verdict judge's job).
- If the executor output contains no sources or citations at all, score citation_coverage as 1.
- Partial citations (some claims sourced, others not) should reduce citation_coverage proportionally."""

RESEARCH_VERDICT_JUDGE_PROMPT = """\
You are a research verdict judge. You have two tasks:

1. Write a structured markdown assessment report
2. Produce a JSON verdict

## Task 1: Assessment Report

Write a markdown report evaluating the research output. Use these sections:

### Executive Summary
A 2-3 sentence summary of the research quality and key findings.

### Key Findings
Bullet list of the most important findings from the research.

### Accuracy Assessment
Evaluate whether the facts and claims appear accurate based on the cited sources.

### Completeness
Assess whether the research adequately covers the scope of the original question. Note any significant gaps.

### Gaps and Limitations
List any important topics that were missed, areas that need deeper investigation, or limitations of the research.

### Recommendations
Suggest any follow-up research or improvements.

## Task 2: JSON Verdict

After the report, output a JSON verdict block:

```json
{
  "verdict": "PASS",
  "reason": "Brief one-line justification for the verdict"
}
```

## Verdict Rules

- PASS if the research provides a substantive, well-structured answer to the question with credible sources
- FAIL if the research is incomplete, inaccurate, unsourced, or does not address the question

## Important

- The markdown report MUST come BEFORE the JSON verdict block
- The JSON verdict MUST be the last thing in your response, inside a ```json code fence
- Do not include any text after the JSON block
- Evaluate the research objectively - focus on substance over style"""


def resolve_executor_agent(
    executor_data: dict,
    scaffold_data: dict,
    scaffold_name: str,
    user_dirs: Optional[list] = None,
) -> Optional[Any]:
    """Resolve the executor agent definition and merge defaults.

    Loads the agent YAML file, applies defaults for system_prompt, tools,
    skills, browser config, and research config when not overridden by scaffold.

    Args:
        executor_data: The executor section from scaffold YAML (mutated in place)
        scaffold_data: The full scaffold data dict (mutated for browser/research config)
        scaffold_name: Name of the scaffold for logging
        user_dirs: Optional list of user agent directories

    Returns:
        The loaded agent definition, or None if no agent specified

    Raises:
        ValueError: If specified agent is not found
    """
    scaffold_agent = executor_data.get("agent")
    if not scaffold_agent:
        return None

    agent_def = _load_agent_def(scaffold_agent, user_dirs=user_dirs)
    if not agent_def:
        raise ValueError(
            f"Agent '{scaffold_agent}' not found. "
            f"Check that the agent YAML file exists in the agents/ directory "
            f"or as a built-in agent."
        )

    _prompt_hash = hashlib.sha256((agent_def.system_prompt or "").encode()).hexdigest()[:12]
    logger.info(
        "agent.definition.loaded",
        agent_name=agent_def.name,
        tools=agent_def.tools,
        prompt_hash=_prompt_hash,
        prompt_length=len(agent_def.system_prompt or ""),
        capabilities=agent_def.capabilities,
    )

    # Use agent's system_prompt as default (scaffold can override)
    if not executor_data.get("system_prompt") and agent_def.system_prompt:
        executor_data["system_prompt"] = agent_def.system_prompt
        logger.info(
            "loader.agent_default.system_prompt",
            scaffold=scaffold_name,
            agent=scaffold_agent,
        )

    # Use agent's tools as default (scaffold can override)
    if not executor_data.get("tools") and agent_def.tools:
        executor_data["tools"] = list(agent_def.tools)
        logger.info(
            "loader.agent_default.tools",
            scaffold=scaffold_name,
            agent=scaffold_agent,
            tools=agent_def.tools,
        )

    # Merge agent skills with scaffold skills
    agent_skills_list = agent_def.skills or []
    scaffold_skills_list = executor_data.get("skills") or []
    merged_executor_skills = list(set(agent_skills_list + scaffold_skills_list))
    if merged_executor_skills:
        executor_data["skills"] = merged_executor_skills
        logger.info(
            "loader.agent_default.skills",
            scaffold=scaffold_name,
            agent=scaffold_agent,
            skills=merged_executor_skills,
        )

    # Use agent's browser config as default (scaffold can override)
    if not scaffold_data.get("browser") and agent_def.browser:
        scaffold_data["browser"] = agent_def.browser.model_dump(exclude_none=True)
        logger.info(
            "loader.agent_default.browser_config",
            scaffold=scaffold_name,
            agent=scaffold_agent,
        )

    # Use agent's research config as default (scaffold can override)
    if not scaffold_data.get("research") and agent_def.research:
        scaffold_data["research"] = agent_def.research.model_dump(exclude_none=True)
        logger.info(
            "loader.agent_default.research_config",
            scaffold=scaffold_name,
            agent=scaffold_agent,
        )

    return agent_def


def resolve_judge_agent(
    judge_data: dict,
    scaffold_name: str,
    judge_name: str,
    user_dirs: Optional[list] = None,
) -> Optional[Any]:
    """Resolve a judge's agent definition and merge defaults.

    Args:
        judge_data: Single judge entry from scaffold YAML (mutated in place)
        scaffold_name: Name of the scaffold for logging
        judge_name: Name of the judge for logging
        user_dirs: Optional list of user agent directories

    Returns:
        The loaded judge agent definition, or None if no agent specified

    Raises:
        ValueError: If specified agent is not found
    """
    judge_agent_name = judge_data.get("agent")
    if not judge_agent_name:
        return None

    judge_agent_def = _load_agent_def(judge_agent_name, user_dirs=user_dirs)
    if not judge_agent_def:
        raise ValueError(f"Judge '{judge_name}' references unknown agent: '{judge_agent_name}'")

    # Inject agent defaults (scaffold overrides)
    if not judge_data.get("system_prompt") and judge_agent_def.system_prompt:
        judge_data["system_prompt"] = judge_agent_def.system_prompt
        logger.info(
            "loader.judge_agent_default.system_prompt",
            scaffold=scaffold_name,
            judge=judge_name,
            agent=judge_agent_name,
        )
    if not judge_data.get("tools") and judge_agent_def.tools:
        judge_data["tools"] = list(judge_agent_def.tools)
        logger.info(
            "loader.judge_agent_default.tools",
            scaffold=scaffold_name,
            judge=judge_name,
            agent=judge_agent_name,
            tools=judge_agent_def.tools,
        )
    # Merge skills
    agent_skills = judge_agent_def.skills or []
    scaffold_skills = judge_data.get("skills") or []
    merged_skills = list(set(agent_skills + scaffold_skills))
    if merged_skills:
        judge_data["skills"] = merged_skills

    return judge_agent_def


def get_agent_capabilities(agent_def: Optional[Any]) -> set:
    """Extract capability flags from agent definition.

    Returns:
        Set of capability strings (e.g., {"browser", "research", "code"})
    """
    if not agent_def:
        return set()
    return set(agent_def.capabilities)


def compute_user_dirs(base_dir: str, agents_config: Optional[Any]) -> Optional[list]:
    """Compute user agent directories from config.

    Args:
        base_dir: Project root directory
        agents_config: AgentsConfig from dokumen.yaml

    Returns:
        List of Path objects for user agent directories, or None
    """
    if not agents_config:
        return None
    agents_dir = Path(base_dir) / agents_config.dir
    if agents_dir.exists():
        logger.info("loader.user_agents_dir", agents_dir=str(agents_dir))
        return [agents_dir]
    return None


def format_skills_for_prompt(
    skills: List[Tuple[str, str, str]],
) -> str:
    """Format reusable instructions for injection into a system prompt.

    Each item is a (name, content, source) tuple where source indicates
    origin: "agent:db", "scaffold", etc.

    Args:
        skills: List of (name, content, source) tuples.

    Returns:
        Formatted string to append to system prompt, or empty string.
    """
    if not skills:
        return ""

    sections = []
    for name, content, source in skills:
        sections.append(f"### {name} (source: {source})\n\n{content}")

    body = "\n\n---\n\n".join(sections)
    return f"\n\n## Available Instructions and SOPs\n\n{body}"


def collect_skills(
    scaffold_skill_names: Optional[List[str]],
    base_dir: str,
) -> List[Tuple[str, str, str]]:
    """Collect and merge reusable instructions from DB and scaffold sources.

    DB skills (from agent_loader) take priority over scaffold instructions
    with the same name.

    Args:
        scaffold_skill_names: Instruction names from scaffold YAML.
        base_dir: Workspace root for SkillLoader scanning.

    Returns:
        List of (name, content, source) tuples.

    Raises:
        ValueError: If a scaffold-referenced skill is not found in workspace.
    """
    result: List[Tuple[str, str, str]] = []
    seen_names: set = set()

    # Source A: DB skills (from agent_loader, when DOKUMEN_AGENT_ID is set)
    db_skills = get_agent_skills()
    for skill in db_skills:
        name = skill.get("name", "")
        content = skill.get("content", "")
        if name and name not in seen_names:
            result.append((name, content, "agent:db"))
            seen_names.add(name)
            logger.info(
                "loader.instruction.db",
                instruction_name=name,
                content_length=len(content),
            )

    # Source B: Scaffold instructions (resolved via SkillLoader from workspace)
    if scaffold_skill_names:
        loader = SkillLoader()
        workspace_skills = loader.load_skills(base_dir)
        workspace_map = {s.name: s for s in workspace_skills}

        for skill_name in scaffold_skill_names:
            if skill_name in seen_names:
                logger.info(
                    "loader.instruction.dedup",
                    instruction_name=skill_name,
                    kept_source="agent:db",
                )
                continue

            skill_info = workspace_map.get(skill_name)
            if skill_info is None:
                raise ValueError(
                    f"Scaffold references instruction '{skill_name}' but it was not found "
                    f"in workspace instruction directories. "
                    f"Searched: {', '.join(SkillLoader()._paths)}"
                )

            result.append((skill_name, skill_info.content, "scaffold"))
            seen_names.add(skill_name)
            logger.info(
                "loader.instruction.scaffold",
                instruction_name=skill_name,
                file_path=skill_info.file_path,
                content_length=len(skill_info.content),
            )

    # Source C: system skills from dokumen.skills.loader (inline mode only)
    # these are auto-injected unless the scaffold already specified them
    try:
        from .skills.loader import get_all_skills, ExecutionMode

        system = get_all_skills(include_system=True)
        for skill in system:
            if skill.name in seen_names:
                continue
            if skill.mode != ExecutionMode.INLINE:
                continue
            result.append((skill.name, skill.prompt, "system"))
            seen_names.add(skill.name)
            logger.info(
                "loader.instruction.system",
                instruction_name=skill.name,
                content_length=len(skill.prompt),
            )
    except ImportError:
        pass

    return result
