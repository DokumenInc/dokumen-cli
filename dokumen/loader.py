"""
Loader module for the Documentation Unit Test Framework.

Thin orchestrator that delegates to focused sub-modules:
- scaffold_parser: YAML parsing, prompt substitution, browser/viewport config
- tool_resolver: Tool name resolution, auto-injection, provenance tracking
- agent_resolver: Agent definition loading, skills, research judges
- test_builder: Provider creation, SDK agent construction

All public symbols are re-exported here for backward compatibility.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
import yaml

from dokumen_schema.constants import KNOWN_MODEL_ALIASES
from dokumen_schema.skills import SkillLoader

from .agent_loader import get_agent_skills
from .agent_object import AgentType
from .config import _preprocess_yaml
from .logging_config import get_logger
from .test_object import TestObject, BrowserConfig
from .tools_object import ToolDefinition, ToolResult
from .user_tool_overrides import load_overrides_from_dir, is_tool_enabled_for_test

# Re-export from scaffold_parser
from .scaffold_parser import (
    substitute_prompt_variables,
    find_project_root,
    normalize_raw_model,
    parse_max_iterations,
    default_executor_iterations,
    parse_browser_config,
    parse_viewport_size,
    extract_test_name,
    extract_file_paths,
)

# Re-export from tool_resolver
from .tool_resolver import (
    ToolProvenance,
    determine_executor_tool_names,
    enforce_allowed_list,
    auto_inject_tools,
    filter_tools_with_overrides,
    filter_judge_tools,
    resolve_tools,
    _create_placeholder_tool,
    _get_agent_tool_config,
)

# Re-export from agent_resolver
from .agent_resolver import (
    resolve_executor_agent,
    resolve_judge_agent,
    get_agent_capabilities,
    compute_user_dirs,
    format_skills_for_prompt,
    collect_skills as _collect_skills_impl,
    RESEARCH_SOURCES_JUDGE_PROMPT,
    RESEARCH_VERDICT_JUDGE_PROMPT,
)

# Re-export from test_builder
from .test_builder import (
    create_provider,
    find_config_file,
    get_configured_provider,
    get_configured_providers,
    build_sdk_executor,
    build_sdk_judge,
    build_research_judge,
)

logger = get_logger(__name__)


# ── Backward-compatible aliases (private names used by existing tests) ────────

_normalize_raw_model = normalize_raw_model
_parse_max_iterations = parse_max_iterations
_default_executor_iterations = default_executor_iterations
_parse_browser_config = parse_browser_config
_parse_viewport_size = parse_viewport_size
_extract_test_name = extract_test_name
_create_provider = create_provider
_find_config_file = find_config_file
_format_skills_for_prompt = format_skills_for_prompt
# Research judge prompt aliases (private names used by loader.py internals)
_RESEARCH_SOURCES_JUDGE_PROMPT = RESEARCH_SOURCES_JUDGE_PROMPT
_RESEARCH_VERDICT_JUDGE_PROMPT = RESEARCH_VERDICT_JUDGE_PROMPT


def _collect_skills(
    scaffold_skill_names: Optional[List[str]],
    base_dir: str,
) -> List[Tuple[str, str, str]]:
    """Collect and merge skills from DB and scaffold sources.

    This wrapper exists so that patching `dokumen.loader.get_agent_skills`
    in tests intercepts the call inside skill collection (backward compat).
    """
    result: List[Tuple[str, str, str]] = []
    seen_names: set = set()

    # Source A: DB skills — use module-level get_agent_skills so patches work
    db_skills = get_agent_skills()
    for skill in db_skills:
        name = skill.get("name", "")
        content = skill.get("content", "")
        if name and name not in seen_names:
            result.append((name, content, "agent:db"))
            seen_names.add(name)
            logger.info(
                "loader.skill.db",
                skill_name=name,
                content_length=len(content),
            )

    # Source B: Scaffold skills (resolved via SkillLoader from workspace)
    if scaffold_skill_names:
        loader = SkillLoader()
        workspace_skills = loader.load_skills(base_dir)
        workspace_map = {s.name: s for s in workspace_skills}

        for skill_name in scaffold_skill_names:
            if skill_name in seen_names:
                logger.info(
                    "loader.skill.dedup",
                    skill_name=skill_name,
                    kept_source="agent:db",
                )
                continue

            skill_info = workspace_map.get(skill_name)
            if skill_info is None:
                raise ValueError(
                    f"Scaffold references skill '{skill_name}' but it was not found "
                    f"in workspace skill directories. "
                    f"Searched: {', '.join(SkillLoader()._paths)}"
                )

            result.append((skill_name, skill_info.content, "scaffold"))
            seen_names.add(skill_name)
            logger.info(
                "loader.skill.scaffold",
                skill_name=skill_name,
                file_path=skill_info.file_path,
                content_length=len(skill_info.content),
            )

    return result


# Public alias for backward compatibility
collect_skills = _collect_skills


# ── Orchestrator: load_scaffold ───────────────────────────────────────────────


def load_scaffold(
    yaml_path: str,
    provider=None,
    project_root: str = None,
    sandbox_profiles: Dict[str, Any] = None,
    executor_provider=None,
    judge_provider=None,
    explore_config: Optional["ExploreConfig"] = None,
    tools_config: Optional["ToolsConfig"] = None,
    execution_config: Optional["ExecutionConfig"] = None,
    code_repos_config: Optional[List[Dict[str, Any]]] = None,
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    agents_config: Optional["AgentsConfig"] = None,
    experimental_config=None,
    compaction_config=None,
    coordinator_config=None,
    tasks_config=None,
    skills_config=None,
) -> TestObject:
    """
    Load a scaffold YAML file and convert to TestObject.

    Orchestrates scaffold_parser, tool_resolver, agent_resolver, and test_builder
    to construct a fully-resolved TestObject ready for execution.

    Args:
        yaml_path: Path to the .test.yaml file
        provider: Optional LLM provider (fallback if executor/judge not set)
        project_root: Optional project root for resolving prompt references
        sandbox_profiles: Ignored in Phase 0
        executor_provider: Optional separate provider for executor
        judge_provider: Optional separate provider for judges
        explore_config: Optional explore configuration
        tools_config: Optional project-level tool configuration
        execution_config: Optional execution configuration
        code_repos_config: Optional list of code repo configs
        provider_name: Optional provider name for per-test model providers
        api_key: Optional API key for per-test model providers
        agents_config: Optional agents configuration
        experimental_config: Reserved for future use

    Returns:
        TestObject ready for execution

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML is invalid or missing required fields
    """
    from .scaffold import validate_scaffold

    # ── Step 1: Parse scaffold YAML ───────────────────────────────────────
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty YAML file: {yaml_path}")

    validation = validate_scaffold(data)
    if not validation.valid:
        raise ValueError(f"Invalid scaffold: {', '.join(validation.errors)}")

    name = data["name"]
    base_dir = project_root or find_project_root(yaml_path)
    prompt_variables = {"working_dir": base_dir}

    # ── Step 2: Load user tool overrides ──────────────────────────────────
    overrides = load_overrides_from_dir(base_dir)
    if overrides is not None:
        logger.info(
            "loader.overrides_loaded",
            scaffold=yaml_path,
            override_count=len(overrides.overrides),
            error_count=len(overrides.errors),
        )
        if overrides.errors:
            logger.warning("loader.overrides_errors", scaffold=yaml_path, errors=overrides.errors)

    # ── Step 3: Resolve per-test model providers ──────────────────────────
    scaffold_executor_model = normalize_raw_model(data.get("executor_model"))
    scaffold_judge_model = normalize_raw_model(data.get("judge_model"))

    actual_executor_provider = _resolve_per_test_provider(
        scaffold_executor_model,
        provider_name,
        api_key,
        executor_provider or provider,
        name,
        "executor",
    )
    actual_judge_provider = _resolve_per_test_provider(
        scaffold_judge_model, provider_name, api_key, judge_provider or provider, name, "judge"
    )

    browser_config = parse_browser_config(data.get("browser"))

    # ── Step 4: Resolve agent definitions ─────────────────────────────────
    executor_data = data["executor"]
    if data.get("type") == "browser" and not data.get("agent") and not executor_data.get("agent"):
        executor_data["agent"] = "browser-tester"
    elif data.get("agent") and not executor_data.get("agent"):
        executor_data["agent"] = data["agent"]
    user_dirs = compute_user_dirs(base_dir, agents_config)

    agent_def = resolve_executor_agent(executor_data, data, name, user_dirs=user_dirs)

    # Re-parse browser config if agent injected one
    if agent_def and data.get("browser") and browser_config is None:
        browser_config = parse_browser_config(data["browser"])

    scaffold_agent = executor_data.get("agent")
    scaffold_outputs = data.get("outputs")

    # ── Step 5: Derive capability flags ───────────────────────────────────
    agent_capabilities = get_agent_capabilities(agent_def)
    is_browser_agent = (
        "browser" in agent_capabilities
        or data.get("type") == "browser"
        or bool(data.get("browser") and scaffold_agent == "browser-tester")
    )
    is_research_agent = "research" in agent_capabilities
    is_code_agent = "code" in agent_capabilities

    # ── Step 6: Resolve executor tools ────────────────────────────────────
    provenance = ToolProvenance(overrides_active=overrides is not None)

    executor_tool_names = determine_executor_tool_names(
        executor_data.get("tools", None), tools_config, name, provenance
    )

    enforce_allowed_list(executor_tool_names, tools_config, name)

    executor_tool_names, auto_injected_tools = auto_inject_tools(
        executor_tool_names,
        is_browser_agent,
        is_research_agent,
        is_code_agent,
        tools_config,
        name,
        provenance,
    )

    executor_tool_names = filter_tools_with_overrides(
        executor_tool_names,
        auto_injected_tools,
        overrides,
        tools_config,
        name,
        scaffold_agent,
        provenance,
    )

    executor_max_iterations = parse_max_iterations(
        executor_data.get("max_iterations"),
        default=default_executor_iterations(executor_tool_names),
    )
    executor_tools = resolve_tools(
        executor_tool_names,
        base_dir=base_dir,
        tools_config=tools_config,
        code_repos_config=code_repos_config,
    )

    # ── Step 7: Build executor system prompt with skills ──────────────────
    raw_system_prompt = executor_data.get("system_prompt", "")
    executor_system_prompt = substitute_prompt_variables(raw_system_prompt, prompt_variables)

    if agent_def:
        logger.info(
            "loader.agent_inline_prompt",
            scaffold=name,
            agent=scaffold_agent,
            prompt_length=len(executor_system_prompt),
        )

    executor_skill_names = executor_data.get("skills", None)
    executor_skills = collect_skills(scaffold_skill_names=executor_skill_names, base_dir=base_dir)
    if executor_skills:
        executor_system_prompt += format_skills_for_prompt(executor_skills)
        logger.info(
            "loader.executor_skills_injected",
            scaffold=name,
            skill_count=len(executor_skills),
            skill_names=[s[0] for s in executor_skills],
        )

    # ── Step 8: Build SDK executor ────────────────────────────────────────
    executor_tool_names_resolved = [t.name for t in executor_tools]
    executor = build_sdk_executor(
        data=data,
        executor_system_prompt=executor_system_prompt,
        executor_tool_names=executor_tool_names_resolved,
        actual_executor_provider=actual_executor_provider,
        executor_max_iterations=executor_max_iterations,
        tools_config=tools_config,
        user_dirs=user_dirs,
        base_dir=base_dir,
    )
    logger.info("loader.sdk_path.enabled", scaffold=name)

    # ── Step 9: Build judges ──────────────────────────────────────────────
    judges = []
    for judge_data in data["judges"]:
        judge = _build_single_judge(
            judge_data=judge_data,
            data=data,
            name=name,
            base_dir=base_dir,
            prompt_variables=prompt_variables,
            user_dirs=user_dirs,
            is_browser_agent=is_browser_agent,
            is_research_agent=is_research_agent,
            scaffold_agent=scaffold_agent,
            overrides=overrides,
            tools_config=tools_config,
            actual_judge_provider=actual_judge_provider,
            provider_name=provider_name,
            api_key=api_key,
            provenance=provenance,
        )
        judges.append(judge)

    # ── Step 10: Auto-inject research judges ──────────────────────────────
    if is_research_agent:
        _inject_research_judges(judges, actual_judge_provider, tools_config, provenance, name)

    # ── Step 11: Extract file paths ───────────────────────────────────────
    file_paths = extract_file_paths(data.get("files", []))

    # ── Step 12: Build TestObject ─────────────────────────────────────────
    logger.info(
        "tool_provenance.built",
        scaffold=name,
        executor_tools=len(provenance.executor_tools),
        judge_count=len(provenance.judge_tools),
        overrides_active=provenance.overrides_active,
        removed_count=len(provenance.removed_tools),
    )

    setup_steps = None
    raw_setup = data.get("setup")
    if raw_setup:
        from dokumen_schema.models import SetupStep

        setup_steps = [SetupStep(**s) for s in raw_setup]
        logger.info("loader.setup_steps.parsed", scaffold=name, step_count=len(setup_steps))

    # Collect all resolved skills for artifact writing
    all_skills = _collect_all_skills(executor_skills, data, base_dir)

    # per-scaffold overrides for coordinator/compaction (optional in yaml)
    scaffold_coordinator = data.get("coordinator")
    if scaffold_coordinator and isinstance(scaffold_coordinator, dict):
        from .config import CoordinatorConfig

        try:
            coordinator_config = CoordinatorConfig(
                **{
                    **(coordinator_config.model_dump() if coordinator_config else {}),
                    **scaffold_coordinator,
                }
            )
        except Exception as e:
            logger.warning("loader.coordinator_config.error", scaffold=name, error=str(e))

    scaffold_compaction = data.get("compaction")
    if scaffold_compaction and isinstance(scaffold_compaction, dict):
        from .config import CompactionConfig

        try:
            compaction_config = CompactionConfig(
                **{
                    **(compaction_config.model_dump() if compaction_config else {}),
                    **scaffold_compaction,
                }
            )
        except Exception as e:
            logger.warning("loader.compaction_config.error", scaffold=name, error=str(e))

    return TestObject(
        id=data["name"],
        reason=data.get("reason", ""),
        executor=executor,
        judges=judges,
        timeout=float(data.get("timeout", 60.0)),
        retries=int(data.get("retries", 0)),
        source_path=yaml_path,
        files=file_paths,
        explore_config=explore_config,
        browser_config=browser_config,
        test_type=data.get("type"),
        tool_overrides=overrides,
        tool_provenance=provenance,
        setup_steps=setup_steps,
        agent=scaffold_agent,
        outputs=scaffold_outputs,
        user_dirs=user_dirs,
        resolved_skills=all_skills if all_skills else None,
        compaction_config=compaction_config,
        coordinator_config=coordinator_config,
        tasks_config=tasks_config,
    )


# Alias for compatibility
load_test_from_yaml = load_scaffold


# ── Orchestrator: load_all_scaffolds ──────────────────────────────────────────


def load_all_scaffolds(
    tests_dir: str = "tests",
    provider=None,
    sandbox_profiles: Dict[str, Any] = None,
    config_path: str = None,
    executor_provider=None,
    judge_provider=None,
    tools_config: Optional["ToolsConfig"] = None,
) -> Tuple[List[TestObject], Dict[str, str]]:
    """
    Load all scaffold files in a directory.

    Args:
        tests_dir: Directory containing .test.yaml files
        provider: Optional provider for all tests
        sandbox_profiles: Ignored in Phase 0
        config_path: Optional config file path
        executor_provider: Optional separate provider for executors
        judge_provider: Optional separate provider for judges
        tools_config: Optional project-level tool configuration

    Returns:
        Tuple of (tests, load_errors)
    """
    from .scaffold import discover_scaffolds

    explore_config = None
    execution_config = None
    compaction_config = None
    coordinator_config = None
    tasks_config = None
    skills_config = None
    config = None
    try:
        from .config import load_config

        config = load_config(config_path)
        explore_config = config.explore
        execution_config = config.execution
        if tools_config is None:
            tools_config = config.tools
        compaction_config = getattr(config, "compaction", None)
        coordinator_config = getattr(config, "coordinator", None)
        tasks_config = getattr(config, "tasks", None)
        skills_config = getattr(config, "skills", None)
    except Exception:
        explore_config = None
        execution_config = None

    # Clone linked code repos
    code_repos_config = None
    if config is not None and getattr(config, "code_repos", None):
        try:
            from .secrets import get_gitlab_token

            token = get_gitlab_token()
            gitlab_url_fallback = os.environ.get("GITLAB_URL") or os.environ.get("CI_SERVER_URL")
            cloned = clone_code_repos(
                config.code_repos,
                token=token,
                gitlab_url_fallback=gitlab_url_fallback,
            )
            if cloned:
                code_repos_config = cloned
                logger.info(
                    "code_repos.setup_complete",
                    count=len(code_repos_config),
                    names=[r["name"] for r in code_repos_config],
                )
        except Exception as e:
            logger.warning("code_repos.setup_failed", error=str(e))

    # Extract provider info for per-test model overrides
    provider_name_for_overrides = None
    api_key_for_overrides = None
    if config and hasattr(config, "provider") and config.provider:
        provider_name_for_overrides = config.provider.name
    if provider_name_for_overrides:
        api_key_for_overrides = os.environ.get(f"{provider_name_for_overrides.upper()}_API_KEY")
    logger.debug(
        "Per-test model override provider info",
        extra={
            "provider_name": provider_name_for_overrides,
            "api_key_available": api_key_for_overrides is not None,
        },
    )

    tests = []
    load_errors = {}
    for scaffold_path in discover_scaffolds(tests_dir):
        try:
            test = load_scaffold(
                scaffold_path,
                provider,
                executor_provider=executor_provider,
                judge_provider=judge_provider,
                explore_config=explore_config,
                tools_config=tools_config,
                execution_config=execution_config,
                code_repos_config=code_repos_config,
                provider_name=provider_name_for_overrides,
                api_key=api_key_for_overrides,
                agents_config=config.agents if config else None,
                compaction_config=compaction_config,
                coordinator_config=coordinator_config,
                tasks_config=tasks_config,
                skills_config=skills_config,
            )
            tests.append(test)
        except (ValueError, IOError, yaml.YAMLError) as e:
            test_name = extract_test_name(scaffold_path)
            load_errors[test_name] = str(e)
            logger.warning(
                "scaffold.load_failed",
                extra={
                    "scaffold_path": scaffold_path,
                    "test_name": test_name,
                    "error": str(e),
                },
            )

    return tests, load_errors


# ── clone_code_repos (kept in loader for backward compat) ─────────────────────


def clone_code_repos(
    code_repos: list,
    token: Optional[str] = None,
    cache_dir: str = ".dokumen-cache",
    gitlab_url_fallback: Optional[str] = None,
) -> list:
    """Clone or update linked code repos and return code_repos_config dicts.

    For each CodeRepoConfig, calls the GitLab API to get the HTTP clone URL,
    then clones (or fetches) into {cache_dir}/repos/{name}/.

    Args:
        code_repos: List of CodeRepoConfig objects from dokumen.yaml
        token: GitLab access token for authentication
        cache_dir: Base cache directory
        gitlab_url_fallback: Default GitLab URL when a repo has no gitlab_url set

    Returns:
        List of code_repos_config dicts
    """
    import json
    import subprocess
    from urllib.request import Request, urlopen
    from urllib.parse import urlparse, urlunparse

    result = []

    for repo in code_repos:
        repo_name = repo.name
        gitlab_url = (getattr(repo, "gitlab_url", None) or gitlab_url_fallback or "").rstrip("/")

        if not gitlab_url:
            logger.warning("code_repos.clone.skip_no_url", repo_name=repo_name)
            continue

        api_url = f"{gitlab_url}/api/v4/projects/{repo.gitlab_project_id}"
        req = Request(api_url)
        if token:
            req.add_header("PRIVATE-TOKEN", token)

        try:
            with urlopen(req, timeout=30) as resp:
                project_info = json.loads(resp.read())
        except Exception as e:
            logger.warning(
                "code_repos.clone.api_failed",
                repo_name=repo_name,
                api_url=api_url,
                error=str(e),
            )
            continue

        http_url = project_info.get("http_url_to_repo", "")
        if not http_url:
            logger.warning(
                "code_repos.clone.no_clone_url",
                repo_name=repo_name,
                project_id=repo.gitlab_project_id,
            )
            continue

        if token:
            parsed = urlparse(http_url)
            port_suffix = f":{parsed.port}" if parsed.port else ""
            authenticated_url = urlunparse(
                parsed._replace(netloc=f"oauth2:{token}@{parsed.hostname}{port_suffix}")
            )
        else:
            authenticated_url = http_url

        branch = getattr(repo, "branch", "main")
        dest = Path(cache_dir) / "repos" / repo_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            logger.info("code_repos.clone.updating", repo_name=repo_name, dest=str(dest))
            subprocess.run(
                ["git", "-C", str(dest), "fetch", "origin", "--depth=1"],
                capture_output=True,
                timeout=120,
            )
            subprocess.run(
                ["git", "-C", str(dest), "checkout", branch],
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["git", "-C", str(dest), "reset", "--hard", f"origin/{branch}"],
                capture_output=True,
                timeout=30,
            )
        else:
            logger.info(
                "code_repos.clone.cloning",
                repo_name=repo_name,
                branch=branch,
                dest=str(dest),
            )
            proc = subprocess.run(
                ["git", "clone", "--depth=1", "-b", branch, authenticated_url, str(dest)],
                capture_output=True,
                timeout=300,
            )
            if proc.returncode != 0:
                logger.warning(
                    "code_repos.clone.clone_failed",
                    repo_name=repo_name,
                    returncode=proc.returncode,
                    stderr=(proc.stderr.decode(errors="replace")[:500] if proc.stderr else ""),
                )
                continue

        if not dest.exists():
            logger.warning("code_repos.clone.dest_missing", repo_name=repo_name, dest=str(dest))
            continue

        result.append(
            {
                "name": repo_name,
                "base_dir": str(dest),
                "include_patterns": list(getattr(repo, "paths_include", [])),
                "exclude_patterns": list(getattr(repo, "paths_exclude", [])),
            }
        )
        logger.info("code_repos.clone.ready", repo_name=repo_name, base_dir=str(dest))

    return result


# ── Private helpers ───────────────────────────────────────────────────────────


def _resolve_per_test_provider(
    scaffold_model: Optional[str],
    provider_name: Optional[str],
    api_key: Optional[str],
    fallback_provider,
    scaffold_name: str,
    role: str,
):
    """Resolve per-test model override to a provider instance.

    Args:
        scaffold_model: Normalized model from scaffold YAML (or None)
        provider_name: Provider name for creating new provider
        api_key: API key for creating new provider
        fallback_provider: Fallback provider if no override
        scaffold_name: Scaffold name for logging
        role: 'executor' or 'judge'

    Returns:
        Provider instance
    """
    if scaffold_model and provider_name and api_key:
        logger.info(
            f"Using per-test {role} model",
            extra={"scaffold": scaffold_name, "model": scaffold_model, "source": "scaffold"},
        )
        return _create_provider(provider_name, api_key, scaffold_model)
    else:
        if scaffold_model and (not provider_name or not api_key):
            logger.debug(
                f"Per-test {role} model ignored: missing provider_name or api_key",
                extra={"scaffold": scaffold_name, "model": scaffold_model},
            )
        else:
            logger.debug(
                f"Using project-level {role} model",
                extra={"scaffold": scaffold_name, "reason": "no scaffold override"},
            )
        return fallback_provider


def _build_single_judge(
    judge_data: dict,
    data: dict,
    name: str,
    base_dir: str,
    prompt_variables: Dict[str, str],
    user_dirs: Optional[list],
    is_browser_agent: bool,
    is_research_agent: bool,
    scaffold_agent: Optional[str],
    overrides,
    tools_config,
    actual_judge_provider,
    provider_name: Optional[str],
    api_key: Optional[str],
    provenance: ToolProvenance,
):
    """Build a single judge from scaffold data.

    Handles agent resolution, tool filtering, skill injection, and
    per-judge model overrides.
    """
    judge_name = judge_data.get("name", "unknown")

    # Judge agent resolution
    resolve_judge_agent(judge_data, name, judge_name, user_dirs=user_dirs)

    # Judge tool names + provenance
    judge_tool_names = judge_data.get("tools", [])
    judge_prov: Dict[str, str] = {}
    for t in judge_tool_names:
        judge_prov[t] = "scaffold"

    auto_added_judge_tools: set = set()
    if (
        not is_browser_agent
        and not is_research_agent
        and "run_shell_command" not in judge_tool_names
    ):
        judge_tool_names = ["run_shell_command"] + list(judge_tool_names)
        auto_added_judge_tools.add("run_shell_command")
        judge_prov["run_shell_command"] = "auto:standard"

    # Apply tool filtering
    judge_tool_names = filter_judge_tools(
        judge_tool_names,
        auto_added_judge_tools,
        overrides,
        tools_config,
        name,
        judge_name,
        scaffold_agent,
        judge_prov,
    )

    judge_tools = resolve_tools(
        judge_tool_names,
        base_dir=base_dir,
        tools_config=tools_config,
    )

    judge_system_prompt = substitute_prompt_variables(
        judge_data.get("system_prompt", ""), prompt_variables
    )

    # Inject skills
    judge_skill_names = judge_data.get("skills", None)
    judge_skills = collect_skills(scaffold_skill_names=judge_skill_names, base_dir=base_dir)
    if judge_skills:
        judge_system_prompt += format_skills_for_prompt(judge_skills)
        logger.info(
            "loader.judge_skills_injected",
            scaffold=name,
            judge=judge_name,
            skill_count=len(judge_skills),
            skill_names=[s[0] for s in judge_skills],
        )

    # Per-judge model override
    raw_judge_model = normalize_raw_model(judge_data.get("model"))
    if raw_judge_model and provider_name and api_key:
        logger.info(
            "Using per-judge model",
            extra={"judge": judge_name, "model": raw_judge_model},
        )
        judge_provider_for_this = _create_provider(provider_name, api_key, raw_judge_model)
    else:
        if raw_judge_model and (not provider_name or not api_key):
            logger.warning(
                "Per-judge model ignored: missing provider_name or api_key",
                extra={"judge": judge_name, "model": raw_judge_model},
            )
        judge_provider_for_this = actual_judge_provider

    judge_max_iterations = parse_max_iterations(judge_data.get("max_iterations"))
    judge_timeout_override = judge_data.get("timeout")

    judge = build_sdk_judge(
        judge_data=judge_data,
        judge_system_prompt=judge_system_prompt,
        judge_tools=judge_tools,
        judge_provider=judge_provider_for_this,
        judge_max_iterations=judge_max_iterations,
        judge_timeout_override=judge_timeout_override,
        tools_config=tools_config,
    )

    provenance.judge_tools[judge_name] = judge_prov
    return judge


def _inject_research_judges(
    judges: list,
    actual_judge_provider,
    tools_config,
    provenance: ToolProvenance,
    scaffold_name: str,
):
    """Auto-inject sources + verdict judges for research agents."""
    existing_judge_names = {j.id for j in judges}

    if "sources" not in existing_judge_names:
        sources_judge = build_research_judge(
            judge_id="sources",
            prompt=RESEARCH_SOURCES_JUDGE_PROMPT,
            judge_provider=actual_judge_provider,
            tools_config=tools_config,
        )
        judges.append(sources_judge)
        provenance.judge_tools["sources"] = {}
        logger.info("sources judge auto-injected for research agent", scaffold=scaffold_name)

    if "verdict" not in existing_judge_names:
        verdict_judge = build_research_judge(
            judge_id="verdict",
            prompt=RESEARCH_VERDICT_JUDGE_PROMPT,
            judge_provider=actual_judge_provider,
            tools_config=tools_config,
        )
        judges.append(verdict_judge)
        provenance.judge_tools["verdict"] = {}
        logger.info("verdict judge auto-injected for research agent", scaffold=scaffold_name)


def _collect_all_skills(
    executor_skills: List[Tuple[str, str, str]],
    data: dict,
    base_dir: str,
) -> Dict[str, Dict[str, str]]:
    """Collect all resolved skills for artifact writing."""
    all_skills: Dict[str, Dict[str, str]] = {}
    if executor_skills:
        for skill_name, skill_content, skill_source in executor_skills:
            all_skills[skill_name] = {
                "content": skill_content,
                "source": skill_source,
                "used_by": "executor",
            }
    for judge_data_item in data["judges"]:
        judge_skill_names_item = judge_data_item.get("skills", None)
        if judge_skill_names_item:
            try:
                judge_skills_item = collect_skills(
                    scaffold_skill_names=judge_skill_names_item,
                    base_dir=base_dir,
                )
                for skill_name, skill_content, skill_source in judge_skills_item:
                    if skill_name not in all_skills:
                        all_skills[skill_name] = {
                            "content": skill_content,
                            "source": skill_source,
                            "used_by": f"judge:{judge_data_item.get('name', 'unknown')}",
                        }
            except (ValueError, Exception):
                pass  # Skills already validated during judge construction

    return all_skills
