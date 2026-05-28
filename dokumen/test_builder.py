"""
Test builder for the Skill Testing Framework.

Handles creating LLM providers, constructing SDK executor/judge agents,
building TestObject instances, and provider configuration from environment/config.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import os
import yaml

from .config import _preprocess_yaml
from .logging_config import get_logger

logger = get_logger(__name__)


def create_provider(name: str, api_key: str = None, model: str = None, **kwargs):
    """
    Create a provider instance by name.

    Args:
        name: Provider name (anthropic, openai, google, gemini, custom, etc.)
        api_key: API key
        model: Model name
        **kwargs: Additional provider-specific options (api_base, etc.)

    Returns:
        Provider instance or None
    """
    if not name:
        return None

    name = name.lower()

    # native anthropic provider (battle-tested, keeps working as before)
    if name == "anthropic" and not kwargs.get("force_dokurouter"):
        from .providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=api_key, model=model)

    # Non-Anthropic direct-provider adapters go through DokuRouter.
    from .providers.dokurouter import DokuRouter

    return DokuRouter(
        model=model,
        provider_name=name,
        api_key=api_key,
        api_base=kwargs.get("api_base") or kwargs.get("base_url"),
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ("api_base", "base_url", "force_dokurouter", "enable_thinking")
        },
    )


def find_config_file(config_path: str = None) -> Optional[str]:
    """Find dokumen.yaml, searching parent directories if needed."""
    if config_path and os.path.exists(config_path):
        return config_path

    # Search current and parent directories for dokumen.yaml
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        candidate = parent / "dokumen.yaml"
        if candidate.exists():
            return str(candidate)
        if parent == parent.parent:
            break

    return None


def get_configured_provider(
    config_path: str = None, env_prefix: str = "DOKUMEN", model_override: str = None
):
    """
    Get the configured LLM provider.

    Checks in order:
    1. Environment variables (DOKUMEN_PROVIDER, DOKUMEN_API_KEY, etc.)
    2. Config file (dokumen.yaml) - searches parent directories
    3. Default (None)

    Returns:
        Provider instance or None if not configured
    """
    provider_name = os.environ.get(f"{env_prefix}_PROVIDER")
    api_key = os.environ.get(f"{env_prefix}_API_KEY")
    model = model_override or os.environ.get(f"{env_prefix}_MODEL")

    if provider_name and api_key:
        return create_provider(provider_name, api_key, model)

    config_path = find_config_file(config_path)
    if config_path:
        try:
            with open(config_path) as f:
                config = yaml.safe_load(_preprocess_yaml(f.read()))
                if config and "provider" in config:
                    provider_config = config["provider"]
                    prov_name = provider_config.get("name")

                    return create_provider(
                        prov_name,
                        provider_config.get("api_key")
                        or os.environ.get(f"{prov_name.upper()}_API_KEY"),
                        model_override or provider_config.get("model"),
                        base_url=provider_config.get("base_url"),
                        enable_thinking=provider_config.get("enable_thinking"),
                    )
        except (IOError, yaml.YAMLError):
            pass

    return None


def get_configured_providers(
    config_path: str = None, env_prefix: str = "DOKUMEN"
) -> Dict[str, Any]:
    """
    Get configured LLM providers for executor and judge.

    Supports separate models for executor and judge via config:
    - executor_model: Model to use for executors
    - judge_model: Model to use for judges

    Returns:
        Dict with 'executor', 'judge', and 'default' provider instances
    """
    config_path = find_config_file(config_path)

    provider_name = os.environ.get(f"{env_prefix}_PROVIDER")
    api_key = os.environ.get(f"{env_prefix}_API_KEY")
    default_model = os.environ.get(f"{env_prefix}_MODEL")

    config = None
    if config_path:
        try:
            with open(config_path) as f:
                config = yaml.safe_load(_preprocess_yaml(f.read()))
        except (IOError, yaml.YAMLError):
            pass

    if config and "provider" in config:
        provider_config = config["provider"]
        provider_name = provider_name or provider_config.get("name")
        api_key = (
            api_key
            or provider_config.get("api_key")
            or os.environ.get(f"{provider_config.get('name', '').upper()}_API_KEY")
        )
        default_model = default_model or provider_config.get("model")

    if not provider_name or not api_key:
        return {"executor": None, "judge": None, "default": None}

    executor_model = default_model
    judge_model = default_model

    if config:
        executor_model = config.get("executor_model") or default_model
        judge_model = config.get("judge_model") or default_model

    default_provider = create_provider(provider_name, api_key, default_model)
    executor_provider = create_provider(provider_name, api_key, executor_model)
    judge_provider = create_provider(provider_name, api_key, judge_model)

    return {"executor": executor_provider, "judge": judge_provider, "default": default_provider}


def build_sdk_executor(
    data: dict,
    executor_system_prompt: str,
    executor_tool_names: List[str],
    actual_executor_provider: Any,
    executor_max_iterations: int,
    tools_config: Optional[Any] = None,
    user_dirs: Optional[List[Any]] = None,
    base_dir: Optional[str] = None,
    executor_tools: Optional[List[Any]] = None,
):
    """Build SDK ExecutorAgent and SdkExecutorWrapper.

    Args:
        data: Full scaffold data dict
        executor_system_prompt: Resolved system prompt with skills
        executor_tool_names: Resolved tool name list
        actual_executor_provider: Provider for the executor
        executor_max_iterations: Max iterations
        tools_config: Optional project-level tool config
        user_dirs: Optional agent definition directories (for delegate_to_agent)
        base_dir: Optional project base directory (for delegate_to_agent)
        executor_tools: Pre-resolved ToolDefinition objects from the loader.

    Returns:
        SdkExecutorWrapper instance
    """
    from .sdk.tools import AgentContext, resolve_sdk_tools
    from .sdk.executor import ExecutorAgent
    from .sdk.agent_wrapper import SdkExecutorWrapper

    # Build AgentContext for delegate_to_agent support
    agent_context = None
    if "delegate_to_agent" in executor_tool_names:
        agent_context = AgentContext(
            user_dirs=user_dirs,
            tools_config=tools_config,
            base_dir=base_dir or ".",
            timeout=float(data.get("timeout", 120.0)),
            is_subagent=False,
        )

    browser_data = data.get("browser") or {}
    resolved = resolve_sdk_tools(
        executor_tool_names,
        tools_config,
        test_name=data.get("name"),
        browser_config=browser_data if browser_data else None,
        agent_context=agent_context,
        dokumen_tool_definitions=executor_tools,
    )

    mcp_servers = {}

    sdk_tool_names = list(resolved.sdk_tool_names)
    mcp_tools = list(resolved.dokumen_mcp_tools) if resolved.dokumen_mcp_tools else []

    # Build SDK agent definitions for native subagent support
    sdk_agents = None
    if agent_context is not None:
        from .sdk.delegate import build_sdk_agent_definitions

        sdk_agents = build_sdk_agent_definitions(agent_context)
        if not sdk_agents:
            sdk_agents = None
            logger.info(
                "No agents found for delegation",
                extra={"test": data.get("name")},
            )

    # Playwright browser tools: use native McpStdioServerConfig
    playwright_tool_names = []
    if resolved.playwright_mcp_config:
        mcp_servers["playwright"] = resolved.playwright_mcp_config
        playwright_tool_names = list(resolved.playwright_tool_names)

    sdk_executor_model = None
    if actual_executor_provider and hasattr(actual_executor_provider, "model"):
        sdk_executor_model = actual_executor_provider.model

    sdk_executor = ExecutorAgent(
        id=f"{data['name']}-executor",
        system_prompt=executor_system_prompt,
        user_prompt=data["executor"].get("user_prompt", ""),
        sdk_tools=sdk_tool_names,
        mcp_tools=mcp_tools if mcp_tools else None,
        mcp_servers=mcp_servers if mcp_servers else None,
        playwright_tool_names=playwright_tool_names if playwright_tool_names else None,
        max_turns=executor_max_iterations,
        timeout=float(data.get("timeout", 60.0)),
        tools_config=tools_config,
        model=sdk_executor_model,
        agents=sdk_agents,
    )
    executor = SdkExecutorWrapper(
        sdk_executor,
        system_prompt=executor_system_prompt,
        user_prompt=data["executor"].get("user_prompt", ""),
    )
    return executor


def build_sdk_judge(
    judge_data: dict,
    judge_system_prompt: str,
    judge_tools: list,
    judge_provider: Any,
    judge_max_iterations: Optional[int],
    judge_timeout_override: Optional[float],
    tools_config: Optional[Any] = None,
):
    """Build SDK JudgeAgent and SdkJudgeWrapper.

    Args:
        judge_data: Single judge entry from scaffold YAML
        judge_system_prompt: Resolved system prompt
        judge_tools: Resolved ToolDefinition list
        judge_provider: Provider for this judge
        judge_max_iterations: Optional max iterations override
        judge_timeout_override: Optional timeout override
        tools_config: Optional project-level tool config

    Returns:
        SdkJudgeWrapper instance
    """
    from .sdk.tools import resolve_sdk_tools
    from .sdk.judge import JudgeAgent
    from .sdk.agent_wrapper import SdkJudgeWrapper

    sdk_judge_model = None
    if judge_provider and hasattr(judge_provider, "model"):
        sdk_judge_model = judge_provider.model

    judge_tool_names_for_sdk = [t.name for t in judge_tools]
    judge_resolved = resolve_sdk_tools(
        judge_tool_names_for_sdk,
        tools_config,
        dokumen_tool_definitions=judge_tools,
    )

    sdk_judge = JudgeAgent(
        id=judge_data["name"],
        system_prompt=judge_system_prompt,
        user_prompt=judge_data.get("user_prompt", ""),
        include_executor_output=judge_data.get("include_executor_output", True),
        sdk_tools=judge_resolved.sdk_tool_names,
        mcp_tools=judge_resolved.dokumen_mcp_tools if judge_resolved.dokumen_mcp_tools else None,
        max_turns=judge_max_iterations or 3,
        timeout=float(judge_timeout_override) if judge_timeout_override else 120.0,
        tools_config=tools_config,
        model=sdk_judge_model,
        decomposed=judge_data.get("decomposed", False),
        decomposed_threshold=float(judge_data.get("decomposed_threshold", 0.5)),
    )
    judge = SdkJudgeWrapper(
        sdk_judge,
        assertion_text=judge_data.get("name", ""),
        system_prompt=judge_system_prompt,
    )
    return judge


def build_research_judge(
    judge_id: str,
    prompt: str,
    judge_provider: Any,
    tools_config: Optional[Any] = None,
):
    """Build a research auto-injected judge (sources or verdict).

    Args:
        judge_id: Judge identifier ('sources' or 'verdict')
        prompt: The judge system prompt
        judge_provider: Provider for the judge
        tools_config: Optional project-level tool config

    Returns:
        SdkJudgeWrapper instance
    """
    from .sdk.judge import JudgeAgent
    from .sdk.agent_wrapper import SdkJudgeWrapper

    sdk_model = None
    if judge_provider and hasattr(judge_provider, "model"):
        sdk_model = judge_provider.model

    sdk_judge = JudgeAgent(
        id=judge_id,
        system_prompt=prompt,
        user_prompt=prompt,
        include_executor_output=True,
        sdk_tools=[],
        max_turns=3,
        timeout=120.0,
        tools_config=tools_config,
        model=sdk_model,
    )
    return SdkJudgeWrapper(
        sdk_judge,
        assertion_text=judge_id,
        system_prompt=prompt,
    )
