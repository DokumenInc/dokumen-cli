"""
Configuration module for dokumen-cli.

Provides Pydantic models for parsing and validating dokumen.yaml configuration files.
"""

import logging
import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)

# Model defaults used by the standalone CLI.
DEFAULT_FAST_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_BALANCED_MODEL = "claude-sonnet-4-6"


def _preprocess_yaml(content: str) -> str:
    """Strip lines starting with ':' (treated as comments in dokumen config).

    Lines where the first non-whitespace character is ':' are removed before
    YAML parsing. This allows dokumen.yaml files to use ':' as a label/comment
    prefix (e.g., ': DOKUMEN').
    """
    lines = content.split("\n")
    filtered = [line for line in lines if not line.lstrip().startswith(":")]
    stripped_count = len(lines) - len(filtered)
    if stripped_count > 0:
        logger.info(
            "Stripped colon-prefixed comment lines from config",
            extra={"lines_stripped": stripped_count},
        )
    return "\n".join(filtered)


class ConfigError(Exception):
    """Configuration-related errors."""

    pass


class ProviderConfig(BaseModel):
    """Provider configuration for AI model."""

    name: Literal[
        "anthropic",
        "openai",
        "google",
        "gemini",
        "mistral",
        "deepseek",
        "groq",
        "together",
        "bedrock",
        "vertex",
        "custom",
        "mock",
    ] = Field(..., description="Provider name (e.g., 'anthropic', 'openai', 'google', 'custom')")
    api_base: Optional[str] = Field(
        None, description="Custom API base URL (for self-hosted or proxy endpoints)"
    )
    model: str = Field(DEFAULT_FAST_MODEL, description="Model name to use")


class ExecutionConfig(BaseModel):
    """Execution configuration for test runs."""

    timeout: int = Field(3600, ge=1, description="Timeout per test in seconds")
    max_tool_result_chars: int = Field(
        50000, description="Max chars per tool result before truncation. 0 disables."
    )
    judge_retries: int = Field(2, ge=0, le=5, description="Max retries for judge structural errors")

    @model_validator(mode="after")
    def validate_max_tool_result_chars(self) -> "ExecutionConfig":
        v = self.max_tool_result_chars
        if v != 0 and (v < 1000 or v > 500000):
            raise ValueError(f"max_tool_result_chars must be 0 (disabled) or 1000-500000, got {v}")
        return self


class CoverageConfig(BaseModel):
    """Coverage configuration for documentation tracking."""

    include: list[str] = Field(
        default_factory=lambda: ["docs/**/*", "README.md"],
        description="Glob patterns for files to include",
    )
    exclude: list[str] = Field(
        default_factory=list, description="Glob patterns for files to exclude"
    )
    min_threshold: Optional[int] = Field(
        None, ge=0, le=100, description="Minimum coverage percentage (0-100)"
    )

    @model_validator(mode="after")
    def validate_coverage_patterns(self) -> "CoverageConfig":
        """Ensure coverage patterns don't target sensitive files."""
        FORBIDDEN_PATTERNS = [".env", "secrets", ".pem", ".key", "credential"]
        for label, patterns in [("include", self.include), ("exclude", self.exclude)]:
            for pattern in patterns:
                if ".." in pattern:
                    raise ValueError(
                        f"Coverage {label} pattern '{pattern}' contains path traversal"
                    )
                lower = pattern.lower()
                if any(fp in lower for fp in FORBIDDEN_PATTERNS):
                    raise ValueError(
                        f"Coverage {label} pattern '{pattern}' targets sensitive files"
                    )
        return self


class ExploreDebugConfig(BaseModel):
    """Settings for explore phase debug output on failure."""

    max_tool_calls: int = Field(10, ge=1, le=100, description="Max tool calls to show on failure")
    max_command_chars: int = Field(200, ge=50, le=1000, description="Max chars for command display")
    max_output_chars: int = Field(500, ge=100, le=5000, description="Max chars for tool output")
    max_output_lines: int = Field(10, ge=1, le=50, description="Max lines for tool output")


DEFAULT_EXPLORE_TOOL_NAMES = [
    "read_file",
    "list_directory",
    "glob",
    "search_file_content",
]


class ExploreConfig(BaseModel):
    """Explore configuration for AI exploration phase.

    The explore phase runs before the main executor to discover
    relevant documentation files.
    """

    enabled: bool = Field(False, description="Enable/disable the explore phase")
    model: str = Field(
        DEFAULT_BALANCED_MODEL,
        description="Model to use for explore agents (can use cheaper/faster model)",
    )
    max_files: int = Field(100, ge=1, le=500, description="Maximum number of files to discover")
    max_iterations: int = Field(
        50, ge=1, le=100, description="Maximum tool call iterations before stopping"
    )
    timeout: int = Field(60, ge=1, le=300, description="Timeout for explore phase in seconds")
    allowed_tools: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EXPLORE_TOOL_NAMES),
        description="Allowed tool names for explore agents (browser tools are not permitted)",
    )
    debug: ExploreDebugConfig = Field(
        default_factory=ExploreDebugConfig, description="Debug output settings for explore failures"
    )


class PerplexityConfig(BaseModel):
    """Configuration for Perplexity web search."""

    api_key: Optional[str] = Field(
        None, description="Perplexity API key (or use PERPLEXITY_API_KEY env var)"
    )
    model: str = Field("sonar", description="Perplexity model to use")
    max_searches_per_test: int = Field(
        5, ge=1, le=20, description="Maximum web searches per test execution"
    )


class ShellToolConfig(BaseModel):
    """Per-tool configuration for run_shell_command."""

    timeout: float = Field(30.0, ge=1.0, le=300.0, description="Command timeout in seconds")


class HttpToolConfig(BaseModel):
    """Per-tool configuration for web_fetch."""

    timeout: float = Field(30.0, ge=1.0, le=120.0, description="Request timeout in seconds")


class WebSearchToolConfig(BaseModel):
    """Per-tool configuration for web_search."""

    model: Optional[str] = Field(
        None, description="Perplexity model (falls back to perplexity.model)"
    )
    max_searches: Optional[int] = Field(
        None,
        ge=1,
        le=200,
        description="Max searches per test (falls back to perplexity.max_searches_per_test)",
    )


class ToolConfigMap(BaseModel):
    """Per-tool configuration overrides."""

    run_shell_command: ShellToolConfig = Field(default_factory=ShellToolConfig)
    web_fetch: HttpToolConfig = Field(default_factory=HttpToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)


def _validate_tool_names(names: list[str], field_name: str) -> list[str]:
    """Validate that all tool names are in VALID_EXECUTOR_TOOLS."""
    from dokumen_schema import VALID_EXECUTOR_TOOLS

    invalid = [n for n in names if n not in VALID_EXECUTOR_TOOLS]
    if invalid:
        raise ValueError(
            f"{field_name} contains unknown tool names: {', '.join(sorted(invalid))}. "
            f"Valid tools: {', '.join(sorted(list(VALID_EXECUTOR_TOOLS)[:8]))}..."
        )
    return names


class ToolsConfig(BaseModel):
    """Project-level tool configuration for dokumen.yaml."""

    defaults: Optional[list[str]] = Field(
        None, description="Default tools when scaffold omits executor.tools"
    )
    allowed: Optional[list[str]] = Field(
        None, description="Project-level allowlist (None = all valid tools allowed)"
    )
    blocked: Optional[list[str]] = Field(None, description="Tools explicitly disabled via UI")
    config: ToolConfigMap = Field(default_factory=ToolConfigMap)

    @model_validator(mode="after")
    def validate_tool_names_and_subset(self) -> "ToolsConfig":
        """Validate tool names and ensure defaults is subset of allowed."""
        if self.defaults is not None:
            _validate_tool_names(self.defaults, "defaults")
        if self.allowed is not None:
            _validate_tool_names(self.allowed, "allowed")
        if self.blocked is not None:
            _validate_tool_names(self.blocked, "blocked")
        if self.defaults is not None and self.allowed is not None:
            not_in_allowed = set(self.defaults) - set(self.allowed)
            if not_in_allowed:
                raise ValueError(
                    f"defaults contains tools not in allowed list: "
                    f"{', '.join(sorted(not_in_allowed))}"
                )
        return self


class MemoryConfig(BaseModel):
    """Configuration for agent memory system."""

    enabled: bool = Field(False, description="Enable persistent agent memory (default: off)")
    store: Literal["mem0"] = Field("mem0", description="Memory store implementation")
    embedding_model: str = Field(
        "gemini/text-embedding-004",
        description="Embedding model for similarity search (default: gemini text-embedding-004)",
    )
    model: str = Field(
        "gemini/gemini-2.0-flash",
        description="LLM model for memory extraction and decisions",
    )
    similarity_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for memory retrieval",
    )
    max_memories_per_query: int = Field(
        10,
        ge=1,
        le=100,
        description="Max memories to retrieve per query",
    )


class CompactionConfig(BaseModel):
    """Configuration for context compaction."""

    enabled: bool = Field(False, description="Enable automatic context compaction during execution")
    token_threshold: float = Field(
        0.9, ge=0.1, le=0.95, description="Compact when token usage exceeds this fraction of budget"
    )
    token_budget: int = Field(
        1000000, ge=1000, le=1000000, description="Total token budget for conversation"
    )
    keep_recent_turns: int = Field(
        20, ge=1, le=50, description="Number of recent turns to preserve during compaction"
    )
    micro_compact_enabled: bool = Field(
        True, description="Enable micro-compaction of old tool results"
    )
    micro_compact_age_seconds: float = Field(
        3600.0, ge=10.0, le=3600.0, description="Age threshold for micro-compacting tool results"
    )
    micro_compact_max_chars: int = Field(
        2000, ge=50, le=10000, description="Max chars for micro-compacted tool results"
    )


class CoordinatorConfig(BaseModel):
    """Configuration for coordinator (multi-agent) mode."""

    enabled: bool = Field(
        False, description="Enable coordinator mode for parallel worker execution"
    )
    max_workers: int = Field(5, ge=1, le=20, description="Maximum number of parallel worker agents")
    synthesis_strategy: str = Field(
        "merge", description="How to combine worker results: merge, vote, or chain"
    )
    worker_timeout: float = Field(
        2700.0, ge=10.0, le=7200.0, description="Timeout per worker in seconds"
    )
    worker_model: Optional[str] = Field(
        None, description="Model override for worker agents (defaults to provider model)"
    )
    decompose_timeout: float = Field(
        60.0, ge=10.0, le=300.0, description="Timeout for auto-decompose planning step"
    )
    decompose_model: Optional[str] = Field(
        None, description="Model for task decomposition (defaults to provider model)"
    )
    executor_mode: str = Field(
        "sdk", description="Worker executor mode: sdk (Claude Agent SDK) or api (direct provider)"
    )


class TasksConfig(BaseModel):
    """Configuration for the task tracking system."""

    enabled: bool = Field(False, description="Enable task tracking during execution")
    persist_to_disk: bool = Field(True, description="Persist tasks to .dokumen-cache/tasks/")
    max_tasks: int = Field(200, ge=1, le=1000, description="Maximum number of tasks per run")


class SkillsConfig(BaseModel):
    """Configuration for reusable instruction injection."""

    enabled: bool = Field(True, description="Enable reusable instruction injection")
    dir: Optional[str] = Field(
        "skills/", description="Directory for project-specific instruction files (yaml/md)"
    )
    include_system: bool = Field(
        True, description="Include bundled reusable instructions (qa-check, link-check)"
    )
    max_skills_per_prompt: int = Field(
        10, ge=1, le=20, description="Maximum reusable instructions to inject into a prompt"
    )


class AgentsConfig(BaseModel):
    """Configuration for user-defined agents."""

    dir: str = Field("agents/", description="Directory for user-defined agent YAML files")

    @field_validator("dir")
    @classmethod
    def validate_agents_dir(cls, v: str) -> str:
        """Reject absolute paths and path traversal in agents.dir."""
        if not v:
            return "agents/"
        if os.path.isabs(v):
            raise ValueError(f"agents.dir must be a relative path, got absolute: '{v}'")
        if ".." in v.split(os.sep):
            raise ValueError(f"agents.dir must not contain '..', got: '{v}'")
        return v


class DokumenConfig(BaseModel):
    """Main configuration model for dokumen.yaml."""

    version: str = Field("1.0", description="Config file version")
    provider: ProviderConfig = Field(..., description="AI provider configuration")
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig, description="Execution settings"
    )
    coverage: CoverageConfig = Field(
        default_factory=CoverageConfig, description="Coverage settings"
    )
    explore: ExploreConfig = Field(
        default_factory=ExploreConfig, description="Explore phase settings"
    )
    perplexity: PerplexityConfig = Field(
        default_factory=PerplexityConfig, description="Perplexity web search settings"
    )
    tools: ToolsConfig = Field(
        default_factory=ToolsConfig, description="Project-level tool configuration"
    )
    agents: AgentsConfig = Field(
        default_factory=AgentsConfig, description="User-defined agent configuration"
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig, description="Agent memory settings (off by default)"
    )
    compaction: CompactionConfig = Field(
        default_factory=CompactionConfig, description="Context compaction settings"
    )
    coordinator: CoordinatorConfig = Field(
        default_factory=CoordinatorConfig, description="Coordinator multi-agent settings"
    )
    tasks: TasksConfig = Field(default_factory=TasksConfig, description="Task tracking settings")
    skills: SkillsConfig = Field(
        default_factory=SkillsConfig, description="Reusable instruction settings"
    )
    # Model overrides (optional - defaults to provider.model if not specified)
    executor_model: Optional[str] = Field(
        None, description="Model to use for test executors (overrides provider.model)"
    )
    judge_model: Optional[str] = Field(
        None, description="Model to use for test judges (overrides provider.model)"
    )

    def get_executor_model(self) -> str:
        """Get the model to use for executors (executor_model or provider.model)."""
        return self.executor_model or self.provider.model

    def get_judge_model(self) -> str:
        """Get the model to use for judges (judge_model or provider.model)."""
        return self.judge_model or self.provider.model


def load_config(config_path: Optional[str] = None) -> DokumenConfig:
    """
    Load and validate dokumen configuration from a YAML file.

    Args:
        config_path: Path to the config file. If None, looks for dokumen.yaml
                    in the current directory.

    Returns:
        Validated DokumenConfig instance.

    Raises:
        ConfigError: If the file is not found, invalid YAML, or validation fails.
    """
    path = Path(config_path) if config_path else Path("dokumen.yaml")

    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(_preprocess_yaml(f.read()))
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file: {e}")

    if raw_config is None:
        raise ConfigError(f"Empty configuration file: {path}")

    try:
        config = DokumenConfig(**raw_config)
    except ValidationError as e:
        # Extract meaningful error message
        errors = e.errors()
        error_msgs = []
        for error in errors:
            loc = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            error_msgs.append(f"{loc}: {msg}")
        raise ConfigError(f"Configuration validation failed: {'; '.join(error_msgs)}")

    return config
