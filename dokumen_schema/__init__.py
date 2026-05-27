"""Schema models used by the Dokumen CLI.

This package keeps the historic ``dokumen_schema`` import path local to the
CLI so a fresh checkout can install without private package indexes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .constants import CLI_RESOLVABLE_TOOLS, KNOWN_MODEL_ALIASES, VALID_EXECUTOR_TOOLS


@dataclass
class ValidationResult:
    """Pure validation result for a scaffold dictionary."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class FileRef(BaseModel):
    """A documentation file referenced by a scaffold."""

    model_config = ConfigDict(extra="allow")

    path: str
    description: Optional[str] = None


class DockerMount(BaseModel):
    """A Docker volume mount for sandboxed setup or execution."""

    source: str
    target: str
    readonly: bool = False


class SandboxConfig(BaseModel):
    """Sandbox configuration for command execution."""

    type: Literal["none", "whitelist", "subprocess", "docker", "virtual_fs"]
    docker_image: str = "python:3.11-slim"
    docker_network: Literal["none", "bridge"] = "none"
    docker_mount_readonly: bool = False
    docker_workdir: str = "/workspace"
    docker_mounts: list[DockerMount] = Field(default_factory=list)
    timeout: int = 60
    max_memory_mb: int = 512


class BrowserScaffoldConfig(BaseModel):
    """Browser settings for browser-type scaffolds."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    headless: bool = True
    viewport: Any = None
    viewport_size: Optional[Any] = None
    save_video: Optional[Any] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_viewport_alias(cls, data: Any) -> Any:
        """Treat viewport_size as an alias for viewport."""
        if isinstance(data, dict) and "viewport_size" in data and "viewport" not in data:
            data = {**data, "viewport": data["viewport_size"]}
        return data


class ExecutorConfig(BaseModel):
    """Executor section from a test scaffold."""

    model_config = ConfigDict(extra="allow")

    system_prompt: Optional[str] = None
    user_prompt: str
    tools: Optional[list[str]] = None
    agent: Optional[str] = None
    model: Optional[str] = None
    skills: Optional[list[str]] = None
    max_iterations: Optional[int] = None

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, value: Optional[str]) -> Optional[str]:
        """Reject empty prompt strings while allowing omitted prompts."""
        if value is not None and not value.strip():
            raise ValueError("system_prompt cannot be empty")
        if value and value.startswith("@prompts/") and "unknown" in value:
            raise ValueError("unknown prompt reference")
        return value

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        """Reject unknown executor tool names early."""
        if value is None:
            return value
        invalid = [tool for tool in value if tool not in VALID_EXECUTOR_TOOLS]
        if invalid:
            raise ValueError(f"Unknown tool: {', '.join(invalid)}")
        return value


class JudgeConfig(BaseModel):
    """Judge section from a test scaffold."""

    model_config = ConfigDict(extra="allow")

    name: str
    system_prompt: Optional[str] = None
    tools: list[str] = Field(default_factory=list)
    include_executor_output: bool = True
    agent: Optional[str] = None
    model: Optional[str] = None
    skills: Optional[list[str]] = None

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, value: Optional[str]) -> Optional[str]:
        """Reject empty prompt strings while allowing omitted prompts."""
        if value is not None and not value.strip():
            raise ValueError("system_prompt cannot be empty")
        return value

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: list[str]) -> list[str]:
        """Reject unknown judge tool names early."""
        invalid = [tool for tool in value if tool not in VALID_EXECUTOR_TOOLS]
        if invalid:
            raise ValueError(f"Unknown tool: {', '.join(invalid)}")
        return value


class TestScaffold(BaseModel):
    """Top-level Dokumen test scaffold model."""

    model_config = ConfigDict(extra="allow")

    name: str
    reason: Optional[str] = None
    type: Optional[Literal["browser"]] = None
    agent: Optional[str] = None
    files: list[FileRef] = Field(..., min_length=1)
    executor: ExecutorConfig
    judges: list[JudgeConfig] = Field(..., min_length=1)
    timeout: float = Field(60.0, ge=1, le=600)
    retries: int = Field(0, ge=0, le=5)
    sandbox: Optional[Union[str, SandboxConfig]] = None
    browser: Optional[BrowserScaffoldConfig] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Require kebab-case names so CLI filtering stays predictable."""
        import re

        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
            raise ValueError("name must be kebab-case")
        return value

    @model_validator(mode="after")
    def validate_browser_type(self) -> "TestScaffold":
        """Browser settings require type: browser."""
        if self.browser is not None and self.type != "browser" and self.agent != "browser-tester":
            raise ValueError("browser config requires type: browser")
        return self


class SetupStep(BaseModel):
    """Deterministic setup command executed before a test."""

    model_config = ConfigDict(extra="allow")

    name: str
    command: str
    working_dir: Optional[str] = None
    timeout: float = 60
    background: bool = False
    ready_url: Optional[str] = None
    ready_timeout: float = 30


@dataclass
class CICompatibilityResult:
    """Result of CI compatibility checks."""

    ci_compatible: bool
    ci_errors: list[str] = field(default_factory=list)
    ci_warnings: list[str] = field(default_factory=list)


def validate_test_data(data: dict[str, Any]) -> ValidationResult:
    """Validate parsed scaffold data without touching the filesystem."""
    try:
        scaffold = TestScaffold(**(data or {}))
    except ValidationError as exc:
        return ValidationResult(
            valid=False,
            errors=[_format_validation_error(error) for error in exc.errors()],
            warnings=[],
        )
    except Exception as exc:
        return ValidationResult(valid=False, errors=[str(exc)], warnings=[])

    warnings: list[str] = []
    for judge in scaffold.judges:
        if not judge.system_prompt:
            warnings.append(f"Judge '{judge.name}' has no system_prompt")

    prompt = scaffold.executor.user_prompt or ""
    if scaffold.type != "browser":
        for file_ref in scaffold.files:
            if file_ref.path in prompt:
                warnings.append(
                    f"Executor prompt contains hardcoded path from files list: {file_ref.path}"
                )

    return ValidationResult(valid=True, errors=[], warnings=warnings)


def check_ci_compatibility(
    data: dict[str, Any],
    allowed_tools: Optional[list[str]] = None,
    existing_files: Optional[list[str]] = None,
) -> CICompatibilityResult:
    """Check scaffold features that commonly fail in CI."""
    errors: list[str] = []
    warnings: list[str] = []
    allowed = set(allowed_tools or VALID_EXECUTOR_TOOLS)
    existing = set(existing_files or [])

    executor_tools = ((data.get("executor") or {}).get("tools")) or []
    for tool in executor_tools:
        if tool not in allowed:
            errors.append(f"Tool '{tool}' is not allowed in CI")

    for file_ref in data.get("files", []) or []:
        path = file_ref.get("path") if isinstance(file_ref, dict) else None
        if path and existing_files is not None and path not in existing:
            warnings.append(f"Referenced file not found in CI checkout: {path}")

    return CICompatibilityResult(
        ci_compatible=not errors,
        ci_errors=errors,
        ci_warnings=warnings,
    )


def _format_validation_error(error: dict[str, Any]) -> str:
    """Convert a Pydantic error into a short CLI-friendly message."""
    loc = ".".join(str(part) for part in error.get("loc", []))
    msg = error.get("msg", "Invalid value")
    return f"{loc}: {msg}" if loc else msg


__all__ = [
    "BrowserScaffoldConfig",
    "CLI_RESOLVABLE_TOOLS",
    "CICompatibilityResult",
    "DockerMount",
    "ExecutorConfig",
    "FileRef",
    "JudgeConfig",
    "KNOWN_MODEL_ALIASES",
    "SandboxConfig",
    "SetupStep",
    "TestScaffold",
    "VALID_EXECUTOR_TOOLS",
    "ValidationResult",
    "check_ci_compatibility",
    "validate_test_data",
]
