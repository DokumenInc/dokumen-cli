"""
Dokumen - Skill Testing Framework

A framework for testing agent skills against project knowledge and tools.
"""

from pathlib import Path

# Canonical result types
from .agent_object import (
    AgentType,
    Provider,
    # Backward-compatible aliases
    ExecutorOutput,
    JudgeResult,
    LogEntry,
)
from .file_object import FileMetrics, FileObject
from .loader import (
    get_configured_provider,
    load_all_scaffolds,
    load_scaffold,
    load_test_from_yaml,
    resolve_tools,
)
from .providers import AnthropicProvider
from .test_object import TestObject, TestResult, TestConfig
from .test_suite import (
    TestSuite,
    TestSuiteConfig,
    TestSuiteResults,
    CoverageReport,
)
from .scaffold import (
    generate_scaffold,
    validate_scaffold,
    validate_scaffold_file,
    discover_scaffolds,
    load_scaffold_yaml,
    ValidationResult,
)
from .sdk.types import ExecutorResult, JudgeVerdict
from .tools.types import (
    ToolDefinition,
    ToolResult,
    ToolCall,
    ToolsObject,
)


def _read_version() -> str:
    """Read version from VERSION file."""
    version_file = Path(__file__).parent.parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.0.0"  # Fallback for development


__version__ = _read_version()

__all__ = [
    # Version
    "__version__",
    # File Object
    "FileObject",
    "FileMetrics",
    # Tools
    "ToolDefinition",
    "ToolResult",
    "ToolCall",
    "ToolsObject",
    # Result types (canonical)
    "ExecutorResult",
    "JudgeVerdict",
    # Agent types
    "AgentType",
    "Provider",
    # Backward-compatible aliases
    "ExecutorOutput",
    "JudgeResult",
    "LogEntry",
    # Test
    "TestObject",
    "TestResult",
    "TestConfig",
    # Suite
    "TestSuite",
    "TestSuiteConfig",
    "TestSuiteResults",
    "CoverageReport",
    # Scaffold
    "generate_scaffold",
    "validate_scaffold",
    "validate_scaffold_file",
    "discover_scaffolds",
    "load_scaffold_yaml",
    "ValidationResult",
    # Loader
    "load_scaffold",
    "load_test_from_yaml",
    "resolve_tools",
    "get_configured_provider",
    "load_all_scaffolds",
    # Providers
    "AnthropicProvider",
]
