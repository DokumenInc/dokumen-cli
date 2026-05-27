"""
Pytest fixtures for dokumen-cli tests.
"""
import os
import sys
from pathlib import Path
from typing import Generator

import pytest

# Make tests/mock_provider.py importable as 'mock_provider'
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def valid_config_path(fixtures_dir: Path) -> Path:
    """Return path to valid config fixture."""
    return fixtures_dir / "valid_config.yaml"


@pytest.fixture
def invalid_config_path(fixtures_dir: Path) -> Path:
    """Return path to invalid config fixture."""
    return fixtures_dir / "invalid_config.yaml"


@pytest.fixture
def missing_provider_config_path(fixtures_dir: Path) -> Path:
    """Return path to config missing required provider section."""
    return fixtures_dir / "missing_provider_config.yaml"


@pytest.fixture
def minimal_config_path(fixtures_dir: Path) -> Path:
    """Return path to minimal valid config (only required fields)."""
    return fixtures_dir / "minimal_config.yaml"


@pytest.fixture
def temp_config(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary config file."""
    config_path = tmp_path / "dokumen.yaml"
    yield config_path


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set mock environment variables for testing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove environment variables that might interfere with tests."""
    for var in [
        "ANTHROPIC_API_KEY",
        "DOKUMEN_PROVIDER",
        "DOKUMEN_API_KEY",
        "DOKUMEN_MODEL",
    ]:
        monkeypatch.delenv(var, raising=False)


# =============================================================================
# Scaffold Test Fixtures
# =============================================================================


@pytest.fixture
def scaffolds_dir(fixtures_dir: Path) -> Path:
    """Return path to scaffold fixtures directory."""
    return fixtures_dir / "scaffolds"


@pytest.fixture
def valid_minimal_scaffold_path(scaffolds_dir: Path) -> Path:
    """Return path to minimal valid scaffold fixture."""
    return scaffolds_dir / "valid_minimal.test.yaml"


@pytest.fixture
def valid_complete_scaffold_path(scaffolds_dir: Path) -> Path:
    """Return path to complete valid scaffold fixture."""
    return scaffolds_dir / "valid_complete.test.yaml"


@pytest.fixture
def valid_sandbox_string_path(scaffolds_dir: Path) -> Path:
    """Return path to scaffold with sandbox as string reference."""
    return scaffolds_dir / "valid_sandbox_string.test.yaml"


@pytest.fixture
def invalid_missing_name_path(scaffolds_dir: Path) -> Path:
    """Return path to scaffold missing required name field."""
    return scaffolds_dir / "invalid_missing_name.yaml"


@pytest.fixture
def invalid_missing_executor_path(scaffolds_dir: Path) -> Path:
    """Return path to scaffold missing required executor field."""
    return scaffolds_dir / "invalid_missing_executor.yaml"


@pytest.fixture
def invalid_empty_judges_path(scaffolds_dir: Path) -> Path:
    """Return path to scaffold with empty judges list."""
    return scaffolds_dir / "invalid_empty_judges.yaml"


@pytest.fixture
def invalid_malformed_scaffold_path(scaffolds_dir: Path) -> Path:
    """Return path to scaffold with malformed YAML."""
    return scaffolds_dir / "invalid_malformed.yaml"


@pytest.fixture
def invalid_bad_name_path(scaffolds_dir: Path) -> Path:
    """Return path to scaffold with invalid name format."""
    return scaffolds_dir / "invalid_bad_name.yaml"


@pytest.fixture
def warning_unknown_tool_path(scaffolds_dir: Path) -> Path:
    """Return path to scaffold with unknown tool (warning)."""
    return scaffolds_dir / "warning_unknown_tool.yaml"


@pytest.fixture
def warning_no_files_path(scaffolds_dir: Path) -> Path:
    """Return path to scaffold without files section (warning)."""
    return scaffolds_dir / "warning_no_files.yaml"


@pytest.fixture
def warning_missing_judge_prompt_path(scaffolds_dir: Path) -> Path:
    """Return path to scaffold with judge missing system_prompt (warning)."""
    return scaffolds_dir / "warning_missing_judge_prompt.yaml"


@pytest.fixture
def missing_doc_file_scaffold_path(scaffolds_dir: Path) -> Path:
    """Return path to scaffold that references a nonexistent doc file."""
    return scaffolds_dir / "missing_doc_file.yaml"


@pytest.fixture
def invalid_tool_scaffold_path(scaffolds_dir: Path) -> Path:
    """Return path to scaffold that uses an invalid tool name."""
    return scaffolds_dir / "invalid_tool.yaml"


# =============================================================================
# Run Command Test Fixtures
# =============================================================================

from unittest.mock import MagicMock


@pytest.fixture
def mock_test_result():
    """Factory for creating mock TestResult objects."""
    def _make(test_id: str, passed: bool = True, duration: float = 1.0, failure_reasons=None):
        return MagicMock(
            test_id=test_id,
            passed=passed,
            duration=duration,
            failure_reasons=failure_reasons or [],
            executor_output=None,
            judge_results=[],
            files=[],
            explore_output=None,
            explore_tool_calls=None,
            executor_model="claude-sonnet-4-5-20250929",
            judge_model="claude-sonnet-4-5-20250929",
            explore_model=None,
            # Token usage fields (OutputWriter accesses via getattr)
            executor_input_tokens=0,
            executor_output_tokens=0,
            judge_input_tokens=0,
            judge_output_tokens=0,
            explore_input_tokens=0,
            explore_output_tokens=0,
            # Additional fields accessed by OutputWriter
            executor_tools=[],
            judge_prompts=None,
            browser_artifacts=None,
            report_artifacts=None,
            output_artifacts=None,
            timestamp=None,
        )
    return _make


@pytest.fixture
def mock_suite_results():
    """Factory for creating mock TestSuiteResults."""
    def _make(total: int = 1, passed: int = 1, failed: int = 0, error: int = 0, test_results=None):
        return MagicMock(
            total_tests=total,
            passed=passed,
            failed=failed,
            error=error,
            skipped=0,
            duration=1.0,
            test_results=test_results or [],
            cached_results=0
        )
    return _make


@pytest.fixture
def project_with_tests(tmp_path: Path, valid_config_path: Path, valid_minimal_scaffold_path: Path) -> Path:
    """Create a complete project with config and tests."""
    # Copy config
    (tmp_path / "dokumen.yaml").write_text(valid_config_path.read_text())
    # Create tests dir with scaffold
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "my-test.test.yaml").write_text(valid_minimal_scaffold_path.read_text())
    return tmp_path


@pytest.fixture
def runner():
    """Click CLI test runner."""
    from click.testing import CliRunner
    return CliRunner()


# =============================================================================
# Executor & Tool Test Fixtures
# =============================================================================

from unittest.mock import AsyncMock
from typing import Any, Dict, List, Callable


@pytest.fixture
def mock_provider():
    """Mock LLM provider for tests that still need Provider (explore, ask, create).

    Returns a provider that returns a simple text response by default.
    Configure via mock_provider.complete.return_value for custom responses.
    """
    from dokumen.agent_object import Provider

    provider = AsyncMock(spec=Provider)
    # Default: return simple text response (no tool calls)
    provider.complete.return_value = {
        "content": "This is a test response from the executor."
    }
    return provider


@pytest.fixture
def mock_tool():
    """Factory for creating mock tool definitions."""
    from dokumen.tools_object import ToolDefinition, ToolResult

    def _make(
        name: str = "test_tool",
        description: str = "A test tool",
        return_value: Any = "Tool executed successfully",
        success: bool = True,
        error: str = None
    ) -> ToolDefinition:
        async def handler(params: Dict[str, Any]) -> ToolResult:
            return ToolResult(
                success=success,
                output=return_value,
                error=error
            )

        return ToolDefinition(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input parameter"}
                },
                "required": []
            },
            handler=handler
        )

    return _make


@pytest.fixture
def executor_agent():
    """Factory for creating SDK executor wrapper instances."""
    from dokumen.sdk.executor import ExecutorAgent
    from dokumen.sdk.agent_wrapper import SdkExecutorWrapper
    from dokumen.sdk.query_runner import MockQueryRunner
    from dokumen.sdk.testing import make_executor_simple

    def _make(
        id: str = "test-executor",
        system_prompt: str = "You are a test executor.",
        user_prompt: str = "Execute the test task.",
        tools: List = None,
        timeout: float = 60.0,
    ) -> SdkExecutorWrapper:
        runner = MockQueryRunner(make_executor_simple("Test response."))
        executor = ExecutorAgent(
            id=id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            sdk_tools=["Read"] if not tools else [t.name if hasattr(t, 'name') else str(t) for t in tools],
            query_runner=runner,
            timeout=timeout,
        )
        return SdkExecutorWrapper(executor, system_prompt=system_prompt, user_prompt=user_prompt)

    return _make


@pytest.fixture
def sample_test_file(tmp_path: Path) -> Path:
    """Create a sample test file for tool testing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    test_file = docs_dir / "test.md"
    test_file.write_text("# Test Document\n\nThis is test content.\n")
    return test_file


@pytest.fixture
def sample_project_dir(tmp_path: Path) -> Path:
    """Create a sample project directory structure for tool testing."""
    # Create directories
    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()

    # Create files
    (tmp_path / "docs" / "readme.md").write_text("# README\n\nProject documentation.")
    (tmp_path / "docs" / "api.md").write_text("# API\n\nAPI documentation.")
    (tmp_path / "src" / "main.py").write_text("# Main module\nprint('hello')")
    (tmp_path / "config.yaml").write_text("version: 1.0")

    return tmp_path


# =============================================================================
# Judge Test Fixtures
# =============================================================================


@pytest.fixture
def judge_agent():
    """Factory for creating SDK judge wrapper instances."""
    from dokumen.sdk.judge import JudgeAgent
    from dokumen.sdk.agent_wrapper import SdkJudgeWrapper
    from dokumen.sdk.query_runner import MockQueryRunner
    from dokumen.sdk.testing import make_judge_pass

    def _make(
        id: str = "test-judge",
        system_prompt: str = "Evaluate if the executor completed the task correctly.",
        include_executor_output: bool = True,
    ) -> SdkJudgeWrapper:
        runner = MockQueryRunner(make_judge_pass(confidence=0.95, reason="Test passed."))
        judge = JudgeAgent(
            id=id,
            system_prompt=system_prompt,
            user_prompt=system_prompt,
            sdk_tools=[],
            query_runner=runner,
            include_executor_output=include_executor_output,
        )
        return SdkJudgeWrapper(judge, assertion_text=id, system_prompt=system_prompt)

    return _make


@pytest.fixture
def sample_executor_output():
    """Sample ExecutorResult for judge testing."""
    from dokumen.sdk.types import ExecutorResult

    return ExecutorResult(
        tool_calls=[
            {
                "tool_name": "read_file",
                "parameters": {"file_path": "docs/api.md"},
                "result": "# API\n\nOAuth and API key authentication supported.",
            }
        ],
        final_response="The API supports OAuth and API key authentication.",
        success=True,
    )




# =============================================================================
# PDF Test Fixtures
# =============================================================================


def create_minimal_pdf() -> bytes:
    """Create minimal valid PDF for testing."""
    return b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
203
%%EOF
"""


def create_large_pdf(size_mb: float = 4.5) -> bytes:
    """Create large PDF at specified size.

    Args:
        size_mb: Target size in MB

    Returns:
        PDF bytes of exactly the specified size
    """
    header = create_minimal_pdf()
    target_size = int(size_mb * 1024 * 1024)
    # Account for the comment marker (%) so total size is exactly target_size
    padding_size = target_size - len(header) - 1  # -1 for the % character
    if padding_size < 0:
        padding_size = 0
    return header + b"%" + b"x" * padding_size


def create_multipage_pdf(page_count: int = 105) -> bytes:
    """Create a PDF with specified number of pages using pypdf.

    Args:
        page_count: Number of pages to create (default 105 to exceed 100 page limit)

    Returns:
        PDF bytes
    """
    try:
        from pypdf import PdfWriter
        import io

        writer = PdfWriter()
        for i in range(page_count):
            # Add blank pages (612x792 is US Letter size)
            writer.add_blank_page(width=612, height=792)

        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()
    except ImportError:
        # Fallback if pypdf not installed - create fake multi-page PDF
        # This won't actually have multiple pages but allows tests to run
        return create_minimal_pdf()


def create_password_protected_pdf() -> bytes:
    """Create password-protected PDF (requires pypdf).

    Note: This is a mock implementation. In a real test,
    we would use PyPDF2 or similar to create an encrypted PDF.
    For now, we just create a minimal PDF with a comment indicating
    it would be password-protected.
    """
    # Mock encrypted PDF - in real implementation would use PyPDF2
    # For testing purposes, we'll use a minimal PDF
    return b"""%PDF-1.4
%\xE2\xE3\xCF\xD3
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
203
%%EOF
"""
