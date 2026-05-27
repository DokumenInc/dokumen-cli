"""
Unit tests for cross-reference test scaffold support.

Tests the cross-reference scaffold type including:
- Scaffold validation with type: cross-reference
- code_files field validation
- Auto-injection of code tools for cross-reference tests
- Cross-reference executor prompt loading
"""
import os
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cross_ref_scaffold_dir(tmp_path: Path) -> Path:
    """Create a directory with a cross-reference scaffold and supporting files."""
    # Create docs directory with a doc file
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "api.md").write_text("# API\n\nDocumentation for the API.\n")

    # Create tests directory
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    # Create a valid cross-reference scaffold
    scaffold = {
        "name": "api-cross-ref",
        "reason": "Cross-reference API docs with implementation",
        "type": "cross-reference",
        "files": [{"path": "docs/api.md"}],
        "code_files": [
            {"repo": "backend", "path": "src/api/routes.py"},
        ],
        "executor": {
            "system_prompt": "@prompts/cross-reference.txt",
            "user_prompt": "Compare the API documentation with the code implementation.",
            "tools": ["read_file", "code_read_file", "code_search"],
        },
        "judges": [
            {
                "name": "accuracy",
                "system_prompt": "Evaluate if the cross-reference check was thorough.",
            }
        ],
        "timeout": 120,
    }
    scaffold_path = tests_dir / "api-cross-ref.test.yaml"
    scaffold_path.write_text(yaml.dump(scaffold))

    # Create dokumen.yaml config
    config = {
        "version": "1.0",
        "provider": {"name": "mock", "model": "test"},
        "code_repos": [
            {
                "name": "backend",
                "gitlab_project_id": 123,
                "branch": "main",
                "paths_include": ["src/**/*.py"],
                "paths_exclude": ["tests/**"],
            }
        ],
    }
    (tmp_path / "dokumen.yaml").write_text(yaml.dump(config))

    return tmp_path


@pytest.fixture
def code_files_without_type_scaffold(tmp_path: Path) -> Path:
    """Create a scaffold with code_files but without type: cross-reference."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "api.md").write_text("# API\n")

    scaffold = {
        "name": "bad-scaffold",
        "files": [{"path": "docs/api.md"}],
        "code_files": [
            {"repo": "backend", "path": "src/routes.py"},
        ],
        "executor": {
            "system_prompt": "@prompts/general.txt",
            "user_prompt": "Test something.",
            "tools": ["read_file"],
        },
        "judges": [
            {
                "name": "check",
                "system_prompt": "Check it.",
            }
        ],
    }
    path = tests_dir / "bad-scaffold.test.yaml"
    path.write_text(yaml.dump(scaffold))
    return path


# ============================================================================
# Tests for cross-reference scaffold validation (shared schema)
# ============================================================================


class TestCrossReferenceScaffoldValidation:
    """Tests for cross-reference scaffold validation."""

    def test_valid_cross_reference_scaffold(self):
        """Scaffold with type: cross-reference validates successfully."""
        from dokumen_schema import validate_test_data

        data = {
            "name": "api-cross-ref",
            "type": "cross-reference",
            "files": [{"path": "docs/api.md"}],
            "code_files": [
                {"repo": "backend", "path": "src/routes.py"},
            ],
            "executor": {
                "system_prompt": "@prompts/cross-reference.txt",
                "user_prompt": "Compare docs with code.",
                "tools": ["read_file", "code_read_file"],
            },
            "judges": [
                {"name": "accuracy", "system_prompt": "Evaluate accuracy."},
            ],
        }
        result = validate_test_data(data)
        assert result.valid, f"Validation failed: {result.errors}"

    def test_cross_reference_with_code_files(self):
        """Scaffold with code_files field validates."""
        from dokumen_schema import TestScaffold

        scaffold = TestScaffold(
            name="api-cross-ref",
            files=[{"path": "docs/api.md"}],
            code_files=[
                {"repo": "backend", "path": "src/routes.py"},
            ],
            executor={
                "agent": "code-reviewer",
                "user_prompt": "Compare docs with code.",
                "tools": ["read_file"],
            },
            judges=[
                {"name": "accuracy", "system_prompt": "Check."},
            ],
        )
        assert len(scaffold.code_files) == 1
        assert scaffold.code_files[0].repo == "backend"
        assert scaffold.code_files[0].path == "src/routes.py"

    def test_code_files_without_code_agent_rejected(self):
        """code_files without a code-capable agent is rejected."""
        from dokumen_schema import TestScaffold

        with pytest.raises(ValueError, match="code_files.*code"):
            TestScaffold(
                name="bad-scaffold",
                files=[{"path": "docs/api.md"}],
                code_files=[
                    {"repo": "backend", "path": "src/routes.py"},
                ],
                executor={
                    "agent": "doc-validator",
                    "user_prompt": "Test.",
                    "tools": ["read_file"],
                },
                judges=[
                    {"name": "check", "system_prompt": "Check."},
                ],
            )

    def test_code_reviewer_agent_without_code_files_is_valid(self):
        """code-reviewer agent without code_files is valid (code_files optional)."""
        from dokumen_schema import TestScaffold

        scaffold = TestScaffold(
            name="simple-cross-ref",
            files=[{"path": "docs/api.md"}],
            executor={
                "agent": "code-reviewer",
                "user_prompt": "Compare.",
                "tools": ["read_file"],
            },
            judges=[
                {"name": "check", "system_prompt": "Check."},
            ],
        )
        assert scaffold.code_files is None


# ============================================================================
# Tests for cross-reference auto-injection of code tools
# ============================================================================


@pytest.mark.xfail(reason="Code tools (code_read_file, etc.) not yet mapped in SDK tool resolver")
class TestCrossReferenceAutoInjection:
    """Tests for auto-injection of code tools in cross-reference scaffolds."""

    def test_cross_reference_auto_injects_code_tools(self, cross_ref_scaffold_dir: Path):
        """Cross-reference scaffolds auto-inject code tools."""
        from dokumen.loader import load_scaffold

        scaffold_path = str(cross_ref_scaffold_dir / "tests" / "api-cross-ref.test.yaml")

        # We need a code repo directory for the code tools
        code_dir = cross_ref_scaffold_dir / "code-backend"
        code_dir.mkdir()
        (code_dir / "src").mkdir(parents=True)
        (code_dir / "src" / "api").mkdir(parents=True)
        (code_dir / "src" / "api" / "routes.py").write_text("# routes\n")

        test_obj = load_scaffold(
            scaffold_path,
            project_root=str(cross_ref_scaffold_dir),
            code_repos_config=[{
                "name": "backend",
                "base_dir": str(code_dir),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )

        executor_tool_names = [t.name for t in test_obj.executor.tools]

        # Should have code tools auto-injected
        assert "code_read_file" in executor_tool_names
        assert "code_search" in executor_tool_names
        assert "code_glob" in executor_tool_names

    def test_cross_reference_does_not_duplicate_explicit_code_tools(self, cross_ref_scaffold_dir: Path):
        """Auto-injection does not duplicate tools already in scaffold."""
        from dokumen.loader import load_scaffold

        scaffold_path = str(cross_ref_scaffold_dir / "tests" / "api-cross-ref.test.yaml")

        code_dir = cross_ref_scaffold_dir / "code-backend"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "src").mkdir(parents=True, exist_ok=True)

        test_obj = load_scaffold(
            scaffold_path,
            project_root=str(cross_ref_scaffold_dir),
            code_repos_config=[{
                "name": "backend",
                "base_dir": str(code_dir),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )

        executor_tool_names = [t.name for t in test_obj.executor.tools]

        # code_read_file was explicitly in scaffold AND would be auto-injected
        # Should only appear once
        assert executor_tool_names.count("code_read_file") == 1

    def test_cross_reference_sets_max_iterations(self, cross_ref_scaffold_dir: Path):
        """Cross-reference scaffolds set appropriate max_iterations."""
        from dokumen.loader import load_scaffold

        scaffold_path = str(cross_ref_scaffold_dir / "tests" / "api-cross-ref.test.yaml")

        code_dir = cross_ref_scaffold_dir / "code-backend"
        code_dir.mkdir(exist_ok=True)

        test_obj = load_scaffold(
            scaffold_path,
            project_root=str(cross_ref_scaffold_dir),
            code_repos_config=[{
                "name": "backend",
                "base_dir": str(code_dir),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )

        # Cross-reference tests need more iterations for thorough comparison
        assert test_obj.executor.max_iterations >= 10


# ============================================================================
# Tests for cross-reference executor prompt
# ============================================================================


class TestCrossReferencePrompt:
    """Tests for the cross-reference executor prompt file."""

    def test_prompt_file_exists(self):
        """Cross-reference prompt file exists."""
        prompt_path = Path(__file__).parent.parent.parent / "dokumen" / "prompts" / "cross-reference.txt"
        assert prompt_path.exists(), f"Prompt file not found: {prompt_path}"

    def test_prompt_content_mentions_cross_reference(self):
        """Cross-reference prompt guides executor to compare docs with code."""
        prompt_path = Path(__file__).parent.parent.parent / "dokumen" / "prompts" / "cross-reference.txt"
        content = prompt_path.read_text()

        # Should mention cross-referencing docs with code
        content_lower = content.lower()
        assert "documentation" in content_lower or "doc" in content_lower
        assert "code" in content_lower
        assert "compar" in content_lower or "cross-reference" in content_lower or "verify" in content_lower

    def test_prompt_can_be_loaded(self):
        """Cross-reference prompt can be loaded via load_executor_prompt."""
        from dokumen.loader import load_executor_prompt

        content = load_executor_prompt("@prompts/cross-reference.txt")
        assert len(content) > 0
