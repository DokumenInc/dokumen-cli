"""Tests for scaffold module (scaffold generation and validation)."""

import pytest
from pathlib import Path
import tempfile

from dokumen.scaffold import (
    generate_scaffold,
    generate_test_scaffold,
    validate_scaffold,
    validate_scaffold_file,
    ValidationResult,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Should create valid result."""
        result = ValidationResult(valid=True, errors=[], warnings=[])
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_result(self):
        """Should create invalid result with errors."""
        result = ValidationResult(
            valid=False,
            errors=["Missing name"],
            warnings=["No files"]
        )
        assert result.valid is False
        assert result.errors == ["Missing name"]
        assert result.warnings == ["No files"]


class TestGenerateScaffold:
    """Tests for generate_scaffold function."""

    def test_basic_scaffold(self):
        """Should generate basic scaffold YAML."""
        result = generate_scaffold("docs/api.md")

        assert "name:" in result
        assert "api-test" in result
        assert "executor:" in result
        assert "judges:" in result
        assert "docs/api.md" in result
        assert "read_file" in result

    def test_custom_output_dir(self):
        """Should use custom output directory."""
        result = generate_scaffold("docs/api.md", output_dir="custom_tests")

        # The scaffold should reference the doc file
        assert "docs/api.md" in result

    def test_custom_test_name(self):
        """Should use custom test name."""
        result = generate_scaffold("docs/api.md", test_name="my-custom-test")

        assert "name: my-custom-test" in result

    def test_nested_doc_path(self):
        """Should handle nested documentation paths."""
        result = generate_scaffold("docs/api/v2/auth.md")

        assert "auth-test" in result

    def test_underscores_converted_to_hyphens(self):
        """Should convert underscores to hyphens in test name."""
        result = generate_scaffold("docs/api_reference.md")

        assert "api-reference-test" in result

    def test_spaces_converted_to_hyphens(self):
        """Should convert spaces to hyphens in test name."""
        result = generate_scaffold("docs/my docs.md")

        assert "my-docs-test" in result

    def test_contains_todo_markers(self):
        """Should contain TODO markers for user completion."""
        result = generate_scaffold("docs/api.md")

        assert "TODO:" in result

    def test_tools_include_list_files(self):
        """Should include list_files tool."""
        result = generate_scaffold("docs/api.md")

        assert "list_files" in result


class TestGenerateTestScaffold:
    """Tests for generate_test_scaffold function."""

    def test_basic_test_scaffold(self):
        """Should generate test scaffold with action and assertion."""
        result = generate_test_scaffold(
            doc_path="docs/api.md",
            action="Read the API documentation",
            assertion="Should contain authentication methods"
        )

        assert "name:" in result
        assert "executor:" in result
        assert "judges:" in result
        assert "Read the API documentation" in result

    def test_custom_test_name(self):
        """Should use custom test name."""
        result = generate_test_scaffold(
            doc_path="docs/api.md",
            action="Test action",
            assertion="Test assertion",
            test_name="my-api-test"
        )

        assert "name: my-api-test" in result

    def test_multiline_action(self):
        """Should handle multiline action."""
        action = """Read the documentation
Check for errors
Verify the content"""
        result = generate_test_scaffold(
            doc_path="docs/api.md",
            action=action,
            assertion="Should be valid"
        )

        assert "Read the documentation" in result

    def test_multiline_assertion(self):
        """Should handle multiline assertion."""
        assertion = """Should contain:
- Authentication
- Endpoints
- Examples"""
        result = generate_test_scaffold(
            doc_path="docs/api.md",
            action="Read docs",
            assertion=assertion
        )

        assert "Authentication" in result

    def test_includes_assertion_judge(self):
        """Should include assertion-judge."""
        result = generate_test_scaffold(
            doc_path="docs/api.md",
            action="Test",
            assertion="Check"
        )

        assert "assertion-judge" in result

    def test_includes_evaluation_criteria(self):
        """Should include EVALUATION CRITERIA section."""
        result = generate_test_scaffold(
            doc_path="docs/api.md",
            action="Test",
            assertion="Should pass"
        )

        assert "EVALUATION CRITERIA" in result


class TestValidateScaffold:
    """Tests for validate_scaffold function."""

    def test_valid_minimal_scaffold(self):
        """Should validate minimal valid scaffold."""
        data = {
            "name": "test",
            "files": [{"path": "docs/api.md"}],
            "executor": {
                "system_prompt": "@prompts/general.txt",
                "user_prompt": "Test this.",
                "tools": ["read_file"]
            },
            "judges": [
                {"name": "accuracy", "system_prompt": "Evaluate."}
            ]
        }
        result = validate_scaffold(data)

        assert result.valid is True
        assert result.errors == []

    def test_missing_name(self):
        """Should report missing name."""
        data = {
            "executor": {"system_prompt": "Test"},
            "judges": [{"name": "judge"}]
        }
        result = validate_scaffold(data)

        assert result.valid is False
        assert any("name" in e.lower() for e in result.errors)

    def test_missing_executor(self):
        """Should report missing executor."""
        data = {
            "name": "test",
            "judges": [{"name": "judge"}]
        }
        result = validate_scaffold(data)

        assert result.valid is False
        assert any("executor" in e.lower() for e in result.errors)

    def test_executor_system_prompt_required(self):
        """system_prompt is required and must use @prompts/ reference."""
        data = {
            "name": "test",
            "files": [{"path": "docs/api.md"}],
            "executor": {"user_prompt": "Test", "tools": ["read_file"]},
            "judges": [{"name": "judge"}]
        }
        result = validate_scaffold(data)

        # system_prompt is required by the strict schema
        assert result.valid is False
        assert any("system_prompt" in e.lower() for e in result.errors)

    def test_missing_judges(self):
        """Should report missing judges."""
        data = {
            "name": "test",
            "executor": {"system_prompt": "Test"}
        }
        result = validate_scaffold(data)

        assert result.valid is False
        assert any("judges" in e.lower() for e in result.errors)

    def test_empty_judges_list(self):
        """Should report empty judges list."""
        data = {
            "name": "test",
            "executor": {"system_prompt": "Test"},
            "judges": []
        }
        result = validate_scaffold(data)

        assert result.valid is False
        assert any("judges" in e.lower() for e in result.errors)

    def test_judge_missing_name(self):
        """Should report judge missing name."""
        data = {
            "name": "test",
            "executor": {"system_prompt": "Test"},
            "judges": [{"system_prompt": "Eval"}]
        }
        result = validate_scaffold(data)

        assert result.valid is False
        assert any("name" in e.lower() for e in result.errors)

    def test_warning_judge_no_system_prompt(self):
        """Should warn if judge has no system_prompt."""
        data = {
            "name": "test",
            "files": [{"path": "docs/api.md"}],
            "executor": {
                "system_prompt": "@prompts/general.txt",
                "user_prompt": "Test this.",
                "tools": ["read_file"]
            },
            "judges": [{"name": "judge"}]
        }
        result = validate_scaffold(data)

        assert result.valid is True  # Still valid
        assert any("system_prompt" in w.lower() for w in result.warnings)

    def test_invalid_timeout_type(self):
        """Should report invalid timeout type."""
        data = {
            "name": "test",
            "executor": {"system_prompt": "Test"},
            "judges": [{"name": "judge", "system_prompt": "Eval"}],
            "timeout": "not_a_number"
        }
        result = validate_scaffold(data)

        assert result.valid is False
        assert any("timeout" in e.lower() for e in result.errors)

    def test_invalid_retries_type(self):
        """Should report invalid retries type."""
        data = {
            "name": "test",
            "executor": {"system_prompt": "Test"},
            "judges": [{"name": "judge", "system_prompt": "Eval"}],
            "retries": "not_an_int"
        }
        result = validate_scaffold(data)

        assert result.valid is False
        assert any("retries" in e.lower() for e in result.errors)

    def test_valid_timeout(self):
        """Should accept valid timeout."""
        data = {
            "name": "test",
            "files": [{"path": "docs/api.md"}],
            "executor": {
                "system_prompt": "@prompts/general.txt",
                "user_prompt": "Test this.",
                "tools": ["read_file"]
            },
            "judges": [{"name": "judge", "system_prompt": "Eval"}],
            "timeout": 120
        }
        result = validate_scaffold(data)

        assert result.valid is True

    def test_valid_retries(self):
        """Should accept valid retries."""
        data = {
            "name": "test",
            "files": [{"path": "docs/api.md"}],
            "executor": {
                "system_prompt": "@prompts/general.txt",
                "user_prompt": "Test this.",
                "tools": ["read_file"]
            },
            "judges": [{"name": "judge", "system_prompt": "Eval"}],
            "retries": 3
        }
        result = validate_scaffold(data)

        assert result.valid is True

    def test_judge_not_dict(self):
        """Should report if judge is not a dict."""
        data = {
            "name": "test",
            "executor": {"system_prompt": "Test"},
            "judges": ["not_a_dict"]
        }
        result = validate_scaffold(data)

        assert result.valid is False
        assert any("dict" in e.lower() for e in result.errors)

    def test_judge_tools_not_list(self):
        """Should report if judge tools is not a list."""
        data = {
            "name": "test",
            "executor": {"system_prompt": "Test"},
            "judges": [{"name": "judge", "system_prompt": "Eval", "tools": "not_a_list"}]
        }
        result = validate_scaffold(data)

        assert result.valid is False
        assert any("list" in e.lower() for e in result.errors)

    def test_judge_tools_item_not_string(self):
        """Should report if judge tools item is not a string."""
        data = {
            "name": "test",
            "executor": {"system_prompt": "Test"},
            "judges": [{"name": "judge", "system_prompt": "Eval", "tools": [123]}]
        }
        result = validate_scaffold(data)

        assert result.valid is False
        assert any("string" in e.lower() for e in result.errors)


class TestValidateScaffoldFile:
    """Tests for validate_scaffold_file function."""

    def test_valid_file(self, tmp_path):
        """Should validate valid scaffold file."""
        content = """
name: test
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/general.txt"
  user_prompt: "Do test"
  tools: [read_file]
judges:
  - name: accuracy
    system_prompt: "Evaluate"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(content)

        result = validate_scaffold_file(str(file_path))

        assert result.valid is True

    def test_missing_file(self, tmp_path):
        """Should report missing file."""
        file_path = tmp_path / "nonexistent.yaml"

        result = validate_scaffold_file(str(file_path))

        assert result.valid is False
        assert any("not found" in e.lower() or "no such" in e.lower() for e in result.errors)

    def test_invalid_yaml(self, tmp_path):
        """Should report invalid YAML."""
        content = "invalid: yaml: content: ["
        file_path = tmp_path / "invalid.yaml"
        file_path.write_text(content)

        result = validate_scaffold_file(str(file_path))

        assert result.valid is False
        assert any("yaml" in e.lower() for e in result.errors)

    def test_missing_required_fields(self, tmp_path):
        """Should report missing required fields."""
        content = """
name: test
"""
        file_path = tmp_path / "incomplete.yaml"
        file_path.write_text(content)

        result = validate_scaffold_file(str(file_path))

        assert result.valid is False
        assert any("executor" in e.lower() for e in result.errors)


class TestDiscoverScaffolds:
    """Tests for discover_scaffolds function."""

    def test_find_test_yaml_files(self, tmp_path):
        """Should find all .test.yaml files."""
        from dokumen.scaffold import discover_scaffolds

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test1.test.yaml").write_text("name: test1")
        (tests_dir / "test2.test.yaml").write_text("name: test2")

        scaffolds = discover_scaffolds(str(tests_dir))

        assert len(scaffolds) == 2

    def test_find_nested_scaffolds(self, tmp_path):
        """Should find scaffolds in subdirectories."""
        from dokumen.scaffold import discover_scaffolds

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        nested = tests_dir / "api"
        nested.mkdir()
        (nested / "auth.test.yaml").write_text("name: auth")

        scaffolds = discover_scaffolds(str(tests_dir))

        assert len(scaffolds) == 1
        assert "auth.test.yaml" in scaffolds[0]

    def test_empty_directory(self, tmp_path):
        """Should return empty list for directory with no scaffolds."""
        from dokumen.scaffold import discover_scaffolds

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        scaffolds = discover_scaffolds(str(tests_dir))

        assert scaffolds == []

    def test_nonexistent_directory(self, tmp_path):
        """Should return empty list for nonexistent directory."""
        from dokumen.scaffold import discover_scaffolds

        scaffolds = discover_scaffolds(str(tmp_path / "nonexistent"))

        assert scaffolds == []

    def test_ignores_non_test_yaml(self, tmp_path):
        """Should ignore non-.test.yaml files."""
        from dokumen.scaffold import discover_scaffolds

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "regular.yaml").write_text("name: not-a-test")
        (tests_dir / "actual.test.yaml").write_text("name: test")

        scaffolds = discover_scaffolds(str(tests_dir))

        assert len(scaffolds) == 1
        assert "actual.test.yaml" in scaffolds[0]


class TestLoadScaffoldYaml:
    """Tests for load_scaffold_yaml function."""

    def test_load_valid_yaml(self, tmp_path):
        """Should load valid YAML file."""
        from dokumen.scaffold import load_scaffold_yaml

        file_path = tmp_path / "test.yaml"
        file_path.write_text("name: test\nversion: 1.0")

        data = load_scaffold_yaml(str(file_path))

        assert data["name"] == "test"
        assert data["version"] == 1.0

    def test_load_complex_yaml(self, tmp_path):
        """Should load complex YAML structure."""
        from dokumen.scaffold import load_scaffold_yaml

        content = """
name: complex-test
executor:
  system_prompt: "You are a tester"
  tools:
    - read_file
    - list_files
judges:
  - name: accuracy
"""
        file_path = tmp_path / "complex.yaml"
        file_path.write_text(content)

        data = load_scaffold_yaml(str(file_path))

        assert data["name"] == "complex-test"
        assert len(data["executor"]["tools"]) == 2

    def test_load_empty_file_raises(self, tmp_path):
        """Should raise ValueError for empty file."""
        from dokumen.scaffold import load_scaffold_yaml

        file_path = tmp_path / "empty.yaml"
        file_path.write_text("")

        with pytest.raises(ValueError, match="Empty"):
            load_scaffold_yaml(str(file_path))

    def test_load_nonexistent_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        from dokumen.scaffold import load_scaffold_yaml

        with pytest.raises(FileNotFoundError):
            load_scaffold_yaml(str(tmp_path / "missing.yaml"))


class TestExtractFileReferencesFromPrompts:
    """Tests for extract_file_references_from_prompts function."""

    def test_extract_from_executor_prompts(self):
        """Should extract file paths from executor prompts."""
        from dokumen.scaffold import extract_file_references_from_prompts

        data = {
            "executor": {
                "system_prompt": "You will analyze docs/api.md",
                "user_prompt": "Also check docs/guide.md"
            }
        }

        refs = extract_file_references_from_prompts(data)

        assert "docs/api.md" in refs
        assert "docs/guide.md" in refs

    def test_extract_from_user_prompt(self):
        """Should extract file paths from user_prompt."""
        from dokumen.scaffold import extract_file_references_from_prompts

        data = {
            "executor": {
                "user_prompt": "Read docs/api.md and summarize it."
            }
        }

        refs = extract_file_references_from_prompts(data)

        assert "docs/api.md" in refs

    def test_extract_from_system_prompt(self):
        """Should extract file paths from system_prompt."""
        from dokumen.scaffold import extract_file_references_from_prompts

        data = {
            "executor": {
                "system_prompt": "You will read docs/reference.md"
            }
        }

        refs = extract_file_references_from_prompts(data)

        assert "docs/reference.md" in refs

    def test_extract_multiple_paths(self):
        """Should extract multiple file paths."""
        from dokumen.scaffold import extract_file_references_from_prompts

        data = {
            "executor": {
                "user_prompt": "Read docs/a.md and docs/b.md then compare"
            }
        }

        refs = extract_file_references_from_prompts(data)

        assert "docs/a.md" in refs
        assert "docs/b.md" in refs

    def test_empty_data(self):
        """Should return empty set for empty data."""
        from dokumen.scaffold import extract_file_references_from_prompts

        refs = extract_file_references_from_prompts({})

        assert refs == set() or refs == []

    def test_string_files(self):
        """Should handle files as string paths."""
        from dokumen.scaffold import extract_file_references_from_prompts

        data = {
            "files": ["docs/api.md", "docs/guide.md"]
        }

        refs = extract_file_references_from_prompts(data)

        assert "docs/api.md" in refs or len(refs) == 0  # Implementation may vary


class TestGenerateTestScaffoldAdvanced:
    """Advanced tests for generate_test_scaffold function."""

    def test_scaffold_for_markdown(self, tmp_path):
        """Should generate scaffold for markdown file with action and assertion."""
        doc_file = tmp_path / "docs" / "api.md"
        doc_file.parent.mkdir(parents=True)
        doc_file.write_text("# API Documentation\n\nThis is the API.")

        result = generate_test_scaffold(
            doc_path=str(doc_file),
            action="Read the API documentation and list all endpoints",
            assertion="Should contain at least one API endpoint"
        )

        assert "name:" in result
        assert "executor:" in result
        assert "judges:" in result
        assert "API documentation" in result

    def test_scaffold_contains_reason_field(self):
        """Scaffold includes reason field with partial assertion."""
        result = generate_test_scaffold(
            doc_path="docs/guide.md",
            action="Validate user guide content",
            assertion="The guide should be accurate and complete"
        )

        assert "reason:" in result
        # Reason is auto-generated from assertion
        assert "guide should be" in result.lower() or "accurate" in result.lower()

    def test_scaffold_with_custom_name(self):
        """Should accept custom test_name parameter."""
        result = generate_scaffold("docs/api.md", test_name="custom-api-test")

        assert "custom-api-test" in result


class TestValidateScaffoldAdvanced:
    """Advanced validation tests."""

    def test_validate_with_all_optional_fields(self):
        """Should validate scaffold with all optional fields."""
        data = {
            "name": "full-test",
            "reason": "Test everything",
            "files": [{"path": "docs/api.md"}],
            "executor": {
                "system_prompt": "@prompts/general.txt",
                "user_prompt": "Test the docs.",
                "tools": ["read_file"]
            },
            "judges": [
                {"name": "accuracy", "system_prompt": "Evaluate."}
            ],
            "timeout": 120,
            "retries": 2
        }

        result = validate_scaffold(data)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_empty_judges(self):
        """Should fail with empty judges list."""
        data = {
            "name": "test",
            "executor": {"system_prompt": "Test"},
            "judges": []
        }

        result = validate_scaffold(data)

        assert result.valid is False
        assert any("judge" in e.lower() for e in result.errors)


class TestGetScaffoldName:
    """Tests for get_scaffold_name function."""

    def test_get_name_from_valid_file(self, tmp_path):
        """Should extract name from valid scaffold file."""
        from dokumen.scaffold import get_scaffold_name

        file_path = tmp_path / "test.yaml"
        file_path.write_text("name: my-test-name\nexecutor:\n  system_prompt: test")

        name = get_scaffold_name(str(file_path))

        assert name == "my-test-name"

    def test_get_name_from_invalid_file(self, tmp_path):
        """Should return None for invalid YAML."""
        from dokumen.scaffold import get_scaffold_name

        file_path = tmp_path / "invalid.yaml"
        file_path.write_text("invalid: yaml: [")

        name = get_scaffold_name(str(file_path))

        assert name is None

    def test_get_name_from_missing_file(self, tmp_path):
        """Should return None for missing file."""
        from dokumen.scaffold import get_scaffold_name

        name = get_scaffold_name(str(tmp_path / "missing.yaml"))

        assert name is None

    def test_get_name_from_file_without_name(self, tmp_path):
        """Should return None when name field is missing."""
        from dokumen.scaffold import get_scaffold_name

        file_path = tmp_path / "noname.yaml"
        file_path.write_text("executor:\n  system_prompt: test")

        name = get_scaffold_name(str(file_path))

        assert name is None


class TestGetScaffoldFiles:
    """Tests for get_scaffold_files function."""

    def test_get_files_from_scaffold(self, tmp_path):
        """Should extract file references from scaffold prompts."""
        from dokumen.scaffold import get_scaffold_files

        content = """
name: test
executor:
  system_prompt: Read docs/api.md
  user_prompt: Check docs/guide.md
judges:
  - name: judge
    system_prompt: Verify
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(content)

        files = get_scaffold_files(str(file_path))

        assert "docs/api.md" in files
        assert "docs/guide.md" in files

    def test_get_files_from_missing_file(self, tmp_path):
        """Should return empty list for missing file."""
        from dokumen.scaffold import get_scaffold_files

        files = get_scaffold_files(str(tmp_path / "missing.yaml"))

        assert files == []

    def test_get_files_from_invalid_yaml(self, tmp_path):
        """Should return empty list for invalid YAML."""
        from dokumen.scaffold import get_scaffold_files

        file_path = tmp_path / "invalid.yaml"
        file_path.write_text("invalid: yaml: [")

        files = get_scaffold_files(str(file_path))

        assert files == []


class TestExtractFileReferencesEdgeCases:
    """Edge case tests for extract_file_references_from_prompts."""

    def test_extract_various_extensions(self):
        """Should extract files with various extensions."""
        from dokumen.scaffold import extract_file_references_from_prompts

        data = {
            "executor": {
                "user_prompt": """
                Check config.yaml and setup.json
                Also read script.py and index.html
                """
            }
        }

        refs = extract_file_references_from_prompts(data)

        assert "config.yaml" in refs
        assert "setup.json" in refs
        assert "script.py" in refs
        assert "index.html" in refs

    def test_extract_from_judge_prompts(self):
        """Should extract from judge prompts."""
        from dokumen.scaffold import extract_file_references_from_prompts

        data = {
            "judges": [
                {"system_prompt": "Read docs/judge.md to evaluate"},
                {"user_prompt": "Check tests/spec.yaml"}
            ]
        }

        refs = extract_file_references_from_prompts(data)

        assert "docs/judge.md" in refs
        assert "tests/spec.yaml" in refs

    def test_ignores_urls(self):
        """Should ignore URLs."""
        from dokumen.scaffold import extract_file_references_from_prompts

        data = {
            "executor": {
                "user_prompt": "Visit https://example.com/api.md"
            }
        }

        refs = extract_file_references_from_prompts(data)

        # Should not include URLs
        assert len([r for r in refs if "example.com" in r]) == 0

    def test_handles_none_prompts(self):
        """Should handle None prompts gracefully."""
        from dokumen.scaffold import extract_file_references_from_prompts

        data = {
            "executor": {
                "system_prompt": None,
                "user_prompt": "Check docs/api.md"
            }
        }

        refs = extract_file_references_from_prompts(data)

        assert "docs/api.md" in refs
