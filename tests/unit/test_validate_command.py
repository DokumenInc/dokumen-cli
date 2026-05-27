"""
Tests for the `dokumen validate` command.

TDD: These tests are written first, before implementation.
"""
import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner


def extract_json_from_output(output: str) -> dict:
    """Extract JSON from CLI output, ignoring log lines."""
    # Find the first '{' which starts JSON
    start_idx = output.find('{')
    if start_idx == -1:
        return json.loads(output)

    # Try to parse from the start position
    decoder = json.JSONDecoder()
    try:
        result, end_idx = decoder.raw_decode(output[start_idx:])
        return result
    except json.JSONDecodeError:
        return json.loads(output)


@pytest.fixture
def runner():
    """Click CLI test runner with separate stderr to avoid log contamination."""
    return CliRunner()


@pytest.fixture
def valid_project(tmp_path: Path, valid_config_path: Path, valid_minimal_scaffold_path: Path):
    """Create valid project structure with config and test scaffolds."""
    # Copy config
    config = tmp_path / "dokumen.yaml"
    config.write_text(valid_config_path.read_text())

    # Create docs directory with referenced files
    docs_dir = tmp_path / "docs" / "policies"
    docs_dir.mkdir(parents=True)
    (docs_dir / "refund.md").write_text("# Refund Policy\n\nRefund content.")

    # Create tests directory with scaffold
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    scaffold = tests_dir / "sample.test.yaml"
    scaffold.write_text(valid_minimal_scaffold_path.read_text())

    return tmp_path


@pytest.fixture
def invalid_config_project(tmp_path: Path, invalid_config_path: Path):
    """Create project with invalid YAML config."""
    config = tmp_path / "dokumen.yaml"
    config.write_text(invalid_config_path.read_text())
    return tmp_path


@pytest.fixture
def missing_provider_project(tmp_path: Path, missing_provider_config_path: Path):
    """Create project missing required provider section."""
    config = tmp_path / "dokumen.yaml"
    config.write_text(missing_provider_config_path.read_text())
    return tmp_path


@pytest.fixture
def invalid_scaffold_project(tmp_path: Path, valid_config_path: Path, invalid_missing_executor_path: Path):
    """Create project with valid config but invalid test scaffold."""
    # Copy config
    config = tmp_path / "dokumen.yaml"
    config.write_text(valid_config_path.read_text())

    # Create tests directory with invalid scaffold
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    scaffold = tests_dir / "broken.test.yaml"
    scaffold.write_text(invalid_missing_executor_path.read_text())

    return tmp_path


@pytest.fixture
def empty_project(tmp_path: Path, valid_config_path: Path):
    """Create project with config but no test scaffolds."""
    config = tmp_path / "dokumen.yaml"
    config.write_text(valid_config_path.read_text())

    # Create empty tests directory
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    return tmp_path


@pytest.fixture
def warning_project(tmp_path: Path, valid_config_path: Path):
    """Create project with scaffold that has warnings but no errors.

    Uses a scaffold with missing judge system_prompt (warning, not error).
    """
    config = tmp_path / "dokumen.yaml"
    config.write_text(valid_config_path.read_text())

    # Create docs directory with referenced files
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.md").write_text("# Test\n\nTest content.")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    scaffold = tests_dir / "warning.test.yaml"
    scaffold.write_text("""
name: warning-test
files:
  - path: docs/test.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Validate the document."
  tools:
    - read_file
judges:
  - name: accuracy
    # Missing system_prompt - should generate warning
""")

    return tmp_path


class TestValidateSuccess:
    """Tests for successful validation scenarios."""

    def test_validate_all_success(self, runner: CliRunner, valid_project: Path):
        """Validate entire project successfully."""
        from dokumen.cli import cli

        os.chdir(valid_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "pass" in result.output.lower()

    def test_validate_config_only_success(self, runner: CliRunner, valid_project: Path):
        """Validate only config, skip test files."""
        from dokumen.cli import cli

        os.chdir(valid_project)
        result = runner.invoke(cli, ["validate", "--config-only"])

        assert result.exit_code == 0

    def test_validate_specific_file_success(
        self, runner: CliRunner, valid_project: Path, valid_minimal_scaffold_path: Path
    ):
        """Validate single test file."""
        from dokumen.cli import cli

        os.chdir(valid_project)
        result = runner.invoke(cli, ["validate", "tests/sample.test.yaml"])

        assert result.exit_code == 0

    def test_validate_empty_tests_dir(self, runner: CliRunner, empty_project: Path):
        """Validate project with no test scaffolds - still valid."""
        from dokumen.cli import cli

        os.chdir(empty_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 0


class TestValidateErrors:
    """Tests for validation error detection."""

    def test_validate_finds_yaml_errors(self, runner: CliRunner, invalid_config_project: Path):
        """Detect YAML syntax errors in config."""
        from dokumen.cli import cli

        os.chdir(invalid_config_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 2
        assert "yaml" in result.output.lower() or "parse" in result.output.lower() or "invalid" in result.output.lower()

    def test_validate_finds_missing_required_fields(self, runner: CliRunner, missing_provider_project: Path):
        """Detect missing required config fields."""
        from dokumen.cli import cli

        os.chdir(missing_provider_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 2
        assert "provider" in result.output.lower() or "required" in result.output.lower()

    def test_validate_finds_invalid_test_schema(self, runner: CliRunner, invalid_scaffold_project: Path):
        """Detect invalid test scaffold schema (missing executor)."""
        from dokumen.cli import cli

        os.chdir(invalid_scaffold_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 2
        assert "executor" in result.output.lower() or "required" in result.output.lower()

    def test_validate_nonexistent_file(self, runner: CliRunner, valid_project: Path):
        """Error for specified file that doesn't exist."""
        from dokumen.cli import cli

        os.chdir(valid_project)
        result = runner.invoke(cli, ["validate", "tests/nonexistent.test.yaml"])

        assert result.exit_code == 2
        assert "not found" in result.output.lower() or "no such file" in result.output.lower()

    def test_validate_no_config_file(self, runner: CliRunner, tmp_path: Path):
        """Error when dokumen.yaml not found."""
        from dokumen.cli import cli

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 2
        assert "not found" in result.output.lower() or "configuration" in result.output.lower()


class TestValidateConfigOnly:
    """Tests for --config-only option."""

    def test_config_only_skips_scaffold_errors(
        self, runner: CliRunner, invalid_scaffold_project: Path
    ):
        """--config-only ignores scaffold errors."""
        from dokumen.cli import cli

        os.chdir(invalid_scaffold_project)
        result = runner.invoke(cli, ["validate", "--config-only"])

        # Config is valid, so should pass even though scaffolds are invalid
        assert result.exit_code == 0

    def test_config_only_still_catches_config_errors(
        self, runner: CliRunner, invalid_config_project: Path
    ):
        """--config-only still catches config errors."""
        from dokumen.cli import cli

        os.chdir(invalid_config_project)
        result = runner.invoke(cli, ["validate", "--config-only"])

        assert result.exit_code == 2


class TestValidateOutput:
    """Tests for validation output formats."""

    def test_validate_output_text_default(self, runner: CliRunner, valid_project: Path):
        """Default text output."""
        from dokumen.cli import cli

        os.chdir(valid_project)
        result = runner.invoke(cli, ["validate"])

        # Should have human-readable output
        assert len(result.output) > 0
        assert result.exit_code == 0

    def test_validate_output_json(self, runner: CliRunner, valid_project: Path):
        """JSON output of validation results."""
        from dokumen.cli import cli

        os.chdir(valid_project)
        result = runner.invoke(cli, ["validate", "--json"])

        assert result.exit_code == 0
        data = extract_json_from_output(result.output)
        assert "valid" in data or "errors" in data

    def test_validate_json_with_errors(self, runner: CliRunner, invalid_scaffold_project: Path):
        """JSON output includes error details."""
        from dokumen.cli import cli

        os.chdir(invalid_scaffold_project)
        result = runner.invoke(cli, ["validate", "--json"])

        assert result.exit_code == 2
        data = extract_json_from_output(result.output)
        assert "errors" in data or "valid" in data
        if "valid" in data:
            assert data["valid"] is False


class TestValidateWarnings:
    """Tests for validation warnings."""

    def test_validate_shows_warnings(self, runner: CliRunner, warning_project: Path):
        """Warnings shown but still valid (exit 0)."""
        from dokumen.cli import cli

        os.chdir(warning_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 0
        # Should mention warning about missing judge system_prompt
        assert "warning" in result.output.lower() or "system_prompt" in result.output.lower() or "judge" in result.output.lower()

    def test_validate_json_includes_warnings(self, runner: CliRunner, warning_project: Path):
        """JSON output includes warnings array."""
        from dokumen.cli import cli

        os.chdir(warning_project)
        result = runner.invoke(cli, ["validate", "--json"])

        assert result.exit_code == 0
        data = extract_json_from_output(result.output)
        assert "warnings" in data


class TestValidateDocFiles:
    """Tests for validating referenced documentation files."""

    @pytest.fixture
    def missing_doc_project(self, tmp_path: Path, valid_config_path: Path, missing_doc_file_scaffold_path: Path):
        """Create project with scaffold that references nonexistent doc file."""
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        scaffold = tests_dir / "missing-doc.test.yaml"
        scaffold.write_text(missing_doc_file_scaffold_path.read_text())

        return tmp_path

    @pytest.fixture
    def invalid_tool_project(self, tmp_path: Path, valid_config_path: Path, invalid_tool_scaffold_path: Path):
        """Create project with scaffold that uses invalid tool."""
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        scaffold = tests_dir / "invalid-tool.test.yaml"
        scaffold.write_text(invalid_tool_scaffold_path.read_text())

        return tmp_path

    @pytest.fixture
    def valid_tools_project(self, tmp_path: Path, valid_config_path: Path, valid_minimal_scaffold_path: Path):
        """Create project with scaffold that uses only valid tools."""
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        # Create docs directory with referenced files
        docs_dir = tmp_path / "docs" / "policies"
        docs_dir.mkdir(parents=True)
        (docs_dir / "refund.md").write_text("# Refund Policy\n\nRefund content.")

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        scaffold = tests_dir / "valid-tools.test.yaml"
        scaffold.write_text(valid_minimal_scaffold_path.read_text())

        return tmp_path

    def test_validate_fails_missing_doc_file(self, runner: CliRunner, missing_doc_project: Path):
        """Validate fails when scaffold references nonexistent doc file."""
        from dokumen.cli import cli

        os.chdir(missing_doc_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 2
        assert "not found" in result.output.lower() or "nonexistent" in result.output.lower()

    def test_validate_fails_invalid_tool_name(self, runner: CliRunner, invalid_tool_project: Path):
        """Validate fails when scaffold uses invalid tool name."""
        from dokumen.cli import cli

        os.chdir(invalid_tool_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 2
        assert "unknown" in result.output.lower() or "invalid" in result.output.lower() or "delete_everything" in result.output.lower()

    def test_validate_passes_valid_tools(self, runner: CliRunner, valid_tools_project: Path):
        """Validate passes when scaffold uses only valid tools."""
        from dokumen.cli import cli

        os.chdir(valid_tools_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 0


class TestValidateFiltering:
    """Tests for test filtering in validate command."""

    @pytest.fixture
    def mixed_project(self, tmp_path: Path, valid_config_path: Path):
        """Create project with multiple scaffolds - one valid, one invalid."""
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create docs directory with test files
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test\n\nTest content.")

        # Valid scaffold
        valid_scaffold = tests_dir / "good-test.test.yaml"
        valid_scaffold.write_text("""
name: good-test
reason: A valid test
files:
  - path: docs/test.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Do something
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate it
""")

        # Invalid scaffold (missing executor)
        invalid_scaffold = tests_dir / "bad-test.test.yaml"
        invalid_scaffold.write_text("""
name: bad-test
reason: Missing executor
files:
  - path: docs/test.md
judges:
  - name: judge
    system_prompt: Evaluate it
""")

        return tmp_path

    def test_validate_by_test_name(self, runner: CliRunner, mixed_project: Path):
        """Validate specific test by name - should pass even if other tests are broken."""
        from dokumen.cli import cli

        os.chdir(mixed_project)
        result = runner.invoke(cli, ["validate", "good-test"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"

    def test_validate_by_grep_pattern(self, runner: CliRunner, mixed_project: Path):
        """Validate tests matching grep pattern."""
        from dokumen.cli import cli

        os.chdir(mixed_project)
        result = runner.invoke(cli, ["validate", "--grep", "good-*"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"

    def test_validate_with_dokumen_tests_env(self, runner: CliRunner, mixed_project: Path, monkeypatch):
        """Validate tests specified in DOKUMEN_TESTS env var."""
        from dokumen.cli import cli

        os.chdir(mixed_project)
        monkeypatch.setenv("DOKUMEN_TESTS", "good-test")
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"

    def test_validate_all_fails_when_any_invalid(self, runner: CliRunner, mixed_project: Path):
        """Validate all tests fails when any test is invalid."""
        from dokumen.cli import cli

        os.chdir(mixed_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 2, f"Expected failure, got: {result.output}"

    def test_validate_cli_args_override_env_var(self, runner: CliRunner, mixed_project: Path, monkeypatch):
        """CLI test names override DOKUMEN_TESTS env var."""
        from dokumen.cli import cli

        os.chdir(mixed_project)
        monkeypatch.setenv("DOKUMEN_TESTS", "bad-test")
        # CLI arg should override env var
        result = runner.invoke(cli, ["validate", "good-test"])

        assert result.exit_code == 0, f"Expected success (CLI override), got: {result.output}"

    def test_validate_file_path_still_works(self, runner: CliRunner, mixed_project: Path):
        """Validate by file path still works (backwards compatibility)."""
        from dokumen.cli import cli

        os.chdir(mixed_project)
        result = runner.invoke(cli, ["validate", "tests/good-test.test.yaml"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"

    def test_validate_multiple_test_names(self, runner: CliRunner, mixed_project: Path):
        """Validate multiple specific tests by name."""
        from dokumen.cli import cli

        # Add another valid test
        another_valid = mixed_project / "tests" / "another-good.test.yaml"
        another_valid.write_text("""
name: another-good
reason: Another valid test
files:
  - path: docs/test.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Do something else
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate it
""")

        os.chdir(mixed_project)
        result = runner.invoke(cli, ["validate", "good-test", "another-good"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"

    def test_validate_grep_with_wildcard(self, runner: CliRunner, mixed_project: Path):
        """Validate tests matching wildcard pattern."""
        from dokumen.cli import cli

        # Add another good test
        another_valid = mixed_project / "tests" / "good-api-test.test.yaml"
        another_valid.write_text("""
name: good-api-test
reason: An API test
files:
  - path: docs/test.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Test API
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate it
""")

        os.chdir(mixed_project)
        result = runner.invoke(cli, ["validate", "--grep", "good-*"])

        # Should match both good-test and good-api-test, both valid
        assert result.exit_code == 0, f"Expected success, got: {result.output}"

    def test_validate_nonexistent_test_name(self, runner: CliRunner, mixed_project: Path):
        """Validate nonexistent test name fails gracefully."""
        from dokumen.cli import cli

        os.chdir(mixed_project)
        result = runner.invoke(cli, ["validate", "nonexistent-test"])

        # Should fail - no tests matched
        assert result.exit_code == 2

    def test_validate_comma_separated_dokumen_tests(self, runner: CliRunner, mixed_project: Path, monkeypatch):
        """DOKUMEN_TESTS env var with comma-separated values."""
        from dokumen.cli import cli

        # Add another valid test
        another_valid = mixed_project / "tests" / "another-good.test.yaml"
        another_valid.write_text("""
name: another-good
reason: Another valid test
files:
  - path: docs/test.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Do something else
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate it
""")

        os.chdir(mixed_project)
        monkeypatch.setenv("DOKUMEN_TESTS", "good-test,another-good")
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"


class TestValidatePDFConstraints:
    """Tests for PDF-specific validation."""

    def test_validate_detects_oversized_pdfs(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """validate reports PDFs exceeding size limit."""
        from dokumen.cli import cli
        from tests.conftest import create_large_pdf

        # Setup project
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        # Create docs directory with large PDF
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        large_pdf = docs_dir / "big.pdf"
        large_pdf.write_bytes(create_large_pdf(5.0))  # 5MB

        # Create test scaffold referencing it
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test.test.yaml"
        test_file.write_text("""
name: test-pdf
reason: Test large PDF
files:
  - path: docs/big.pdf
executor:
  system_prompt: Test
  user_prompt: Test
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Judge
""")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["validate"])

        # Should show warning about large PDF
        assert "warning" in result.output.lower() or "⚠" in result.output
        assert "big.pdf" in result.output
        assert "5.0" in result.output
        assert "4.5" in result.output

    def test_validate_detects_multiple_pdfs_per_test(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """validate warns about tests with >5 PDFs."""
        from dokumen.cli import cli
        from tests.conftest import create_minimal_pdf

        # Setup project
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        # Create docs directory with 6 small PDFs
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        for i in range(6):
            pdf_path = docs_dir / f"doc{i}.pdf"
            pdf_path.write_bytes(create_minimal_pdf())

        # Create test scaffold referencing all 6
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test.test.yaml"
        test_file.write_text("""
name: multi-pdf-test
reason: Test multiple PDFs
files:
  - path: docs/doc0.pdf
  - path: docs/doc1.pdf
  - path: docs/doc2.pdf
  - path: docs/doc3.pdf
  - path: docs/doc4.pdf
  - path: docs/doc5.pdf
executor:
  system_prompt: Test
  user_prompt: Test
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Judge
""")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["validate"])

        # Should show warning about too many PDFs
        assert "warning" in result.output.lower() or "⚠" in result.output
        assert "6" in result.output
        assert "5" in result.output

    def test_validate_accepts_pdfs_within_limits(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """validate accepts PDFs within size and count limits."""
        from dokumen.cli import cli
        from tests.conftest import create_minimal_pdf

        # Setup project
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        # Create docs directory with 3 small PDFs
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        for i in range(3):
            pdf_path = docs_dir / f"doc{i}.pdf"
            pdf_path.write_bytes(create_minimal_pdf())

        # Create test scaffold referencing 3 PDFs
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test.test.yaml"
        test_file.write_text("""
name: good-pdf-test
reason: Test valid PDFs
files:
  - path: docs/doc0.pdf
  - path: docs/doc1.pdf
  - path: docs/doc2.pdf
executor:
  system_prompt: "@prompts/general.txt"
  user_prompt: Test validation
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate the output
""")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["validate"])

        # Should succeed without warnings about PDFs
        assert result.exit_code == 0
        assert "✓ Validation passed" in result.output

    def test_validate_pdf_at_size_limit(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """validate accepts PDF at exactly 4.5MB."""
        from dokumen.cli import cli
        from tests.conftest import create_large_pdf

        # Setup project
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        # Create docs directory with PDF at limit
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        pdf = docs_dir / "at_limit.pdf"
        pdf.write_bytes(create_large_pdf(4.5))

        # Create test scaffold
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test.test.yaml"
        test_file.write_text("""
name: limit-test
reason: Test PDF at limit
files:
  - path: docs/at_limit.pdf
executor:
  system_prompt: "@prompts/general.txt"
  user_prompt: Test validation
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate the output
""")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["validate"])

        # Should succeed without size warnings
        assert result.exit_code == 0
        assert "exceeds" not in result.output.lower()

    def test_validate_nonexistent_pdf_referenced(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """validate handles missing PDF files gracefully."""
        from dokumen.cli import cli

        # Setup project
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        # Create test scaffold referencing nonexistent PDF
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test.test.yaml"
        test_file.write_text("""
name: missing-pdf-test
reason: Test missing PDF
files:
  - path: docs/missing.pdf
executor:
  system_prompt: "@prompts/general.txt"
  user_prompt: Test validation
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate the output
""")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["validate"])

        # Should still validate (file existence is checked at runtime, not validation time)
        # Just ensure it doesn't crash
        assert result.exit_code == 0 or "missing.pdf" in result.output


class TestValidateVerbose:
    """Tests for --verbose flag showing detailed error context."""

    @pytest.fixture
    def verbose_error_project(self, tmp_path: Path, valid_config_path: Path):
        """Create project with various validation errors for verbose testing."""
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create scaffold with missing file reference
        scaffold = tests_dir / "missing-file.test.yaml"
        scaffold.write_text("""
name: missing-file-test
reason: Test missing file
files:
  - path: docs/nonexistent.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Validate the doc
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate it
""")

        return tmp_path

    @pytest.fixture
    def verbose_unknown_tool_project(self, tmp_path: Path, valid_config_path: Path):
        """Create project with unknown tool error."""
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test\n\nTest content.")

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        scaffold = tests_dir / "unknown-tool.test.yaml"
        scaffold.write_text("""
name: unknown-tool-test
reason: Test unknown tool
files:
  - path: docs/test.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Validate the doc
  tools:
    - read_file
    - magic_wand
judges:
  - name: judge
    system_prompt: Evaluate it
""")

        return tmp_path

    def test_verbose_flag_exists(self, runner: CliRunner, valid_project: Path):
        """--verbose flag is accepted."""
        from dokumen.cli import cli

        os.chdir(valid_project)
        result = runner.invoke(cli, ["validate", "--verbose"])

        assert result.exit_code == 0

    def test_verbose_shows_more_detail(self, runner: CliRunner, verbose_error_project: Path):
        """--verbose shows more detailed error information."""
        from dokumen.cli import cli

        os.chdir(verbose_error_project)

        # Without verbose
        result_normal = runner.invoke(cli, ["validate"])

        # With verbose
        result_verbose = runner.invoke(cli, ["validate", "--verbose"])

        # Both should fail
        assert result_normal.exit_code == 2
        assert result_verbose.exit_code == 2

        # Verbose output should be longer with more context
        assert len(result_verbose.output) >= len(result_normal.output)

    def test_verbose_shows_file_content_snippet(self, runner: CliRunner, verbose_error_project: Path):
        """--verbose shows relevant file content snippet for context."""
        from dokumen.cli import cli

        os.chdir(verbose_error_project)
        result = runner.invoke(cli, ["validate", "--verbose"])

        # Should show the problematic line or context
        assert "docs/nonexistent.md" in result.output or "nonexistent" in result.output.lower()

    def test_verbose_shows_suggested_fix(self, runner: CliRunner, verbose_unknown_tool_project: Path):
        """--verbose shows suggested fixes for common errors."""
        from dokumen.cli import cli

        os.chdir(verbose_unknown_tool_project)
        result = runner.invoke(cli, ["validate", "--verbose"])

        # Should show suggestion for unknown tool
        assert result.exit_code == 2
        # Either shows "magic_wand" or suggests valid tools
        assert "magic_wand" in result.output or "valid" in result.output.lower()

    def test_verbose_json_includes_suggestions(self, runner: CliRunner, verbose_error_project: Path):
        """--verbose with --json includes suggestions in output."""
        from dokumen.cli import cli

        os.chdir(verbose_error_project)
        result = runner.invoke(cli, ["validate", "--verbose", "--json"])

        assert result.exit_code == 2
        data = extract_json_from_output(result.output)

        # Should have verbose details in JSON
        assert "errors" in data
        # Verbose mode may include additional context
        assert len(data.get("errors", [])) > 0

    def test_verbose_short_flag(self, runner: CliRunner, valid_project: Path):
        """-v short flag works for verbose."""
        from dokumen.cli import cli

        os.chdir(valid_project)
        result = runner.invoke(cli, ["validate", "-v"])

        assert result.exit_code == 0


class TestValidateCICompat:
    """Tests for CI compatibility checks in validate command."""

    @pytest.fixture
    def ci_disallowed_tool_project(self, tmp_path: Path):
        """Create project with allowed tools config and a scaffold using a disallowed tool."""
        config = tmp_path / "dokumen.yaml"
        config.write_text("""
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
tools:
  allowed:
    - read_file
    - glob
""")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test\n\nContent.")

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        scaffold = tests_dir / "ci-tool.test.yaml"
        scaffold.write_text("""
name: ci-tool-test
reason: Test with disallowed tool
files:
  - path: docs/test.md
executor:
  system_prompt: "@prompts/general.txt"
  user_prompt: Validate the doc
  tools:
    - read_file
    - web_fetch
judges:
  - name: judge
    system_prompt: Evaluate it
""")
        return tmp_path

    @pytest.fixture
    def ci_missing_file_project(self, tmp_path: Path):
        """Create project with scaffold referencing a file that doesn't exist."""
        config = tmp_path / "dokumen.yaml"
        config.write_text("""
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
""")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        scaffold = tests_dir / "ci-missing.test.yaml"
        scaffold.write_text("""
name: ci-missing-test
reason: Test with missing file
files:
  - path: docs/nonexistent-ci.md
executor:
  system_prompt: "@prompts/general.txt"
  user_prompt: Validate the doc
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate it
""")
        return tmp_path

    def test_ci_error_for_disallowed_tool(self, runner: CliRunner, ci_disallowed_tool_project: Path):
        """CI compat check detects tools not in tools.allowed."""
        from dokumen.cli import cli

        os.chdir(ci_disallowed_tool_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 2
        assert "[CI]" in result.output
        assert "web_fetch" in result.output

    def test_ci_error_for_missing_file(self, runner: CliRunner, ci_missing_file_project: Path):
        """CI compat check detects files that don't exist in the repo."""
        from dokumen.cli import cli

        os.chdir(ci_missing_file_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 2
        assert "[CI]" in result.output
        assert "nonexistent-ci.md" in result.output

    def test_ci_error_verbose_shows_suggestion(self, runner: CliRunner, ci_disallowed_tool_project: Path):
        """CI errors show suggestions in verbose mode."""
        from dokumen.cli import cli

        os.chdir(ci_disallowed_tool_project)
        result = runner.invoke(cli, ["validate", "--verbose"])

        assert result.exit_code == 2
        assert "[CI]" in result.output
        # Should show suggestion for disallowed tool
        assert "dokumen.yaml" in result.output
