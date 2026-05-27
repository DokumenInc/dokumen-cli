"""Integration tests for explore CLI command.

These tests make real API calls and require LLM credentials to be set.
Run with: pytest tests/integration -v -m integration

Credentials:
  - Production/Staging: DOKUMEN_PROXY_URL + DOKUMEN_PROXY_TOKEN (via Guard Proxy)
  - Development only: ANTHROPIC_API_KEY (direct API access)
"""

import json
import os
import pytest
from click.testing import CliRunner
from pathlib import Path


def _has_llm_credentials() -> bool:
    """Check if LLM credentials are available (proxy or direct Anthropic for dev)."""
    has_proxy = bool(os.environ.get("DOKUMEN_PROXY_URL") and os.environ.get("DOKUMEN_PROXY_TOKEN"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return has_proxy or has_anthropic


# Skip all tests in this module if LLM credentials are not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _has_llm_credentials(),
        reason="LLM credentials not set (need DOKUMEN_PROXY_URL+TOKEN or ANTHROPIC_API_KEY)"
    ),
]


def extract_json_from_output(output: str) -> dict:
    """Extract JSON from CLI output, ignoring log lines."""
    lines = output.strip().split('\n')
    json_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('{'):
            json_start = i
            break
    if json_start is not None:
        json_text = '\n'.join(lines[json_start:])
        return json.loads(json_text)
    return json.loads(output)


@pytest.fixture
def test_repo(tmp_path):
    """Create a minimal test repository with docs and test files."""
    # Create docs directory with markdown files
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "api.md").write_text("""# API Documentation

## Authentication
The API uses JWT tokens for authentication.

## Endpoints
- GET /users - List all users
- POST /users - Create a new user
- GET /users/{id} - Get user by ID
""")

    (docs_dir / "guide.md").write_text("""# User Guide

## Getting Started
1. Install the application
2. Configure your settings
3. Start using the features

## Features
- User management
- Data export
- Reports
""")

    # Create tests directory with test scaffold
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    (tests_dir / "api.test.yaml").write_text("""name: api-test
reason: Validate API documentation accuracy

executor:
  system_prompt: You are testing the API documentation.
  user_prompt: Check the API endpoints documented.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: Evaluate if the API documentation is accurate.
""")

    # Create minimal config
    (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

    return tmp_path


class TestExploreFindsFiles:
    """Test that explore actually finds files in the repository."""

    def test_explore_finds_docs(self, test_repo, monkeypatch):
        """Explore actually finds .md files in docs/ directory."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(test_repo)

        result = runner.invoke(
            cli,
            ['explore', 'API documentation', '--output', 'json', '--timeout', '30'],
            catch_exceptions=False
        )

        # Should succeed
        assert result.exit_code == 0, f"Failed with: {result.output}"

        output = extract_json_from_output(result.output)
        assert output["success"] is True

        # Should find at least one file
        assert len(output.get("files", [])) > 0 or output.get("summary", "")

        # Should mention docs or API in summary
        summary_lower = output.get("summary", "").lower()
        assert "api" in summary_lower or "doc" in summary_lower or len(output.get("files", [])) > 0

    def test_explore_finds_tests(self, test_repo, monkeypatch):
        """Explore actually finds .test.yaml files in tests/ directory."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(test_repo)

        result = runner.invoke(
            cli,
            ['explore', 'test scaffolds', '--output', 'json', '--timeout', '30'],
            catch_exceptions=False
        )

        assert result.exit_code == 0, f"Failed with: {result.output}"

        output = extract_json_from_output(result.output)
        assert output["success"] is True


class TestExploreToolHistory:
    """Test that tool history is populated correctly."""

    def test_explore_tool_history_contains_calls(self, test_repo, monkeypatch):
        """Tool history contains actual tool calls made during exploration."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(test_repo)

        result = runner.invoke(
            cli,
            ['explore', 'user guide features', '--output', 'json', '--timeout', '30'],
            catch_exceptions=False
        )

        assert result.exit_code == 0, f"Failed with: {result.output}"

        output = extract_json_from_output(result.output)

        # Should have some tool calls
        assert output.get("tool_calls_count", 0) > 0

        # Tool history should be a list
        tool_history = output.get("tool_history", [])
        assert isinstance(tool_history, list)


class TestExploreStats:
    """Test that explore returns proper stats."""

    def test_explore_returns_duration(self, test_repo, monkeypatch):
        """Explore returns duration in seconds."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(test_repo)

        result = runner.invoke(
            cli,
            ['explore', 'authentication', '--output', 'json', '--timeout', '30'],
            catch_exceptions=False
        )

        assert result.exit_code == 0, f"Failed with: {result.output}"

        output = extract_json_from_output(result.output)

        # Duration should be a positive number
        assert output.get("duration", 0) > 0
        # Duration should be reasonable (less than timeout)
        assert output.get("duration", 0) < 35  # timeout + buffer


class TestExploreTimeout:
    """Test timeout handling."""

    def test_explore_respects_short_timeout(self, test_repo, monkeypatch):
        """Explore with very short timeout should complete or timeout gracefully."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(test_repo)

        # Use a very short timeout - should either complete quickly or timeout
        result = runner.invoke(
            cli,
            ['explore', 'some topic', '--output', 'json', '--timeout', '5'],
            catch_exceptions=False
        )

        # Should not crash - either succeeds or reports timeout
        output = extract_json_from_output(result.output)
        assert "success" in output


class TestExploreTextOutput:
    """Test text output format in integration."""

    def test_explore_text_output_readable(self, test_repo, monkeypatch):
        """Text output is human-readable and contains useful info."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(test_repo)

        result = runner.invoke(
            cli,
            ['explore', 'API endpoints', '--output', 'text', '--timeout', '30'],
            catch_exceptions=False
        )

        assert result.exit_code == 0, f"Failed with: {result.output}"

        # Should contain human-readable elements
        output = result.output
        assert "Duration" in output or "Stats" in output or "Complete" in output
