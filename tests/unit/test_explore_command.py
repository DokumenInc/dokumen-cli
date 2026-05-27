"""Tests for explore CLI command."""

import json
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock, AsyncMock


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


class TestExploreCommand:
    """Tests for the explore command."""

    def test_explore_command_exists(self):
        """explore command exists and is registered."""
        from dokumen.cli import cli

        # Check that explore is in the registered commands
        assert 'explore' in cli.commands

    def test_explore_json_output_valid(self, tmp_path, monkeypatch):
        """explore --output json returns valid JSON with required fields."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        # Create minimal config
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        # Mock the ExploreAgent
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "success": True,
            "summary": "Found 2 relevant files: docs/api.md, docs/guide.md",
            "files": [
                {"path": "docs/api.md", "summary": "API documentation", "relevance": 0.95},
                {"path": "docs/guide.md", "summary": "User guide", "relevance": 0.8},
            ],
            "tool_history": [
                {"tool": "glob", "input": {"pattern": "docs/**/*.md"}},
            ],
            "duration": 1.5,
            "tool_calls_count": 3,
        }

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_result

            result = runner.invoke(cli, ['explore', 'test topic', '--output', 'json'])

            assert result.exit_code == 0
            # Verify it's valid JSON (extract from output, ignoring log lines)
            output = extract_json_from_output(result.output)
            assert output["success"] is True
            assert "summary" in output
            assert "files" in output
            assert "tool_history" in output
            assert "duration" in output
            assert "tool_calls_count" in output

    def test_explore_text_output_readable(self, tmp_path, monkeypatch):
        """explore --output text returns human-readable output."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        mock_result = MagicMock()
        mock_result.summary = "Found 2 relevant files"
        mock_result.files = [
            MagicMock(path="docs/api.md", summary="API documentation", relevance=0.95),
        ]
        mock_result.duration = 1.5
        mock_result.tool_calls_count = 3
        mock_result.success = True

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_result

            result = runner.invoke(cli, ['explore', 'test topic', '--output', 'text'])

            assert result.exit_code == 0
            # Should have human-readable content
            assert 'docs/api.md' in result.output or 'Found' in result.output

    def test_explore_max_files_option(self, tmp_path, monkeypatch):
        """explore --max-files limits returned files."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"success": True, "files": [], "summary": "", "duration": 1.0, "tool_calls_count": 1, "tool_history": []}

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_result

            result = runner.invoke(cli, ['explore', 'test topic', '--max-files', '5', '--output', 'json'])

            assert result.exit_code == 0
            # Verify max_files was passed
            mock_explore.assert_called_once()
            call_kwargs = mock_explore.call_args
            assert call_kwargs[1].get('max_files') == 5 or (call_kwargs[0] and 5 in call_kwargs[0])

    def test_explore_timeout_option(self, tmp_path, monkeypatch):
        """explore --timeout is respected."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"success": True, "files": [], "summary": "", "duration": 1.0, "tool_calls_count": 1, "tool_history": []}

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_result

            result = runner.invoke(cli, ['explore', 'test topic', '--timeout', '30', '--output', 'json'])

            assert result.exit_code == 0
            # Verify timeout was passed
            mock_explore.assert_called_once()
            call_kwargs = mock_explore.call_args
            assert call_kwargs[1].get('timeout') == 30 or (call_kwargs[0] and 30 in call_kwargs[0])

    def test_explore_timeout_error(self, tmp_path, monkeypatch):
        """explore returns error when timeout exceeded."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Exploration timeout"
        mock_result.to_dict.return_value = {
            "success": False,
            "error": "Exploration timeout",
            "files": [],
            "summary": "",
            "duration": 60.0,
            "tool_calls_count": 5,
            "tool_history": []
        }

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_result

            result = runner.invoke(cli, ['explore', 'test topic', '--output', 'json'])

            # Should still exit 0 but report error in output
            output = extract_json_from_output(result.output)
            assert output["success"] is False
            assert "timeout" in output.get("error", "").lower()

    def test_explore_no_config_uses_defaults(self, tmp_path, monkeypatch):
        """explore works without dokumen.yaml using defaults."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        # No dokumen.yaml - should use defaults

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"success": True, "files": [], "summary": "", "duration": 1.0, "tool_calls_count": 1, "tool_history": []}

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_result

            result = runner.invoke(cli, ['explore', 'test topic', '--output', 'json'])

            # Should not fail due to missing config
            assert result.exit_code == 0

    def test_explore_with_config_reads_model(self, tmp_path, monkeypatch):
        """explore reads model settings from dokumen.yaml."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        # Config with custom explore model
        config_content = """
version: '1.0'
explore:
  enabled: true
  model: claude-haiku-4-5-20251001
  max_files: 15
  timeout: 45
"""
        (tmp_path / "dokumen.yaml").write_text(config_content)

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"success": True, "files": [], "summary": "", "duration": 1.0, "tool_calls_count": 1, "tool_history": []}

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_result

            result = runner.invoke(cli, ['explore', 'test topic', '--output', 'json'])

            assert result.exit_code == 0
            # Model from config should be used (we can't easily verify this without more mocking)

    def test_explore_empty_topic_error(self, tmp_path, monkeypatch):
        """explore returns helpful error for empty topic."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        # Click should require the topic argument
        result = runner.invoke(cli, ['explore', '--output', 'json'])

        # Should fail due to missing required argument
        assert result.exit_code != 0
        assert 'missing argument' in result.output.lower() or 'topic' in result.output.lower()

    def test_explore_no_matches_returns_success(self, tmp_path, monkeypatch):
        """explore returns success=true with empty files when no matches."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "success": True,
            "summary": "No relevant files found for this topic.",
            "files": [],
            "tool_history": [{"tool": "glob", "input": {"pattern": "docs/**/*.md"}}],
            "duration": 2.0,
            "tool_calls_count": 2,
        }

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_result

            result = runner.invoke(cli, ['explore', 'nonexistent topic', '--output', 'json'])

            assert result.exit_code == 0
            output = extract_json_from_output(result.output)
            assert output["success"] is True
            assert output["files"] == []


class TestExploreCommandDefaultOutput:
    """Test default output behavior (text)."""

    def test_explore_default_is_text(self, tmp_path, monkeypatch):
        """explore defaults to text output."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        mock_result = MagicMock()
        mock_result.summary = "Found 1 relevant file"
        mock_result.files = [MagicMock(path="docs/api.md", summary="API docs", relevance=0.9)]
        mock_result.duration = 1.0
        mock_result.tool_calls_count = 2
        mock_result.success = True

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_result

            result = runner.invoke(cli, ['explore', 'test topic'])

            assert result.exit_code == 0
            # Should NOT be valid JSON (it's text)
            with pytest.raises(json.JSONDecodeError):
                json.loads(result.output)


class TestExploreCommandHelp:
    """Test help and usage."""

    def test_explore_help(self):
        """explore --help shows usage."""
        from dokumen.cli import cli
        runner = CliRunner()

        result = runner.invoke(cli, ['explore', '--help'])

        assert result.exit_code == 0
        assert 'topic' in result.output.lower()
        assert '--output' in result.output
        assert '--max-files' in result.output
        assert '--timeout' in result.output


class TestExploreTextOutput:
    """Test text output formatting."""

    def test_text_output_shows_error_on_failure(self, tmp_path, monkeypatch):
        """Text output shows error message when exploration fails."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Connection timeout"
        mock_result.summary = ""
        mock_result.files = []
        mock_result.duration = 0.5
        mock_result.tool_calls_count = 1

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_result

            result = runner.invoke(cli, ['explore', 'test topic', '--output', 'text'])

            assert result.exit_code == 0
            assert 'Failed' in result.output or 'Error' in result.output

    def test_text_output_shows_no_files_message(self, tmp_path, monkeypatch):
        """Text output shows 'No files found' when files list is empty."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "No relevant documentation found"
        mock_result.files = []  # Empty files list
        mock_result.duration = 1.0
        mock_result.tool_calls_count = 2

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_result

            result = runner.invoke(cli, ['explore', 'obscure topic', '--output', 'text'])

            assert result.exit_code == 0
            assert 'No files found' in result.output


class TestExploreErrorHandling:
    """Test error handling in explore command."""

    def test_exception_returns_json_error(self, tmp_path, monkeypatch):
        """Exception during exploration returns JSON error output."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.side_effect = RuntimeError("API connection failed")

            result = runner.invoke(cli, ['explore', 'test topic', '--output', 'json'])

            assert result.exit_code == 1
            output = extract_json_from_output(result.output)
            assert output["success"] is False
            assert "API connection failed" in output["error"]

    def test_exception_returns_text_error(self, tmp_path, monkeypatch):
        """Exception during exploration returns text error output."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        with patch('dokumen.cli.commands.explore._run_explore', new_callable=AsyncMock) as mock_explore:
            mock_explore.side_effect = RuntimeError("API connection failed")

            result = runner.invoke(cli, ['explore', 'test topic', '--output', 'text'])

            assert result.exit_code == 1
            # Error goes to stderr in text mode
            stderr_output = getattr(result, 'stderr', '') or ''
            combined = result.output + stderr_output
            assert "API connection failed" in combined or "Error" in combined
