"""Tests for summarize CLI command - TDD tests written first."""

import json
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path


class TestSummarizeCommandRegistration:
    """Tests for command registration."""

    def test_summarize_command_exists(self):
        """summarize command exists and is registered."""
        from dokumen.cli import cli

        assert "summarize" in cli.commands

    def test_summarize_in_main_commands(self):
        """summarize is listed in MAIN_COMMANDS."""
        from dokumen.cli import DokumenGroup

        assert "summarize" in DokumenGroup.MAIN_COMMANDS


class TestSummarizeCommand:
    """Tests for the summarize command execution."""

    def test_creates_index_file(self, tmp_path, monkeypatch):
        """summarize creates DOKUMEN_SUMMARIES_INDEX.md."""
        from dokumen.cli import cli
        from dokumen.summary_index import INDEX_FILENAME

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        # Create minimal config and a doc file
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'\ncoverage:\n  include: ['docs/**/*.md']")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "api.md").write_text("# API\nAuth docs.")

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "API documentation for auth.\n\n- Covers PAT auth",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

        with patch("dokumen.cli.commands.summarize.get_configured_provider", return_value=mock_provider):
            result = runner.invoke(cli, ["summarize"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert (tmp_path / INDEX_FILENAME).exists()

    def test_force_regenerates_all(self, tmp_path, monkeypatch):
        """--force regenerates all summaries."""
        from dokumen.cli import cli
        from dokumen.summary_index import INDEX_FILENAME

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        # Create config, doc, and existing index
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'\ncoverage:\n  include: ['docs/**/*.md']")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "api.md").write_text("# API\nAuth docs.")
        # Write an existing index
        (tmp_path / INDEX_FILENAME).write_text(
            "<!-- DOKUMEN SUMMARIES INDEX -->\n"
            "<!-- Generated at: old -->\n"
            "<!-- File count: 1 -->\n\n"
            "## docs/api.md\n"
            "<!-- hash: sha256:old_hash -->\n\n"
            "Old summary.\n"
        )

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "Fresh summary.",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

        with patch("dokumen.cli.commands.summarize.get_configured_provider", return_value=mock_provider):
            result = runner.invoke(cli, ["summarize", "--force"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        # Provider should be called (force = regenerate)
        assert mock_provider.complete.call_count >= 1

    def test_dry_run_does_not_write(self, tmp_path, monkeypatch):
        """--dry-run reports without writing."""
        from dokumen.cli import cli
        from dokumen.summary_index import INDEX_FILENAME

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'\ncoverage:\n  include: ['docs/**/*.md']")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "api.md").write_text("# API\nAuth docs.")

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "Summary.",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

        with patch("dokumen.cli.commands.summarize.get_configured_provider", return_value=mock_provider):
            result = runner.invoke(cli, ["summarize", "--dry-run"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        # Index file should NOT be created
        assert not (tmp_path / INDEX_FILENAME).exists()
        # Should report what would be done
        assert "dry run" in result.output.lower() or "would" in result.output.lower()

    def test_empty_project_no_files(self, tmp_path, monkeypatch):
        """Handles empty project (no matching files) gracefully."""
        from dokumen.cli import cli

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'\ncoverage:\n  include: ['docs/**/*.md']")

        mock_provider = AsyncMock()

        with patch("dokumen.cli.commands.summarize.get_configured_provider", return_value=mock_provider):
            result = runner.invoke(cli, ["summarize"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        # Provider should NOT be called (no files)
        assert mock_provider.complete.call_count == 0

    def test_incremental_update_skips_unchanged(self, tmp_path, monkeypatch):
        """Incremental update only processes changed files."""
        from dokumen.cli import cli
        from dokumen.summary_index import (
            INDEX_FILENAME,
            compute_content_hash,
            render_summary_index,
            SummaryIndex,
            FileSummaryEntry,
        )

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        (tmp_path / "dokumen.yaml").write_text("version: '1.0'\ncoverage:\n  include: ['docs/**/*.md']")
        (tmp_path / "docs").mkdir()

        unchanged_content = "# Unchanged\nStill the same."
        (tmp_path / "docs" / "unchanged.md").write_text(unchanged_content)
        (tmp_path / "docs" / "new.md").write_text("# New\nBrand new file.")

        # Write existing index with unchanged.md already summarized
        existing = SummaryIndex(
            entries={
                "docs/unchanged.md": FileSummaryEntry(
                    file_path="docs/unchanged.md",
                    content_hash=compute_content_hash(unchanged_content),
                    summary_text="Existing summary.",
                ),
            },
            generated_at="old",
            version="1.0",
        )
        (tmp_path / INDEX_FILENAME).write_text(render_summary_index(existing))

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "New file summary.",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

        with patch("dokumen.cli.commands.summarize.get_configured_provider", return_value=mock_provider):
            result = runner.invoke(cli, ["summarize"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        # Should only call provider once (for the new file)
        assert mock_provider.complete.call_count == 1

    def test_respects_coverage_include_patterns(self, tmp_path, monkeypatch):
        """Respects coverage.include patterns from config."""
        from dokumen.cli import cli

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        # Only include docs/policies/
        (tmp_path / "dokumen.yaml").write_text(
            "version: '1.0'\ncoverage:\n  include: ['docs/policies/**/*.md']"
        )
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "policies").mkdir()
        (tmp_path / "docs" / "policies" / "refund.md").write_text("# Refund\nPolicy.")
        (tmp_path / "docs" / "other.md").write_text("# Other\nShould be ignored.")

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "Refund policy summary.",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

        with patch("dokumen.cli.commands.summarize.get_configured_provider", return_value=mock_provider):
            result = runner.invoke(cli, ["summarize"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        # Should only process the refund.md file
        assert mock_provider.complete.call_count == 1
