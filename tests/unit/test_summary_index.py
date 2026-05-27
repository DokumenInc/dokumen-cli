"""Tests for summary_index module - TDD tests written first."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_deterministic(self):
        """Same content produces same hash."""
        from dokumen.summary_index import compute_content_hash

        content = "Hello, world!"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content produces different hash."""
        from dokumen.summary_index import compute_content_hash

        hash1 = compute_content_hash("Hello")
        hash2 = compute_content_hash("World")
        assert hash1 != hash2

    def test_format_sha256_prefix(self):
        """Hash includes sha256: prefix."""
        from dokumen.summary_index import compute_content_hash

        result = compute_content_hash("test")
        assert result.startswith("sha256:")
        # SHA-256 hex is 64 chars
        assert len(result.split(":", 1)[1]) == 64

    def test_empty_content(self):
        """Empty content produces a valid hash."""
        from dokumen.summary_index import compute_content_hash

        result = compute_content_hash("")
        assert result.startswith("sha256:")

    def test_matches_backend_format(self):
        """Hash matches the backend compute_content_hash format."""
        from dokumen.summary_index import compute_content_hash
        import hashlib

        content = "test content for hashing"
        expected = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
        assert compute_content_hash(content) == expected


class TestFileSummaryEntry:
    """Tests for FileSummaryEntry dataclass."""

    def test_creation(self):
        """FileSummaryEntry can be created."""
        from dokumen.summary_index import FileSummaryEntry

        entry = FileSummaryEntry(
            file_path="docs/api.md",
            content_hash="sha256:abc123",
            summary_text="API documentation for authentication.",
        )
        assert entry.file_path == "docs/api.md"
        assert entry.content_hash == "sha256:abc123"
        assert entry.summary_text == "API documentation for authentication."


class TestSummaryIndex:
    """Tests for SummaryIndex dataclass."""

    def test_creation(self):
        """SummaryIndex can be created."""
        from dokumen.summary_index import SummaryIndex, FileSummaryEntry

        entry = FileSummaryEntry(
            file_path="docs/api.md",
            content_hash="sha256:abc",
            summary_text="API docs.",
        )
        index = SummaryIndex(
            entries={"docs/api.md": entry},
            generated_at="2026-02-19T14:30:00Z",
            version="1.0",
        )
        assert len(index.entries) == 1
        assert index.generated_at == "2026-02-19T14:30:00Z"
        assert index.version == "1.0"

    def test_empty_index(self):
        """SummaryIndex with no entries."""
        from dokumen.summary_index import SummaryIndex

        index = SummaryIndex(entries={}, generated_at="now", version="1.0")
        assert len(index.entries) == 0


class TestRenderSummaryIndex:
    """Tests for render_summary_index function."""

    def test_render_single_entry(self):
        """Renders single file entry correctly."""
        from dokumen.summary_index import (
            SummaryIndex,
            FileSummaryEntry,
            render_summary_index,
        )

        entry = FileSummaryEntry(
            file_path="docs/api.md",
            content_hash="sha256:abc123def456",
            summary_text="API documentation for the platform.\n\n- Covers authentication\n- Covers rate limiting",
        )
        index = SummaryIndex(
            entries={"docs/api.md": entry},
            generated_at="2026-02-19T14:30:00Z",
            version="1.0",
        )
        result = render_summary_index(index)

        assert "<!-- DOKUMEN SUMMARIES INDEX -->" in result
        assert "<!-- Generated at: 2026-02-19T14:30:00Z -->" in result
        assert "<!-- File count: 1 -->" in result
        assert "## docs/api.md" in result
        assert "<!-- hash: sha256:abc123def456 -->" in result
        assert "API documentation for the platform." in result
        assert "- Covers authentication" in result

    def test_render_multiple_entries_sorted(self):
        """Entries are rendered sorted by file path."""
        from dokumen.summary_index import (
            SummaryIndex,
            FileSummaryEntry,
            render_summary_index,
        )

        entries = {
            "docs/z-file.md": FileSummaryEntry(
                file_path="docs/z-file.md",
                content_hash="sha256:zzz",
                summary_text="Z file.",
            ),
            "docs/a-file.md": FileSummaryEntry(
                file_path="docs/a-file.md",
                content_hash="sha256:aaa",
                summary_text="A file.",
            ),
        }
        index = SummaryIndex(
            entries=entries,
            generated_at="2026-02-19T14:30:00Z",
            version="1.0",
        )
        result = render_summary_index(index)

        # a-file should appear before z-file
        a_pos = result.index("## docs/a-file.md")
        z_pos = result.index("## docs/z-file.md")
        assert a_pos < z_pos

    def test_render_empty_index(self):
        """Empty index renders header only."""
        from dokumen.summary_index import SummaryIndex, render_summary_index

        index = SummaryIndex(entries={}, generated_at="now", version="1.0")
        result = render_summary_index(index)

        assert "<!-- DOKUMEN SUMMARIES INDEX -->" in result
        assert "<!-- File count: 0 -->" in result

    def test_render_entries_separated_by_divider(self):
        """Entries are separated by --- dividers."""
        from dokumen.summary_index import (
            SummaryIndex,
            FileSummaryEntry,
            render_summary_index,
        )

        entries = {
            "docs/a.md": FileSummaryEntry("docs/a.md", "sha256:a", "A."),
            "docs/b.md": FileSummaryEntry("docs/b.md", "sha256:b", "B."),
        }
        index = SummaryIndex(entries=entries, generated_at="now", version="1.0")
        result = render_summary_index(index)

        assert "\n---\n" in result


class TestParseSummaryIndex:
    """Tests for parse_summary_index function."""

    def test_round_trip(self):
        """parse(render(x)) == x."""
        from dokumen.summary_index import (
            SummaryIndex,
            FileSummaryEntry,
            render_summary_index,
            parse_summary_index,
        )

        entry = FileSummaryEntry(
            file_path="docs/api.md",
            content_hash="sha256:abc123",
            summary_text="API documentation for the platform.\n\n- Covers authentication\n- Covers rate limiting",
        )
        original = SummaryIndex(
            entries={"docs/api.md": entry},
            generated_at="2026-02-19T14:30:00Z",
            version="1.0",
        )

        rendered = render_summary_index(original)
        parsed = parse_summary_index(rendered)

        assert len(parsed.entries) == len(original.entries)
        assert "docs/api.md" in parsed.entries
        assert parsed.entries["docs/api.md"].content_hash == "sha256:abc123"
        assert "Covers authentication" in parsed.entries["docs/api.md"].summary_text
        assert parsed.generated_at == "2026-02-19T14:30:00Z"

    def test_parse_empty_content(self):
        """Parsing empty string returns empty index."""
        from dokumen.summary_index import parse_summary_index

        result = parse_summary_index("")
        assert len(result.entries) == 0

    def test_parse_malformed_content(self):
        """Parsing random text returns empty index."""
        from dokumen.summary_index import parse_summary_index

        result = parse_summary_index("Some random text\nwithout structure")
        assert len(result.entries) == 0

    def test_parse_multiple_entries(self):
        """Parsing index with multiple entries."""
        from dokumen.summary_index import (
            SummaryIndex,
            FileSummaryEntry,
            render_summary_index,
            parse_summary_index,
        )

        entries = {
            "docs/a.md": FileSummaryEntry("docs/a.md", "sha256:aaa", "File A summary.\n\n- Point 1"),
            "docs/b.md": FileSummaryEntry("docs/b.md", "sha256:bbb", "File B summary.\n\n- Point 2"),
        }
        original = SummaryIndex(entries=entries, generated_at="now", version="1.0")
        rendered = render_summary_index(original)
        parsed = parse_summary_index(rendered)

        assert len(parsed.entries) == 2
        assert "docs/a.md" in parsed.entries
        assert "docs/b.md" in parsed.entries


class TestComputeStaleness:
    """Tests for compute_staleness function."""

    def test_all_new_files(self):
        """All files are new when index is empty."""
        from dokumen.summary_index import SummaryIndex, compute_staleness

        index = SummaryIndex(entries={}, generated_at="now", version="1.0")
        current_files = {"docs/a.md": "sha256:aaa", "docs/b.md": "sha256:bbb"}

        new, changed, removed = compute_staleness(index, current_files)

        assert set(new) == {"docs/a.md", "docs/b.md"}
        assert changed == []
        assert removed == []

    def test_all_unchanged(self):
        """No changes when hashes match."""
        from dokumen.summary_index import (
            SummaryIndex,
            FileSummaryEntry,
            compute_staleness,
        )

        entries = {
            "docs/a.md": FileSummaryEntry("docs/a.md", "sha256:aaa", "A summary."),
        }
        index = SummaryIndex(entries=entries, generated_at="now", version="1.0")
        current_files = {"docs/a.md": "sha256:aaa"}

        new, changed, removed = compute_staleness(index, current_files)

        assert new == []
        assert changed == []
        assert removed == []

    def test_changed_file(self):
        """File with different hash is detected as changed."""
        from dokumen.summary_index import (
            SummaryIndex,
            FileSummaryEntry,
            compute_staleness,
        )

        entries = {
            "docs/a.md": FileSummaryEntry("docs/a.md", "sha256:old", "A summary."),
        }
        index = SummaryIndex(entries=entries, generated_at="now", version="1.0")
        current_files = {"docs/a.md": "sha256:new"}

        new, changed, removed = compute_staleness(index, current_files)

        assert new == []
        assert changed == ["docs/a.md"]
        assert removed == []

    def test_removed_file(self):
        """File in index but not on disk is detected as removed."""
        from dokumen.summary_index import (
            SummaryIndex,
            FileSummaryEntry,
            compute_staleness,
        )

        entries = {
            "docs/a.md": FileSummaryEntry("docs/a.md", "sha256:aaa", "A summary."),
        }
        index = SummaryIndex(entries=entries, generated_at="now", version="1.0")
        current_files = {}

        new, changed, removed = compute_staleness(index, current_files)

        assert new == []
        assert changed == []
        assert removed == ["docs/a.md"]

    def test_mixed_new_changed_removed(self):
        """Mixed scenario with new, changed, and removed files."""
        from dokumen.summary_index import (
            SummaryIndex,
            FileSummaryEntry,
            compute_staleness,
        )

        entries = {
            "docs/unchanged.md": FileSummaryEntry("docs/unchanged.md", "sha256:same", "Same."),
            "docs/changed.md": FileSummaryEntry("docs/changed.md", "sha256:old", "Old."),
            "docs/removed.md": FileSummaryEntry("docs/removed.md", "sha256:gone", "Gone."),
        }
        index = SummaryIndex(entries=entries, generated_at="now", version="1.0")
        current_files = {
            "docs/unchanged.md": "sha256:same",
            "docs/changed.md": "sha256:new",
            "docs/new.md": "sha256:brand_new",
        }

        new, changed, removed = compute_staleness(index, current_files)

        assert set(new) == {"docs/new.md"}
        assert set(changed) == {"docs/changed.md"}
        assert set(removed) == {"docs/removed.md"}


class TestGenerateFileSummary:
    """Tests for generate_file_summary function."""

    @pytest.mark.asyncio
    async def test_calls_provider_correctly(self):
        """generate_file_summary calls provider with correct prompt."""
        from dokumen.summary_index import generate_file_summary

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "API docs for auth.\n\n- PAT-based auth\n- Token expiry",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        result = await generate_file_summary(
            provider=mock_provider,
            file_path="docs/api.md",
            content="# API\nAuthentication uses PATs...",
        )

        assert result is not None
        assert "API docs" in result or "auth" in result.lower()
        mock_provider.complete.assert_called_once()
        call_args = mock_provider.complete.call_args
        messages = call_args[0][0]
        # System message should contain the summary prompt
        assert any("documentation analyst" in str(m).lower() for m in messages)

    @pytest.mark.asyncio
    async def test_returns_summary_text(self):
        """generate_file_summary returns the LLM response text."""
        from dokumen.summary_index import generate_file_summary

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "This is a summary.",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

        result = await generate_file_summary(
            provider=mock_provider,
            file_path="docs/test.md",
            content="# Test\nSome content.",
        )
        assert result == "This is a summary."

    @pytest.mark.asyncio
    async def test_handles_provider_error(self):
        """generate_file_summary returns None on provider error."""
        from dokumen.summary_index import generate_file_summary

        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = Exception("API error")

        result = await generate_file_summary(
            provider=mock_provider,
            file_path="docs/test.md",
            content="content",
        )
        assert result is None


class TestGenerateSummaryIndex:
    """Tests for generate_summary_index function."""

    @pytest.mark.asyncio
    async def test_generates_for_all_files(self):
        """Generates summaries for all doc files."""
        from dokumen.summary_index import generate_summary_index, SummaryIndex

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "File summary here.",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

        doc_files = {"docs/a.md": "Content A", "docs/b.md": "Content B"}

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files=doc_files,
            existing_index=None,
        )

        assert isinstance(result, SummaryIndex)
        assert len(result.entries) == 2
        assert "docs/a.md" in result.entries
        assert "docs/b.md" in result.entries

    @pytest.mark.asyncio
    async def test_incremental_skips_unchanged(self):
        """Incremental update skips files with unchanged hashes."""
        from dokumen.summary_index import (
            generate_summary_index,
            SummaryIndex,
            FileSummaryEntry,
            compute_content_hash,
        )

        content_a = "Content A"
        hash_a = compute_content_hash(content_a)

        existing_entry = FileSummaryEntry(
            file_path="docs/a.md",
            content_hash=hash_a,
            summary_text="Existing summary for A.",
        )
        existing_index = SummaryIndex(
            entries={"docs/a.md": existing_entry},
            generated_at="old",
            version="1.0",
        )

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "New summary for B.",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

        doc_files = {"docs/a.md": content_a, "docs/b.md": "Content B"}

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files=doc_files,
            existing_index=existing_index,
        )

        assert len(result.entries) == 2
        # A should keep its existing summary (unchanged)
        assert result.entries["docs/a.md"].summary_text == "Existing summary for A."
        # B should have the new summary
        assert result.entries["docs/b.md"].summary_text == "New summary for B."
        # Provider should only be called once (for B)
        assert mock_provider.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_removes_deleted_files(self):
        """Files no longer on disk are removed from index."""
        from dokumen.summary_index import (
            generate_summary_index,
            SummaryIndex,
            FileSummaryEntry,
        )

        existing_entry = FileSummaryEntry(
            file_path="docs/deleted.md",
            content_hash="sha256:old",
            summary_text="Old summary.",
        )
        existing_index = SummaryIndex(
            entries={"docs/deleted.md": existing_entry},
            generated_at="old",
            version="1.0",
        )

        mock_provider = AsyncMock()

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={},
            existing_index=existing_index,
        )

        assert len(result.entries) == 0
        # Provider should not be called (no files to summarize)
        assert mock_provider.complete.call_count == 0

    @pytest.mark.asyncio
    async def test_regenerates_changed_files(self):
        """Changed files get new summaries."""
        from dokumen.summary_index import (
            generate_summary_index,
            SummaryIndex,
            FileSummaryEntry,
        )

        existing_entry = FileSummaryEntry(
            file_path="docs/a.md",
            content_hash="sha256:old_hash",
            summary_text="Old summary.",
        )
        existing_index = SummaryIndex(
            entries={"docs/a.md": existing_entry},
            generated_at="old",
            version="1.0",
        )

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "Updated summary.",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

        doc_files = {"docs/a.md": "New content for A"}

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files=doc_files,
            existing_index=existing_index,
        )

        assert result.entries["docs/a.md"].summary_text == "Updated summary."
        assert mock_provider.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_progress_callback_invoked(self):
        """Progress callback is invoked for each file."""
        from dokumen.summary_index import generate_summary_index

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "Summary.",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

        progress_calls = []

        def on_progress(event, data):
            progress_calls.append((event, data))

        doc_files = {"docs/a.md": "Content A", "docs/b.md": "Content B"}

        await generate_summary_index(
            provider=mock_provider,
            doc_files=doc_files,
            existing_index=None,
            on_progress=on_progress,
        )

        # Should have progress calls for each file
        assert len(progress_calls) >= 2

    @pytest.mark.asyncio
    async def test_handles_api_error_gracefully(self):
        """API errors for individual files don't crash the entire run."""
        from dokumen.summary_index import generate_summary_index

        call_count = 0

        async def mock_complete(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("API error")
            return {
                "content": "Summary for second file.",
                "usage": {"input_tokens": 50, "output_tokens": 20},
            }

        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = mock_complete

        doc_files = {"docs/a.md": "Content A", "docs/b.md": "Content B"}

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files=doc_files,
            existing_index=None,
        )

        # Should still have entries (at least for the successful one)
        # Failed file should be skipped (not in index)
        assert len(result.entries) >= 1


class TestIsBinaryFile:
    """Tests for is_binary_file detection."""

    def test_text_file_not_binary(self):
        """Regular text content is not binary."""
        from dokumen.summary_index import is_binary_content

        assert is_binary_content("Hello world\nThis is text.") is False

    def test_binary_content_detected(self):
        """Content with null bytes is detected as binary."""
        from dokumen.summary_index import is_binary_content

        assert is_binary_content("Hello\x00World") is True

    def test_empty_content_not_binary(self):
        """Empty content is not binary."""
        from dokumen.summary_index import is_binary_content

        assert is_binary_content("") is False


class TestIndexFilename:
    """Tests for the index filename constant."""

    def test_filename_constant(self):
        """INDEX_FILENAME constant is set correctly."""
        from dokumen.summary_index import INDEX_FILENAME

        assert INDEX_FILENAME == "DOKUMEN_SUMMARIES_INDEX.md"
