"""Tests for PDF summarization in summary_index module."""

import base64
from unittest.mock import AsyncMock

import pytest

from dokumen.summary_index import (
    PDF_SUMMARY_SYSTEM_PROMPT,
    compute_content_hash,
    FileSummaryEntry,
    SummaryIndex,
    generate_pdf_summary,
    generate_summary_index,
    is_pdf_file,
)


class TestIsPdfFile:
    """Tests for is_pdf_file helper."""

    def test_pdf_is_detected(self):
        """PDF files should be detected."""
        assert is_pdf_file("docs/report.pdf") is True

    def test_uppercase_pdf(self):
        """Detection should be case-insensitive."""
        assert is_pdf_file("docs/Report.PDF") is True

    def test_mixed_case_pdf(self):
        """Detection should be case-insensitive for mixed case."""
        assert is_pdf_file("docs/Manual.Pdf") is True

    def test_markdown_not_pdf(self):
        """Markdown files should not be detected as PDF."""
        assert is_pdf_file("docs/readme.md") is False

    def test_image_not_pdf(self):
        """Image files should not be detected as PDF."""
        assert is_pdf_file("docs/diagram.png") is False

    def test_empty_path_not_pdf(self):
        """Empty path should not be detected as PDF."""
        assert is_pdf_file("") is False

    def test_no_extension_not_pdf(self):
        """Files without extension should not be detected as PDF."""
        assert is_pdf_file("docs/README") is False

    def test_pdf_in_subdirectory(self):
        """PDF files in subdirectories should be detected."""
        assert is_pdf_file("docs/manuals/guide.pdf") is True


class TestPdfSummaryPrompt:
    """Tests for PDF summary system prompt."""

    def test_prompt_exists_and_mentions_pdf(self):
        """PDF summary prompt should exist and be PDF-specific."""
        assert PDF_SUMMARY_SYSTEM_PROMPT
        assert "pdf" in PDF_SUMMARY_SYSTEM_PROMPT.lower() or "document" in PDF_SUMMARY_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_visual_content(self):
        """PDF summary prompt should mention visual content like diagrams."""
        prompt_lower = PDF_SUMMARY_SYSTEM_PROMPT.lower()
        assert "diagram" in prompt_lower or "visual" in prompt_lower or "image" in prompt_lower


class TestGeneratePdfSummary:
    """Tests for generate_pdf_summary function."""

    @pytest.mark.asyncio
    async def test_calls_provider_with_document_block(self):
        """Should call provider.complete() with document content block."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "A technical manual covering fire resistance standards."
        }

        base64_data = base64.b64encode(b"fake-pdf-data").decode("ascii")
        result = await generate_pdf_summary(
            mock_provider, "docs/manual.pdf", base64_data
        )

        assert result == "A technical manual covering fire resistance standards."
        mock_provider.complete.assert_called_once()

        # Verify the messages structure
        call_args = mock_provider.complete.call_args[0][0]
        assert len(call_args) == 2
        # System message
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == PDF_SUMMARY_SYSTEM_PROMPT
        # User message with multimodal content
        assert call_args[1]["role"] == "user"
        user_content = call_args[1]["content"]
        assert isinstance(user_content, list)
        assert len(user_content) == 2
        # Text block
        assert user_content[0]["type"] == "text"
        assert "docs/manual.pdf" in user_content[0]["text"]
        # Document block (not image)
        assert user_content[1]["type"] == "document"
        assert user_content[1]["source"]["type"] == "base64"
        assert user_content[1]["source"]["media_type"] == "application/pdf"
        assert user_content[1]["source"]["data"] == base64_data

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_response(self):
        """Should return None when provider returns empty content."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {"content": ""}

        base64_data = base64.b64encode(b"fake-data").decode("ascii")
        result = await generate_pdf_summary(
            mock_provider, "docs/report.pdf", base64_data
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        """Should return None when provider raises an exception."""
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("API error")

        base64_data = base64.b64encode(b"fake-data").decode("ascii")
        result = await generate_pdf_summary(
            mock_provider, "docs/report.pdf", base64_data
        )

        assert result is None


class TestGenerateSummaryIndexWithPdfs:
    """Tests for generate_summary_index with pdf_files parameter."""

    @pytest.mark.asyncio
    async def test_processes_pdf_files(self):
        """Should generate summaries for PDF files using document blocks."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "A building code specification."
        }

        base64_data = base64.b64encode(b"fake-pdf-data").decode("ascii")
        pdf_files = {
            "docs/code.pdf": base64_data,
        }

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={},
            pdf_files=pdf_files,
        )

        assert "docs/code.pdf" in result.entries
        entry = result.entries["docs/code.pdf"]
        assert entry.summary_text == "A building code specification."
        assert entry.content_hash.startswith("sha256:")

    @pytest.mark.asyncio
    async def test_mixed_text_images_and_pdfs(self):
        """Should process text, image, and PDF files together."""
        mock_provider = AsyncMock()
        # Files sorted alphabetically: code.pdf, diagram.png, readme.md
        mock_provider.complete.side_effect = [
            {"content": "PDF summary."},
            {"content": "Image summary."},
            {"content": "Text summary."},
        ]

        img_b64 = base64.b64encode(b"fake-png-data").decode("ascii")
        pdf_b64 = base64.b64encode(b"fake-pdf-data").decode("ascii")

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={"docs/readme.md": "# README"},
            image_files={"docs/diagram.png": (img_b64, "image/png")},
            pdf_files={"docs/code.pdf": pdf_b64},
        )

        assert "docs/readme.md" in result.entries
        assert "docs/diagram.png" in result.entries
        assert "docs/code.pdf" in result.entries

    @pytest.mark.asyncio
    async def test_pdf_hash_uses_base64_content(self):
        """PDF content hash should be based on the base64 string."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "A PDF document."
        }

        base64_data = base64.b64encode(b"fake-pdf-data").decode("ascii")
        expected_hash = compute_content_hash(base64_data)

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={},
            pdf_files={"docs/manual.pdf": base64_data},
        )

        assert result.entries["docs/manual.pdf"].content_hash == expected_hash

    @pytest.mark.asyncio
    async def test_incremental_update_with_pdfs(self):
        """Unchanged PDFs should be kept from existing index."""
        mock_provider = AsyncMock()

        base64_data = base64.b64encode(b"same-pdf-data").decode("ascii")
        existing_hash = compute_content_hash(base64_data)

        existing_index = SummaryIndex(
            entries={
                "docs/old.pdf": FileSummaryEntry(
                    file_path="docs/old.pdf",
                    content_hash=existing_hash,
                    summary_text="Old PDF summary.",
                ),
            },
            generated_at="2026-01-01T00:00:00Z",
            version="1.0",
        )

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={},
            pdf_files={"docs/old.pdf": base64_data},
            existing_index=existing_index,
        )

        # Should keep existing entry without re-generating
        assert result.entries["docs/old.pdf"].summary_text == "Old PDF summary."
        mock_provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_pdf_files_backward_compatible(self):
        """When pdf_files is None, behavior should be unchanged."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "Text summary."
        }

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={"docs/readme.md": "Hello world"},
        )

        assert "docs/readme.md" in result.entries
