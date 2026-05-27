"""Tests for image summarization in summary_index module."""

import base64
from unittest.mock import AsyncMock

import pytest

from dokumen.summary_index import (
    IMAGE_SUMMARY_SYSTEM_PROMPT,
    IMAGE_TYPES,
    FileSummaryEntry,
    SummaryIndex,
    compute_content_hash,
    generate_image_summary,
    generate_summary_index,
    is_image_file,
)


class TestIsImageFile:
    """Tests for is_image_file helper."""

    def test_png_is_image(self):
        """PNG files should be detected as images."""
        assert is_image_file("docs/diagram.png") is True

    def test_jpg_is_image(self):
        """JPG files should be detected as images."""
        assert is_image_file("docs/photo.jpg") is True

    def test_jpeg_is_image(self):
        """JPEG files should be detected as images."""
        assert is_image_file("docs/photo.jpeg") is True

    def test_gif_is_image(self):
        """GIF files should be detected as images."""
        assert is_image_file("docs/animation.gif") is True

    def test_webp_is_image(self):
        """WebP files should be detected as images."""
        assert is_image_file("docs/photo.webp") is True

    def test_markdown_is_not_image(self):
        """Markdown files should not be detected as images."""
        assert is_image_file("docs/readme.md") is False

    def test_pdf_is_not_image(self):
        """PDF files should not be detected as images."""
        assert is_image_file("docs/report.pdf") is False

    def test_case_insensitive(self):
        """Image detection should be case-insensitive."""
        assert is_image_file("docs/PHOTO.PNG") is True
        assert is_image_file("docs/Image.JPG") is True
        assert is_image_file("docs/art.WebP") is True

    def test_no_extension(self):
        """Files without extensions should not be detected as images."""
        assert is_image_file("docs/README") is False

    def test_empty_path(self):
        """Empty path should not be detected as image."""
        assert is_image_file("") is False


class TestGenerateImageSummary:
    """Tests for generate_image_summary function."""

    @pytest.mark.asyncio
    async def test_calls_provider_with_multimodal_message(self):
        """Should call provider.complete() with image content block."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "A diagram showing the system architecture."
        }

        base64_data = base64.b64encode(b"fake-png-data").decode("ascii")
        result = await generate_image_summary(
            mock_provider, "docs/arch.png", base64_data, "image/png"
        )

        assert result == "A diagram showing the system architecture."
        mock_provider.complete.assert_called_once()

        # Verify the messages structure
        call_args = mock_provider.complete.call_args[0][0]
        assert len(call_args) == 2
        # System message
        assert call_args[0]["role"] == "system"
        # User message with multimodal content
        assert call_args[1]["role"] == "user"
        user_content = call_args[1]["content"]
        assert isinstance(user_content, list)
        assert len(user_content) == 2
        # Text block
        assert user_content[0]["type"] == "text"
        assert "docs/arch.png" in user_content[0]["text"]
        # Image block
        assert user_content[1]["type"] == "image"
        assert user_content[1]["source"]["type"] == "base64"
        assert user_content[1]["source"]["media_type"] == "image/png"
        assert user_content[1]["source"]["data"] == base64_data

    @pytest.mark.asyncio
    async def test_uses_image_system_prompt(self):
        """Should use IMAGE_SUMMARY_SYSTEM_PROMPT, not the text prompt."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "An image of a flowchart."
        }

        base64_data = base64.b64encode(b"fake-data").decode("ascii")
        await generate_image_summary(
            mock_provider, "docs/flow.png", base64_data, "image/png"
        )

        call_args = mock_provider.complete.call_args[0][0]
        assert call_args[0]["content"] == IMAGE_SUMMARY_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_response(self):
        """Should return None when provider returns empty content."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {"content": ""}

        base64_data = base64.b64encode(b"fake-data").decode("ascii")
        result = await generate_image_summary(
            mock_provider, "docs/img.png", base64_data, "image/png"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        """Should return None when provider raises an exception."""
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("API error")

        base64_data = base64.b64encode(b"fake-data").decode("ascii")
        result = await generate_image_summary(
            mock_provider, "docs/img.png", base64_data, "image/png"
        )

        assert result is None


class TestGenerateSummaryIndexWithImages:
    """Tests for generate_summary_index with image_files parameter."""

    @pytest.mark.asyncio
    async def test_processes_image_files(self):
        """Should generate summaries for image files using multimodal."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "An architecture diagram."
        }

        base64_data = base64.b64encode(b"fake-png-data").decode("ascii")
        image_files = {
            "docs/arch.png": (base64_data, "image/png"),
        }

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={},
            image_files=image_files,
        )

        assert "docs/arch.png" in result.entries
        entry = result.entries["docs/arch.png"]
        assert entry.summary_text == "An architecture diagram."
        assert entry.content_hash.startswith("sha256:")

    @pytest.mark.asyncio
    async def test_mixed_text_and_images(self):
        """Should process both text and image files."""
        mock_provider = AsyncMock()
        # Files are sorted alphabetically: diagram.png before readme.md
        mock_provider.complete.side_effect = [
            {"content": "Image description."},
            {"content": "Text file summary."},
        ]

        base64_data = base64.b64encode(b"fake-png-data").decode("ascii")
        image_files = {
            "docs/diagram.png": (base64_data, "image/png"),
        }

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={"docs/readme.md": "# README\nSome content."},
            image_files=image_files,
        )

        assert "docs/readme.md" in result.entries
        assert "docs/diagram.png" in result.entries
        assert result.entries["docs/readme.md"].summary_text == "Text file summary."
        assert result.entries["docs/diagram.png"].summary_text == "Image description."

    @pytest.mark.asyncio
    async def test_skips_non_image_binary(self):
        """Binary text files (with null bytes) should still be skipped."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {"content": "Summary."}

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={"docs/binary.dat": "data\x00with\x00nulls"},
        )

        # Binary file should be skipped
        assert "docs/binary.dat" not in result.entries
        mock_provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_image_hash_uses_base64_content(self):
        """Image content hash should be based on the base64 string."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "An image description."
        }

        base64_data = base64.b64encode(b"fake-png-data").decode("ascii")
        expected_hash = compute_content_hash(base64_data)

        image_files = {
            "docs/img.png": (base64_data, "image/png"),
        }

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={},
            image_files=image_files,
        )

        assert result.entries["docs/img.png"].content_hash == expected_hash

    @pytest.mark.asyncio
    async def test_incremental_update_with_images(self):
        """Unchanged images should be kept from existing index."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "New image summary."
        }

        base64_data = base64.b64encode(b"same-data").decode("ascii")
        existing_hash = compute_content_hash(base64_data)

        existing_index = SummaryIndex(
            entries={
                "docs/old.png": FileSummaryEntry(
                    file_path="docs/old.png",
                    content_hash=existing_hash,
                    summary_text="Old image summary.",
                ),
            },
            generated_at="2026-01-01T00:00:00Z",
            version="1.0",
        )

        image_files = {
            "docs/old.png": (base64_data, "image/png"),
        }

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={},
            image_files=image_files,
            existing_index=existing_index,
        )

        # Should keep existing entry without re-generating
        assert result.entries["docs/old.png"].summary_text == "Old image summary."
        mock_provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_image_files_backward_compatible(self):
        """When image_files is None, behavior should be unchanged."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = {
            "content": "Text summary."
        }

        result = await generate_summary_index(
            provider=mock_provider,
            doc_files={"docs/readme.md": "Hello world"},
        )

        assert "docs/readme.md" in result.entries
        assert result.entries["docs/readme.md"].summary_text == "Text summary."
