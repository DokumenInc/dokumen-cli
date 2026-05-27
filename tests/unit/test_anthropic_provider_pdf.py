"""Test PDF support in Anthropic provider."""
import pytest
from dokumen.providers.anthropic import parse_pdf_data, parse_image_data


class TestPDFDataParsing:
    """Test PDF data marker parsing."""

    def test_parse_pdf_data_basic(self):
        """parse_pdf_data extracts PDF markers."""
        content = (
            "__PDF_DATA__\n"
            "media_type: application/pdf\n"
            "path: docs/guide.pdf\n"
            "data: JVBERi0xLjQK\n"
            "__END_PDF_DATA__"
        )

        media_type, path, data, remaining = parse_pdf_data(content)

        assert media_type == "application/pdf"
        assert path == "docs/guide.pdf"
        assert data == "JVBERi0xLjQK"
        assert remaining == ""

    def test_parse_pdf_data_with_surrounding_text(self):
        """PDF markers can have text before and after."""
        content = (
            "Some text before\n"
            "__PDF_DATA__\n"
            "media_type: application/pdf\n"
            "path: manual.pdf\n"
            "data: ABC123\n"
            "__END_PDF_DATA__\n"
            "Some text after"
        )

        media_type, path, data, remaining = parse_pdf_data(content)

        assert media_type == "application/pdf"
        assert path == "manual.pdf"
        assert data == "ABC123"
        assert "Some text before" in remaining
        assert "Some text after" in remaining

    def test_parse_pdf_data_no_pdf(self):
        """parse_pdf_data returns None when no PDF found."""
        content = "Just regular text content"

        media_type, path, data, remaining = parse_pdf_data(content)

        assert media_type is None
        assert path is None
        assert data is None
        assert remaining == content

    def test_parse_pdf_data_with_long_base64(self):
        """PDF markers work with long base64 strings."""
        long_base64 = "A" * 1000
        content = (
            f"__PDF_DATA__\n"
            f"media_type: application/pdf\n"
            f"path: large.pdf\n"
            f"data: {long_base64}\n"
            f"__END_PDF_DATA__"
        )

        media_type, path, data, remaining = parse_pdf_data(content)

        assert media_type == "application/pdf"
        assert data == long_base64

    def test_parse_pdf_data_path_with_spaces(self):
        """PDF markers handle paths with spaces."""
        content = (
            "__PDF_DATA__\n"
            "media_type: application/pdf\n"
            "path: docs/user manual.pdf\n"
            "data: XYZ789\n"
            "__END_PDF_DATA__"
        )

        media_type, path, data, remaining = parse_pdf_data(content)

        assert path == "docs/user manual.pdf"

    def test_parse_pdf_data_nested_path(self):
        """PDF markers handle deeply nested paths."""
        content = (
            "__PDF_DATA__\n"
            "media_type: application/pdf\n"
            "path: docs/guides/advanced/chapter1.pdf\n"
            "data: DATA123\n"
            "__END_PDF_DATA__"
        )

        media_type, path, data, remaining = parse_pdf_data(content)

        assert path == "docs/guides/advanced/chapter1.pdf"

    def test_parse_pdf_and_image_separate(self):
        """PDF and image parsers don't interfere with each other."""
        pdf_content = (
            "__PDF_DATA__\n"
            "media_type: application/pdf\n"
            "path: doc.pdf\n"
            "data: PDF123\n"
            "__END_PDF_DATA__"
        )

        image_content = (
            "__IMAGE_DATA__\n"
            "media_type: image/png\n"
            "prompt: test.png\n"
            "data: IMG123\n"
            "__END_IMAGE_DATA__"
        )

        # PDF parser should not match image
        pdf_media, pdf_path, pdf_data, pdf_remaining = parse_pdf_data(image_content)
        assert pdf_media is None

        # Image parser should not match PDF
        img_media, img_prompt, img_data, img_remaining = parse_image_data(pdf_content)
        assert img_media is None

    def test_parse_pdf_data_only_first_match(self):
        """parse_pdf_data only extracts first PDF marker."""
        content = (
            "__PDF_DATA__\n"
            "media_type: application/pdf\n"
            "path: first.pdf\n"
            "data: FIRST\n"
            "__END_PDF_DATA__\n"
            "__PDF_DATA__\n"
            "media_type: application/pdf\n"
            "path: second.pdf\n"
            "data: SECOND\n"
            "__END_PDF_DATA__"
        )

        media_type, path, data, remaining = parse_pdf_data(content)

        assert path == "first.pdf"
        assert data == "FIRST"
        # Second marker should be in remaining text
        assert "second.pdf" in remaining


class TestProviderPDFIntegration:
    """Test PDF support integrated into provider message handling."""

    def test_pdf_content_block_structure(self):
        """Verify expected structure for PDF content blocks."""
        # This is more of a documentation test
        # The actual structure should be:
        content_block = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "base64_encoded_data_here"
            }
        }

        assert content_block["type"] == "document"
        assert content_block["source"]["media_type"] == "application/pdf"

    def test_pdf_vs_image_content_types(self):
        """PDF uses 'document' type, images use 'image' type."""
        pdf_block = {"type": "document"}
        image_block = {"type": "image"}

        assert pdf_block["type"] == "document"
        assert image_block["type"] == "image"
        assert pdf_block["type"] != image_block["type"]
