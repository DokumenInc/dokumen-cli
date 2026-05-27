"""Tests for CLI PDF tree-index reading."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

# We test the PDF handling logic via the constants and behavior patterns
from dokumen.tools_object import PDF_EXTENSION


class TestPdfConstants:
    """Test PDF-related constants."""

    def test_pdf_extension(self):
        assert PDF_EXTENSION == ".pdf"

    def test_no_legacy_constants(self):
        """MAX_PDF_SIZE and MAX_PDF_PAGES should no longer exist."""
        import dokumen.tools_object as module
        assert not hasattr(module, "MAX_PDF_SIZE")
        assert not hasattr(module, "MAX_PDF_PAGES")
