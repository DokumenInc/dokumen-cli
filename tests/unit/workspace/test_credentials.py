"""Tests for PAT credential handling."""

import os

import pytest

from dokumen.workspace.credentials import get_pat, mask_pat


class TestGetPAT:
    """Test PAT retrieval from environment."""

    def test_get_pat_from_env(self, monkeypatch):
        """Reads PAT from DOKUMEN_PAT env var."""
        monkeypatch.setenv("DOKUMEN_PAT", "glpat-abc123")
        assert get_pat() == "glpat-abc123"

    def test_get_pat_missing(self, monkeypatch):
        """Raises RuntimeError when DOKUMEN_PAT is not set."""
        monkeypatch.delenv("DOKUMEN_PAT", raising=False)
        with pytest.raises(RuntimeError, match="DOKUMEN_PAT"):
            get_pat()

    def test_get_pat_empty(self, monkeypatch):
        """Raises RuntimeError when DOKUMEN_PAT is empty."""
        monkeypatch.setenv("DOKUMEN_PAT", "")
        with pytest.raises(RuntimeError, match="DOKUMEN_PAT"):
            get_pat()


class TestMaskPAT:
    """Test PAT masking for log output."""

    def test_mask_pat_normal(self):
        """Masks PAT showing only first 8 chars."""
        assert mask_pat("glpat-abc123def456") == "glpat-ab********"

    def test_mask_pat_short(self):
        """Short PAT gets fully masked."""
        assert mask_pat("glpat") == "********"

    def test_mask_pat_empty(self):
        """Empty string returns masked."""
        assert mask_pat("") == "********"
