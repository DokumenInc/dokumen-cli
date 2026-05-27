"""Tests for dokumen content filter (clean/smudge)."""

import io
import sys

import pytest
from unittest.mock import patch

from dokumen.workspace.filter import (
    DOKUMEN_PREFIX,
    _is_binary,
    clean,
    main,
    smudge,
)


class TestSmudge:
    """Test smudge (GitLab → local): strip prefix."""

    def test_smudge_strips_prefix(self):
        """Prefixed content has prefix stripped."""
        data = DOKUMEN_PREFIX + b"Hello world\n"
        assert smudge(data) == b"Hello world\n"

    def test_smudge_no_prefix_passthrough(self):
        """Content without prefix passes through unchanged."""
        data = b"No prefix here\n"
        assert smudge(data) == b"No prefix here\n"


class TestClean:
    """Test clean (local → GitLab): add prefix."""

    def test_clean_adds_prefix(self):
        """Clean prepends the DOKUMEN prefix."""
        data = b"Hello world\n"
        assert clean(data) == DOKUMEN_PREFIX + b"Hello world\n"

    def test_clean_already_prefixed(self):
        """Already prefixed content is unchanged (idempotent)."""
        data = DOKUMEN_PREFIX + b"Hello world\n"
        assert clean(data) == data


class TestBinaryPassthrough:
    """Test binary content detection and passthrough."""

    def test_binary_passthrough_smudge(self):
        """Binary content passes through smudge unchanged."""
        data = b"\x00\x01\x02\x03binary content"
        assert smudge(data) == data

    def test_binary_passthrough_clean(self):
        """Binary content passes through clean unchanged."""
        data = b"some\x00binary\x01data"
        assert clean(data) == data


class TestEmptyContent:
    """Test empty content handling."""

    def test_empty_content_smudge(self):
        """Empty stdin produces empty stdout for smudge."""
        assert smudge(b"") == b""

    def test_empty_content_clean(self):
        """Empty stdin produces prefix only for clean."""
        assert clean(b"") == DOKUMEN_PREFIX


class TestIsBinary:
    """Test binary detection heuristic."""

    def test_text_is_not_binary(self):
        """Normal text is not detected as binary."""
        assert _is_binary(b"Hello world\n") is False

    def test_null_byte_is_binary(self):
        """Null byte in first 8KB marks as binary."""
        assert _is_binary(b"hello\x00world") is True

    def test_null_byte_after_8k_not_detected(self):
        """Null byte beyond 8KB window is not detected."""
        data = b"a" * 8192 + b"\x00"
        assert _is_binary(data) is False

    def test_empty_is_not_binary(self):
        """Empty content is not binary."""
        assert _is_binary(b"") is False


class TestMain:
    """Test main() entry point."""

    def test_main_smudge_flag(self):
        """main() with --smudge reads stdin and writes smudged output."""
        input_data = DOKUMEN_PREFIX + b"content\n"
        stdin = io.BytesIO(input_data)
        stdout = io.BytesIO()

        with (
            patch.object(sys, "argv", ["dokumen-filter", "--smudge"]),
            patch.object(sys, "stdin", type("", (), {"buffer": stdin})()),
            patch.object(sys, "stdout", type("", (), {"buffer": stdout})()),
        ):
            main()

        assert stdout.getvalue() == b"content\n"

    def test_main_clean_flag(self):
        """main() with --clean reads stdin and writes cleaned output."""
        input_data = b"content\n"
        stdin = io.BytesIO(input_data)
        stdout = io.BytesIO()

        with (
            patch.object(sys, "argv", ["dokumen-filter", "--clean"]),
            patch.object(sys, "stdin", type("", (), {"buffer": stdin})()),
            patch.object(sys, "stdout", type("", (), {"buffer": stdout})()),
        ):
            main()

        assert stdout.getvalue() == DOKUMEN_PREFIX + b"content\n"

    def test_main_no_flag_exits(self):
        """main() with no flag exits with code 1."""
        with patch.object(sys, "argv", ["dokumen-filter"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_main_invalid_flag_exits(self):
        """main() with invalid flag exits with code 1."""
        with patch.object(sys, "argv", ["dokumen-filter", "--invalid"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1


class TestRoundtrip:
    """Test clean→smudge roundtrip."""

    def test_roundtrip(self):
        """clean(content) then smudge(result) returns original content."""
        content = b"# Hello\n\nThis is a **doc** file.\n"
        cleaned = clean(content)
        assert cleaned.startswith(DOKUMEN_PREFIX)
        smudged = smudge(cleaned)
        assert smudged == content

    def test_roundtrip_multiline(self):
        """Roundtrip preserves multiline content."""
        content = b"line 1\nline 2\nline 3\n"
        assert smudge(clean(content)) == content

    def test_roundtrip_unicode(self):
        """Roundtrip preserves unicode content."""
        content = "Hello 世界 🌍\n".encode("utf-8")
        assert smudge(clean(content)) == content
