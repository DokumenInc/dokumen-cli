"""Tests for read_file tool security limits (file size and symlinks)."""
import asyncio
import os

import pytest

from dokumen.tools_object import create_glob_tool, create_list_directory_tool, create_read_file_tool


@pytest.fixture
def tool(tmp_path):
    """Create a read_file tool scoped to tmp_path."""
    return create_read_file_tool(base_dir=str(tmp_path))


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestReadFileTextSizeLimit:
    """Text files over 1MB must be rejected."""

    ONE_MB = 1024 * 1024

    def test_read_file_under_size_limit(self, tool, tmp_path):
        """A small text file should be read successfully."""
        f = tmp_path / "small.txt"
        f.write_text("hello world\n")
        result = _run(tool.handler({"file_path": "small.txt"}))
        assert result.success is True
        assert "hello world" in result.output

    def test_read_file_over_size_limit(self, tool, tmp_path):
        """A 2MB text file should be rejected with a size error."""
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * (2 * self.ONE_MB))
        result = _run(tool.handler({"file_path": "big.txt"}))
        assert result.success is False
        assert "too large" in result.error.lower() or "File too large" in result.error

    def test_read_file_exactly_at_limit(self, tool, tmp_path):
        """A file exactly 1MB should be accepted (boundary)."""
        f = tmp_path / "exact.txt"
        f.write_bytes(b"a" * self.ONE_MB)
        result = _run(tool.handler({"file_path": "exact.txt"}))
        assert result.success is True

    def test_read_file_just_over_limit(self, tool, tmp_path):
        """A file 1MB + 1 byte should be rejected."""
        f = tmp_path / "over.txt"
        f.write_bytes(b"a" * (self.ONE_MB + 1))
        result = _run(tool.handler({"file_path": "over.txt"}))
        assert result.success is False
        assert "File too large" in result.error

    def test_read_file_pdf_uses_own_limit(self, tool, tmp_path):
        """PDF size check is unchanged (4.5MB limit, not 1MB text limit).

        A 2MB PDF should NOT be rejected by the text file size check.
        It may fail for other reasons (invalid PDF), but the error should
        NOT mention the 1MB text file limit.
        """
        f = tmp_path / "doc.pdf"
        # Write a 2MB fake PDF (will fail PDF parsing but NOT the text size check)
        f.write_bytes(b"%PDF-" + b"x" * (2 * self.ONE_MB))
        result = _run(tool.handler({"file_path": "doc.pdf"}))
        # Should not hit the text file size limit
        if not result.success:
            assert "File too large" not in result.error
            assert "max 1,048,576" not in result.error

    def test_read_file_image_no_text_size_limit(self, tool, tmp_path):
        """Images bypass the text file size limit (they have their own handling)."""
        f = tmp_path / "photo.png"
        # Write a 2MB fake PNG (will be read as image, not text)
        f.write_bytes(b"\x89PNG" + b"x" * (2 * self.ONE_MB))
        result = _run(tool.handler({"file_path": "photo.png"}))
        # Images are handled separately - should succeed or fail for image reasons,
        # but NOT hit the text file size limit
        if not result.success:
            assert "File too large" not in result.error
            assert "max 1,048,576" not in result.error


class TestReadFileImageSizeLimit:
    """Images over 10MB must be rejected to prevent OOM."""

    TEN_MB = 10 * 1024 * 1024

    def test_image_under_size_limit(self, tool, tmp_path):
        """A small image should be read successfully."""
        f = tmp_path / "small.png"
        f.write_bytes(b"\x89PNG" + b"x" * 1000)
        result = _run(tool.handler({"file_path": "small.png"}))
        assert result.success is True

    def test_image_over_size_limit(self, tool, tmp_path):
        """An image over 10MB should be rejected."""
        f = tmp_path / "huge.png"
        f.write_bytes(b"\x89PNG" + b"x" * (self.TEN_MB + 1))
        result = _run(tool.handler({"file_path": "huge.png"}))
        assert result.success is False
        assert "Image too large" in result.error


class TestSymlinkTraversalProtection:
    """Symlinks pointing outside the workspace must be rejected."""

    def test_read_file_symlink_outside_workspace_rejected(self, tool, tmp_path):
        """A symlink pointing outside the workspace must be blocked."""
        target = tmp_path.parent / "outside_secret.txt"
        target.write_text("secret data")
        link = tmp_path / "sneaky.txt"
        link.symlink_to(target)
        result = _run(tool.handler({"file_path": "sneaky.txt"}))
        assert result.success is False
        assert "Access denied" in result.error or "traversal" in result.error.lower()

    def test_read_file_symlink_within_workspace_allowed(self, tool, tmp_path):
        """A symlink pointing within the workspace should be allowed."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        real_file = subdir / "real.txt"
        real_file.write_text("allowed content")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)
        result = _run(tool.handler({"file_path": "link.txt"}))
        assert result.success is True
        assert "allowed content" in result.output

    def test_read_file_nested_symlink_chain_outside_rejected(self, tool, tmp_path):
        """A chain of symlinks ultimately resolving outside must be blocked."""
        outside = tmp_path.parent / "chain_target.txt"
        outside.write_text("chain secret")
        mid_link = tmp_path / "mid.txt"
        mid_link.symlink_to(outside)
        final_link = tmp_path / "final.txt"
        final_link.symlink_to(mid_link)
        result = _run(tool.handler({"file_path": "final.txt"}))
        assert result.success is False
        assert "Access denied" in result.error or "traversal" in result.error.lower()

    def test_glob_symlink_path_outside_workspace_rejected(self, tmp_path):
        """Glob tool must reject symlinks pointing outside workspace."""
        glob_tool = create_glob_tool(base_dir=str(tmp_path))
        outside_dir = tmp_path.parent / "outside_dir"
        outside_dir.mkdir(exist_ok=True)
        (outside_dir / "secret.txt").write_text("glob secret")
        link = tmp_path / "linked_dir"
        link.symlink_to(outside_dir)
        result = _run(glob_tool.handler({"pattern": "linked_dir/*.txt"}))
        # Should either fail or return no results (not expose outside files)
        if result.success:
            assert "secret" not in (result.output or "").lower()
        else:
            assert "Access denied" in result.error or "traversal" in result.error.lower()

    def test_list_directory_symlink_outside_workspace_rejected(self, tmp_path):
        """list_directory must reject symlinks pointing outside workspace."""
        list_tool = create_list_directory_tool(base_dir=str(tmp_path))
        outside_dir = tmp_path.parent / "outside_list_dir"
        outside_dir.mkdir(exist_ok=True)
        (outside_dir / "secret.txt").write_text("list secret")
        link = tmp_path / "linked"
        link.symlink_to(outside_dir)
        result = _run(list_tool.handler({"path": "linked"}))
        assert result.success is False
        assert "Access denied" in result.error or "traversal" in result.error.lower()

    def test_list_directory_symlink_within_workspace_succeeds(self, tmp_path):
        """list_directory should allow symlinks within workspace."""
        list_tool = create_list_directory_tool(base_dir=str(tmp_path))
        subdir = tmp_path / "real_dir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("hello")
        link = tmp_path / "link_dir"
        link.symlink_to(subdir)
        result = _run(list_tool.handler({"path": "link_dir"}))
        assert result.success is True
        assert "file.txt" in result.output
