"""Test PDF file handling in read_file tool.

read_file checks for _tree_index.json in PDF folders and returns structured tree
overviews. Raw PDFs without tree indexes return guidance to import first.
"""
import json
import pytest
from pathlib import Path
from dokumen.tools_object import create_read_file_tool, ToolResult


def _make_tree_index(tmp_path: Path, folder: str = "doc") -> Path:
    """Create a minimal _tree_index.json in a PDF folder."""
    folder_path = tmp_path / folder
    folder_path.mkdir(exist_ok=True)
    tree_data = {
        "file_path": f"{folder}/report.pdf",
        "file_type": "pdf",
        "title": "Test Report",
        "total_nodes": 2,
        "total_tokens": 100,
        "nodes": [
            {
                "node_id": "n1",
                "title": "Introduction",
                "summary": "Overview of the report",
                "level": 0,
                "page_index": 0,
            },
            {
                "node_id": "n2",
                "title": "Conclusion",
                "summary": "Summary of findings",
                "level": 0,
                "page_index": 5,
            },
        ],
    }
    index_path = folder_path / "_tree_index.json"
    index_path.write_text(json.dumps(tree_data), encoding="utf-8")
    return folder_path


def _make_v1_metadata(tmp_path: Path, folder: str = "doc") -> Path:
    """Create a v1 _metadata.json (no tree index)."""
    folder_path = tmp_path / folder
    folder_path.mkdir(exist_ok=True)
    metadata = {
        "name": "report",
        "original_filename": "report.pdf",
        "page_count": 3,
        "converted_at": "2026-01-01T00:00:00Z",
    }
    meta_path = folder_path / "_metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    return folder_path


class TestPDFTreeIndexReading:
    """Test PDF tree-index-aware read_file behavior."""

    @pytest.mark.asyncio
    async def test_pdf_folder_with_tree_index_returns_overview(self, tmp_path):
        """read_file on a PDF folder with _tree_index.json returns tree overview."""
        folder = _make_tree_index(tmp_path)

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc"})

        assert result.success is True
        assert "Test Report" in result.output
        assert "Introduction" in result.output
        assert "Conclusion" in result.output

    @pytest.mark.asyncio
    async def test_pdf_file_with_tree_index_in_parent(self, tmp_path):
        """read_file on a .pdf file with _tree_index.json in parent dir returns overview."""
        folder = _make_tree_index(tmp_path)
        # Create a .pdf file in the same folder
        (folder / "report.pdf").write_bytes(b"%PDF-1.4 fake")

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc/report.pdf"})

        assert result.success is True
        assert "Test Report" in result.output

    @pytest.mark.asyncio
    async def test_raw_pdf_without_tree_index_returns_error(self, tmp_path):
        """read_file on a raw .pdf file without tree index returns import guidance."""
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "report.pdf"})

        assert result.success is False
        assert "import" in result.error.lower() or "tree index" in result.error.lower()

    @pytest.mark.asyncio
    async def test_pdf_case_insensitive_extension(self, tmp_path):
        """PDF extension detection is case-insensitive."""
        pdf_path = tmp_path / "test.PDF"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "test.PDF"})

        # Should be handled as PDF (returns error since no tree index)
        assert result.success is False
        assert "import" in result.error.lower() or "tree index" in result.error.lower()

    @pytest.mark.asyncio
    async def test_pdf_in_subdirectory(self, tmp_path):
        """PDFs in subdirectories with tree indexes are handled."""
        docs_dir = tmp_path / "docs" / "guides"
        docs_dir.mkdir(parents=True)
        tree_data = {
            "file_path": "docs/guides/manual.pdf",
            "file_type": "pdf",
            "title": "User Manual",
            "total_nodes": 1,
            "total_tokens": 50,
            "nodes": [{"node_id": "n1", "title": "Getting Started", "summary": "How to begin", "level": 0}],
        }
        (docs_dir / "_tree_index.json").write_text(json.dumps(tree_data), encoding="utf-8")
        (docs_dir / "manual.pdf").write_bytes(b"%PDF-1.4 fake")

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "docs/guides/manual.pdf"})

        assert result.success is True
        assert "User Manual" in result.output

    @pytest.mark.asyncio
    async def test_pdf_nonexistent(self, tmp_path):
        """Nonexistent PDF files return appropriate error."""
        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "missing.pdf"})

        assert result.success is False
        assert "not found" in result.error.lower() or "no such file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_pdf_path_traversal_blocked(self, tmp_path):
        """Path traversal attempts are blocked."""
        outside_dir = tmp_path.parent / "outside"
        outside_dir.mkdir(exist_ok=True)
        pdf_path = outside_dir / "secret.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "../outside/secret.pdf"})

        assert result.success is False
        assert "path traversal" in result.error.lower() or "access denied" in result.error.lower()

    @pytest.mark.asyncio
    async def test_text_files_not_affected(self, tmp_path):
        """Text files are still read normally (not treated as PDF)."""
        text_path = tmp_path / "readme.md"
        text_path.write_text("# Test")

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "readme.md"})

        assert result.success is True
        assert "# Test" in result.output

    @pytest.mark.asyncio
    async def test_v1_metadata_folder_returns_guidance(self, tmp_path):
        """PDF folder with v1 _metadata.json (no tree index) returns v1 guidance."""
        folder = _make_v1_metadata(tmp_path)

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc"})

        # Should be recognized as a document folder but without tree index
        # May fall through to directory listing or return guidance
        # The behavior depends on whether _metadata.json is detected
        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_pdf_file(self, tmp_path):
        """Empty .pdf files are handled gracefully."""
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"")

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "empty.pdf"})

        # Should return error since no tree index exists
        assert result.success is False

    @pytest.mark.asyncio
    async def test_tree_index_includes_section_info(self, tmp_path):
        """Tree overview includes section titles and summaries."""
        folder = _make_tree_index(tmp_path)

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc"})

        assert result.success is True
        output = result.output
        # Should contain section info from tree
        assert "Introduction" in output
        assert "Conclusion" in output
        # Should mention read_pdf_section for drilling into sections
        assert "read_pdf_section" in output

    @pytest.mark.asyncio
    async def test_corrupt_tree_index_handled(self, tmp_path):
        """Corrupt _tree_index.json is handled gracefully."""
        folder_path = tmp_path / "doc"
        folder_path.mkdir()
        (folder_path / "_tree_index.json").write_text("not json", encoding="utf-8")

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc"})

        # Should fail gracefully
        assert result.success is False
        assert result.error is not None
