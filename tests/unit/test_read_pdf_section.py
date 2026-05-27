"""Tests for the read_pdf_section CLI tool.

Tests cover:
- Tool creation and schema
- Section lookup by title (case-insensitive)
- Section lookup by node_id
- Error when tree index doesn't exist
- Error when section not found
- Path traversal protection
- Nested sections with children summaries
- PDF file path resolution (resolves to parent folder)
- Directory path handling
- Missing parameters
"""

import json
import os

import pytest

from dokumen.tools_object import create_read_pdf_section_tool, ToolResult


def _make_tree_index(nodes=None, title="Test Document", page_count=5):
    """Build a minimal _tree_index.json dict for testing."""
    if nodes is None:
        nodes = [
            {
                "node_id": "sec-1",
                "title": "Introduction",
                "summary": "Overview of the document",
                "text": "This is the introduction text with details about the document.",
                "level": 0,
                "page_index": 0,
                "token_count": 50,
            },
            {
                "node_id": "sec-2",
                "title": "Configuration Guide",
                "summary": "How to configure the system",
                "text": "Configuration requires setting up environment variables.",
                "level": 0,
                "page_index": 2,
                "token_count": 40,
                "nodes": [
                    {
                        "node_id": "sec-2-1",
                        "title": "Environment Variables",
                        "summary": "Required env vars for the app",
                        "text": "Set API_KEY and DB_URL in your .env file.",
                        "level": 1,
                        "page_index": 3,
                        "token_count": 30,
                    },
                    {
                        "node_id": "sec-2-2",
                        "title": "Database Setup",
                        "summary": "How to set up PostgreSQL",
                        "text": "",
                        "level": 1,
                        "page_index": 4,
                        "token_count": 0,
                    },
                ],
            },
            {
                "node_id": "sec-3",
                "title": "API Reference",
                "summary": "",
                "text": "",
                "level": 0,
                "token_count": 0,
            },
        ]
    return {
        "file_path": "test.pdf",
        "file_type": "pdf",
        "title": title,
        "page_count": page_count,
        "total_nodes": 5,
        "total_tokens": 120,
        "nodes": nodes,
    }


def _write_tree_index(pdf_folder, tree_data=None):
    """Write a _tree_index.json into the given folder."""
    if tree_data is None:
        tree_data = _make_tree_index()
    index_path = os.path.join(str(pdf_folder), "_tree_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(tree_data, f)
    return index_path


class TestReadPdfSectionToolCreation:
    """Tests for tool creation and schema validation."""

    def test_tool_creates_successfully(self, tmp_path):
        """Tool factory returns a valid ToolDefinition."""
        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        assert tool.name == "read_pdf_section"
        assert "section" in tool.description.lower()
        assert callable(tool.handler)

    def test_tool_schema_has_required_params(self, tmp_path):
        """Tool schema requires file_path and section."""
        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        assert tool.parameters["type"] == "object"
        assert "file_path" in tool.parameters["properties"]
        assert "section" in tool.parameters["properties"]
        assert set(tool.parameters["required"]) == {"file_path", "section"}


class TestReadPdfSectionByTitle:
    """Tests for finding sections by title (case-insensitive)."""

    @pytest.mark.asyncio
    async def test_find_section_by_exact_title(self, tmp_path):
        """Finds a section when title matches exactly."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "Introduction"})

        assert result.success is True
        assert "Introduction" in result.output
        assert "introduction text" in result.output

    @pytest.mark.asyncio
    async def test_find_section_by_title_case_insensitive(self, tmp_path):
        """Title search is case-insensitive."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "introduction"})

        assert result.success is True
        assert "Introduction" in result.output

    @pytest.mark.asyncio
    async def test_find_section_by_partial_title(self, tmp_path):
        """Title search matches substring."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "Config"})

        assert result.success is True
        assert "Configuration Guide" in result.output


class TestReadPdfSectionByNodeId:
    """Tests for finding sections by node_id."""

    @pytest.mark.asyncio
    async def test_find_section_by_node_id(self, tmp_path):
        """Finds a section by exact node_id."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "sec-1"})

        assert result.success is True
        assert "Introduction" in result.output
        assert "sec-1" in result.output

    @pytest.mark.asyncio
    async def test_find_nested_section_by_node_id(self, tmp_path):
        """Finds a nested child section by node_id."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "sec-2-1"})

        assert result.success is True
        assert "Environment Variables" in result.output
        assert "API_KEY" in result.output


class TestReadPdfSectionErrors:
    """Tests for error conditions."""

    @pytest.mark.asyncio
    async def test_error_when_tree_index_missing(self, tmp_path):
        """Returns error when _tree_index.json doesn't exist."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        # No _tree_index.json written

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "Introduction"})

        assert result.success is False
        assert "No tree index found" in result.error

    @pytest.mark.asyncio
    async def test_error_when_section_not_found(self, tmp_path):
        """Returns error with hints when section doesn't exist."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "nonexistent-section"})

        assert result.success is False
        assert "not found" in result.error
        # Should include available sections as hints
        assert "Introduction" in result.error
        assert "Configuration Guide" in result.error

    @pytest.mark.asyncio
    async def test_error_missing_file_path_param(self, tmp_path):
        """Returns error when file_path parameter is missing."""
        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"section": "Introduction"})

        assert result.success is False
        assert "file_path" in result.error

    @pytest.mark.asyncio
    async def test_error_missing_section_param(self, tmp_path):
        """Returns error when section parameter is missing."""
        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf"})

        assert result.success is False
        assert "section" in result.error

    @pytest.mark.asyncio
    async def test_error_invalid_tree_index_json(self, tmp_path):
        """Returns error when _tree_index.json contains invalid JSON."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        index_path = pdf_folder / "_tree_index.json"
        index_path.write_text("not valid json {{{")

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "Introduction"})

        assert result.success is False
        assert "Failed to read tree index" in result.error


class TestReadPdfSectionPathTraversal:
    """Tests for path traversal protection."""

    @pytest.mark.asyncio
    async def test_blocks_path_traversal_dotdot(self, tmp_path):
        """Blocks ../../../etc/passwd style traversal."""
        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({
            "file_path": "../../../etc/passwd",
            "section": "Introduction",
        })

        assert result.success is False
        assert "path traversal" in result.error.lower() or "Access denied" in result.error

    @pytest.mark.asyncio
    async def test_blocks_absolute_path_outside_base(self, tmp_path):
        """Blocks absolute paths outside the base directory."""
        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({
            "file_path": "/etc/shadow",
            "section": "users",
        })

        assert result.success is False
        assert "Access denied" in result.error


class TestReadPdfSectionNestedContent:
    """Tests for nested section handling and output formatting."""

    @pytest.mark.asyncio
    async def test_section_with_children_shows_subsections(self, tmp_path):
        """Section with children includes subsection listing."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "sec-2"})

        assert result.success is True
        assert "Configuration Guide" in result.output
        assert "Subsections" in result.output
        assert "Environment Variables" in result.output
        assert "Database Setup" in result.output
        assert "sec-2-1" in result.output

    @pytest.mark.asyncio
    async def test_section_with_page_info(self, tmp_path):
        """Output includes page number when available."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "sec-1"})

        assert result.success is True
        # page_index=0 should display as "Page: 1" (1-indexed for humans)
        assert "Page: 1" in result.output

    @pytest.mark.asyncio
    async def test_section_without_text_shows_summary(self, tmp_path):
        """Section with no text but a summary shows the summary."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        # sec-2-2 has empty text but a summary
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "sec-2-2"})

        assert result.success is True
        assert "Database Setup" in result.output
        assert "PostgreSQL" in result.output

    @pytest.mark.asyncio
    async def test_section_without_text_or_summary(self, tmp_path):
        """Section with neither text nor summary shows placeholder."""
        pdf_folder = tmp_path / "doc.pdf"
        pdf_folder.mkdir()
        # sec-3 has no text and no summary
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "doc.pdf", "section": "sec-3"})

        assert result.success is True
        assert "No text content available" in result.output


class TestReadPdfSectionPathResolution:
    """Tests for PDF path resolution (file path vs directory path)."""

    @pytest.mark.asyncio
    async def test_handles_directory_path_directly(self, tmp_path):
        """Works when given a directory path containing _tree_index.json."""
        pdf_folder = tmp_path / "my-document"
        pdf_folder.mkdir()
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "my-document", "section": "Introduction"})

        assert result.success is True
        assert "Introduction" in result.output

    @pytest.mark.asyncio
    async def test_handles_pdf_folder_name(self, tmp_path):
        """Works with a folder named like a PDF file (e.g., 'report.pdf/')."""
        pdf_folder = tmp_path / "report.pdf"
        pdf_folder.mkdir()
        _write_tree_index(pdf_folder)

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "report.pdf", "section": "sec-2"})

        assert result.success is True
        assert "Configuration Guide" in result.output

    @pytest.mark.asyncio
    async def test_handles_pdf_file_path_resolves_to_parent(self, tmp_path):
        """When given path to a .pdf file, resolves to its parent directory."""
        # Create structure: tmp_path/docs/report.pdf (file) and tmp_path/docs/_tree_index.json
        docs_folder = tmp_path / "docs"
        docs_folder.mkdir()
        _write_tree_index(docs_folder)
        # Create a dummy .pdf file so the path exists as a file
        pdf_file = docs_folder / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake pdf")

        tool = create_read_pdf_section_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "docs/report.pdf", "section": "Introduction"})

        assert result.success is True
        assert "Introduction" in result.output


class TestReadPdfSectionRegistration:
    """Tests that the tool is properly registered in BUILTIN_TOOLS."""

    def test_tool_in_builtin_tools(self):
        """read_pdf_section is registered in BUILTIN_TOOLS."""
        from dokumen.tools_object import BUILTIN_TOOLS
        assert "read_pdf_section" in BUILTIN_TOOLS

    def test_tool_in_all_tool_names(self):
        """read_pdf_section appears in get_all_tool_names()."""
        from dokumen.tools_object import get_all_tool_names
        assert "read_pdf_section" in get_all_tool_names()

    def test_tool_resolved_by_loader(self, tmp_path):
        """Loader resolve_tools can resolve read_pdf_section."""
        from dokumen.loader import resolve_tools
        tools = resolve_tools(["read_pdf_section"], base_dir=str(tmp_path))
        assert len(tools) == 1
        assert tools[0].name == "read_pdf_section"
