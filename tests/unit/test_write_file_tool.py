"""Tests for the write_file executor tool."""
import asyncio
import os
import pytest
from dokumen.tools_object import create_write_file_tool


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    return str(tmp_path)


def run(coro):
    """Helper to run async coroutines in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestWriteFileTool:
    """Tests for create_write_file_tool."""

    def test_creates_tool_definition(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        assert tool.name == "write_file"
        assert tool.description
        assert tool.parameters
        assert "file_path" in tool.parameters["properties"]
        assert "content" in tool.parameters["properties"]

    def test_writes_new_file(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({"file_path": "test.md", "content": "hello world"}))
        assert result.success is True
        with open(os.path.join(tmp_workspace, "test.md")) as f:
            assert f.read() == "hello world"

    def test_overwrites_existing_file(self, tmp_workspace):
        path = os.path.join(tmp_workspace, "existing.md")
        with open(path, "w") as f:
            f.write("old content")
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({"file_path": "existing.md", "content": "new content"}))
        assert result.success is True
        with open(path) as f:
            assert f.read() == "new content"

    def test_creates_parent_directories(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({
            "file_path": "docs/research/report.md",
            "content": "# Report"
        }))
        assert result.success is True
        with open(os.path.join(tmp_workspace, "docs/research/report.md")) as f:
            assert f.read() == "# Report"

    def test_append_mode(self, tmp_workspace):
        path = os.path.join(tmp_workspace, "log.md")
        with open(path, "w") as f:
            f.write("line 1\n")
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({
            "file_path": "log.md",
            "content": "line 2\n",
            "append": True
        }))
        assert result.success is True
        with open(path) as f:
            assert f.read() == "line 1\nline 2\n"

    def test_missing_file_path(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({"content": "hello"}))
        assert result.success is False
        assert "file_path" in result.error

    def test_missing_content(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({"file_path": "test.md"}))
        assert result.success is False
        assert "content" in result.error

    def test_path_traversal_blocked(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({
            "file_path": "../../etc/passwd",
            "content": "hacked"
        }))
        assert result.success is False
        assert "traversal" in result.error.lower() or "denied" in result.error.lower()

    def test_absolute_path_outside_base_blocked(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({
            "file_path": "/tmp/outside.md",
            "content": "hacked"
        }))
        assert result.success is False

    def test_large_content_writes_successfully(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        large_content = "x" * 100_000
        result = run(tool.handler({
            "file_path": "large.md",
            "content": large_content
        }))
        assert result.success is True
        with open(os.path.join(tmp_workspace, "large.md")) as f:
            assert len(f.read()) == 100_000

    def test_output_confirms_path_and_size(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({
            "file_path": "test.md",
            "content": "hello"
        }))
        assert result.success is True
        assert "test.md" in result.output
        assert "5" in result.output  # 5 bytes


    def test_symlink_outside_base_blocked(self, tmp_workspace):
        """Symlink pointing outside base_dir must be blocked."""
        import tempfile
        external_file = os.path.join(tempfile.gettempdir(), "write_file_test_target.txt")
        # Create the external target
        with open(external_file, "w") as f:
            f.write("original")
        # Create symlink inside workspace pointing to external file
        link_path = os.path.join(tmp_workspace, "sneaky.txt")
        os.symlink(external_file, link_path)
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({
            "file_path": "sneaky.txt",
            "content": "hacked"
        }))
        assert result.success is False
        assert "symlink" in result.error.lower() or "denied" in result.error.lower()
        # Verify external file was NOT modified
        with open(external_file) as f:
            assert f.read() == "original"
        os.unlink(external_file)

    def test_empty_content_creates_file(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({"file_path": "empty.md", "content": ""}))
        assert result.success is True
        assert os.path.getsize(os.path.join(tmp_workspace, "empty.md")) == 0

    def test_append_to_nonexistent_creates_file(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({
            "file_path": "new.md",
            "content": "first line",
            "append": True
        }))
        assert result.success is True
        with open(os.path.join(tmp_workspace, "new.md")) as f:
            assert f.read() == "first line"

    def test_nested_traversal_blocked(self, tmp_workspace):
        """Nested path traversal like subdir/../../etc/passwd must be blocked."""
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({
            "file_path": "subdir/../../etc/passwd",
            "content": "hacked"
        }))
        assert result.success is False

    def test_null_byte_in_path_handled(self, tmp_workspace):
        tool = create_write_file_tool(tmp_workspace)
        result = run(tool.handler({
            "file_path": "test\x00.md",
            "content": "hello"
        }))
        assert result.success is False


class TestWriteFileInToolDicts:
    """Test that write_file is properly registered."""

    def test_in_builtin_tools(self):
        from dokumen.tools_object import BUILTIN_TOOLS
        assert "write_file" in BUILTIN_TOOLS

    def test_in_cli_resolvable_tools(self):
        from dokumen_schema.constants import CLI_RESOLVABLE_TOOLS
        assert "write_file" in CLI_RESOLVABLE_TOOLS

    def test_in_valid_executor_tools(self):
        from dokumen_schema.constants import VALID_EXECUTOR_TOOLS
        assert "write_file" in VALID_EXECUTOR_TOOLS
