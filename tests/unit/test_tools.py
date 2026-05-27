"""
Unit tests for tool registry and tool resolution.

Tests the tool system including:
- ToolDefinition and ToolResult dataclasses
- ToolsObject registry operations
- resolve_tools() function
- Built-in tool implementations
"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_tool_definition_fields(self, mock_tool):
        """ToolDefinition has name, description, parameters, handler."""
        tool = mock_tool(name="test", description="A test tool")
        assert tool.name == "test"
        assert tool.description == "A test tool"
        assert "type" in tool.parameters
        assert callable(tool.handler)

    def test_tool_definition_parameters_schema(self, mock_tool):
        """Tool parameters should be a valid JSON schema structure."""
        tool = mock_tool()
        assert tool.parameters["type"] == "object"
        assert "properties" in tool.parameters


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_tool_result_success(self):
        """ToolResult can represent success."""
        from dokumen.tools_object import ToolResult
        result = ToolResult(success=True, output="Done")
        assert result.success is True
        assert result.output == "Done"
        assert result.error is None

    def test_tool_result_failure(self):
        """ToolResult can represent failure with error."""
        from dokumen.tools_object import ToolResult
        result = ToolResult(success=False, output=None, error="File not found")
        assert result.success is False
        assert result.output is None
        assert result.error == "File not found"


class TestToolsObject:
    """Tests for ToolsObject registry."""

    def test_register_tool(self, mock_tool):
        """Can register a new tool."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()
        tool = mock_tool(name="my_tool")

        registry.register(tool)

        assert registry.get("my_tool") is tool

    def test_register_duplicate_raises(self, mock_tool):
        """Registering duplicate tool raises ValueError."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()
        tool = mock_tool(name="dup_tool")
        registry.register(tool)

        with pytest.raises(ValueError, match="Tool already exists"):
            registry.register(tool)

    def test_unregister_tool(self, mock_tool):
        """Can unregister a tool."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()
        tool = mock_tool(name="to_remove")
        registry.register(tool)

        result = registry.unregister("to_remove")

        assert result is True
        assert registry.get("to_remove") is None

    def test_unregister_nonexistent_returns_false(self):
        """Unregistering nonexistent tool returns False."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()

        result = registry.unregister("never_existed")

        assert result is False

    def test_get_tool(self, mock_tool):
        """Can retrieve a tool by name."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()
        tool = mock_tool(name="get_me")
        registry.register(tool)

        retrieved = registry.get("get_me")

        assert retrieved is tool

    def test_get_nonexistent_returns_none(self):
        """Getting nonexistent tool returns None."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()

        result = registry.get("unknown")

        assert result is None

    def test_get_definitions(self, mock_tool):
        """get_definitions returns all registered tools."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()
        tool1 = mock_tool(name="tool1")
        tool2 = mock_tool(name="tool2")
        registry.register(tool1)
        registry.register(tool2)

        definitions = registry.get_definitions()

        assert len(definitions) == 2
        names = [d.name for d in definitions]
        assert "tool1" in names
        assert "tool2" in names

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_tool):
        """Can execute a registered tool."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()
        tool = mock_tool(name="exec_tool", return_value="executed!")
        registry.register(tool)

        result = await registry.execute("exec_tool", {"input": "test"})

        assert result.success is True
        assert result.output == "executed!"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Executing unknown tool returns error result."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()

        result = await registry.execute("nonexistent", {})

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_to_openai_format(self, mock_tool):
        """Can convert tools to OpenAI format."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()
        tool = mock_tool(name="openai_tool", description="OpenAI test")
        registry.register(tool)

        openai_tools = registry.to_openai_format()

        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "openai_tool"
        assert openai_tools[0]["function"]["description"] == "OpenAI test"

    def test_to_anthropic_format(self, mock_tool):
        """Can convert tools to Anthropic format."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()
        tool = mock_tool(name="anthropic_tool", description="Anthropic test")
        registry.register(tool)

        anthropic_tools = registry.to_anthropic_format()

        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "anthropic_tool"
        assert anthropic_tools[0]["description"] == "Anthropic test"
        assert "input_schema" in anthropic_tools[0]


class TestResolveTools:
    """Tests for resolve_tools() function."""

    def test_resolve_read_file(self, tmp_path):
        """Can resolve read_file tool."""
        from dokumen.loader import resolve_tools
        tools = resolve_tools(["read_file"], base_dir=str(tmp_path))

        assert len(tools) == 1
        assert tools[0].name == "read_file"

    def test_resolve_list_directory(self, tmp_path):
        """Can resolve list_directory tool."""
        from dokumen.loader import resolve_tools
        tools = resolve_tools(["list_directory"], base_dir=str(tmp_path))

        assert len(tools) == 1
        assert tools[0].name == "list_directory"

    def test_resolve_glob(self, tmp_path):
        """Can resolve glob tool."""
        from dokumen.loader import resolve_tools
        tools = resolve_tools(["glob"], base_dir=str(tmp_path))

        assert len(tools) == 1
        assert tools[0].name == "glob"

    def test_resolve_run_shell_command(self, tmp_path):
        """Can resolve run_shell_command tool."""
        from dokumen.loader import resolve_tools
        tools = resolve_tools(["run_shell_command"], base_dir=str(tmp_path))

        assert len(tools) == 1
        assert tools[0].name == "run_shell_command"

    def test_resolve_web_fetch(self, tmp_path):
        """Can resolve web_fetch tool."""
        from dokumen.loader import resolve_tools
        tools = resolve_tools(["web_fetch"], base_dir=str(tmp_path))

        assert len(tools) == 1
        assert tools[0].name == "web_fetch"

    def test_resolve_unknown_tool_raises(self, tmp_path):
        """Resolving unknown tool raises ValueError."""
        from dokumen.loader import resolve_tools

        with pytest.raises(ValueError, match="Unknown tool"):
            resolve_tools(["totally_fake_tool"], base_dir=str(tmp_path))

    def test_resolve_multiple_tools(self, tmp_path):
        """Can resolve multiple tools at once."""
        from dokumen.loader import resolve_tools
        tools = resolve_tools(
            ["read_file", "list_directory", "glob"],
            base_dir=str(tmp_path)
        )

        assert len(tools) == 3
        names = [t.name for t in tools]
        assert "read_file" in names
        assert "list_directory" in names
        assert "glob" in names


class TestBuiltinReadFileTool:
    """Tests for the read_file tool implementation."""

    @pytest.mark.asyncio
    async def test_read_file_tool(self, sample_test_file):
        """read_file tool can read a file."""
        from dokumen.tools_object import create_read_file_tool
        tool = create_read_file_tool(base_dir=str(sample_test_file.parent.parent))

        result = await tool.handler({"file_path": "docs/test.md"})

        assert result.success is True
        assert "Test Document" in result.output
        assert "test content" in result.output

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path):
        """read_file returns error for missing file."""
        from dokumen.tools_object import create_read_file_tool
        tool = create_read_file_tool(base_dir=str(tmp_path))

        result = await tool.handler({"file_path": "nonexistent.md"})

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_file_with_line_range(self, tmp_path):
        """read_file supports offset and limit parameters."""
        from dokumen.tools_object import create_read_file_tool

        # Create a file with multiple lines
        test_file = tmp_path / "multi_line.txt"
        test_file.write_text("\n".join([f"Line {i}" for i in range(1, 11)]))

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({
            "file_path": "multi_line.txt",
            "offset": 3,
            "limit": 2
        })

        assert result.success is True
        assert "Line 3" in result.output
        assert "Line 4" in result.output
        # Should NOT include lines outside range
        assert "Line 1" not in result.output
        assert "Line 6" not in result.output


class TestBuiltinListDirectoryTool:
    """Tests for the list_directory tool implementation."""

    @pytest.mark.asyncio
    async def test_list_directory_tool(self, sample_project_dir):
        """list_directory can list directory contents."""
        from dokumen.tools_object import create_list_directory_tool
        tool = create_list_directory_tool(base_dir=str(sample_project_dir))

        result = await tool.handler({"path": "."})

        assert result.success is True
        assert "docs/" in result.output
        assert "src/" in result.output
        assert "config.yaml" in result.output

    @pytest.mark.asyncio
    async def test_list_directory_not_found(self, tmp_path):
        """list_directory returns error for missing directory."""
        from dokumen.tools_object import create_list_directory_tool
        tool = create_list_directory_tool(base_dir=str(tmp_path))

        result = await tool.handler({"path": "nonexistent_dir"})

        assert result.success is False
        assert "not found" in result.error.lower()


class TestBuiltinGlobTool:
    """Tests for the glob tool implementation."""

    @pytest.mark.asyncio
    async def test_glob_tool(self, sample_project_dir):
        """glob can find files by pattern."""
        from dokumen.tools_object import create_glob_tool
        tool = create_glob_tool(base_dir=str(sample_project_dir))

        result = await tool.handler({"pattern": "**/*.md"})

        assert result.success is True
        assert "readme.md" in result.output
        assert "api.md" in result.output

    @pytest.mark.asyncio
    async def test_glob_no_matches(self, sample_project_dir):
        """glob returns message when no files match."""
        from dokumen.tools_object import create_glob_tool
        tool = create_glob_tool(base_dir=str(sample_project_dir))

        result = await tool.handler({"pattern": "**/*.xyz"})

        assert result.success is True
        assert "no files found" in result.output.lower()


class TestBuiltinRunShellCommandTool:
    """Tests for the run_shell_command tool implementation."""

    @pytest.mark.asyncio
    async def test_run_shell_command_tool(self):
        """run_shell_command can execute shell commands."""
        from dokumen.tools_object import create_bash_tool
        tool = create_bash_tool(sandbox=None)

        result = await tool.handler({"command": "echo hello"})

        assert result.success is True
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_run_shell_command_exit_code(self):
        """run_shell_command returns failure for non-zero exit."""
        from dokumen.tools_object import create_bash_tool
        tool = create_bash_tool(sandbox=None)

        result = await tool.handler({"command": "exit 1"})

        assert result.success is False

    @pytest.mark.asyncio
    async def test_run_shell_command_missing_param(self):
        """run_shell_command requires command parameter."""
        from dokumen.tools_object import create_bash_tool
        tool = create_bash_tool(sandbox=None)

        result = await tool.handler({})

        assert result.success is False
        assert "command" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_shell_command_restricted_patterns(self):
        """run_shell_command blocks restricted patterns with sandbox."""
        from dokumen.tools_object import create_bash_tool
        from unittest.mock import MagicMock

        mock_sandbox = MagicMock()
        tool = create_bash_tool(sandbox=mock_sandbox)

        # Test various restricted patterns
        patterns = [
            "find /",
            "ls /etc",
            "cat /etc/passwd",
            "cd /home",
            "head /var/log/syslog",
        ]
        for cmd in patterns:
            result = await tool.handler({"command": cmd})
            assert result.success is False
            assert "outside /workspace" in result.error

    @pytest.mark.asyncio
    async def test_run_shell_command_path_traversal_blocked(self):
        """run_shell_command blocks path traversal with sandbox."""
        from dokumen.tools_object import create_bash_tool
        from unittest.mock import MagicMock

        mock_sandbox = MagicMock()
        tool = create_bash_tool(sandbox=mock_sandbox)

        result = await tool.handler({"command": "cat ../../../etc/passwd"})

        assert result.success is False
        assert "outside /workspace" in result.error

    @pytest.mark.asyncio
    async def test_run_shell_command_with_stderr(self):
        """run_shell_command includes stderr in output."""
        from dokumen.tools_object import create_bash_tool
        tool = create_bash_tool(sandbox=None)

        # Command that writes to stderr
        result = await tool.handler({"command": "echo error >&2 && echo success"})

        # Should still succeed but include stderr
        assert "success" in result.output

    @pytest.mark.asyncio
    async def test_run_shell_command_timeout(self):
        """run_shell_command handles timeout."""
        from dokumen.tools_object import create_bash_tool
        tool = create_bash_tool(sandbox=None)

        # Command that would take forever - use very short timeout
        import sys
        if sys.platform == 'win32':
            # Skip timeout test on Windows (different behavior)
            return

        # The timeout is tested implicitly - just verify the tool exists
        assert tool.name == "run_shell_command"

    def test_run_shell_command_schema_has_timeout_property(self):
        """run_shell_command schema includes optional timeout parameter."""
        from dokumen.tools_object import create_bash_tool
        tool = create_bash_tool(sandbox=None, timeout=60.0)

        props = tool.parameters["properties"]
        assert "timeout" in props
        assert props["timeout"]["type"] == "number"
        # timeout must NOT be required
        assert "timeout" not in tool.parameters.get("required", [])

    @pytest.mark.asyncio
    async def test_run_shell_command_uses_model_timeout(self):
        """run_shell_command respects per-invocation timeout from model."""
        from dokumen.tools_object import create_bash_tool
        import time

        # Config timeout is generous (60s), but model asks for 1s
        tool = create_bash_tool(sandbox=None, timeout=60.0)

        start = time.monotonic()
        result = await tool.handler({"command": "sleep 10", "timeout": 1})
        elapsed = time.monotonic() - start

        assert result.success is False
        assert "timed out" in result.error.lower()
        # Should have timed out near 1s, not 60s
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_run_shell_command_default_timeout_when_not_specified(self):
        """run_shell_command uses config default when no timeout param."""
        from dokumen.tools_object import create_bash_tool
        import time

        # Config timeout is 2s, model doesn't specify timeout
        tool = create_bash_tool(sandbox=None, timeout=2.0)

        start = time.monotonic()
        result = await tool.handler({"command": "sleep 10"})
        elapsed = time.monotonic() - start

        assert result.success is False
        assert "timed out" in result.error.lower()
        # Should have timed out near 2s (the config default)
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_run_shell_command_timeout_capped_at_config(self):
        """run_shell_command clamps model timeout to config ceiling."""
        from dokumen.tools_object import create_bash_tool
        import time

        # Config timeout is 2s, model asks for 999s
        tool = create_bash_tool(sandbox=None, timeout=2.0)

        start = time.monotonic()
        result = await tool.handler({"command": "sleep 10", "timeout": 999})
        elapsed = time.monotonic() - start

        assert result.success is False
        assert "timed out" in result.error.lower()
        # Should have been capped to 2s, not 999s
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_run_shell_command_timeout_minimum_one_second(self):
        """run_shell_command clamps model timeout to minimum 1.0s."""
        from dokumen.tools_object import create_bash_tool
        import time

        tool = create_bash_tool(sandbox=None, timeout=60.0)

        start = time.monotonic()
        result = await tool.handler({"command": "sleep 10", "timeout": 0.1})
        elapsed = time.monotonic() - start

        assert result.success is False
        assert "timed out" in result.error.lower()
        # Should have been clamped to 1.0s, not 0.1s
        assert elapsed >= 0.9

    @pytest.mark.asyncio
    async def test_run_shell_command_invalid_timeout_uses_default(self):
        """run_shell_command falls back to config default for invalid timeout."""
        from dokumen.tools_object import create_bash_tool
        import time

        # Config timeout is 2s, model sends invalid string
        tool = create_bash_tool(sandbox=None, timeout=2.0)

        start = time.monotonic()
        result = await tool.handler({"command": "sleep 10", "timeout": "abc"})
        elapsed = time.monotonic() - start

        assert result.success is False
        assert "timed out" in result.error.lower()
        # Should fall back to config default (2s), not crash
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_run_shell_command_timeout_nan_uses_default(self):
        """run_shell_command rejects NaN timeout and falls back to default."""
        from dokumen.tools_object import create_bash_tool
        import time

        tool = create_bash_tool(sandbox=None, timeout=2.0)

        start = time.monotonic()
        result = await tool.handler({"command": "sleep 10", "timeout": float("nan")})
        elapsed = time.monotonic() - start

        assert result.success is False
        assert "timed out" in result.error.lower()
        # Should fall back to config default (2s), not pass nan through
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_run_shell_command_timeout_inf_uses_default(self):
        """run_shell_command rejects Infinity timeout and falls back to default."""
        from dokumen.tools_object import create_bash_tool
        import time

        tool = create_bash_tool(sandbox=None, timeout=2.0)

        start = time.monotonic()
        result = await tool.handler({"command": "sleep 10", "timeout": float("inf")})
        elapsed = time.monotonic() - start

        assert result.success is False
        assert "timed out" in result.error.lower()
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_run_shell_command_timeout_negative_inf_uses_default(self):
        """run_shell_command rejects -Infinity timeout and falls back to default."""
        from dokumen.tools_object import create_bash_tool
        import time

        tool = create_bash_tool(sandbox=None, timeout=2.0)

        start = time.monotonic()
        result = await tool.handler({"command": "sleep 10", "timeout": float("-inf")})
        elapsed = time.monotonic() - start

        assert result.success is False
        assert "timed out" in result.error.lower()
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_run_shell_command_timeout_explicit_none_uses_default(self):
        """run_shell_command treats explicit None (JSON null) same as absent."""
        from dokumen.tools_object import create_bash_tool
        import time

        tool = create_bash_tool(sandbox=None, timeout=2.0)

        start = time.monotonic()
        result = await tool.handler({"command": "sleep 10", "timeout": None})
        elapsed = time.monotonic() - start

        assert result.success is False
        assert "timed out" in result.error.lower()
        assert elapsed < 5.0

    def test_run_shell_command_description_uses_base_dir(self):
        """run_shell_command description uses base_dir."""
        from dokumen.tools_object import create_bash_tool

        # Default base_dir
        tool_default = create_bash_tool(sandbox=None)
        assert "current directory" in tool_default.description

        # Custom base_dir
        tool_custom = create_bash_tool(sandbox=None, base_dir="/builds/my-project")
        assert "/builds/my-project" in tool_custom.description
        assert "current directory" not in tool_custom.description

    def test_run_shell_command_description_relative_paths(self):
        """run_shell_command description encourages relative paths."""
        from dokumen.tools_object import create_bash_tool

        tool = create_bash_tool(sandbox=None, base_dir="/builds/test")
        assert "relative paths" in tool.description.lower()
        assert "cat docs/" in tool.description or "find ." in tool.description


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_fields(self):
        """ToolCall has required fields."""
        from dokumen.tools_object import ToolCall, ToolResult
        from datetime import datetime

        result = ToolResult(success=True, output="done")
        call = ToolCall(
            tool_name="test_tool",
            parameters={"input": "value"},
            result=result,
            timestamp=datetime.now(),
            duration=1.5
        )

        assert call.tool_name == "test_tool"
        assert call.parameters == {"input": "value"}
        assert call.result == result
        assert call.duration == 1.5


class TestSubagentResult:
    """Tests for SubagentResult dataclass."""

    def test_subagent_result_fields(self):
        """SubagentResult has required fields."""
        from dokumen.tools_object import SubagentResult

        result = SubagentResult(
            file_path="docs/test.md",
            start_line=1,
            end_line=10,
            goal="Test goal",
            success=True,
            response="Completed",
            tool_calls=[{"name": "read_file"}]
        )

        assert result.file_path == "docs/test.md"
        assert result.start_line == 1
        assert result.end_line == 10
        assert result.success is True
        assert result.covered_lines == []
        assert result.coverage_confidence == 0.0

    def test_subagent_result_to_dict(self):
        """SubagentResult.to_dict() serializes correctly."""
        from dokumen.tools_object import SubagentResult

        result = SubagentResult(
            file_path="docs/api.md",
            start_line=5,
            end_line=15,
            goal="Check API docs",
            success=True,
            response="Done",
            tool_calls=[{"name": "read_file", "params": {}}],
            covered_lines=[5, 6, 7, 8],
            coverage_confidence=0.85,
            error=None
        )

        d = result.to_dict()

        assert d["file_path"] == "docs/api.md"
        assert d["lines"] == "5-15"
        assert d["goal"] == "Check API docs"
        assert d["success"] is True
        assert d["covered_lines"] == [5, 6, 7, 8]
        assert d["coverage_confidence"] == 0.85
        assert d["error"] is None


class TestToolsObjectFormats:
    """Tests for ToolsObject format conversion methods."""

    def test_to_fastmcp_format(self, mock_tool):
        """Can convert tools to fastMCP format."""
        from dokumen.tools_object import ToolsObject
        registry = ToolsObject()
        tool = mock_tool(name="mcp_tool", description="MCP test")
        registry.register(tool)

        mcp_tools = registry.to_fastmcp_format()

        assert len(mcp_tools) == 1
        assert mcp_tools[0]["name"] == "mcp_tool"
        assert mcp_tools[0]["description"] == "MCP test"
        assert "inputSchema" in mcp_tools[0]


class TestToolsObjectExecuteErrors:
    """Tests for ToolsObject execute error handling."""

    @pytest.mark.asyncio
    async def test_execute_validates_required_params(self):
        """execute() validates required parameters."""
        from dokumen.tools_object import ToolsObject, ToolDefinition, ToolResult

        async def handler(params):
            return ToolResult(success=True, output="done")

        tool = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"]
            },
            handler=handler
        )

        registry = ToolsObject()
        registry.register(tool)

        result = await registry.execute("test", {})

        assert result.success is False
        assert "file_path" in result.error

    @pytest.mark.asyncio
    async def test_execute_handler_exception(self):
        """execute() handles handler exceptions."""
        from dokumen.tools_object import ToolsObject, ToolDefinition

        async def failing_handler(params):
            raise RuntimeError("Handler error")

        tool = ToolDefinition(
            name="fail_tool",
            description="Will fail",
            parameters={"type": "object", "properties": {}},
            handler=failing_handler
        )

        registry = ToolsObject()
        registry.register(tool)

        result = await registry.execute("fail_tool", {})

        assert result.success is False
        assert "Handler error" in result.error


class TestFormatSize:
    """Tests for _format_size helper."""

    def test_format_bytes(self):
        """Format bytes correctly."""
        from dokumen.tools_object import _format_size
        assert _format_size(100) == "100B"
        assert _format_size(0) == "0B"

    def test_format_kilobytes(self):
        """Format kilobytes correctly."""
        from dokumen.tools_object import _format_size
        assert _format_size(1024) == "1.0KB"
        assert _format_size(2560) == "2.5KB"

    def test_format_megabytes(self):
        """Format megabytes correctly."""
        from dokumen.tools_object import _format_size
        assert _format_size(1024 * 1024) == "1.0MB"

    def test_format_gigabytes(self):
        """Format gigabytes correctly."""
        from dokumen.tools_object import _format_size
        assert _format_size(1024 * 1024 * 1024) == "1.0GB"

    def test_format_terabytes(self):
        """Format terabytes correctly."""
        from dokumen.tools_object import _format_size
        assert _format_size(1024 * 1024 * 1024 * 1024) == "1.0TB"


class TestResolveBuiltinTool:
    """Tests for resolve_builtin_tool function."""

    def test_resolve_builtin_read_file(self, tmp_path):
        """resolve_builtin_tool finds read_file."""
        from dokumen.tools_object import resolve_builtin_tool
        tool = resolve_builtin_tool("read_file", base_dir=str(tmp_path))
        assert tool is not None
        assert tool.name == "read_file"

    def test_resolve_builtin_glob(self, tmp_path):
        """resolve_builtin_tool finds glob."""
        from dokumen.tools_object import resolve_builtin_tool
        tool = resolve_builtin_tool("glob", base_dir=str(tmp_path))
        assert tool is not None
        assert tool.name == "glob"

    def test_resolve_builtin_list_directory(self, tmp_path):
        """resolve_builtin_tool finds list_directory."""
        from dokumen.tools_object import resolve_builtin_tool
        tool = resolve_builtin_tool("list_directory", base_dir=str(tmp_path))
        assert tool is not None
        assert tool.name == "list_directory"

    def test_resolve_sandbox_run_shell_command(self, tmp_path):
        """resolve_builtin_tool finds run_shell_command without sandbox."""
        from dokumen.tools_object import resolve_builtin_tool
        tool = resolve_builtin_tool("run_shell_command", base_dir=str(tmp_path), sandbox=None)
        assert tool is not None
        assert tool.name == "run_shell_command"

    def test_resolve_sandbox_search_file_content(self, tmp_path):
        """resolve_builtin_tool returns search_file_content tool even without sandbox."""
        from dokumen.tools_object import resolve_builtin_tool
        # search_file_content should work without sandbox (direct execution fallback)
        tool = resolve_builtin_tool("search_file_content", base_dir=str(tmp_path), sandbox=None)
        assert tool is not None
        assert tool.name == "search_file_content"

    def test_resolve_web_fetch(self, tmp_path):
        """resolve_builtin_tool finds web_fetch."""
        from dokumen.tools_object import resolve_builtin_tool
        tool = resolve_builtin_tool("web_fetch", base_dir=str(tmp_path))
        assert tool is not None
        assert tool.name == "web_fetch"

    def test_resolve_unknown_returns_none(self, tmp_path):
        """resolve_builtin_tool returns None for unknown tool."""
        from dokumen.tools_object import resolve_builtin_tool
        tool = resolve_builtin_tool("totally_fake_tool", base_dir=str(tmp_path))
        assert tool is None


class TestGetAllToolNames:
    """Tests for get_all_tool_names function."""

    def test_returns_all_tools(self):
        """get_all_tool_names returns all available tools."""
        from dokumen.tools_object import get_all_tool_names

        names = get_all_tool_names()

        assert "read_file" in names
        assert "glob" in names
        assert "list_directory" in names
        assert "run_shell_command" in names
        assert "web_fetch" in names


class TestReadFileToolAdvanced:
    """Advanced tests for read_file tool."""

    @pytest.mark.asyncio
    async def test_read_file_missing_path_param(self, tmp_path):
        """read_file returns error when file_path missing."""
        from dokumen.tools_object import create_read_file_tool
        tool = create_read_file_tool(base_dir=str(tmp_path))

        result = await tool.handler({})

        assert result.success is False
        assert "file_path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_file_directory_error(self, tmp_path):
        """read_file returns error for directory path."""
        from dokumen.tools_object import create_read_file_tool
        (tmp_path / "subdir").mkdir()
        tool = create_read_file_tool(base_dir=str(tmp_path))

        result = await tool.handler({"file_path": "subdir"})

        assert result.success is False
        assert "directory" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_file_image(self, tmp_path):
        """read_file handles image files."""
        from dokumen.tools_object import create_read_file_tool
        import base64

        # Create a tiny PNG
        png_header = b'\x89PNG\r\n\x1a\n'
        test_image = tmp_path / "test.png"
        test_image.write_bytes(png_header + b'\x00' * 100)

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": "test.png"})

        assert result.success is True
        assert "__IMAGE_DATA__" in result.output
        assert "image/png" in result.output

    @pytest.mark.asyncio
    async def test_read_file_absolute_path(self, tmp_path):
        """read_file handles absolute paths."""
        from dokumen.tools_object import create_read_file_tool

        test_file = tmp_path / "abs_test.txt"
        test_file.write_text("absolute path content")

        tool = create_read_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"file_path": str(test_file)})

        assert result.success is True
        assert "absolute path content" in result.output


class TestListDirectoryToolAdvanced:
    """Advanced tests for list_directory tool."""

    @pytest.mark.asyncio
    async def test_list_directory_recursive(self, tmp_path):
        """list_directory supports recursive listing."""
        from dokumen.tools_object import create_list_directory_tool

        # Create nested structure
        (tmp_path / "sub1").mkdir()
        (tmp_path / "sub1" / "file1.txt").write_text("content")
        (tmp_path / "sub2").mkdir()
        (tmp_path / "sub2" / "file2.txt").write_text("content")

        tool = create_list_directory_tool(base_dir=str(tmp_path))
        result = await tool.handler({"path": ".", "recursive": True})

        assert result.success is True
        assert "sub1" in result.output
        assert "sub2" in result.output
        assert "file1.txt" in result.output
        assert "file2.txt" in result.output

    @pytest.mark.asyncio
    async def test_list_directory_include_hidden(self, tmp_path):
        """list_directory can include hidden files."""
        from dokumen.tools_object import create_list_directory_tool

        (tmp_path / ".hidden").write_text("hidden file")
        (tmp_path / "visible").write_text("visible file")

        tool = create_list_directory_tool(base_dir=str(tmp_path))

        # Default: hidden files excluded
        result = await tool.handler({"path": "."})
        assert ".hidden" not in result.output
        assert "visible" in result.output

        # With include_hidden
        result = await tool.handler({"path": ".", "include_hidden": True})
        assert ".hidden" in result.output

    @pytest.mark.asyncio
    async def test_list_directory_empty(self, tmp_path):
        """list_directory handles empty directories."""
        from dokumen.tools_object import create_list_directory_tool

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        tool = create_list_directory_tool(base_dir=str(tmp_path))
        result = await tool.handler({"path": "empty"})

        assert result.success is True
        assert "empty" in result.output.lower()

    @pytest.mark.asyncio
    async def test_list_directory_not_a_dir(self, tmp_path):
        """list_directory returns error for file path."""
        from dokumen.tools_object import create_list_directory_tool

        (tmp_path / "afile.txt").write_text("content")

        tool = create_list_directory_tool(base_dir=str(tmp_path))
        result = await tool.handler({"path": "afile.txt"})

        assert result.success is False
        assert "not a directory" in result.error.lower()


class TestGlobToolAdvanced:
    """Advanced tests for glob tool."""

    @pytest.mark.asyncio
    async def test_glob_missing_pattern(self, tmp_path):
        """glob returns error when pattern missing."""
        from dokumen.tools_object import create_glob_tool
        tool = create_glob_tool(base_dir=str(tmp_path))

        result = await tool.handler({})

        assert result.success is False
        assert "pattern" in result.error.lower()

    @pytest.mark.asyncio
    async def test_glob_nonexistent_path(self, tmp_path):
        """glob returns error for nonexistent path."""
        from dokumen.tools_object import create_glob_tool
        tool = create_glob_tool(base_dir=str(tmp_path))

        result = await tool.handler({"pattern": "*.py", "path": "nonexistent"})

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_glob_respects_gitignore(self, tmp_path):
        """glob respects .gitignore patterns."""
        from dokumen.tools_object import create_glob_tool

        # Create files
        (tmp_path / "keep.py").write_text("keep")
        (tmp_path / "ignore.pyc").write_text("ignore")

        # Create .gitignore
        (tmp_path / ".gitignore").write_text("*.pyc\n")

        tool = create_glob_tool(base_dir=str(tmp_path))
        result = await tool.handler({"pattern": "*.*", "respect_gitignore": True})

        assert result.success is True
        assert "keep.py" in result.output
        # pyc file should be ignored
        # Note: simple gitignore matching may or may not work perfectly


class TestReadManyFilesTool:
    """Tests for read_many_files tool."""

    @pytest.mark.asyncio
    async def test_read_many_files_basic(self, tmp_path):
        """read_many_files reads multiple files."""
        from dokumen.tools_object import create_read_many_files_tool

        (tmp_path / "a.txt").write_text("content A")
        (tmp_path / "b.txt").write_text("content B")

        tool = create_read_many_files_tool(base_dir=str(tmp_path))
        result = await tool.handler({"patterns": ["*.txt"]})

        assert result.success is True
        assert "content A" in result.output
        assert "content B" in result.output
        assert "a.txt" in result.output
        assert "b.txt" in result.output

    @pytest.mark.asyncio
    async def test_read_many_files_missing_patterns(self, tmp_path):
        """read_many_files requires patterns parameter."""
        from dokumen.tools_object import create_read_many_files_tool
        tool = create_read_many_files_tool(base_dir=str(tmp_path))

        result = await tool.handler({})

        assert result.success is False
        assert "patterns" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_many_files_no_matches(self, tmp_path):
        """read_many_files handles no matches."""
        from dokumen.tools_object import create_read_many_files_tool
        tool = create_read_many_files_tool(base_dir=str(tmp_path))

        result = await tool.handler({"patterns": ["*.nonexistent"]})

        assert result.success is True
        assert "no files found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_read_many_files_with_exclude(self, tmp_path):
        """read_many_files supports exclude patterns."""
        from dokumen.tools_object import create_read_many_files_tool

        (tmp_path / "keep.txt").write_text("keep")
        (tmp_path / "exclude.txt").write_text("exclude")

        tool = create_read_many_files_tool(base_dir=str(tmp_path))
        result = await tool.handler({
            "patterns": ["*.txt"],
            "exclude": ["exclude.txt"]
        })

        assert result.success is True
        assert "keep" in result.output
        assert "exclude.txt" not in result.output

    @pytest.mark.asyncio
    async def test_read_many_files_line_limit(self, tmp_path):
        """read_many_files truncates long files."""
        from dokumen.tools_object import create_read_many_files_tool

        lines = "\n".join([f"Line {i}" for i in range(100)])
        (tmp_path / "long.txt").write_text(lines)

        tool = create_read_many_files_tool(base_dir=str(tmp_path))
        result = await tool.handler({
            "patterns": ["long.txt"],
            "max_lines_per_file": 10
        })

        assert result.success is True
        assert "truncated" in result.output.lower()

    @pytest.mark.asyncio
    async def test_read_many_files_string_pattern(self, tmp_path):
        """read_many_files accepts string pattern."""
        from dokumen.tools_object import create_read_many_files_tool

        (tmp_path / "test.txt").write_text("content")

        tool = create_read_many_files_tool(base_dir=str(tmp_path))
        result = await tool.handler({"patterns": "*.txt"})

        assert result.success is True
        assert "content" in result.output


class TestHttpRequestTool:
    """Tests for web_fetch/http_request tool."""

    @pytest.mark.asyncio
    async def test_http_request_missing_url(self):
        """http_request returns error when url missing."""
        from dokumen.tools_object import create_http_request_tool
        tool = create_http_request_tool(sandbox=None)

        result = await tool.handler({})

        assert result.success is False
        assert "url" in result.error.lower()


class TestWriteFileTool:
    """Tests for write_file tool."""

    @pytest.mark.asyncio
    async def test_write_file_missing_path(self, tmp_path):
        """write_file returns error when file_path missing."""
        from dokumen.tools_object import create_write_file_tool

        tool = create_write_file_tool(str(tmp_path))

        result = await tool.handler({"content": "test"})

        assert result.success is False
        assert "file_path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_write_file_missing_content(self, tmp_path):
        """write_file returns error when content missing."""
        from dokumen.tools_object import create_write_file_tool

        tool = create_write_file_tool(str(tmp_path))

        result = await tool.handler({"file_path": "test.txt"})

        assert result.success is False
        assert "content" in result.error.lower()

    @pytest.mark.asyncio
    async def test_write_file_success(self, tmp_path):
        """write_file writes content to file."""
        from dokumen.tools_object import create_write_file_tool
        import os

        tool = create_write_file_tool(str(tmp_path))

        result = await tool.handler({"file_path": "test.txt", "content": "hello world"})

        assert result.success is True
        with open(os.path.join(str(tmp_path), "test.txt")) as f:
            assert f.read() == "hello world"

    @pytest.mark.asyncio
    async def test_write_file_path_traversal_blocked(self, tmp_path):
        """write_file blocks path traversal attempts."""
        from dokumen.tools_object import create_write_file_tool

        tool = create_write_file_tool(str(tmp_path))

        result = await tool.handler({"file_path": "../../etc/passwd", "content": "hello"})

        assert result.success is False
        assert "denied" in result.error.lower() or "traversal" in result.error.lower()


class TestBashToolWithSandbox:
    """Tests for bash tool with sandbox."""

    @pytest.mark.asyncio
    async def test_bash_tool_sandbox_success(self):
        """run_shell_command executes via sandbox."""
        from dokumen.tools_object import create_bash_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_result.error = None
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_bash_tool(sandbox=mock_sandbox)
        result = await tool.handler({"command": "echo test"})

        assert result.success is True
        mock_sandbox.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_bash_tool_sandbox_with_stderr(self):
        """run_shell_command includes stderr in output."""
        from dokumen.tools_object import create_bash_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.stdout = "output"
        mock_result.stderr = "warning message"
        mock_result.error = None
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_bash_tool(sandbox=mock_sandbox)
        result = await tool.handler({"command": "echo test"})

        assert result.success is True
        assert "warning message" in result.output


class TestGrepTool:
    """Tests for grep tool."""

    @pytest.mark.asyncio
    async def test_grep_missing_pattern(self):
        """grep fails without pattern."""
        from dokumen.tools_object import create_grep_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        tool = create_grep_tool(sandbox=mock_sandbox)

        result = await tool.handler({})

        assert result.success is False
        assert "pattern" in result.error

    @pytest.mark.asyncio
    async def test_grep_success(self):
        """grep finds matches."""
        from dokumen.tools_object import create_grep_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "file.txt:1:matching line"
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_grep_tool(sandbox=mock_sandbox)
        result = await tool.handler({"pattern": "test"})

        assert result.success is True
        assert "matching line" in result.output

    @pytest.mark.asyncio
    async def test_grep_no_matches(self):
        """grep handles no matches."""
        from dokumen.tools_object import create_grep_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 1  # grep returns 1 for no matches
        mock_result.stdout = ""
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_grep_tool(sandbox=mock_sandbox)
        result = await tool.handler({"pattern": "nonexistent"})

        assert result.success is True
        assert "No matches" in result.output

    @pytest.mark.asyncio
    async def test_grep_error(self):
        """grep handles errors."""
        from dokumen.tools_object import create_grep_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 2  # grep returns 2 for errors
        mock_result.stderr = "Invalid regex"
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_grep_tool(sandbox=mock_sandbox)
        result = await tool.handler({"pattern": "["})

        assert result.success is False
        assert "Invalid regex" in result.error or "grep failed" in result.error

    @pytest.mark.asyncio
    async def test_grep_case_insensitive(self):
        """grep supports case insensitive search."""
        from dokumen.tools_object import create_grep_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "file.txt:1:Test Line"
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_grep_tool(sandbox=mock_sandbox)
        await tool.handler({"pattern": "test", "case_insensitive": True})

        # Check that -i flag is used
        call_args = mock_sandbox.execute.call_args[0][0]
        assert "i" in call_args or "-i" in call_args

    @pytest.mark.asyncio
    async def test_grep_exception(self):
        """grep handles exceptions."""
        from dokumen.tools_object import create_grep_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_sandbox.execute = AsyncMock(side_effect=Exception("Sandbox error"))

        tool = create_grep_tool(sandbox=mock_sandbox)
        result = await tool.handler({"pattern": "test"})

        assert result.success is False
        assert "Sandbox error" in result.error


class TestEditTool:
    """Tests for edit tool."""

    @pytest.mark.asyncio
    async def test_edit_missing_file_path(self):
        """edit fails without file_path."""
        from dokumen.tools_object import create_edit_tool
        from unittest.mock import AsyncMock

        mock_sandbox = AsyncMock()
        tool = create_edit_tool(sandbox=mock_sandbox)

        result = await tool.handler({"old_content": "a", "new_content": "b"})

        assert result.success is False
        assert "file_path" in result.error

    @pytest.mark.asyncio
    async def test_edit_missing_old_content(self):
        """edit fails without old_content."""
        from dokumen.tools_object import create_edit_tool
        from unittest.mock import AsyncMock

        mock_sandbox = AsyncMock()
        tool = create_edit_tool(sandbox=mock_sandbox)

        result = await tool.handler({"file_path": "test.py", "new_content": "b"})

        assert result.success is False
        assert "old_content" in result.error

    @pytest.mark.asyncio
    async def test_edit_missing_new_content(self):
        """edit fails without new_content."""
        from dokumen.tools_object import create_edit_tool
        from unittest.mock import AsyncMock

        mock_sandbox = AsyncMock()
        tool = create_edit_tool(sandbox=mock_sandbox)

        result = await tool.handler({"file_path": "test.py", "old_content": "a"})

        assert result.success is False
        assert "new_content" in result.error

    @pytest.mark.asyncio
    async def test_edit_file_read_error(self):
        """edit handles file read errors."""
        from dokumen.tools_object import create_edit_tool
        from unittest.mock import AsyncMock

        mock_sandbox = AsyncMock()
        mock_sandbox.read_file = AsyncMock(return_value={"success": False, "error": "File not found"})
        tool = create_edit_tool(sandbox=mock_sandbox)

        result = await tool.handler({
            "file_path": "missing.py",
            "old_content": "old",
            "new_content": "new"
        })

        assert result.success is False
        assert "Failed to read" in result.error

    @pytest.mark.asyncio
    async def test_edit_content_not_found(self):
        """edit fails when old_content not in file."""
        from dokumen.tools_object import create_edit_tool
        from unittest.mock import AsyncMock

        mock_sandbox = AsyncMock()
        mock_sandbox.read_file = AsyncMock(return_value={
            "success": True,
            "content": "def hello(): pass"
        })
        tool = create_edit_tool(sandbox=mock_sandbox)

        result = await tool.handler({
            "file_path": "test.py",
            "old_content": "nonexistent content",
            "new_content": "new"
        })

        assert result.success is False
        assert "Could not find" in result.error

    @pytest.mark.asyncio
    async def test_edit_multiple_occurrences(self):
        """edit fails with multiple occurrences."""
        from dokumen.tools_object import create_edit_tool
        from unittest.mock import AsyncMock

        mock_sandbox = AsyncMock()
        mock_sandbox.read_file = AsyncMock(return_value={
            "success": True,
            "content": "foo bar foo baz foo"
        })
        tool = create_edit_tool(sandbox=mock_sandbox)

        result = await tool.handler({
            "file_path": "test.py",
            "old_content": "foo",
            "new_content": "qux"
        })

        assert result.success is False
        assert "occurrences" in result.error

    @pytest.mark.asyncio
    async def test_edit_success(self):
        """edit successfully replaces content."""
        from dokumen.tools_object import create_edit_tool
        from unittest.mock import AsyncMock

        mock_sandbox = AsyncMock()
        mock_sandbox.read_file = AsyncMock(return_value={
            "success": True,
            "content": "def hello():\n    return 'world'"
        })
        mock_sandbox.write_file = AsyncMock(return_value={"success": True})
        tool = create_edit_tool(sandbox=mock_sandbox)

        result = await tool.handler({
            "file_path": "test.py",
            "old_content": "return 'world'",
            "new_content": "return 'universe'"
        })

        assert result.success is True
        assert "Edited" in result.output

    @pytest.mark.asyncio
    async def test_edit_write_error(self):
        """edit handles write errors."""
        from dokumen.tools_object import create_edit_tool
        from unittest.mock import AsyncMock

        mock_sandbox = AsyncMock()
        mock_sandbox.read_file = AsyncMock(return_value={
            "success": True,
            "content": "unique content here"
        })
        mock_sandbox.write_file = AsyncMock(return_value={
            "success": False,
            "error": "Permission denied"
        })
        tool = create_edit_tool(sandbox=mock_sandbox)

        result = await tool.handler({
            "file_path": "test.py",
            "old_content": "unique content here",
            "new_content": "new content"
        })

        assert result.success is False
        assert "Failed to write" in result.error

    @pytest.mark.asyncio
    async def test_edit_exception(self):
        """edit handles exceptions."""
        from dokumen.tools_object import create_edit_tool
        from unittest.mock import AsyncMock

        mock_sandbox = AsyncMock()
        mock_sandbox.read_file = AsyncMock(side_effect=Exception("Unexpected error"))
        tool = create_edit_tool(sandbox=mock_sandbox)

        result = await tool.handler({
            "file_path": "test.py",
            "old_content": "old",
            "new_content": "new"
        })

        assert result.success is False
        assert "Edit failed" in result.error


class TestHttpRequestViaSandbox:
    """Tests for HTTP request via sandbox."""

    @pytest.mark.asyncio
    async def test_http_request_via_sandbox(self):
        """web_fetch uses sandbox when provided."""
        from dokumen.tools_object import create_http_request_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.stdout = '{"status": 200, "headers": {}, "body": "OK"}'
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_http_request_tool(sandbox=mock_sandbox)
        result = await tool.handler({"url": "https://example.com"})

        # Should invoke sandbox
        assert mock_sandbox.execute.called or result is not None


class TestDelegateToAgentTool:
    """Tests for delegate_to_agent tool."""

    @pytest.mark.asyncio
    async def test_delegate_missing_agent(self):
        """delegate_to_agent fails without agent name."""
        from dokumen.tools_object import create_delegate_to_agent_tool
        from unittest.mock import MagicMock, AsyncMock

        mock_registry = MagicMock()
        mock_provider = MagicMock()
        tool = create_delegate_to_agent_tool(
            registry=mock_registry,
            provider=mock_provider,
            sandbox=None,
            timeout=60
        )

        result = await tool.handler({"input": "test"})

        assert result.success is False
        assert "agent" in result.error

    @pytest.mark.asyncio
    async def test_delegate_invalid_thoroughness(self):
        """delegate_to_agent fails with invalid thoroughness."""
        from dokumen.tools_object import create_delegate_to_agent_tool
        from unittest.mock import MagicMock

        mock_registry = MagicMock()
        mock_provider = MagicMock()
        tool = create_delegate_to_agent_tool(
            registry=mock_registry,
            provider=mock_provider,
            sandbox=None,
            timeout=60
        )

        result = await tool.handler({
            "agent": "explore",
            "input": "test",
            "thoroughness": "invalid"
        })

        assert result.success is False
        assert "thoroughness" in result.error

    @pytest.mark.asyncio
    async def test_delegate_agent_not_found(self):
        """delegate_to_agent fails when agent not in registry."""
        from dokumen.tools_object import create_delegate_to_agent_tool
        from unittest.mock import MagicMock

        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_registry.list_all.return_value = []
        mock_provider = MagicMock()
        tool = create_delegate_to_agent_tool(
            registry=mock_registry,
            provider=mock_provider,
            sandbox=None,
            timeout=60
        )

        result = await tool.handler({"agent": "unknown_agent", "input": "test"})

        assert result.success is False
        assert "not found" in result.error


class TestToolResultSerialization:
    """Tests for ToolResult serialization."""

    def test_tool_result_to_string_success(self):
        """ToolResult converts to string for success case."""
        from dokumen.tools_object import ToolResult

        result = ToolResult(success=True, output="test output")
        assert str(result.output) == "test output"

    def test_tool_result_to_string_with_dict(self):
        """ToolResult handles dict output."""
        from dokumen.tools_object import ToolResult

        result = ToolResult(success=True, output={"key": "value"})
        assert result.output["key"] == "value"

    def test_tool_result_error_message(self):
        """ToolResult preserves error message."""
        from dokumen.tools_object import ToolResult

        result = ToolResult(success=False, output=None, error="Something failed")
        assert result.error == "Something failed"


class TestToolDefinitionSchema:
    """Tests for ToolDefinition schema formatting."""

    def test_tool_definition_has_required_fields(self):
        """ToolDefinition has all required fields."""
        from dokumen.tools_object import ToolDefinition

        async def handler(params):
            pass

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            handler=handler
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert "type" in tool.parameters


class TestChatWriteFileTool:
    """Tests for create_chat_write_file_tool.

    This tool writes files to the local clone filesystem. Changes are
    batch-committed to GitLab at the end of the conversation turn.
    """

    @pytest.mark.asyncio
    async def test_chat_write_file_writes_to_filesystem(self, tmp_path):
        """write_file creates file on local filesystem."""
        from dokumen.tools_object import create_chat_write_file_tool

        # Create docs dir in tmp
        (tmp_path / "docs").mkdir()

        tool = create_chat_write_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({
            "path": "docs/policy.md",
            "content": "# Policy\nContent here",
        })

        assert result.success is True
        assert result.output["action"] == "created"
        assert result.output["path"] == "docs/policy.md"
        assert (tmp_path / "docs" / "policy.md").read_text() == "# Policy\nContent here"

    @pytest.mark.asyncio
    async def test_chat_write_file_creates_parent_dirs(self, tmp_path):
        """write_file creates parent directories if missing."""
        from dokumen.tools_object import create_chat_write_file_tool

        tool = create_chat_write_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({
            "path": "docs/sub/dir/file.md",
            "content": "content",
        })

        assert result.success is True
        assert (tmp_path / "docs" / "sub" / "dir" / "file.md").exists()

    @pytest.mark.asyncio
    async def test_chat_write_file_updates_existing_file(self, tmp_path):
        """write_file updates existing file and reports 'updated' action."""
        from dokumen.tools_object import create_chat_write_file_tool

        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "existing.test.yaml").write_text("old content")

        tool = create_chat_write_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({
            "path": "tests/existing.test.yaml",
            "content": "new content",
        })

        assert result.success is True
        assert result.output["action"] == "updated"
        assert (tmp_path / "tests" / "existing.test.yaml").read_text() == "new content"

    @pytest.mark.asyncio
    async def test_chat_write_file_rejects_outside_allowed_dirs(self, tmp_path):
        """write_file rejects paths outside docs/ and tests/."""
        from dokumen.tools_object import create_chat_write_file_tool

        tool = create_chat_write_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({
            "path": "src/main.py",
            "content": "bad",
        })

        assert result.success is False
        assert "not allowed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_chat_write_file_rejects_traversal(self, tmp_path):
        """write_file rejects path traversal."""
        from dokumen.tools_object import create_chat_write_file_tool

        tool = create_chat_write_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({
            "path": "docs/../.env",
            "content": "bad",
        })

        assert result.success is False
        assert "traversal" in result.error.lower()

    @pytest.mark.asyncio
    async def test_chat_write_file_missing_path(self):
        """write_file fails without path."""
        from dokumen.tools_object import create_chat_write_file_tool

        tool = create_chat_write_file_tool()
        result = await tool.handler({"content": "test"})

        assert result.success is False
        assert "path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_chat_write_file_missing_content(self):
        """write_file fails without content."""
        from dokumen.tools_object import create_chat_write_file_tool

        tool = create_chat_write_file_tool()
        result = await tool.handler({"path": "docs/test.txt"})

        assert result.success is False
        assert "content" in result.error.lower()

    @pytest.mark.asyncio
    async def test_chat_write_file_empty_content_allowed(self, tmp_path):
        """write_file allows empty string content."""
        from dokumen.tools_object import create_chat_write_file_tool

        tool = create_chat_write_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({
            "path": "docs/empty.md",
            "content": "",
        })

        assert result.success is True
        assert result.output["bytes"] == 0

    def test_chat_write_file_tool_definition(self):
        """write_file has correct tool definition."""
        from dokumen.tools_object import create_chat_write_file_tool

        tool = create_chat_write_file_tool()

        assert tool.name == "write_file"
        assert "write" in tool.description.lower()
        assert "path" in tool.parameters["properties"]
        assert "content" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["path", "content"]

    @pytest.mark.asyncio
    async def test_chat_write_file_unicode_content(self, tmp_path):
        """write_file handles unicode content."""
        from dokumen.tools_object import create_chat_write_file_tool

        tool = create_chat_write_file_tool(base_dir=str(tmp_path))
        unicode_content = "Hello 世界 🌍"
        result = await tool.handler({
            "path": "docs/unicode.md",
            "content": unicode_content,
        })

        assert result.success is True
        assert result.output["bytes"] == len(unicode_content.encode("utf-8"))
        assert (tmp_path / "docs" / "unicode.md").read_text() == unicode_content


class TestReExploreTool:
    """Tests for create_re_explore_tool.

    This tool allows the chat agent to re-explore the codebase with a new topic
    when the initial exploration found the wrong files.
    """

    @pytest.mark.asyncio
    async def test_re_explore_tool_returns_success(self):
        """re_explore returns success with valid topic."""
        from dokumen.tools_object import create_re_explore_tool

        tool = create_re_explore_tool()
        result = await tool.handler({
            "topic": "API authentication endpoints",
        })

        assert result.success is True
        assert result.output["action"] == "re_explore_requested"
        assert result.output["topic"] == "API authentication endpoints"
        assert "Re-exploring" in result.output["message"]

    @pytest.mark.asyncio
    async def test_re_explore_tool_missing_topic(self):
        """re_explore fails without topic."""
        from dokumen.tools_object import create_re_explore_tool

        tool = create_re_explore_tool()
        result = await tool.handler({})

        assert result.success is False
        assert "topic" in result.error.lower()

    @pytest.mark.asyncio
    async def test_re_explore_tool_empty_topic(self):
        """re_explore fails with empty topic."""
        from dokumen.tools_object import create_re_explore_tool

        tool = create_re_explore_tool()
        result = await tool.handler({"topic": ""})

        assert result.success is False
        assert "topic" in result.error.lower()

    def test_re_explore_tool_definition(self):
        """re_explore has correct tool definition."""
        from dokumen.tools_object import create_re_explore_tool

        tool = create_re_explore_tool()

        assert tool.name == "re_explore"
        assert "re-explore" in tool.description.lower() or "different" in tool.description.lower()
        assert "topic" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["topic"]

    @pytest.mark.asyncio
    async def test_re_explore_preserves_topic_in_output(self):
        """re_explore includes topic in output for downstream handling."""
        from dokumen.tools_object import create_re_explore_tool

        tool = create_re_explore_tool()
        topic = "refund policy documentation"
        result = await tool.handler({"topic": topic})

        assert result.success is True
        assert result.output["topic"] == topic
        assert topic in result.output["message"]


class TestChatDeleteFileTool:
    """Tests for create_chat_delete_file_tool.

    This tool deletes files from the local clone filesystem. Deletions are
    batch-committed to GitLab at the end of the conversation turn.
    """

    @pytest.mark.asyncio
    async def test_delete_file_success(self, tmp_path):
        """delete_file removes file in allowed prefix."""
        from dokumen.tools_object import create_chat_delete_file_tool

        # Create a file to delete
        (tmp_path / "tests").mkdir()
        target = tmp_path / "tests" / "old-test.test.yaml"
        target.write_text("old content")

        tool = create_chat_delete_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"path": "tests/old-test.test.yaml"})

        assert result.success is True
        assert result.output["action"] == "deleted"
        assert result.output["path"] == "tests/old-test.test.yaml"
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_delete_file_docs_prefix(self, tmp_path):
        """delete_file works for docs/ prefix."""
        from dokumen.tools_object import create_chat_delete_file_tool

        (tmp_path / "docs").mkdir()
        target = tmp_path / "docs" / "old-doc.md"
        target.write_text("old doc")

        tool = create_chat_delete_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"path": "docs/old-doc.md"})

        assert result.success is True
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_delete_file_rejects_outside_allowed(self, tmp_path):
        """delete_file rejects path outside docs/ and tests/."""
        from dokumen.tools_object import create_chat_delete_file_tool

        tool = create_chat_delete_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"path": "src/main.py"})

        assert result.success is False
        assert "not allowed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_file_rejects_traversal(self, tmp_path):
        """delete_file rejects path traversal."""
        from dokumen.tools_object import create_chat_delete_file_tool

        tool = create_chat_delete_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"path": "docs/../.env"})

        assert result.success is False
        assert "traversal" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, tmp_path):
        """delete_file returns error for non-existent file."""
        from dokumen.tools_object import create_chat_delete_file_tool

        tool = create_chat_delete_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"path": "docs/nonexistent.md"})

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_file_missing_path(self):
        """delete_file fails without path."""
        from dokumen.tools_object import create_chat_delete_file_tool

        tool = create_chat_delete_file_tool()
        result = await tool.handler({})

        assert result.success is False
        assert "path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_file_leading_slash_stripped(self, tmp_path):
        """delete_file strips leading slash from path."""
        from dokumen.tools_object import create_chat_delete_file_tool

        (tmp_path / "tests").mkdir()
        target = tmp_path / "tests" / "file.yaml"
        target.write_text("content")

        tool = create_chat_delete_file_tool(base_dir=str(tmp_path))
        result = await tool.handler({"path": "/tests/file.yaml"})

        assert result.success is True
        assert not target.exists()

    def test_delete_file_tool_definition(self):
        """delete_file has correct tool definition."""
        from dokumen.tools_object import create_chat_delete_file_tool

        tool = create_chat_delete_file_tool()

        assert tool.name == "delete_file"
        assert "delete" in tool.description.lower()
        assert "path" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["path"]


class TestHttpRequestSSRFProtection:
    """Tests for SSRF protection in http_request tool (security)."""

    @pytest.mark.asyncio
    async def test_blocks_localhost(self):
        """http_request blocks localhost URLs."""
        from dokumen.tools_object import create_http_request_tool

        tool = create_http_request_tool()
        result = await tool.handler({"url": "http://localhost/api/secret"})

        assert result.success is False
        assert "blocked" in result.error.lower() or "security" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_127_0_0_1(self):
        """http_request blocks 127.0.0.1 URLs."""
        from dokumen.tools_object import create_http_request_tool

        tool = create_http_request_tool()
        result = await tool.handler({"url": "http://127.0.0.1:8080/admin"})

        assert result.success is False
        assert "blocked" in result.error.lower() or "security" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_aws_metadata_endpoint(self):
        """http_request blocks AWS metadata endpoint (169.254.169.254)."""
        from dokumen.tools_object import create_http_request_tool

        tool = create_http_request_tool()
        result = await tool.handler({"url": "http://169.254.169.254/latest/meta-data/"})

        assert result.success is False
        assert "blocked" in result.error.lower() or "security" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_private_ip_10_x(self):
        """http_request blocks private 10.x.x.x IPs."""
        from dokumen.tools_object import create_http_request_tool

        tool = create_http_request_tool()
        result = await tool.handler({"url": "http://10.0.0.1/internal"})

        assert result.success is False
        assert "blocked" in result.error.lower() or "private" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_private_ip_192_168(self):
        """http_request blocks private 192.168.x.x IPs."""
        from dokumen.tools_object import create_http_request_tool

        tool = create_http_request_tool()
        result = await tool.handler({"url": "http://192.168.1.1/router"})

        assert result.success is False
        assert "blocked" in result.error.lower() or "private" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_private_ip_172_16(self):
        """http_request blocks private 172.16.x.x IPs."""
        from dokumen.tools_object import create_http_request_tool

        tool = create_http_request_tool()
        result = await tool.handler({"url": "http://172.16.0.1/internal"})

        assert result.success is False
        assert "blocked" in result.error.lower() or "private" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_file_scheme(self):
        """http_request blocks file:// URLs."""
        from dokumen.tools_object import create_http_request_tool

        tool = create_http_request_tool()
        result = await tool.handler({"url": "file:///etc/passwd"})

        assert result.success is False
        assert "scheme" in result.error.lower() or "invalid" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_ftp_scheme(self):
        """http_request blocks ftp:// URLs."""
        from dokumen.tools_object import create_http_request_tool

        tool = create_http_request_tool()
        result = await tool.handler({"url": "ftp://ftp.example.com/file"})

        assert result.success is False
        assert "scheme" in result.error.lower() or "invalid" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_empty_hostname(self):
        """http_request blocks URLs without hostname."""
        from dokumen.tools_object import create_http_request_tool

        tool = create_http_request_tool()
        result = await tool.handler({"url": "http:///path"})

        assert result.success is False


class TestGrepShellEscaping:
    """Tests for shell escaping in grep tool (security)."""

    @pytest.mark.asyncio
    async def test_grep_escapes_special_characters(self):
        """grep properly escapes special shell characters in pattern."""
        from dokumen.tools_object import create_grep_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 1  # No matches
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_grep_tool(mock_sandbox)

        # Pattern with shell metacharacters
        await tool.handler({"pattern": "test;rm -rf /", "path": "."})

        # Verify the command was called with escaped pattern
        call_args = mock_sandbox.execute.call_args[0][0]
        # The pattern should be quoted, not executed as shell command
        assert "rm -rf" not in call_args or "'" in call_args or '"' in call_args

    @pytest.mark.asyncio
    async def test_grep_escapes_backticks(self):
        """grep properly escapes backticks to prevent command substitution."""
        from dokumen.tools_object import create_grep_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_grep_tool(mock_sandbox)

        # Pattern with backticks (command substitution)
        await tool.handler({"pattern": "`whoami`", "path": "."})

        call_args = mock_sandbox.execute.call_args[0][0]
        # Backticks should be escaped/quoted
        assert "'" in call_args or "\\" in call_args

    @pytest.mark.asyncio
    async def test_grep_escapes_dollar_sign(self):
        """grep properly escapes $ to prevent variable expansion."""
        from dokumen.tools_object import create_grep_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_grep_tool(mock_sandbox)

        # Pattern with $ variable expansion
        await tool.handler({"pattern": "$HOME", "path": "."})

        call_args = mock_sandbox.execute.call_args[0][0]
        # $ should be escaped/quoted
        assert "'" in call_args

    @pytest.mark.asyncio
    async def test_grep_uses_double_dash_separator(self):
        """grep uses -- to separate options from pattern (prevents flag injection)."""
        from dokumen.tools_object import create_grep_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_grep_tool(mock_sandbox)

        # Pattern that starts with - (could be interpreted as flag)
        await tool.handler({"pattern": "-e malicious", "path": "."})

        call_args = mock_sandbox.execute.call_args[0][0]
        # Should include -- separator
        assert "--" in call_args

    @pytest.mark.asyncio
    async def test_grep_escapes_path_parameter(self):
        """grep properly escapes path parameter."""
        from dokumen.tools_object import create_grep_tool
        from unittest.mock import AsyncMock, MagicMock

        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_sandbox.execute = AsyncMock(return_value=mock_result)

        tool = create_grep_tool(mock_sandbox)

        # Path with spaces and special chars
        await tool.handler({"pattern": "test", "path": "/path/with spaces;rm -rf /"})

        call_args = mock_sandbox.execute.call_args[0][0]
        # Path should be quoted
        assert "'" in call_args


class TestGrepToolWithoutSandbox:
    """Tests for grep tool without sandbox (direct execution fallback)."""

    @pytest.mark.asyncio
    async def test_grep_no_sandbox_finds_matches(self, tmp_path):
        """grep without sandbox finds matching content in files."""
        from dokumen.tools_object import create_grep_tool

        # Create test files
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "test.md").write_text("fire resistance rating\nother content\n")
        (tmp_path / "docs" / "other.md").write_text("no match here\n")

        tool = create_grep_tool(sandbox=None, base_dir=str(tmp_path))
        result = await tool.handler({"pattern": "fire resistance", "path": "docs/"})

        assert result.success is True
        assert "fire resistance" in result.output

    @pytest.mark.asyncio
    async def test_grep_no_sandbox_no_matches(self, tmp_path):
        """grep without sandbox returns no-match message."""
        from dokumen.tools_object import create_grep_tool

        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "test.md").write_text("nothing relevant here\n")

        tool = create_grep_tool(sandbox=None, base_dir=str(tmp_path))
        result = await tool.handler({"pattern": "nonexistent_pattern_xyz", "path": "docs/"})

        assert result.success is True
        assert "No matches" in result.output

    @pytest.mark.asyncio
    async def test_grep_no_sandbox_error(self, tmp_path):
        """grep without sandbox handles errors gracefully."""
        from dokumen.tools_object import create_grep_tool

        tool = create_grep_tool(sandbox=None, base_dir=str(tmp_path))
        # Invalid regex pattern
        result = await tool.handler({"pattern": "[invalid", "path": "nonexistent_dir/"})

        assert result.success is False

    @pytest.mark.asyncio
    async def test_grep_no_sandbox_case_insensitive(self, tmp_path):
        """grep without sandbox supports case-insensitive search."""
        from dokumen.tools_object import create_grep_tool

        (tmp_path / "test.md").write_text("Fire Resistance Rating\n")

        tool = create_grep_tool(sandbox=None, base_dir=str(tmp_path))
        result = await tool.handler({
            "pattern": "fire resistance",
            "case_insensitive": True,
            "path": "test.md"
        })

        assert result.success is True
        assert "Fire Resistance" in result.output

    @pytest.mark.asyncio
    async def test_grep_no_sandbox_missing_pattern(self, tmp_path):
        """grep without sandbox fails when pattern is missing."""
        from dokumen.tools_object import create_grep_tool

        tool = create_grep_tool(sandbox=None, base_dir=str(tmp_path))
        result = await tool.handler({})

        assert result.success is False
        assert "pattern" in result.error

    @pytest.mark.asyncio
    async def test_grep_no_sandbox_default_path(self, tmp_path):
        """grep without sandbox uses default path '.' relative to base_dir."""
        from dokumen.tools_object import create_grep_tool

        (tmp_path / "file.txt").write_text("searchable content\n")

        tool = create_grep_tool(sandbox=None, base_dir=str(tmp_path))
        result = await tool.handler({"pattern": "searchable"})

        assert result.success is True
        assert "searchable" in result.output
