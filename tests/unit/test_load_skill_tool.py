"""
Unit tests for the load_skill CLI tool.

Tests the tool that lets agents dynamically discover and load
workspace skills from SKILL.md files at runtime.

Test cases:
- List all skills (empty name)
- Load specific skill by name
- Skill not found error
- No workspace directory error
- Argument substitution
- Empty workspace (no skills)
"""
import pytest
from pathlib import Path


def _create_skill_file(base_dir: Path, rel_path: str, content: str) -> Path:
    """Helper to create a SKILL.md file in the given directory structure."""
    full_path = base_dir / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    return full_path


# Sample SKILL.md content for tests
SKILL_COMMIT = """\
---
name: commit
description: Create a git commit with a descriptive message
argument-hint: "[message]"
---

# Commit Skill

Follow these steps to create a commit:
1. Run `git add -A`
2. Run `git commit -m "$ARGUMENTS"`
"""

SKILL_REVIEW = """\
---
name: review-pr
description: Review a pull request for code quality
---

# Review PR Skill

Review the pull request thoroughly:
- Check for code quality
- Verify test coverage
- Look for security issues
"""

SKILL_NO_FRONTMATTER = """\
# Just a regular markdown file

This has no YAML frontmatter and should be ignored.
"""

SKILL_MISSING_NAME = """\
---
description: A skill without a name
---

# No Name Skill
"""


class TestLoadSkillListAll:
    """Tests for listing all skills (name is empty)."""

    @pytest.mark.asyncio
    async def test_load_skill_list_all(self, tmp_path):
        """When name is empty, lists all skills in workspace."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        # Create workspace with two skills in default paths
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _create_skill_file(workspace, ".skills/commit/SKILL.md", SKILL_COMMIT)
        _create_skill_file(workspace, ".skills/review/SKILL.md", SKILL_REVIEW)

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({"name": ""})

        assert result.success is True
        assert "commit" in result.output
        assert "review-pr" in result.output
        assert "Create a git commit" in result.output
        assert "Review a pull request" in result.output

    @pytest.mark.asyncio
    async def test_load_skill_list_all_no_name_key(self, tmp_path):
        """When name key is missing from params, lists all skills."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _create_skill_file(workspace, ".skills/commit/SKILL.md", SKILL_COMMIT)

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({})

        assert result.success is True
        assert "commit" in result.output

    @pytest.mark.asyncio
    async def test_load_skill_list_empty_workspace(self, tmp_path):
        """When workspace has no skills, returns informative message."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({"name": ""})

        assert result.success is True
        assert "no skills" in result.output.lower() or "No skills" in result.output


class TestLoadSkillByName:
    """Tests for loading a specific skill by name."""

    @pytest.mark.asyncio
    async def test_load_skill_by_name(self, tmp_path):
        """When name is provided, returns that specific skill content."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _create_skill_file(workspace, ".skills/commit/SKILL.md", SKILL_COMMIT)
        _create_skill_file(workspace, ".skills/review/SKILL.md", SKILL_REVIEW)

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({"name": "commit"})

        assert result.success is True
        assert "Commit Skill" in result.output
        assert "git add" in result.output
        assert "git commit" in result.output

    @pytest.mark.asyncio
    async def test_load_skill_by_name_includes_metadata(self, tmp_path):
        """Loaded skill includes source file path in output."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _create_skill_file(workspace, ".skills/commit/SKILL.md", SKILL_COMMIT)

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({"name": "commit"})

        assert result.success is True
        # Should include source path
        assert "SKILL.md" in result.output


class TestLoadSkillNotFound:
    """Tests for skill not found error."""

    @pytest.mark.asyncio
    async def test_load_skill_not_found(self, tmp_path):
        """Returns error message when skill name not found."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _create_skill_file(workspace, ".skills/commit/SKILL.md", SKILL_COMMIT)

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({"name": "nonexistent-skill"})

        assert result.success is False
        assert "not found" in result.error.lower()
        # Should mention available skills
        assert "commit" in result.error

    @pytest.mark.asyncio
    async def test_load_skill_not_found_no_skills_available(self, tmp_path):
        """Returns error when no skills exist and a name is requested."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({"name": "anything"})

        assert result.success is False
        assert "not found" in result.error.lower()


class TestLoadSkillNoWorkspace:
    """Tests for missing workspace directory."""

    @pytest.mark.asyncio
    async def test_load_skill_no_workspace(self, tmp_path):
        """Returns error when workspace directory doesn't exist."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        nonexistent = str(tmp_path / "does-not-exist")

        tool = create_load_skill_tool(nonexistent)
        result = await tool.handler({"name": ""})

        assert result.success is False
        assert result.error is not None
        assert "does not exist" in result.error.lower() or "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_load_skill_no_workspace_with_name(self, tmp_path):
        """Returns error when workspace directory doesn't exist and name is provided."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        nonexistent = str(tmp_path / "does-not-exist")

        tool = create_load_skill_tool(nonexistent)
        result = await tool.handler({"name": "commit"})

        assert result.success is False
        assert result.error is not None


class TestLoadSkillArguments:
    """Tests for argument substitution."""

    @pytest.mark.asyncio
    async def test_load_skill_with_arguments(self, tmp_path):
        """Arguments are substituted into $ARGUMENTS placeholders."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _create_skill_file(workspace, ".skills/commit/SKILL.md", SKILL_COMMIT)

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({
            "name": "commit",
            "arguments": "fix: resolve login bug",
        })

        assert result.success is True
        assert "fix: resolve login bug" in result.output

    @pytest.mark.asyncio
    async def test_load_skill_without_arguments(self, tmp_path):
        """Without arguments, $ARGUMENTS placeholders remain or are empty."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _create_skill_file(workspace, ".skills/commit/SKILL.md", SKILL_COMMIT)

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({"name": "commit"})

        assert result.success is True
        # Content should still be present
        assert "Commit Skill" in result.output


class TestLoadSkillToolDefinition:
    """Tests for the tool definition shape."""

    def test_tool_definition_name(self, tmp_path):
        """Tool is named 'load_skill'."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        tool = create_load_skill_tool(str(tmp_path))
        assert tool.name == "load_skill"

    def test_tool_definition_has_parameters(self, tmp_path):
        """Tool has proper JSON Schema parameters."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        tool = create_load_skill_tool(str(tmp_path))
        assert tool.parameters["type"] == "object"
        assert "name" in tool.parameters["properties"]
        assert "arguments" in tool.parameters["properties"]

    def test_tool_definition_has_description(self, tmp_path):
        """Tool has a non-empty description."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        tool = create_load_skill_tool(str(tmp_path))
        assert len(tool.description) > 0
        assert "skill" in tool.description.lower()


class TestLoadSkillEdgeCases:
    """Edge case tests for robustness."""

    @pytest.mark.asyncio
    async def test_skill_with_invalid_frontmatter_skipped(self, tmp_path):
        """SKILL.md files without valid frontmatter are ignored."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _create_skill_file(workspace, ".skills/bad/SKILL.md", SKILL_NO_FRONTMATTER)
        _create_skill_file(workspace, ".skills/commit/SKILL.md", SKILL_COMMIT)

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({"name": ""})

        assert result.success is True
        # Only the valid skill should appear
        assert "commit" in result.output
        # The invalid file should not cause an error

    @pytest.mark.asyncio
    async def test_skill_missing_name_field_skipped(self, tmp_path):
        """SKILL.md files with missing name field are ignored."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _create_skill_file(workspace, ".skills/noname/SKILL.md", SKILL_MISSING_NAME)

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({"name": ""})

        assert result.success is True
        assert "no skills" in result.output.lower() or "No skills" in result.output

    @pytest.mark.asyncio
    async def test_multiple_skill_directories_scanned(self, tmp_path):
        """Skills are discovered from all default scan paths."""
        from dokumen.tools.load_skill_tool import create_load_skill_tool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        # Put skills in different default directories
        _create_skill_file(workspace, ".skills/commit/SKILL.md", SKILL_COMMIT)
        _create_skill_file(workspace, "skills/review/SKILL.md", SKILL_REVIEW)

        tool = create_load_skill_tool(str(workspace))
        result = await tool.handler({"name": ""})

        assert result.success is True
        assert "commit" in result.output
        assert "review-pr" in result.output


class TestLoadSkillResolveTools:
    """Tests for load_skill wiring into resolve_tools."""

    def test_resolve_tools_load_skill(self, tmp_path):
        """resolve_tools resolves 'load_skill' to a ToolDefinition."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(["load_skill"], base_dir=str(tmp_path))

        assert len(tools) == 1
        assert tools[0].name == "load_skill"

    def test_resolve_tools_load_skill_with_other_tools(self, tmp_path):
        """resolve_tools resolves load_skill alongside other tools."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["read_file", "load_skill"],
            base_dir=str(tmp_path),
        )

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names
        assert "load_skill" in tool_names
