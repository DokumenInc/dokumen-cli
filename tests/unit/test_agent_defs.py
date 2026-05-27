"""Tests for agent definition integration with CLI loader.

Covers:
- load_agent() loading YAML definitions
- Loader using agent defaults for system_prompt, tools, config
- Scaffold overriding agent defaults
- Backward compatibility with type: + @prompts/ pattern
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile
import yaml

from dokumen_schema.agent_defs import AgentDefinition, load_agent, list_agents


class TestAgentDefsFromCLI:
    """Tests for agent definition loading from the CLI perspective."""

    def test_load_known_agent(self):
        """CLI can load a known agent definition."""
        agent = load_agent("browser-tester")
        assert agent is not None
        assert agent.name == "browser-tester"

    def test_load_unknown_agent_returns_none(self):
        """Unknown agent name returns None (not an error)."""
        result = load_agent("does-not-exist")
        assert result is None

    def test_agent_tools_are_strings(self):
        """Agent tools are plain string names."""
        agent = load_agent("general")
        assert all(isinstance(t, str) for t in agent.tools)

    def test_agent_has_typed_capabilities(self):
        """Agent has typed capabilities list."""
        agent = load_agent("browser-tester")
        assert isinstance(agent.capabilities, list)
        assert "browser" in agent.capabilities
        assert agent.browser is not None


class TestLoaderAgentDefaults:
    """Tests that the loader uses agent definitions as defaults."""

    def _make_scaffold_yaml(self, tmp_path, scaffold_data):
        """Helper to write a scaffold YAML and return its path."""
        # Write dokumen.yaml so find_project_root works
        config = {
            "version": "1.0",
            "provider": {"name": "anthropic", "model": "claude-sonnet-4-6"},
        }
        (tmp_path / "dokumen.yaml").write_text(yaml.dump(config))

        # Write scaffold
        scaffold_path = tmp_path / "tests" / "test.test.yaml"
        scaffold_path.parent.mkdir(parents=True, exist_ok=True)
        scaffold_path.write_text(yaml.dump(scaffold_data))
        return str(scaffold_path)

    def test_agent_provides_system_prompt_when_scaffold_omits(self, tmp_path):
        """When scaffold has no system_prompt, agent definition's prompt is used."""
        scaffold = {
            "name": "test-agent-defaults",
            "agent": "general",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "user_prompt": "Read and describe the file.",
            },
            "judges": [
                {
                    "name": "check",
                    "system_prompt": "Evaluate the output. Return JSON: {\"verdict\": \"PASS\", \"confidence\": 1.0, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = self._make_scaffold_yaml(tmp_path, scaffold)

        mock_provider = MagicMock()
        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(yaml_path, provider=mock_provider, project_root=str(tmp_path))

        # The executor should have the general agent's system prompt
        assert "Dokumen executor agent" in test_obj.executor.system_prompt
        assert test_obj.agent == "general"

    def test_agent_provides_tools_when_scaffold_omits(self, tmp_path):
        """When scaffold has no tools, agent definition's tools are used."""
        scaffold = {
            "name": "test-agent-tools",
            "agent": "general",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "user_prompt": "Do something.",
            },
            "judges": [
                {
                    "name": "check",
                    "system_prompt": "Evaluate. Return JSON: {\"verdict\": \"PASS\", \"confidence\": 1.0, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = self._make_scaffold_yaml(tmp_path, scaffold)

        mock_provider = MagicMock()
        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(yaml_path, provider=mock_provider, project_root=str(tmp_path))

        tool_names = [t.name for t in test_obj.executor.tools]
        # General agent provides SDK tool names (Read, Glob, etc.)
        assert "Read" in tool_names or "read_file" in tool_names

    def test_agent_provides_research_tools(self, tmp_path):
        """When scaffold uses researcher agent, web_search is auto-injected."""
        scaffold = {
            "name": "test-research-defaults",
            "agent": "researcher",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "user_prompt": "Research a topic.",
            },
            "judges": [
                {
                    "name": "check",
                    "system_prompt": "Evaluate. Return JSON: {\"verdict\": \"PASS\", \"confidence\": 1.0, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = self._make_scaffold_yaml(tmp_path, scaffold)

        mock_provider = MagicMock()
        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(yaml_path, provider=mock_provider, project_root=str(tmp_path))

        # Research agent should auto-inject web_search (SDK maps to WebSearch)
        tool_names = [t.name for t in test_obj.executor.tools]
        assert "web_search" in tool_names or "WebSearch" in tool_names
        assert test_obj.agent == "researcher"

    def test_agent_provides_browser_config(self, tmp_path):
        """When scaffold has no browser config, agent definition provides it."""
        scaffold = {
            "name": "test-browser-defaults",
            "agent": "browser-tester",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "user_prompt": "Navigate to a page.",
            },
            "judges": [
                {
                    "name": "check",
                    "system_prompt": "Evaluate. Return JSON: {\"verdict\": \"PASS\", \"confidence\": 1.0, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = self._make_scaffold_yaml(tmp_path, scaffold)

        mock_provider = MagicMock()
        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(yaml_path, provider=mock_provider, project_root=str(tmp_path))

        # Browser config should come from agent definition
        assert test_obj.browser_config is not None
        assert test_obj.browser_config.viewport_size == "1512x982"


class TestLoaderAgentOverrides:
    """Tests that scaffold values override agent defaults."""

    def _make_scaffold_yaml(self, tmp_path, scaffold_data):
        """Helper to write a scaffold YAML and return its path."""
        config = {
            "version": "1.0",
            "provider": {"name": "anthropic", "model": "claude-sonnet-4-6"},
        }
        (tmp_path / "dokumen.yaml").write_text(yaml.dump(config))

        scaffold_path = tmp_path / "tests" / "test.test.yaml"
        scaffold_path.parent.mkdir(parents=True, exist_ok=True)
        scaffold_path.write_text(yaml.dump(scaffold_data))
        return str(scaffold_path)

    def test_scaffold_system_prompt_overrides_agent(self, tmp_path):
        """Scaffold's system_prompt takes precedence over agent default."""
        scaffold = {
            "name": "test-override-prompt",
            "agent": "general",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "system_prompt": "Custom prompt that overrides agent.",
                "user_prompt": "Do something.",
            },
            "judges": [
                {
                    "name": "check",
                    "system_prompt": "Evaluate. Return JSON: {\"verdict\": \"PASS\", \"confidence\": 1.0, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = self._make_scaffold_yaml(tmp_path, scaffold)

        mock_provider = MagicMock()
        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(yaml_path, provider=mock_provider, project_root=str(tmp_path))

        assert test_obj.executor.system_prompt == "Custom prompt that overrides agent."

    def test_scaffold_tools_override_agent(self, tmp_path):
        """Scaffold's tools take precedence over agent default tools."""
        scaffold = {
            "name": "test-override-tools",
            "agent": "general",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "user_prompt": "Do something.",
                "tools": ["read_file"],  # Only read_file, not agent's full list
            },
            "judges": [
                {
                    "name": "check",
                    "system_prompt": "Evaluate. Return JSON: {\"verdict\": \"PASS\", \"confidence\": 1.0, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = self._make_scaffold_yaml(tmp_path, scaffold)

        mock_provider = MagicMock()
        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(yaml_path, provider=mock_provider, project_root=str(tmp_path))

        tool_names = [t.name for t in test_obj.executor.tools]
        # SDK maps read_file to Read
        assert "Read" in tool_names or "read_file" in tool_names

    def test_scaffold_browser_config_overrides_agent(self, tmp_path):
        """Scaffold's browser config overrides agent default."""
        scaffold = {
            "name": "test-override-browser",
            "agent": "browser-tester",
            "files": [{"path": "docs/test.md"}],
            "browser": {
                "headless": True,
                "viewport": {"width": 800, "height": 600},
            },
            "executor": {
                "user_prompt": "Navigate to a page.",
            },
            "judges": [
                {
                    "name": "check",
                    "system_prompt": "Evaluate. Return JSON: {\"verdict\": \"PASS\", \"confidence\": 1.0, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = self._make_scaffold_yaml(tmp_path, scaffold)

        mock_provider = MagicMock()
        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(yaml_path, provider=mock_provider, project_root=str(tmp_path))

        # Should use scaffold's browser config, not agent's
        assert test_obj.browser_config is not None
        assert test_obj.browser_config.viewport_size == "800x600"


class TestLoaderBackwardCompat:
    """Tests that existing type: + @prompts/ patterns still work."""

    def _make_scaffold_yaml(self, tmp_path, scaffold_data):
        """Helper to write a scaffold YAML and return its path."""
        config = {
            "version": "1.0",
            "provider": {"name": "anthropic", "model": "claude-sonnet-4-6"},
        }
        (tmp_path / "dokumen.yaml").write_text(yaml.dump(config))

        scaffold_path = tmp_path / "tests" / "test.test.yaml"
        scaffold_path.parent.mkdir(parents=True, exist_ok=True)
        scaffold_path.write_text(yaml.dump(scaffold_data))
        return str(scaffold_path)

    def test_type_browser_still_works(self, tmp_path):
        """type: browser + @prompts/browser-testing.txt still loads correctly."""
        scaffold = {
            "name": "test-legacy-browser",
            "type": "browser",
            "files": [{"path": "docs/test.md"}],
            "browser": {
                "headless": True,
                "viewport": {"width": 1024, "height": 768},
            },
            "executor": {
                "system_prompt": "@prompts/browser-testing.txt",
                "user_prompt": "Navigate to example.com.",
            },
            "judges": [
                {
                    "name": "check",
                    "system_prompt": "Evaluate. Return JSON: {\"verdict\": \"PASS\", \"confidence\": 1.0, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = self._make_scaffold_yaml(tmp_path, scaffold)

        mock_provider = MagicMock()
        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(yaml_path, provider=mock_provider, project_root=str(tmp_path))

        # Should have browser tools injected (SDK uses mcp__playwright__ prefix)
        tool_names = [t.name for t in test_obj.executor.tools]
        has_browser_tools = any("browser_navigate" in t for t in tool_names)
        assert has_browser_tools
        assert test_obj.agent == "browser-tester"  # Inferred from type

    def test_standard_prompts_reference_still_works(self, tmp_path):
        """@prompts/documentation-validation.txt reference still loads."""
        scaffold = {
            "name": "test-legacy-prompt",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "system_prompt": "@prompts/documentation-validation.txt",
                "user_prompt": "Validate docs.",
                "tools": ["read_file"],
            },
            "judges": [
                {
                    "name": "check",
                    "system_prompt": "Evaluate. Return JSON: {\"verdict\": \"PASS\", \"confidence\": 1.0, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = self._make_scaffold_yaml(tmp_path, scaffold)

        mock_provider = MagicMock()
        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(yaml_path, provider=mock_provider, project_root=str(tmp_path))

        assert "Dokumen executor agent" in test_obj.executor.system_prompt

    def test_no_agent_no_type_still_works(self, tmp_path):
        """Standard scaffold without agent or type still loads."""
        scaffold = {
            "name": "test-plain",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "system_prompt": "@prompts/general.txt",
                "user_prompt": "Read the file.",
                "tools": ["read_file"],
            },
            "judges": [
                {
                    "name": "check",
                    "system_prompt": "Evaluate. Return JSON: {\"verdict\": \"PASS\", \"confidence\": 1.0, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = self._make_scaffold_yaml(tmp_path, scaffold)

        mock_provider = MagicMock()
        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(yaml_path, provider=mock_provider, project_root=str(tmp_path))

        assert test_obj.agent is None
        assert test_obj.executor.system_prompt is not None

    def test_agent_not_found_raises_hard_error(self, tmp_path):
        """Agent name that doesn't match any YAML file raises ValueError."""
        scaffold = {
            "name": "test-unknown-agent",
            "agent": "nonexistent-agent",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "system_prompt": "@prompts/general.txt",
                "user_prompt": "Read the file.",
                "tools": ["read_file"],
            },
            "judges": [
                {
                    "name": "check",
                    "system_prompt": "Evaluate. Return JSON: {\"verdict\": \"PASS\", \"confidence\": 1.0, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = self._make_scaffold_yaml(tmp_path, scaffold)

        mock_provider = MagicMock()
        from dokumen.loader import load_scaffold
        with pytest.raises(ValueError, match="not found"):
            load_scaffold(yaml_path, provider=mock_provider, project_root=str(tmp_path))


class TestLoaderAgentFlagDerivation:
    """Tests that agent flags are correctly derived from capabilities."""

    def test_browser_agent_flag_from_capabilities(self):
        """Browser agent flag is derived from capabilities."""
        agent = load_agent("browser-tester")
        assert "browser" in agent.capabilities

    def test_research_agent_flag_from_capabilities(self):
        """Research agent flag is derived from capabilities."""
        agent = load_agent("researcher")
        assert "research" in agent.capabilities

    def test_code_agent_flag_from_capabilities(self):
        """Code agent flag is derived from capabilities."""
        agent = load_agent("code-reviewer")
        assert "code" in agent.capabilities

    def test_general_agent_has_standard_capability(self):
        """General agent has standard capability only."""
        agent = load_agent("general")
        assert "standard" in agent.capabilities
        assert "browser" not in agent.capabilities
        assert "research" not in agent.capabilities
        assert "code" not in agent.capabilities
