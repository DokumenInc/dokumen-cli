"""Tests for SDK-native subagent delegation via AgentDefinition.

Tests the build_sdk_agent_definitions() function that converts Dokumen
agent YAML definitions into SDK AgentDefinition objects for native
subagent support.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from claude_agent_sdk import AgentDefinition


class TestBuildSdkAgentDefinitions:
    """Tests for build_sdk_agent_definitions()."""

    def test_returns_empty_dict_when_no_agents(self):
        """Returns empty dict when no user-defined agents are found."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        ctx = AgentContext(user_dirs=[], base_dir=".")
        with patch("dokumen_schema.agent_defs.list_agents", return_value=[]):
            result = build_sdk_agent_definitions(ctx)

        assert result == {}

    def test_converts_agent_def_to_sdk_agent_definition(self):
        """Converts a Dokumen AgentDefinition to SDK AgentDefinition."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        mock_agent = MagicMock()
        mock_agent.name = "site-searcher"
        mock_agent.description = "Searches a biomass news site"
        mock_agent.system_prompt = "You search ONE site for announcements."
        mock_agent.tools = ["run_shell_command", "read_file"]
        mock_agent.model = "sonnet"

        ctx = AgentContext(user_dirs=[Path("/fake/agents")], base_dir=".")

        with patch("dokumen_schema.agent_defs.list_agents", return_value=["site-searcher"]), \
             patch("dokumen_schema.agent_defs.load_agent", return_value=mock_agent):
            result = build_sdk_agent_definitions(ctx)

        assert "site-searcher" in result
        agent_def = result["site-searcher"]
        assert isinstance(agent_def, AgentDefinition)
        assert agent_def.description == "Searches a biomass news site"
        assert agent_def.prompt == "You search ONE site for announcements."

    def test_maps_tools_to_sdk_names(self):
        """Tool names are mapped from Dokumen names to SDK names."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        mock_agent = MagicMock()
        mock_agent.name = "helper"
        mock_agent.description = "Helper agent"
        mock_agent.system_prompt = "Help."
        mock_agent.tools = ["run_shell_command", "read_file", "glob"]
        mock_agent.model = None

        ctx = AgentContext(user_dirs=[Path("/fake")], base_dir=".")

        with patch("dokumen_schema.agent_defs.list_agents", return_value=["helper"]), \
             patch("dokumen_schema.agent_defs.load_agent", return_value=mock_agent):
            result = build_sdk_agent_definitions(ctx)

        assert result["helper"].tools == ["Bash", "Read", "Glob"]

    def test_skips_compaction_agent(self):
        """Compaction agent is internal and should be skipped."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        compaction_agent = MagicMock()
        compaction_agent.name = "compaction"
        compaction_agent.description = "Internal compaction"
        compaction_agent.system_prompt = "Compact."
        compaction_agent.tools = []
        compaction_agent.model = None

        ctx = AgentContext(user_dirs=[], base_dir=".")

        with patch("dokumen_schema.agent_defs.list_agents", return_value=["compaction"]), \
             patch("dokumen_schema.agent_defs.load_agent", return_value=compaction_agent):
            result = build_sdk_agent_definitions(ctx)

        assert "compaction" not in result

    def test_skips_explore_agent(self):
        """Explore agent is internal and should be skipped."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        explore_agent = MagicMock()
        explore_agent.name = "explore"
        explore_agent.description = "Explore phase"
        explore_agent.system_prompt = "Explore."
        explore_agent.tools = []
        explore_agent.model = None

        ctx = AgentContext(user_dirs=[], base_dir=".")

        with patch("dokumen_schema.agent_defs.list_agents", return_value=["explore"]), \
             patch("dokumen_schema.agent_defs.load_agent", return_value=explore_agent):
            result = build_sdk_agent_definitions(ctx)

        assert "explore" not in result

    def test_maps_model_alias_to_sdk_literal(self):
        """Model aliases are mapped to SDK-compatible literals."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        mock_agent = MagicMock()
        mock_agent.name = "fast-helper"
        mock_agent.description = "Fast helper"
        mock_agent.system_prompt = "Help fast."
        mock_agent.tools = []
        mock_agent.model = "haiku"

        ctx = AgentContext(user_dirs=[Path("/fake")], base_dir=".")

        with patch("dokumen_schema.agent_defs.list_agents", return_value=["fast-helper"]), \
             patch("dokumen_schema.agent_defs.load_agent", return_value=mock_agent):
            result = build_sdk_agent_definitions(ctx)

        assert result["fast-helper"].model == "haiku"

    def test_none_model_becomes_none(self):
        """Agent with no model gets None (inherits from parent)."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        mock_agent = MagicMock()
        mock_agent.name = "basic"
        mock_agent.description = "Basic"
        mock_agent.system_prompt = "Be basic."
        mock_agent.tools = ["read_file"]
        mock_agent.model = None

        ctx = AgentContext(user_dirs=[Path("/fake")], base_dir=".")

        with patch("dokumen_schema.agent_defs.list_agents", return_value=["basic"]), \
             patch("dokumen_schema.agent_defs.load_agent", return_value=mock_agent):
            result = build_sdk_agent_definitions(ctx)

        assert result["basic"].model is None

    def test_skips_agent_when_load_returns_none(self):
        """Gracefully skips agents that fail to load."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        ctx = AgentContext(user_dirs=[Path("/fake")], base_dir=".")

        with patch("dokumen_schema.agent_defs.list_agents", return_value=["missing"]), \
             patch("dokumen_schema.agent_defs.load_agent", return_value=None):
            result = build_sdk_agent_definitions(ctx)

        assert result == {}

    def test_multiple_agents_converted(self):
        """Multiple agents are all converted."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        agent_a = MagicMock()
        agent_a.name = "agent-a"
        agent_a.description = "A"
        agent_a.system_prompt = "A prompt"
        agent_a.tools = ["read_file"]
        agent_a.model = "sonnet"

        agent_b = MagicMock()
        agent_b.name = "agent-b"
        agent_b.description = "B"
        agent_b.system_prompt = "B prompt"
        agent_b.tools = ["run_shell_command"]
        agent_b.model = "haiku"

        agents = {"agent-a": agent_a, "agent-b": agent_b}

        ctx = AgentContext(user_dirs=[Path("/fake")], base_dir=".")

        with patch("dokumen_schema.agent_defs.list_agents", return_value=["agent-a", "agent-b"]), \
             patch("dokumen_schema.agent_defs.load_agent", side_effect=lambda n, **kw: agents[n]):
            result = build_sdk_agent_definitions(ctx)

        assert len(result) == 2
        assert "agent-a" in result
        assert "agent-b" in result

    def test_empty_system_prompt_defaults_to_empty_string(self):
        """Agent with None system_prompt gets empty string in SDK definition."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        mock_agent = MagicMock()
        mock_agent.name = "no-prompt"
        mock_agent.description = "No prompt agent"
        mock_agent.system_prompt = None
        mock_agent.tools = []
        mock_agent.model = None

        ctx = AgentContext(user_dirs=[], base_dir=".")

        with patch("dokumen_schema.agent_defs.list_agents", return_value=["no-prompt"]), \
             patch("dokumen_schema.agent_defs.load_agent", return_value=mock_agent):
            result = build_sdk_agent_definitions(ctx)

        assert result["no-prompt"].prompt == ""

    def test_unmapped_tools_kept_as_is(self):
        """Tools not in SDK_MAPPING are kept with their original name."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        mock_agent = MagicMock()
        mock_agent.name = "custom-tools"
        mock_agent.description = "Has custom tools"
        mock_agent.system_prompt = "Use tools."
        mock_agent.tools = ["read_file", "some_custom_tool"]
        mock_agent.model = None

        ctx = AgentContext(user_dirs=[Path("/fake")], base_dir=".")

        with patch("dokumen_schema.agent_defs.list_agents", return_value=["custom-tools"]), \
             patch("dokumen_schema.agent_defs.load_agent", return_value=mock_agent):
            result = build_sdk_agent_definitions(ctx)

        assert "Read" in result["custom-tools"].tools
        assert "some_custom_tool" in result["custom-tools"].tools

    def test_passes_user_dirs_to_list_agents(self):
        """user_dirs from AgentContext are passed to list_agents."""
        from dokumen.sdk.delegate import build_sdk_agent_definitions
        from dokumen.sdk.tools import AgentContext

        dirs = [Path("/custom/agents")]
        ctx = AgentContext(user_dirs=dirs, base_dir=".")

        with patch("dokumen_schema.agent_defs.list_agents", return_value=[]) as mock_list:
            build_sdk_agent_definitions(ctx)

        mock_list.assert_called_once_with(user_dirs=dirs)


class TestMapToolsToSdk:
    """Tests for _map_tools_to_sdk helper."""

    def test_maps_known_tools(self):
        """Known Dokumen tools are mapped to SDK names."""
        from dokumen.sdk.delegate import _map_tools_to_sdk

        result = _map_tools_to_sdk(["read_file", "run_shell_command", "glob"])
        assert result == ["Read", "Bash", "Glob"]

    def test_preserves_unknown_tools(self):
        """Unknown tools are passed through unchanged."""
        from dokumen.sdk.delegate import _map_tools_to_sdk

        result = _map_tools_to_sdk(["read_file", "custom_tool"])
        assert result == ["Read", "custom_tool"]

    def test_empty_list(self):
        """Empty list returns empty list."""
        from dokumen.sdk.delegate import _map_tools_to_sdk

        assert _map_tools_to_sdk([]) == []

    def test_deduplicates_mapped_tools(self):
        """Duplicate SDK names after mapping are deduplicated."""
        from dokumen.sdk.delegate import _map_tools_to_sdk

        # glob and list_directory both map to Glob
        result = _map_tools_to_sdk(["glob", "list_directory"])
        assert result == ["Glob"]

    def test_already_sdk_names_preserved(self):
        """Tools that are already SDK names are preserved as-is."""
        from dokumen.sdk.delegate import _map_tools_to_sdk

        result = _map_tools_to_sdk(["Read", "Bash"])
        assert result == ["Read", "Bash"]


class TestMapModel:
    """Tests for _map_model helper."""

    def test_sonnet_maps_to_sonnet(self):
        """sonnet alias maps to SDK literal 'sonnet'."""
        from dokumen.sdk.delegate import _map_model

        assert _map_model("sonnet") == "sonnet"

    def test_haiku_maps_to_haiku(self):
        """haiku alias maps to SDK literal 'haiku'."""
        from dokumen.sdk.delegate import _map_model

        assert _map_model("haiku") == "haiku"

    def test_opus_maps_to_opus(self):
        """opus alias maps to SDK literal 'opus'."""
        from dokumen.sdk.delegate import _map_model

        assert _map_model("opus") == "opus"

    def test_none_returns_none(self):
        """None model returns None (inherit from parent)."""
        from dokumen.sdk.delegate import _map_model

        assert _map_model(None) is None

    def test_full_model_id_maps_to_alias(self):
        """Full model ID is mapped back to SDK literal alias."""
        from dokumen.sdk.delegate import _map_model

        assert _map_model("claude-sonnet-4-6") == "sonnet"
        assert _map_model("claude-haiku-4-5-20251001") == "haiku"
        assert _map_model("claude-opus-4-6") == "opus"

    def test_unknown_model_returns_none(self):
        """Unknown full model ID that doesn't match any alias returns None."""
        from dokumen.sdk.delegate import _map_model

        assert _map_model("unknown-model-id") is None


class TestSdkToolsResolveDelegateToAgent:
    """Tests that resolve_sdk_tools handles delegate_to_agent → Agent tool."""

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", {})
    def test_delegate_to_agent_skipped_without_context(self):
        """delegate_to_agent falls through to dokumen tool resolution without context."""
        from dokumen.sdk.tools import resolve_sdk_tools

        # Without agent_context, delegate_to_agent goes to resolve_dokumen_tool which raises
        with pytest.raises(ValueError, match="Unknown Dokumen tool"):
            resolve_sdk_tools(["delegate_to_agent"])

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", {})
    def test_delegate_to_agent_skipped_for_subagent(self):
        """delegate_to_agent is silently skipped when is_subagent=True."""
        from dokumen.sdk.tools import resolve_sdk_tools, AgentContext

        ctx = AgentContext(is_subagent=True)
        result = resolve_sdk_tools(["read_file", "delegate_to_agent"], agent_context=ctx)

        assert "Read" in result.sdk_tool_names
        # delegate_to_agent should not appear anywhere
        assert "Agent" not in result.sdk_tool_names
        assert len(result.dokumen_mcp_tools) == 0


    @patch("dokumen.sdk.tools.BROWSER_TOOLS", {})
    def test_delegate_to_agent_adds_agent_tool_with_context(self):
        """delegate_to_agent maps to SDK Agent tool when agent_context is valid."""
        from dokumen.sdk.tools import resolve_sdk_tools, AgentContext

        ctx = AgentContext(is_subagent=False, base_dir=".")
        result = resolve_sdk_tools(
            ["read_file", "delegate_to_agent"],
            agent_context=ctx,
        )

        assert "Read" in result.sdk_tool_names
        assert "Agent" in result.sdk_tool_names
        assert len(result.dokumen_mcp_tools) == 0

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", {})
    def test_delegate_to_agent_deduplicates_agent_tool(self):
        """Multiple delegate_to_agent entries don't duplicate Agent tool."""
        from dokumen.sdk.tools import resolve_sdk_tools, AgentContext

        ctx = AgentContext(is_subagent=False, base_dir=".")
        result = resolve_sdk_tools(
            ["delegate_to_agent", "read_file", "delegate_to_agent"],
            agent_context=ctx,
        )

        assert result.sdk_tool_names.count("Agent") == 1


class TestBuildSdkExecutorAgentIntegration:
    """Tests that build_sdk_executor wires agent definitions correctly."""

    @patch("dokumen.sdk.tools.BROWSER_TOOLS", {})
    def test_build_sdk_executor_passes_agents_when_delegate_present(self):
        """build_sdk_executor passes agents dict to ExecutorAgent when delegate_to_agent is used."""
        from dokumen.test_builder import build_sdk_executor
        from dokumen.sdk.tools import AgentContext

        mock_agent = MagicMock()
        mock_agent.name = "helper-agent"
        mock_agent.description = "A helper"
        mock_agent.system_prompt = "Help."
        mock_agent.tools = ["read_file"]
        mock_agent.model = "sonnet"

        data = {
            "name": "test-with-delegate",
            "executor": {"user_prompt": "Test prompt"},
            "timeout": 60,
        }

        with patch("dokumen_schema.agent_defs.list_agents", return_value=["helper-agent"]), \
             patch("dokumen_schema.agent_defs.load_agent", return_value=mock_agent), \
             patch("dokumen.sdk.executor.ExecutorAgent.__init__", return_value=None) as mock_init, \
             patch("dokumen.sdk.agent_wrapper.SdkExecutorWrapper.__init__", return_value=None):
            build_sdk_executor(
                data=data,
                executor_system_prompt="Test system",
                executor_tool_names=["read_file", "delegate_to_agent"],
                actual_executor_provider=None,
                executor_max_iterations=10,
                user_dirs=[Path("/fake/agents")],
                base_dir=".",
            )

        # Verify ExecutorAgent received agents parameter
        call_kwargs = mock_init.call_args
        assert call_kwargs is not None
        # Check that agents kwarg was passed
        if call_kwargs.kwargs:
            assert "agents" in call_kwargs.kwargs
            agents_dict = call_kwargs.kwargs["agents"]
            assert "helper-agent" in agents_dict
        else:
            # positional args - agents should be there
            assert any(isinstance(a, dict) and "helper-agent" in a for a in call_kwargs.args)


class TestDokumenAgentWithAgents:
    """Tests that DokumenAgent passes agents to ClaudeAgentOptions."""

    def test_agents_none_by_default(self):
        """When no agents param, ClaudeAgentOptions.agents is None."""
        from dokumen.sdk.base import DokumenAgent
        from dokumen.sdk.query_runner import MockQueryRunner

        runner = MockQueryRunner([])
        agent = DokumenAgent(
            id="test",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=["Read"],
            query_runner=runner,
        )

        assert agent._options.agents is None

    def test_agents_passed_to_options(self):
        """When agents param provided, ClaudeAgentOptions.agents is set."""
        from dokumen.sdk.base import DokumenAgent
        from dokumen.sdk.query_runner import MockQueryRunner

        runner = MockQueryRunner([])
        agents = {
            "helper": AgentDefinition(
                description="A helper",
                prompt="Help with tasks.",
                tools=["Read", "Bash"],
                model="sonnet",
            ),
        }
        agent = DokumenAgent(
            id="test",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=["Read", "Agent"],
            agents=agents,
            query_runner=runner,
        )

        assert agent._options.agents is not None
        assert "helper" in agent._options.agents
        assert agent._options.agents["helper"].description == "A helper"

    def test_agent_tool_in_allowed_tools(self):
        """When agents provided, Agent should be in allowed_tools if passed in sdk_tools."""
        from dokumen.sdk.base import DokumenAgent
        from dokumen.sdk.query_runner import MockQueryRunner

        runner = MockQueryRunner([])
        agents = {
            "sub": AgentDefinition(
                description="Sub agent",
                prompt="Do sub things.",
            ),
        }
        agent = DokumenAgent(
            id="test",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=["Read", "Agent"],
            agents=agents,
            query_runner=runner,
        )

        assert "Agent" in agent._options.allowed_tools


class TestSubagentHooks:
    """Tests for SubagentStart/Stop hook logging."""

    def test_subagent_hooks_added_to_validation_hooks(self):
        """build_validation_hooks includes SubagentStart and SubagentStop hooks."""
        from dokumen.sdk.hooks import build_validation_hooks

        hooks = build_validation_hooks()

        assert "SubagentStart" in hooks
        assert "SubagentStop" in hooks
        assert len(hooks["SubagentStart"]) == 1
        assert len(hooks["SubagentStop"]) == 1

    @pytest.mark.asyncio
    async def test_subagent_start_hook_returns_empty(self):
        """SubagentStart hook logs and returns empty dict (no-op)."""
        from dokumen.sdk.hooks import build_validation_hooks

        hooks = build_validation_hooks()
        start_hook = hooks["SubagentStart"][0].hooks[0]

        result = await start_hook(
            {"agent_id": "sub-1", "agent_type": "site-searcher"},
            "tool-use-123",
            {},
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_subagent_stop_hook_returns_empty(self):
        """SubagentStop hook logs and returns empty dict (no-op)."""
        from dokumen.sdk.hooks import build_validation_hooks

        hooks = build_validation_hooks()
        stop_hook = hooks["SubagentStop"][0].hooks[0]

        result = await stop_hook(
            {
                "agent_id": "sub-1",
                "agent_type": "site-searcher",
                "agent_transcript_path": "/tmp/transcript.jsonl",
            },
            "tool-use-456",
            {},
        )
        assert result == {}
