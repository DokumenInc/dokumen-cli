"""Tests for delegate_to_agent tool integration with judges.

Issue #587: Enable subagent spawning for CLI executors and judges.
Verifies that delegate_to_agent can be resolved, used by judges,
and that subagent tools are restricted to the parent's tool set.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestDelegateToAgentResolution:
    """Tests that delegate_to_agent resolves correctly in resolve_tools."""

    def test_delegate_to_agent_resolves_with_registry(self):
        """delegate_to_agent resolves to a real tool (not placeholder) when registry provided."""
        from dokumen.loader import resolve_tools

        mock_registry = MagicMock()
        mock_provider = MagicMock()

        tools = resolve_tools(
            ["delegate_to_agent"],
            base_dir=".",
            agent_registry=mock_registry,
            agent_provider=mock_provider,
        )

        assert len(tools) == 1
        assert tools[0].name == "delegate_to_agent"
        # Should NOT be a placeholder - real tool has a proper description
        assert "placeholder" not in tools[0].description.lower()
        assert "not available" not in tools[0].description.lower()

    def test_delegate_to_agent_placeholder_without_registry(self):
        """delegate_to_agent creates placeholder when no registry provided."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["delegate_to_agent"],
            base_dir=".",
        )

        assert len(tools) == 1
        assert tools[0].name == "delegate_to_agent"

    @pytest.mark.asyncio
    async def test_delegate_placeholder_errors_on_call(self):
        """Placeholder delegate_to_agent returns error when called."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["delegate_to_agent"],
            base_dir=".",
        )

        result = await tools[0].handler({"agent": "explore", "input": "test"})
        assert result.success is False

    def test_existing_tools_still_resolve(self):
        """Other tools still resolve correctly when delegate_to_agent params are added."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["read_file", "glob"],
            base_dir=".",
            agent_registry=None,
            agent_provider=None,
        )

        assert len(tools) == 2
        assert tools[0].name == "read_file"
        assert tools[1].name == "glob"


class TestDelegateToAgentInImplementableIn:
    """Tests that delegate_to_agent is in the IMPLEMENTABLE_IN map."""

    def test_delegate_to_agent_in_implementable_in(self):
        """delegate_to_agent is registered in IMPLEMENTABLE_IN."""
        from dokumen.user_tool_overrides import IMPLEMENTABLE_IN

        assert "delegate_to_agent" in IMPLEMENTABLE_IN

    def test_delegate_to_agent_available_in_test(self):
        """delegate_to_agent is available in the test system."""
        from dokumen.user_tool_overrides import IMPLEMENTABLE_IN

        assert "test" in IMPLEMENTABLE_IN["delegate_to_agent"]

    def test_delegate_to_agent_not_in_explore(self):
        """delegate_to_agent should NOT be available in explore phase."""
        from dokumen.user_tool_overrides import IMPLEMENTABLE_IN

        assert "explore" not in IMPLEMENTABLE_IN["delegate_to_agent"]

    def test_delegate_to_agent_not_in_chat(self):
        """delegate_to_agent is CLI-only, not in chat (chat uses spawn_agent)."""
        from dokumen.user_tool_overrides import IMPLEMENTABLE_IN

        assert "chat" not in IMPLEMENTABLE_IN["delegate_to_agent"]

    def test_delegate_to_agent_not_explore_safe(self):
        """delegate_to_agent should not be in EXPLORE_SAFE_TOOLS."""
        from dokumen.user_tool_overrides import EXPLORE_SAFE_TOOLS

        assert "delegate_to_agent" not in EXPLORE_SAFE_TOOLS

    def test_is_tool_enabled_for_test(self):
        """is_tool_enabled_for_test returns True for delegate_to_agent by default."""
        from dokumen.user_tool_overrides import is_tool_enabled_for_test

        # No overrides = legacy mode = all tools enabled
        assert is_tool_enabled_for_test("delegate_to_agent", overrides=None) is True

    def test_user_override_can_disable_delegate(self):
        """User overrides can disable delegate_to_agent for test."""
        from dokumen.user_tool_overrides import (
            validate_tool_overrides,
            is_tool_enabled_for_test,
        )

        result = validate_tool_overrides({"delegate_to_agent": []})
        assert is_tool_enabled_for_test("delegate_to_agent", overrides=result) is False


class TestJudgeWithDelegateToAgent:
    """Tests that judges can use delegate_to_agent tool."""

    def test_judge_scaffold_with_delegate_to_agent(self):
        """Loading a scaffold with judge tools including delegate_to_agent succeeds."""
        from dokumen.loader import resolve_tools

        mock_registry = MagicMock()
        mock_provider = MagicMock()

        # Simulate judge tool resolution with delegate_to_agent
        judge_tool_names = ["run_shell_command", "read_file", "delegate_to_agent"]
        tools = resolve_tools(
            judge_tool_names,
            base_dir=".",
            agent_registry=mock_registry,
            agent_provider=mock_provider,
        )

        tool_names = [t.name for t in tools]
        assert "delegate_to_agent" in tool_names
        assert "run_shell_command" in tool_names
        assert "read_file" in tool_names

    def test_executor_delegate_still_works(self):
        """Executor tool resolution with delegate_to_agent still works."""
        from dokumen.loader import resolve_tools

        mock_registry = MagicMock()
        mock_provider = MagicMock()

        executor_tool_names = ["read_file", "run_shell_command", "delegate_to_agent"]
        tools = resolve_tools(
            executor_tool_names,
            base_dir=".",
            agent_registry=mock_registry,
            agent_provider=mock_provider,
        )

        tool_names = [t.name for t in tools]
        assert "delegate_to_agent" in tool_names
        assert len(tools) == 3

    @pytest.mark.asyncio
    async def test_delegate_tool_handler_calls_run_agent(self):
        """delegate_to_agent handler invokes run_agent when agent is found."""
        import sys
        import types

        from dokumen.tools_object import create_delegate_to_agent_tool

        mock_registry = MagicMock()
        mock_provider = MagicMock()

        mock_definition = MagicMock()
        mock_definition.name = "explore"
        mock_registry.get.return_value = mock_definition

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "Found 3 files"
        mock_result.duration = 2.5
        mock_result.tool_calls = []

        tool = create_delegate_to_agent_tool(
            registry=mock_registry,
            provider=mock_provider,
            sandbox=None,
            timeout=60,
        )

        # Create a mock agents module with run_agent
        mock_agents_module = types.ModuleType("dokumen.agents")
        mock_run_agent = AsyncMock(return_value=mock_result)
        mock_agents_module.run_agent = mock_run_agent
        sys.modules["dokumen.agents"] = mock_agents_module

        try:
            result = await tool.handler({
                "agent": "explore",
                "input": "find test files",
            })

            assert result.success is True
            assert result.output["agent"] == "explore"
            assert mock_run_agent.called
        finally:
            del sys.modules["dokumen.agents"]

    @pytest.mark.asyncio
    async def test_delegate_tool_with_parent_tools_restriction(self):
        """delegate_to_agent created with parent_tools restricts subagent tools."""
        from dokumen.tools_object import create_delegate_to_agent_tool

        mock_registry = MagicMock()
        mock_provider = MagicMock()

        # Create a list of parent tools (simulating judge's read-only tools)
        parent_tool_1 = MagicMock()
        parent_tool_1.name = "read_file"
        parent_tool_2 = MagicMock()
        parent_tool_2.name = "glob"
        parent_tools = [parent_tool_1, parent_tool_2]

        tool = create_delegate_to_agent_tool(
            registry=mock_registry,
            provider=mock_provider,
            sandbox=None,
            timeout=60,
            parent_tools=parent_tools,
        )

        # Tool should be created successfully
        assert tool.name == "delegate_to_agent"
        # Description should mention tool restriction
        assert "restricted" in tool.description.lower() or "parent" in tool.description.lower() or "delegate" in tool.description.lower()


class TestDelegateToAgentEdgeCases:
    """Edge case tests for delegate_to_agent integration."""

    def test_resolve_tools_with_mixed_tools_including_delegate(self):
        """resolve_tools handles a mix of standard tools and delegate_to_agent."""
        from dokumen.loader import resolve_tools

        mock_registry = MagicMock()
        mock_provider = MagicMock()

        tools = resolve_tools(
            ["read_file", "delegate_to_agent", "glob"],
            base_dir=".",
            agent_registry=mock_registry,
            agent_provider=mock_provider,
        )

        assert len(tools) == 3
        names = [t.name for t in tools]
        assert names == ["read_file", "delegate_to_agent", "glob"]

    def test_resolve_tools_delegate_without_registry_params_defaults_none(self):
        """resolve_tools works when agent_registry/agent_provider not passed at all."""
        from dokumen.loader import resolve_tools

        # Should not raise - just creates placeholder
        tools = resolve_tools(
            ["delegate_to_agent"],
            base_dir=".",
        )

        assert len(tools) == 1
        assert tools[0].name == "delegate_to_agent"
