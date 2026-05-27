"""
Tests for CreateAgent - generates test scaffolds from natural language goals.

TDD tests for the CreateAgent, including:
- Scaffold generation (name, executor, judges)
- File discovery (explore phase integration)
- Unique name generation (conflict avoidance)
- Validation of generated scaffolds
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_explore_result():
    """Factory for creating mock ExploreResult objects."""
    def _make(
        files: List[str] = None,
        summary: str = "Found relevant documentation.",
        success: bool = True,
    ):
        result = MagicMock()
        result.files = [
            MagicMock(path=f, summary=f"Content from {f}", relevance=0.9)
            for f in (files or ["docs/policy.md"])
        ]
        result.summary = summary
        result.success = success
        result.duration = 1.5
        result.tool_calls_count = 3
        result.error = None
        return result
    return _make


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.complete = AsyncMock()
    return provider


# =============================================================================
# CreateResult Dataclass Tests
# =============================================================================


class TestCreateResultDataclass:
    """Tests for CreateResult dataclass."""

    def test_create_result_has_required_fields(self):
        """CreateResult should have all required fields."""
        from dokumen.create_agent import CreateResult

        result = CreateResult(
            success=True,
            scaffold_yaml="name: test\n...",
            scaffold_dict={"name": "test"},
            name="test",
            discovered_files=["docs/api.md"],
            duration=2.5,
            error=None,
        )

        assert result.success is True
        assert result.name == "test"
        assert "name: test" in result.scaffold_yaml
        assert result.discovered_files == ["docs/api.md"]

    def test_create_result_to_dict(self):
        """CreateResult.to_dict() should serialize correctly."""
        from dokumen.create_agent import CreateResult

        result = CreateResult(
            success=True,
            scaffold_yaml="name: test\n...",
            scaffold_dict={"name": "test"},
            name="test",
            discovered_files=["docs/api.md"],
            duration=2.5,
            error=None,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["name"] == "test"
        assert "scaffold_yaml" in d
        assert d["discovered_files"] == ["docs/api.md"]


# =============================================================================
# CreateAgent Initialization Tests
# =============================================================================


class TestCreateAgentInit:
    """Tests for CreateAgent initialization."""

    def test_create_agent_init(self, mock_provider):
        """CreateAgent should initialize with required parameters."""
        from dokumen.create_agent import CreateAgent

        agent = CreateAgent(
            provider=mock_provider,
            base_dir=".",
        )

        assert agent.provider == mock_provider
        assert agent.base_dir == "."
        assert agent.timeout == 120.0  # Default timeout

    def test_create_agent_init_with_timeout(self, mock_provider):
        """CreateAgent should accept custom timeout."""
        from dokumen.create_agent import CreateAgent

        agent = CreateAgent(
            provider=mock_provider,
            base_dir=".",
            timeout=60.0,
        )

        assert agent.timeout == 60.0

    def test_create_agent_init_with_tools(self, mock_provider):
        """CreateAgent should accept custom tools."""
        from dokumen.create_agent import CreateAgent

        mock_tool = MagicMock()
        mock_tool.name = "read_file"

        agent = CreateAgent(
            provider=mock_provider,
            base_dir=".",
            tools=[mock_tool],
        )

        assert len(agent.tools) == 1
        assert agent.tools[0].name == "read_file"


# =============================================================================
# Name Generation Tests
# =============================================================================


class TestNameGeneration:
    """Tests for unique name generation."""

    def test_goal_to_kebab_simple(self, mock_provider):
        """_goal_to_kebab should convert simple goals."""
        from dokumen.create_agent import CreateAgent

        agent = CreateAgent(provider=mock_provider, base_dir=".")

        name = agent._goal_to_kebab("Verify refund policy")

        assert name == "verify-refund-policy"

    def test_goal_to_kebab_strips_special_chars(self, mock_provider):
        """_goal_to_kebab should strip special characters."""
        from dokumen.create_agent import CreateAgent

        agent = CreateAgent(provider=mock_provider, base_dir=".")

        name = agent._goal_to_kebab("Verify the API's authentication!")

        # Should remove apostrophes and exclamation marks
        assert "'" not in name
        assert "!" not in name
        assert re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name)

    def test_goal_to_kebab_handles_long_goals(self, mock_provider):
        """_goal_to_kebab should truncate very long goals."""
        from dokumen.create_agent import CreateAgent

        agent = CreateAgent(provider=mock_provider, base_dir=".")

        long_goal = "Verify that the documentation correctly describes " * 5
        name = agent._goal_to_kebab(long_goal)

        # Name should not be excessively long (max 30 chars)
        assert len(name) <= 30

    def test_generate_unique_name_no_conflict(self, mock_provider):
        """_generate_unique_name should return base name if no conflict."""
        from dokumen.create_agent import CreateAgent

        agent = CreateAgent(provider=mock_provider, base_dir=".")

        name = agent._generate_unique_name(
            "Verify refund policy",
            existing_tests=["other-test", "another-test"]
        )

        assert name == "verify-refund-policy"

    def test_generate_unique_name_with_conflict(self, mock_provider):
        """_generate_unique_name should add suffix on conflict."""
        from dokumen.create_agent import CreateAgent

        agent = CreateAgent(provider=mock_provider, base_dir=".")

        name = agent._generate_unique_name(
            "Verify refund policy",
            existing_tests=["verify-refund-policy", "other-test"]
        )

        assert name == "verify-refund-policy-2"

    def test_generate_unique_name_multiple_conflicts(self, mock_provider):
        """_generate_unique_name should handle multiple conflicts."""
        from dokumen.create_agent import CreateAgent

        agent = CreateAgent(provider=mock_provider, base_dir=".")

        name = agent._generate_unique_name(
            "Verify refund policy",
            existing_tests=[
                "verify-refund-policy",
                "verify-refund-policy-2",
                "verify-refund-policy-3"
            ]
        )

        assert name == "verify-refund-policy-4"


# =============================================================================
# Scaffold Generation Tests
# =============================================================================


class TestScaffoldGeneration:
    """Tests for test scaffold generation."""

    @pytest.mark.asyncio
    async def test_create_generates_valid_scaffold(self, mock_provider, mock_explore_result):
        """create() should generate a valid test scaffold."""
        from dokumen.create_agent import CreateAgent

        # Mock explore result
        explore_result = mock_explore_result(files=["docs/refund-policy.md"])

        # Mock provider to return scaffold generation response
        mock_provider.complete = AsyncMock(side_effect=[
            # First call: doc analysis
            {"content": "The refund policy document describes return procedures.", "tool_calls": []},
            # Second call: scaffold generation
            {"content": """```yaml
name: verify-refund-policy
reason: Verify refund policy

files:
  - path: docs/refund-policy.md

executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: |
    Read and verify the refund policy document.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: |
      Verify the executor correctly validated the document.
      Return: {"verdict": "PASS|FAIL", "confidence": 0.0-1.0, "reason": "..."}

timeout: 120
```""", "tool_calls": []}
        ])

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            result = await agent.create(goal="Verify refund policy")

        assert result.success is True
        assert result.name == "verify-refund-policy"
        assert "verify-refund-policy" in result.scaffold_yaml
        assert "docs/refund-policy.md" in result.discovered_files

    @pytest.mark.asyncio
    async def test_create_uses_provided_files(self, mock_provider):
        """create() should use provided files instead of exploring."""
        from dokumen.create_agent import CreateAgent

        # Mock provider for scaffold generation
        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "API authentication describes OAuth and API keys.", "tool_calls": []},
            {"content": """```yaml
name: verify-api-auth
reason: Verify API authentication documentation

files:
  - path: docs/api/auth.md

executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Verify the API authentication document.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: Verify documentation accuracy.

timeout: 120
```""", "tool_calls": []}
        ])

        agent = CreateAgent(provider=mock_provider, base_dir=".")

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            result = await agent.create(
                goal="Verify API authentication",
                files=["docs/api/auth.md"]
            )

            # Explore should NOT be called when files are provided
            mock_explore_class.assert_not_called()

        assert result.success is True
        assert "docs/api/auth.md" in result.discovered_files

    @pytest.mark.asyncio
    async def test_create_discovers_files_when_not_provided(self, mock_provider, mock_explore_result):
        """create() should run explore phase when files not provided."""
        from dokumen.create_agent import CreateAgent

        explore_result = mock_explore_result(files=["docs/margin.md", "docs/trading.md"])

        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "Margin documentation analysis.", "tool_calls": []},
            {"content": """```yaml
name: verify-margin-requirements
reason: Verify margin requirements

files:
  - path: docs/margin.md

executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Verify margin requirements.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: Verify accuracy.

timeout: 120
```""", "tool_calls": []}
        ])

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            result = await agent.create(goal="Verify margin requirements")

            # Explore SHOULD be called
            mock_explore.explore.assert_called_once()

        assert result.success is True
        assert len(result.discovered_files) > 0

    @pytest.mark.asyncio
    async def test_create_avoids_name_conflicts(self, mock_provider, mock_explore_result):
        """create() should avoid name conflicts with existing tests."""
        from dokumen.create_agent import CreateAgent

        explore_result = mock_explore_result(files=["docs/policy.md"])

        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "Policy analysis.", "tool_calls": []},
            {"content": """```yaml
name: verify-policy
reason: Verify policy

files:
  - path: docs/policy.md

executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Verify policy.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: Verify accuracy.

timeout: 120
```""", "tool_calls": []}
        ])

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            result = await agent.create(
                goal="Verify policy",
                existing_tests=["verify-policy"]  # Conflict!
            )

        assert result.success is True
        # Name should be different from existing
        assert result.name != "verify-policy"
        assert result.name.startswith("verify-policy")

    @pytest.mark.asyncio
    async def test_create_handles_explore_failure(self, mock_provider):
        """create() should handle explore phase failure gracefully."""
        from dokumen.create_agent import CreateAgent

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(side_effect=Exception("Explore failed"))

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            result = await agent.create(goal="Verify something")

        assert result.success is False
        assert "Explore failed" in result.error

    @pytest.mark.asyncio
    async def test_create_handles_generation_failure(self, mock_provider, mock_explore_result):
        """create() should handle scaffold generation failure."""
        from dokumen.create_agent import CreateAgent

        explore_result = mock_explore_result(files=["docs/test.md"])

        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "Analysis.", "tool_calls": []},
            Exception("API error during generation")
        ])

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            result = await agent.create(goal="Verify something")

        assert result.success is False
        assert "API error" in result.error


# =============================================================================
# YAML Parsing Tests
# =============================================================================


class TestYamlParsing:
    """Tests for YAML extraction from LLM responses."""

    def test_extract_yaml_from_code_block(self, mock_provider):
        """_extract_yaml should extract YAML from code block."""
        from dokumen.create_agent import CreateAgent

        agent = CreateAgent(provider=mock_provider, base_dir=".")

        response = """Here's the test scaffold:

```yaml
name: test-name
reason: Test reason

files:
  - path: docs/test.md

executor:
  system_prompt: Prompt
  user_prompt: Task
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: Evaluate
```

This scaffold validates the documentation."""

        yaml_content = agent._extract_yaml_from_response(response)

        assert "name: test-name" in yaml_content
        assert "files:" in yaml_content
        assert "executor:" in yaml_content
        assert "judges:" in yaml_content

    def test_extract_yaml_without_code_block(self, mock_provider):
        """_extract_yaml should handle response without code block."""
        from dokumen.create_agent import CreateAgent

        agent = CreateAgent(provider=mock_provider, base_dir=".")

        response = """name: test-name
reason: Test reason

files:
  - path: docs/test.md

executor:
  system_prompt: Prompt
  user_prompt: Task
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: Evaluate"""

        yaml_content = agent._extract_yaml_from_response(response)

        assert "name: test-name" in yaml_content


# =============================================================================
# Progress Callback Tests
# =============================================================================


class TestProgressCallbacks:
    """Tests for progress callback functionality."""

    @pytest.mark.asyncio
    async def test_create_emits_progress_events(self, mock_provider, mock_explore_result):
        """create() should emit progress events."""
        from dokumen.create_agent import CreateAgent

        explore_result = mock_explore_result(files=["docs/test.md"])

        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "Analysis.", "tool_calls": []},
            {"content": """```yaml
name: test
reason: Test

files:
  - path: docs/test.md

executor:
  system_prompt: Prompt
  user_prompt: Task
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: Evaluate
```""", "tool_calls": []}
        ])

        progress_events = []

        def on_progress(event_type: str, data: dict):
            progress_events.append((event_type, data))

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            await agent.create(goal="Test", on_progress=on_progress)

        event_types = [e[0] for e in progress_events]
        assert "create_start" in event_types
        assert "explore_start" in event_types or "explore_complete" in event_types


# =============================================================================
# Validation Tests
# =============================================================================


class TestScaffoldValidation:
    """Tests for generated scaffold validation."""

    @pytest.mark.asyncio
    async def test_generated_scaffold_is_valid(self, mock_provider, mock_explore_result):
        """Generated scaffold should be valid according to schema."""
        from dokumen.create_agent import CreateAgent

        explore_result = mock_explore_result(files=["docs/test.md"])

        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "Analysis.", "tool_calls": []},
            {"content": """```yaml
name: verify-test-doc
reason: Verify test documentation

files:
  - path: docs/test.md

executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: |
    Verify the test documentation is accurate.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: |
      Verify the executor correctly validated the document.
      Return: {"verdict": "PASS|FAIL", "confidence": 0.0-1.0, "reason": "..."}

timeout: 120
```""", "tool_calls": []}
        ])

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            result = await agent.create(goal="Verify test documentation")

        assert result.success is True

        # Check scaffold structure
        scaffold = result.scaffold_dict
        assert "name" in scaffold
        assert "executor" in scaffold
        assert "judges" in scaffold
        assert isinstance(scaffold["judges"], list)
        assert len(scaffold["judges"]) > 0


# =============================================================================
# Browser Scaffold Generation Tests
# =============================================================================


class TestBrowserScaffoldGeneration:
    """Tests for browser test scaffold generation."""

    @pytest.mark.asyncio
    async def test_create_browser_scaffold_uses_browser_prompt(self, mock_provider, mock_explore_result):
        """create() with test_type='browser' should use the browser prompt."""
        from dokumen.create_agent import CreateAgent, CREATE_BROWSER_SCAFFOLD_PROMPT

        explore_result = mock_explore_result(files=["docs/credentials/pat.txt"])

        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "Credentials file for login testing.", "tool_calls": []},
            {"content": """```yaml
name: login-flow-test
type: browser
reason: Verify login flow

files:
  - path: docs/credentials/pat.txt

browser:
  headless: false
  save_video: "1920x1080"

executor:
  system_prompt: "@prompts/browser-testing.txt"
  user_prompt: |
    Navigate to https://app.dokumen.app and verify login.
  tools:
    - browser_navigate
    - browser_click
    - browser_type
    - read_file

judges:
  - name: login-check
    system_prompt: |
      Verify login was successful.
      Return: {"verdict": "PASS|FAIL", "confidence": 0.0-1.0, "reason": "..."}

timeout: 120
```""", "tool_calls": []}
        ])

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            result = await agent.create(goal="Test login flow", test_type="browser")

        assert result.success is True
        # Verify the browser prompt was used (check second call to provider)
        second_call = mock_provider.complete.call_args_list[1]
        system_msg = second_call[0][0][0]["content"]
        assert "browser" in system_msg.lower()

    @pytest.mark.asyncio
    async def test_create_browser_scaffold_has_type_and_browser_fields(self, mock_provider, mock_explore_result):
        """create() with test_type='browser' should ensure type and browser fields."""
        from dokumen.create_agent import CreateAgent

        explore_result = mock_explore_result(files=["docs/test.md"])

        # LLM returns scaffold WITHOUT type/browser - should be added by sanitization
        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "Analysis.", "tool_calls": []},
            {"content": """```yaml
name: ui-test
reason: Test UI

files:
  - path: docs/test.md

executor:
  system_prompt: "@prompts/browser-testing.txt"
  user_prompt: Navigate to the page.
  tools:
    - browser_navigate

judges:
  - name: check
    system_prompt: Verify.

timeout: 120
```""", "tool_calls": []}
        ])

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            result = await agent.create(goal="Test UI", test_type="browser")

        assert result.success is True
        assert result.scaffold_dict.get("type") == "browser"
        assert result.scaffold_dict.get("browser") is not None

    @pytest.mark.asyncio
    async def test_create_browser_scaffold_uses_browser_tools(self, mock_provider, mock_explore_result):
        """Browser scaffold should reference browser tools in the prompt."""
        from dokumen.create_agent import CreateAgent, CREATE_BROWSER_SCAFFOLD_PROMPT

        # Verify browser tools are mentioned in the prompt
        assert "browser_navigate" in CREATE_BROWSER_SCAFFOLD_PROMPT
        assert "browser_click" in CREATE_BROWSER_SCAFFOLD_PROMPT
        assert "browser_type" in CREATE_BROWSER_SCAFFOLD_PROMPT

    @pytest.mark.asyncio
    async def test_create_standard_unchanged(self, mock_provider, mock_explore_result):
        """create() with default test_type should not include browser fields."""
        from dokumen.create_agent import CreateAgent

        explore_result = mock_explore_result(files=["docs/policy.md"])

        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "Policy analysis.", "tool_calls": []},
            {"content": """```yaml
name: verify-policy
reason: Verify policy

files:
  - path: docs/policy.md

executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Verify the policy.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: Verify accuracy.

timeout: 120
```""", "tool_calls": []}
        ])

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            result = await agent.create(goal="Verify policy")

        assert result.success is True
        assert result.scaffold_dict.get("type") is None
        assert result.scaffold_dict.get("browser") is None

    @pytest.mark.asyncio
    async def test_create_invalid_test_type_raises(self, mock_provider):
        """create() with invalid test_type should raise ValueError."""
        from dokumen.create_agent import CreateAgent

        agent = CreateAgent(provider=mock_provider, base_dir=".")

        result = await agent.create(goal="Test something", test_type="invalid")
        assert result.success is False
        assert "invalid" in result.error.lower() or "test_type" in result.error.lower()

    @pytest.mark.asyncio
    async def test_create_standard_strips_browser_fields(self, mock_provider, mock_explore_result):
        """Standard create should strip type/browser fields if LLM includes them."""
        from dokumen.create_agent import CreateAgent

        explore_result = mock_explore_result(files=["docs/test.md"])

        # LLM erroneously includes type/browser in standard scaffold
        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "Analysis.", "tool_calls": []},
            {"content": """```yaml
name: verify-doc
type: browser
reason: Verify doc

files:
  - path: docs/test.md

browser:
  headless: false

executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Verify the doc.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: Verify.

timeout: 120
```""", "tool_calls": []}
        ])

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            # Default test_type is "standard"
            result = await agent.create(goal="Verify doc")

        assert result.success is True
        assert result.scaffold_dict.get("type") is None
        assert result.scaffold_dict.get("browser") is None

    @pytest.mark.asyncio
    async def test_create_uses_llm_generated_name(self, mock_provider, mock_explore_result):
        """create() should use the LLM-generated name from scaffold when valid."""
        from dokumen.create_agent import CreateAgent

        explore_result = mock_explore_result(files=["docs/test.md"])

        # LLM returns scaffold with a short, valid name different from goal-kebab
        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "Analysis.", "tool_calls": []},
            {"content": """```yaml
name: api-auth-check
reason: Verify API auth

files:
  - path: docs/test.md

executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Verify API authentication works correctly.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: Evaluate accuracy.

timeout: 120
```""", "tool_calls": []}
        ])

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            # Goal would produce "verify-api-authentication-works" via _goal_to_kebab
            result = await agent.create(goal="Verify API authentication works")

        assert result.success is True
        # Should use LLM name, not goal-kebab
        assert result.name == "api-auth-check"

    @pytest.mark.asyncio
    async def test_create_result_includes_test_type(self, mock_provider, mock_explore_result):
        """CreateResult should include test_type field."""
        from dokumen.create_agent import CreateAgent

        explore_result = mock_explore_result(files=["docs/test.md"])

        mock_provider.complete = AsyncMock(side_effect=[
            {"content": "Analysis.", "tool_calls": []},
            {"content": """```yaml
name: browser-test
reason: Test

files:
  - path: docs/test.md

executor:
  system_prompt: "@prompts/browser-testing.txt"
  user_prompt: Test.
  tools:
    - browser_navigate

judges:
  - name: check
    system_prompt: Evaluate.

timeout: 120
```""", "tool_calls": []}
        ])

        with patch('dokumen.create_agent.ExploreAgent') as mock_explore_class:
            mock_explore = MagicMock()
            mock_explore_class.return_value = mock_explore
            mock_explore.explore = AsyncMock(return_value=explore_result)

            agent = CreateAgent(provider=mock_provider, base_dir=".")
            result = await agent.create(goal="Test browser", test_type="browser")

        assert result.test_type == "browser"
