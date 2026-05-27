"""Tests for CreateAgent core logic.

Tests the key methods of CreateAgent including scaffold generation,
name generation, YAML extraction, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dokumen.create_agent import CreateAgent, CreateResult
from dokumen.explore_agent import ExploreResult, FileDiscovery


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    provider = MagicMock()
    provider.complete = AsyncMock(return_value={
        "content": """```yaml
name: test-scaffold
reason: Test reason

files:
  - path: docs/api.md

executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: |
    Test the API documentation.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: |
      Evaluate accuracy.
      Return: {"verdict": "PASS", "confidence": 0.9, "reason": "Correct"}

timeout: 120
```""",
    })
    return provider


@pytest.fixture
def mock_explore_result():
    """Create a mock ExploreResult."""
    files = [
        FileDiscovery(path="docs/api.md", summary="API documentation", relevance=0.9),
        FileDiscovery(path="docs/auth.md", summary="Auth documentation", relevance=0.7),
    ]
    return ExploreResult(
        files=files,
        duration=1.5,
        tool_calls_count=3,
        success=True,
        summary="Found API and auth documentation",
        tool_history=[],
    )


@pytest.fixture
def create_agent(mock_provider):
    """Create a CreateAgent instance for testing."""
    return CreateAgent(
        provider=mock_provider,
        base_dir=".",
        timeout=60.0,
    )


# =============================================================================
# Create Method Tests
# =============================================================================


class TestCreateMethod:
    """Tests for the create() method."""

    @pytest.mark.asyncio
    async def test_create_returns_successful_result(
        self, create_agent, mock_provider, mock_explore_result
    ):
        """create() returns successful result with scaffold."""
        with patch.object(create_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            result = await create_agent.create("Validate API documentation")

            assert result.success is True
            assert result.scaffold_yaml != ""
            assert result.name != ""
            assert len(result.discovered_files) > 0

    @pytest.mark.asyncio
    async def test_create_uses_provided_files(
        self, create_agent, mock_provider
    ):
        """create() uses provided files instead of exploring."""
        with patch.object(create_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            result = await create_agent.create(
                "Validate API",
                files=["docs/custom.md", "docs/other.md"],
            )

            # Explore should not be called when files are provided
            mock_explore.assert_not_called()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_create_generates_unique_name(
        self, create_agent, mock_explore_result
    ):
        """create() generates unique name avoiding existing tests."""
        with patch.object(create_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            result = await create_agent.create(
                "Validate API",
                existing_tests=["validate-api", "test-scaffold"],
            )

            # Should add suffix to avoid conflict with both goal-based and LLM names
            assert result.success is True
            assert result.name != "validate-api"
            assert result.name == "test-scaffold-2"

    @pytest.mark.asyncio
    async def test_create_returns_error_on_exception(
        self, create_agent, mock_provider, mock_explore_result
    ):
        """create() returns error result on exception."""
        with patch.object(create_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.side_effect = Exception("Explore failed")

            result = await create_agent.create("Validate API")

            assert result.success is False
            assert "Explore failed" in result.error

    @pytest.mark.asyncio
    async def test_create_calls_progress_callbacks(
        self, create_agent, mock_explore_result
    ):
        """create() calls progress callbacks at appropriate stages."""
        progress_events = []

        def on_progress(event, data):
            progress_events.append((event, data))

        with patch.object(create_agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            await create_agent.create("Validate API", on_progress=on_progress)

            event_names = [e[0] for e in progress_events]
            assert "create_start" in event_names
            assert "explore_start" in event_names
            assert "scaffold_generating" in event_names


# =============================================================================
# Analyze Docs Tests
# =============================================================================


class TestAnalyzeDocs:
    """Tests for _analyze_docs() method."""

    @pytest.mark.asyncio
    async def test_analyze_docs_returns_analysis(self, create_agent, mock_provider):
        """_analyze_docs returns analysis string."""
        mock_provider.complete = AsyncMock(return_value={
            "content": "The API documentation describes REST endpoints."
        })

        analysis = await create_agent._analyze_docs(
            ["docs/api.md", "docs/auth.md"],
            "Validate API endpoints"
        )

        assert "REST endpoints" in analysis

    @pytest.mark.asyncio
    async def test_analyze_docs_includes_goal_in_prompt(self, create_agent, mock_provider):
        """_analyze_docs includes goal in the prompt to provider."""
        mock_provider.complete = AsyncMock(return_value={"content": "Analysis"})

        await create_agent._analyze_docs(
            ["docs/api.md"],
            "Validate refund policy"
        )

        # Check that the call included the goal
        call_args = mock_provider.complete.call_args
        messages = call_args[0][0]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "Validate refund policy" in user_message["content"]


# =============================================================================
# Generate Scaffold Tests
# =============================================================================


class TestGenerateScaffold:
    """Tests for _generate_scaffold() method."""

    @pytest.mark.asyncio
    async def test_generate_scaffold_returns_yaml(self, create_agent, mock_provider):
        """_generate_scaffold returns valid YAML string."""
        yaml_content = await create_agent._generate_scaffold(
            goal="Validate API",
            name="validate-api",
            files=["docs/api.md"],
            doc_analysis="API documentation",
        )

        # Should be valid YAML
        import yaml
        parsed = yaml.safe_load(yaml_content)
        assert parsed is not None
        assert "name" in parsed


# =============================================================================
# Extract YAML From Response Tests
# =============================================================================


class TestExtractYamlFromResponse:
    """Tests for _extract_yaml_from_response() method."""

    def test_extract_yaml_from_code_block(self, create_agent):
        """_extract_yaml_from_response extracts from ```yaml block."""
        response = """Here's the scaffold:

```yaml
name: test-scaffold
reason: Test reason
```

That's the scaffold."""

        yaml_str = create_agent._extract_yaml_from_response(response)

        assert "name: test-scaffold" in yaml_str
        assert "reason: Test reason" in yaml_str
        assert "Here's the scaffold" not in yaml_str

    def test_extract_yaml_from_code_block_without_yaml_tag(self, create_agent):
        """_extract_yaml_from_response extracts from ``` block without yaml tag."""
        response = """```
name: test-scaffold
timeout: 120
```"""

        yaml_str = create_agent._extract_yaml_from_response(response)

        assert "name: test-scaffold" in yaml_str
        assert "timeout: 120" in yaml_str

    def test_extract_yaml_finds_name_line(self, create_agent):
        """_extract_yaml_from_response finds YAML starting at 'name:' line."""
        response = """Some preamble text.

name: my-test
reason: Test reason
files:
  - path: docs/file.md"""

        yaml_str = create_agent._extract_yaml_from_response(response)

        assert yaml_str.startswith("name: my-test")
        assert "Some preamble text" not in yaml_str


# =============================================================================
# Goal To Kebab Tests
# =============================================================================


class TestGoalToKebab:
    """Tests for _goal_to_kebab() method."""

    def test_goal_to_kebab_simple(self, create_agent):
        """_goal_to_kebab converts simple goal to kebab-case."""
        assert create_agent._goal_to_kebab("Validate API") == "validate-api"

    def test_goal_to_kebab_removes_special_chars(self, create_agent):
        """_goal_to_kebab removes special characters."""
        assert create_agent._goal_to_kebab("Test API!@#$%") == "test-api"

    def test_goal_to_kebab_handles_multiple_spaces(self, create_agent):
        """_goal_to_kebab handles multiple consecutive spaces."""
        assert create_agent._goal_to_kebab("Validate   API   Docs") == "validate-api-docs"

    def test_goal_to_kebab_truncates_long_names(self, create_agent):
        """_goal_to_kebab truncates names longer than 30 characters."""
        long_goal = "This is a very long test goal that exceeds the maximum length allowed for test names"
        result = create_agent._goal_to_kebab(long_goal)
        assert len(result) <= 30

    def test_goal_to_kebab_strips_leading_trailing_hyphens(self, create_agent):
        """_goal_to_kebab strips leading and trailing hyphens."""
        assert create_agent._goal_to_kebab("-Test API-") == "test-api"

    def test_goal_to_kebab_handles_numbers(self, create_agent):
        """_goal_to_kebab preserves numbers."""
        assert create_agent._goal_to_kebab("Test API v2") == "test-api-v2"


# =============================================================================
# Generate Unique Name Tests
# =============================================================================


class TestGenerateUniqueName:
    """Tests for _generate_unique_name() method."""

    def test_generate_unique_name_no_conflict(self, create_agent):
        """_generate_unique_name returns base name when no conflict."""
        name = create_agent._generate_unique_name("Validate API", [])
        assert name == "validate-api"

    def test_generate_unique_name_with_conflict(self, create_agent):
        """_generate_unique_name adds suffix when name exists."""
        name = create_agent._generate_unique_name(
            "Validate API",
            ["validate-api"]
        )
        assert name == "validate-api-2"

    def test_generate_unique_name_multiple_conflicts(self, create_agent):
        """_generate_unique_name increments suffix for multiple conflicts."""
        name = create_agent._generate_unique_name(
            "Validate API",
            ["validate-api", "validate-api-2", "validate-api-3"]
        )
        assert name == "validate-api-4"

    def test_generate_unique_name_raises_on_too_many_conflicts(self, create_agent):
        """_generate_unique_name raises error after 99 conflicts."""
        existing = ["validate-api"] + [f"validate-api-{i}" for i in range(2, 100)]

        with pytest.raises(ValueError, match="Could not generate unique name"):
            create_agent._generate_unique_name("Validate API", existing)


# =============================================================================
# Is Valid Name Tests
# =============================================================================


class TestIsValidName:
    """Tests for _is_valid_name() method."""

    def test_valid_kebab_name(self, create_agent):
        """_is_valid_name accepts valid kebab-case names."""
        assert create_agent._is_valid_name("refund-policy-validation") is True

    def test_rejects_empty_name(self, create_agent):
        """_is_valid_name rejects empty string."""
        assert create_agent._is_valid_name("") is False

    def test_rejects_none(self, create_agent):
        """_is_valid_name rejects None."""
        assert create_agent._is_valid_name(None) is False

    def test_rejects_too_long_name(self, create_agent):
        """_is_valid_name rejects names longer than 40 chars."""
        long_name = "a-very-long-test-name-that-exceeds-forty-characters"
        assert len(long_name) > 40
        assert create_agent._is_valid_name(long_name) is False

    def test_rejects_uppercase(self, create_agent):
        """_is_valid_name rejects names with uppercase letters."""
        assert create_agent._is_valid_name("Refund-Policy") is False

    def test_rejects_spaces(self, create_agent):
        """_is_valid_name rejects names with spaces."""
        assert create_agent._is_valid_name("refund policy") is False

    def test_rejects_single_word(self, create_agent):
        """_is_valid_name rejects single-word names (too vague)."""
        assert create_agent._is_valid_name("refund") is False

    def test_accepts_two_word_name(self, create_agent):
        """_is_valid_name accepts two-word kebab names."""
        assert create_agent._is_valid_name("refund-policy") is True

    def test_rejects_trailing_hyphen(self, create_agent):
        """_is_valid_name rejects names with trailing hyphen."""
        assert create_agent._is_valid_name("refund-") is False

    def test_rejects_leading_hyphen(self, create_agent):
        """_is_valid_name rejects names with leading hyphen."""
        assert create_agent._is_valid_name("-refund") is False

    def test_accepts_numbers(self, create_agent):
        """_is_valid_name accepts names with numbers."""
        assert create_agent._is_valid_name("api-v2-auth") is True

    def test_accepts_max_length_name(self, create_agent):
        """_is_valid_name accepts names exactly at 40 chars."""
        name = "a" * 19 + "-" + "b" * 19 + "c"  # 40 chars
        assert len(name) == 40
        assert create_agent._is_valid_name(name) is True


# =============================================================================
# Extract Name From Scaffold Tests
# =============================================================================


class TestExtractNameFromScaffold:
    """Tests for _extract_name_from_scaffold() method."""

    def test_uses_llm_name_when_valid(self, create_agent):
        """_extract_name_from_scaffold uses valid LLM name."""
        scaffold = {"name": "refund-policy-check"}
        name = create_agent._extract_name_from_scaffold(
            scaffold, "Verify the refund policy", []
        )
        assert name == "refund-policy-check"

    def test_falls_back_to_goal_when_llm_name_invalid(self, create_agent):
        """_extract_name_from_scaffold falls back when LLM name is too long."""
        scaffold = {"name": "this-is-a-very-long-name-that-exceeds-the-forty-character-limit"}
        name = create_agent._extract_name_from_scaffold(
            scaffold, "Verify API", []
        )
        # Should fall back to goal-based name
        assert name == "verify-api"

    def test_falls_back_when_name_missing(self, create_agent):
        """_extract_name_from_scaffold falls back when no name in scaffold."""
        scaffold = {}
        name = create_agent._extract_name_from_scaffold(
            scaffold, "Verify API docs", []
        )
        assert name == "verify-api-docs"

    def test_adds_suffix_for_uniqueness(self, create_agent):
        """_extract_name_from_scaffold adds suffix when LLM name conflicts."""
        scaffold = {"name": "refund-policy-check"}
        name = create_agent._extract_name_from_scaffold(
            scaffold, "Verify refund", ["refund-policy-check"]
        )
        assert name == "refund-policy-check-2"

    def test_falls_back_when_llm_name_has_uppercase(self, create_agent):
        """_extract_name_from_scaffold falls back when LLM name has uppercase."""
        scaffold = {"name": "Refund-Policy"}
        name = create_agent._extract_name_from_scaffold(
            scaffold, "Verify refund", []
        )
        assert name == "verify-refund"

    def test_falls_back_when_llm_name_is_single_word(self, create_agent):
        """_extract_name_from_scaffold falls back when LLM name is single word."""
        scaffold = {"name": "refund"}
        name = create_agent._extract_name_from_scaffold(
            scaffold, "Verify refund policy", []
        )
        assert name == "verify-refund-policy"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in CreateAgent."""

    @pytest.mark.asyncio
    async def test_create_handles_provider_error(self, mock_provider, mock_explore_result):
        """create() handles provider errors gracefully."""
        mock_provider.complete = AsyncMock(side_effect=Exception("API error"))

        agent = CreateAgent(provider=mock_provider)

        with patch.object(agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            result = await agent.create("Validate API")

            assert result.success is False
            assert "API error" in result.error

    @pytest.mark.asyncio
    async def test_create_handles_invalid_yaml_response(self, mock_provider, mock_explore_result):
        """create() handles invalid YAML from provider."""
        mock_provider.complete = AsyncMock(return_value={
            "content": "This is not valid YAML: [{"
        })

        agent = CreateAgent(provider=mock_provider)

        with patch.object(agent, "_run_explore", new_callable=AsyncMock) as mock_explore:
            mock_explore.return_value = mock_explore_result

            result = await agent.create("Validate API")

            # Should fail gracefully
            assert result.success is False


# =============================================================================
# CreateResult Tests
# =============================================================================


class TestCreateResult:
    """Tests for CreateResult dataclass."""

    def test_create_result_to_dict(self):
        """CreateResult.to_dict() returns correct structure."""
        result = CreateResult(
            success=True,
            scaffold_yaml="name: test\n",
            scaffold_dict={"name": "test"},
            name="test",
            discovered_files=["docs/api.md"],
            duration=1.5,
            error=None,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["scaffold_yaml"] == "name: test\n"
        assert d["scaffold_dict"] == {"name": "test"}
        assert d["name"] == "test"
        assert d["discovered_files"] == ["docs/api.md"]
        assert d["duration"] == 1.5
        assert d["error"] is None

    def test_create_result_to_dict_with_error(self):
        """CreateResult.to_dict() includes error when present."""
        result = CreateResult(
            success=False,
            scaffold_yaml="",
            scaffold_dict={},
            name="",
            discovered_files=[],
            duration=0.5,
            error="Something went wrong",
        )

        d = result.to_dict()

        assert d["success"] is False
        assert d["error"] == "Something went wrong"

    def test_create_result_to_dict_includes_test_type(self):
        """CreateResult.to_dict() should include test_type field."""
        result = CreateResult(
            success=True,
            scaffold_yaml="name: test\n",
            scaffold_dict={"name": "test"},
            name="test",
            discovered_files=["docs/api.md"],
            duration=1.5,
            error=None,
            test_type="browser",
        )

        d = result.to_dict()

        assert d["test_type"] == "browser"

    def test_create_result_test_type_defaults_standard(self):
        """CreateResult.test_type should default to 'standard'."""
        result = CreateResult(
            success=True,
            scaffold_yaml="name: test\n",
            scaffold_dict={"name": "test"},
            name="test",
            discovered_files=[],
            duration=1.0,
        )

        assert result.test_type == "standard"
        assert result.to_dict()["test_type"] == "standard"
