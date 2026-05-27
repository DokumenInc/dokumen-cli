"""Tests for agent_resolver module — agent loading, skills, and research judges."""
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from dokumen.agent_resolver import (
    resolve_executor_agent,
    resolve_judge_agent,
    get_agent_capabilities,
    compute_user_dirs,
    format_skills_for_prompt,
    collect_skills,
    RESEARCH_SOURCES_JUDGE_PROMPT,
    RESEARCH_VERDICT_JUDGE_PROMPT,
)


class TestResolveExecutorAgent:
    """Tests for resolve_executor_agent."""

    @patch("dokumen.agent_resolver._load_agent_def")
    def test_no_agent_field_returns_none(self, mock_load):
        executor_data = {"system_prompt": "test"}
        result = resolve_executor_agent(executor_data, {}, "test-scaffold")
        assert result is None
        mock_load.assert_not_called()

    @patch("dokumen.agent_resolver._load_agent_def")
    def test_agent_not_found_raises(self, mock_load):
        mock_load.return_value = None
        executor_data = {"agent": "nonexistent"}
        with pytest.raises(ValueError, match="not found"):
            resolve_executor_agent(executor_data, {}, "test-scaffold")

    @patch("dokumen.agent_resolver._load_agent_def")
    def test_applies_system_prompt_default(self, mock_load):
        agent_def = SimpleNamespace(
            name="test-agent",
            system_prompt="Agent default prompt",
            tools=["read_file"],
            skills=[],
            capabilities=[],
            browser=None,
            research=None,
        )
        mock_load.return_value = agent_def
        executor_data = {"agent": "test-agent"}
        scaffold_data = {}
        result = resolve_executor_agent(executor_data, scaffold_data, "test-scaffold")
        assert result == agent_def
        assert executor_data["system_prompt"] == "Agent default prompt"

    @patch("dokumen.agent_resolver._load_agent_def")
    def test_scaffold_overrides_system_prompt(self, mock_load):
        agent_def = SimpleNamespace(
            name="test-agent",
            system_prompt="Agent prompt",
            tools=["read_file"],
            skills=[],
            capabilities=[],
            browser=None,
            research=None,
        )
        mock_load.return_value = agent_def
        executor_data = {"agent": "test-agent", "system_prompt": "Custom prompt"}
        scaffold_data = {}
        resolve_executor_agent(executor_data, scaffold_data, "test-scaffold")
        assert executor_data["system_prompt"] == "Custom prompt"

    @patch("dokumen.agent_resolver._load_agent_def")
    def test_applies_tools_default(self, mock_load):
        agent_def = SimpleNamespace(
            name="test-agent",
            system_prompt="prompt",
            tools=["read_file", "glob"],
            skills=[],
            capabilities=[],
            browser=None,
            research=None,
        )
        mock_load.return_value = agent_def
        executor_data = {"agent": "test-agent"}
        resolve_executor_agent(executor_data, {}, "test-scaffold")
        assert executor_data["tools"] == ["read_file", "glob"]

    @patch("dokumen.agent_resolver._load_agent_def")
    def test_merges_skills(self, mock_load):
        agent_def = SimpleNamespace(
            name="test-agent",
            system_prompt="prompt",
            tools=[],
            skills=["skill-a"],
            capabilities=[],
            browser=None,
            research=None,
        )
        mock_load.return_value = agent_def
        executor_data = {"agent": "test-agent", "skills": ["skill-b"]}
        resolve_executor_agent(executor_data, {}, "test-scaffold")
        assert set(executor_data["skills"]) == {"skill-a", "skill-b"}


class TestResolveJudgeAgent:
    """Tests for resolve_judge_agent."""

    @patch("dokumen.agent_resolver._load_agent_def")
    def test_no_agent_returns_none(self, mock_load):
        result = resolve_judge_agent({"name": "accuracy"}, "test", "accuracy")
        assert result is None

    @patch("dokumen.agent_resolver._load_agent_def")
    def test_unknown_agent_raises(self, mock_load):
        mock_load.return_value = None
        with pytest.raises(ValueError, match="unknown agent"):
            resolve_judge_agent({"agent": "bad"}, "test", "accuracy")

    @patch("dokumen.agent_resolver._load_agent_def")
    def test_applies_judge_defaults(self, mock_load):
        agent_def = SimpleNamespace(
            system_prompt="Judge prompt",
            tools=["read_file"],
            skills=[],
        )
        mock_load.return_value = agent_def
        judge_data = {"agent": "judge-agent", "name": "accuracy"}
        resolve_judge_agent(judge_data, "test", "accuracy")
        assert judge_data["system_prompt"] == "Judge prompt"
        assert judge_data["tools"] == ["read_file"]


class TestGetAgentCapabilities:
    """Tests for get_agent_capabilities."""

    def test_none_returns_empty(self):
        assert get_agent_capabilities(None) == set()

    def test_extracts_capabilities(self):
        agent_def = SimpleNamespace(capabilities=["browser", "research"])
        assert get_agent_capabilities(agent_def) == {"browser", "research"}

    def test_empty_capabilities(self):
        agent_def = SimpleNamespace(capabilities=[])
        assert get_agent_capabilities(agent_def) == set()


class TestComputeUserDirs:
    """Tests for compute_user_dirs."""

    def test_none_config_returns_none(self):
        assert compute_user_dirs("/tmp", None) is None

    def test_nonexistent_dir_returns_none(self):
        config = SimpleNamespace(dir="agents")
        result = compute_user_dirs("/nonexistent/path", config)
        assert result is None

    def test_existing_dir_returns_list(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        config = SimpleNamespace(dir="agents")
        result = compute_user_dirs(str(tmp_path), config)
        assert result is not None
        assert len(result) == 1
        assert str(result[0]) == str(agents_dir)


class TestFormatSkillsForPrompt:
    """Tests for format_skills_for_prompt."""

    def test_empty_skills_returns_empty(self):
        assert format_skills_for_prompt([]) == ""

    def test_single_skill(self):
        result = format_skills_for_prompt([("test-skill", "content here", "scaffold")])
        assert "## Available Skills" in result
        assert "### test-skill (source: scaffold)" in result
        assert "content here" in result

    def test_multiple_skills(self):
        skills = [
            ("skill-a", "A content", "agent:db"),
            ("skill-b", "B content", "scaffold"),
        ]
        result = format_skills_for_prompt(skills)
        assert "skill-a" in result
        assert "skill-b" in result
        assert "---" in result  # separator


class TestCollectSkills:
    """Tests for collect_skills."""

    @patch("dokumen.agent_resolver.get_agent_skills", return_value=[])
    def test_no_skills_returns_empty(self, mock_db):
        result = collect_skills(None, "/tmp")
        assert result == []

    @patch("dokumen.agent_resolver.get_agent_skills")
    def test_db_skills_collected(self, mock_db):
        mock_db.return_value = [
            {"name": "db-skill", "content": "DB skill content"}
        ]
        result = collect_skills(None, "/tmp")
        assert len(result) == 1
        assert result[0] == ("db-skill", "DB skill content", "agent:db")

    @patch("dokumen.agent_resolver.get_agent_skills", return_value=[
        {"name": "dup-skill", "content": "DB version"}
    ])
    @patch("dokumen.agent_resolver.SkillLoader")
    def test_db_wins_on_duplicate(self, mock_loader_cls, mock_db):
        mock_loader = MagicMock()
        mock_loader.load_skills.return_value = [
            SimpleNamespace(name="dup-skill", content="Scaffold version", file_path="/tmp/SKILL.md")
        ]
        mock_loader_cls.return_value = mock_loader
        mock_loader_cls.return_value._paths = ["/tmp"]

        result = collect_skills(["dup-skill"], "/tmp")
        assert len(result) == 1
        assert result[0][2] == "agent:db"  # DB wins

    @patch("dokumen.agent_resolver.get_agent_skills", return_value=[])
    @patch("dokumen.agent_resolver.SkillLoader")
    def test_scaffold_skill_not_found_raises(self, mock_loader_cls, mock_db):
        mock_loader = MagicMock()
        mock_loader.load_skills.return_value = []
        mock_loader_cls.return_value = mock_loader
        mock_loader_cls.return_value._paths = ["/tmp/skills"]

        with pytest.raises(ValueError, match="not found"):
            collect_skills(["missing-skill"], "/tmp")


class TestResearchJudgePrompts:
    """Tests for research judge prompt constants."""

    def test_sources_prompt_not_empty(self):
        assert len(RESEARCH_SOURCES_JUDGE_PROMPT) > 100
        assert "source" in RESEARCH_SOURCES_JUDGE_PROMPT.lower()

    def test_verdict_prompt_not_empty(self):
        assert len(RESEARCH_VERDICT_JUDGE_PROMPT) > 100
        assert "verdict" in RESEARCH_VERDICT_JUDGE_PROMPT.lower()
