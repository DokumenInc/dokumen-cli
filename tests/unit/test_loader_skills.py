"""Tests for skill injection into system prompts via loader.py."""

import os
from unittest.mock import patch, MagicMock

import pytest


class TestFormatSkillsForPrompt:
    """Tests for _format_skills_for_prompt helper."""

    def test_formats_single_skill(self):
        """Formats a single skill into a structured prompt section."""
        from dokumen.loader import _format_skills_for_prompt

        skills = [("review-docs", "# Review\nCheck docs...", "scaffold")]
        result = _format_skills_for_prompt(skills)

        assert "review-docs" in result
        assert "# Review\nCheck docs..." in result
        assert "scaffold" in result

    def test_formats_multiple_skills(self):
        """Formats multiple skills, each in its own block."""
        from dokumen.loader import _format_skills_for_prompt

        skills = [
            ("review-docs", "Check docs content", "scaffold"),
            ("summarize", "Summarize content", "agent:db"),
        ]
        result = _format_skills_for_prompt(skills)

        assert "review-docs" in result
        assert "Check docs content" in result
        assert "summarize" in result
        assert "Summarize content" in result

    def test_empty_list_returns_empty_string(self):
        """Empty skill list returns empty string (no-op)."""
        from dokumen.loader import _format_skills_for_prompt

        result = _format_skills_for_prompt([])
        assert result == ""

    def test_format_includes_header(self):
        """Output includes a clear section header."""
        from dokumen.loader import _format_skills_for_prompt

        skills = [("s1", "content", "scaffold")]
        result = _format_skills_for_prompt(skills)

        assert "Available Skills" in result or "SKILLS" in result


class TestCollectSkills:
    """Tests for _collect_skills — merges DB + scaffold skills."""

    def test_returns_empty_when_no_skills(self):
        """Returns empty list when no DB or scaffold skills."""
        from dokumen.loader import _collect_skills

        result = _collect_skills(
            scaffold_skill_names=None,
            base_dir="/tmp/test",
        )
        assert result == []

    def test_returns_db_skills_when_agent_set(self):
        """Returns DB skills when DOKUMEN_AGENT_ID is set."""
        from dokumen.loader import _collect_skills

        db_skills = [
            {"name": "review-docs", "content": "# Review\nCheck...", "description": "Review docs"},
        ]

        with patch("dokumen.loader.get_agent_skills", return_value=db_skills):
            result = _collect_skills(
                scaffold_skill_names=None,
                base_dir="/tmp/test",
            )

        assert len(result) == 1
        assert result[0][0] == "review-docs"
        assert result[0][2] == "agent:db"

    def test_returns_scaffold_skills_from_workspace(self):
        """Resolves scaffold skill names via SkillLoader."""
        from dokumen.loader import _collect_skills
        from dokumen_schema.skills import SkillInfo

        mock_skill = SkillInfo(
            name="summarize",
            description="Summarize docs",
            file_path="skills/summarize/SKILL.md",
            content="# Summarize\nSummarize content...",
        )

        with patch("dokumen.loader.get_agent_skills", return_value=[]), \
             patch("dokumen.loader.SkillLoader") as MockLoader:
            MockLoader.return_value.load_skills.return_value = [mock_skill]

            result = _collect_skills(
                scaffold_skill_names=["summarize"],
                base_dir="/tmp/test",
            )

        assert len(result) == 1
        assert result[0][0] == "summarize"
        assert result[0][2] == "scaffold"

    def test_scaffold_skill_not_found_raises(self):
        """Raises ValueError when scaffold references a skill not in workspace."""
        from dokumen.loader import _collect_skills

        with patch("dokumen.loader.get_agent_skills", return_value=[]), \
             patch("dokumen.loader.SkillLoader") as MockLoader:
            MockLoader.return_value.load_skills.return_value = []

            with pytest.raises(ValueError, match="missing-skill"):
                _collect_skills(
                    scaffold_skill_names=["missing-skill"],
                    base_dir="/tmp/test",
                )

    def test_db_wins_on_duplicate_name(self):
        """DB skills take priority over scaffold skills with same name."""
        from dokumen.loader import _collect_skills
        from dokumen_schema.skills import SkillInfo

        db_skills = [
            {"name": "review-docs", "content": "DB version", "description": "From DB"},
        ]
        scaffold_skill = SkillInfo(
            name="review-docs",
            description="From scaffold",
            file_path="skills/review-docs/SKILL.md",
            content="Scaffold version",
        )

        with patch("dokumen.loader.get_agent_skills", return_value=db_skills), \
             patch("dokumen.loader.SkillLoader") as MockLoader:
            MockLoader.return_value.load_skills.return_value = [scaffold_skill]

            result = _collect_skills(
                scaffold_skill_names=["review-docs"],
                base_dir="/tmp/test",
            )

        # Only one entry, from DB
        assert len(result) == 1
        assert result[0][0] == "review-docs"
        assert result[0][1] == "DB version"
        assert result[0][2] == "agent:db"

    def test_merges_db_and_scaffold_skills(self):
        """DB and scaffold skills with different names are both included."""
        from dokumen.loader import _collect_skills
        from dokumen_schema.skills import SkillInfo

        db_skills = [
            {"name": "review-docs", "content": "DB review", "description": "DB"},
        ]
        scaffold_skill = SkillInfo(
            name="summarize",
            description="Summarize",
            file_path="skills/summarize/SKILL.md",
            content="Scaffold summarize",
        )

        with patch("dokumen.loader.get_agent_skills", return_value=db_skills), \
             patch("dokumen.loader.SkillLoader") as MockLoader:
            MockLoader.return_value.load_skills.return_value = [scaffold_skill]

            result = _collect_skills(
                scaffold_skill_names=["summarize"],
                base_dir="/tmp/test",
            )

        assert len(result) == 2
        names = [r[0] for r in result]
        assert "review-docs" in names
        assert "summarize" in names


class TestSkillInjectionInLoader:
    """Integration tests: skills are injected into executor/judge system prompts."""

    def _make_scaffold_yaml(self, tmp_path, executor_skills=None, judge_skills=None):
        """Create a minimal scaffold YAML for testing."""
        import yaml

        scaffold = {
            "name": "test-with-skills",
            "reason": "Test skill injection",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "system_prompt": "You are a test executor.",
                "user_prompt": "Check the docs.",
                "tools": ["read_file"],
            },
            "judges": [
                {
                    "name": "accuracy",
                    "system_prompt": "You are a test judge.",
                },
            ],
        }
        if executor_skills:
            scaffold["executor"]["skills"] = executor_skills
        if judge_skills:
            scaffold["judges"][0]["skills"] = judge_skills

        yaml_path = tmp_path / "test-with-skills.test.yaml"
        yaml_path.write_text(yaml.dump(scaffold))

        # Create docs/test.md so file validation passes
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        (docs_dir / "test.md").write_text("# Test Doc")

        # Create dokumen.yaml so project root detection works
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'\n")

        return str(yaml_path)

    def test_executor_skills_injected_into_system_prompt(self, tmp_path):
        """Executor system prompt includes skill content when skills are configured."""
        from dokumen.loader import load_scaffold
        from dokumen_schema.skills import SkillInfo

        yaml_path = self._make_scaffold_yaml(tmp_path, executor_skills=["review-docs"])

        mock_skill = SkillInfo(
            name="review-docs",
            description="Review docs",
            file_path="skills/review-docs/SKILL.md",
            content="# Review\nCheck documentation accuracy.",
        )

        mock_provider = MagicMock()

        with patch("dokumen.loader.get_agent_skills", return_value=[]), \
             patch("dokumen.loader.SkillLoader") as MockLoader:
            MockLoader.return_value.load_skills.return_value = [mock_skill]

            test_obj = load_scaffold(
                yaml_path,
                provider=mock_provider,
                project_root=str(tmp_path),
            )

        assert "review-docs" in test_obj.executor.system_prompt
        assert "Check documentation accuracy" in test_obj.executor.system_prompt

    def test_no_skills_no_injection(self, tmp_path):
        """System prompt unchanged when no skills are configured."""
        from dokumen.loader import load_scaffold

        yaml_path = self._make_scaffold_yaml(tmp_path)

        mock_provider = MagicMock()

        with patch("dokumen.loader.get_agent_skills", return_value=[]):
            test_obj = load_scaffold(
                yaml_path,
                provider=mock_provider,
                project_root=str(tmp_path),
            )

        # Original prompt should be intact without skill section
        assert "Available Skills" not in test_obj.executor.system_prompt

    def test_judge_skills_injected_into_system_prompt(self, tmp_path):
        """Judge system prompt includes skill content when judge has skills."""
        from dokumen.loader import load_scaffold
        from dokumen_schema.skills import SkillInfo

        yaml_path = self._make_scaffold_yaml(tmp_path, judge_skills=["summarize"])

        mock_skill = SkillInfo(
            name="summarize",
            description="Summarize",
            file_path="skills/summarize/SKILL.md",
            content="# Summarize\nCreate a summary.",
        )

        mock_provider = MagicMock()

        with patch("dokumen.loader.get_agent_skills", return_value=[]), \
             patch("dokumen.loader.SkillLoader") as MockLoader:
            MockLoader.return_value.load_skills.return_value = [mock_skill]

            test_obj = load_scaffold(
                yaml_path,
                provider=mock_provider,
                project_root=str(tmp_path),
            )

        judge = test_obj.judges[0]
        assert "summarize" in judge.system_prompt
        assert "Create a summary" in judge.system_prompt

    def test_db_skills_injected_for_executor(self, tmp_path):
        """DB agent skills are injected into executor prompt when DOKUMEN_AGENT_ID is set."""
        from dokumen.loader import load_scaffold

        yaml_path = self._make_scaffold_yaml(tmp_path)

        db_skills = [
            {"name": "review-docs", "content": "# Review\nDB skill content.", "description": "Review"},
        ]

        mock_provider = MagicMock()

        with patch("dokumen.loader.get_agent_skills", return_value=db_skills):
            test_obj = load_scaffold(
                yaml_path,
                provider=mock_provider,
                project_root=str(tmp_path),
            )

        assert "review-docs" in test_obj.executor.system_prompt
        assert "DB skill content" in test_obj.executor.system_prompt
