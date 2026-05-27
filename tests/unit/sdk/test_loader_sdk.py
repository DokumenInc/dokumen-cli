"""Tests for loader.py SDK path — verifies loader always produces SDK wrappers.

After removing the legacy AgentObject execution path, the loader should
always produce SdkExecutorWrapper and SdkJudgeWrapper instances.
"""

from pathlib import Path

import pytest
import yaml

from dokumen.agent_object import AgentType


def _write_scaffold(tmp_path: Path, name: str = "test-sdk", **overrides) -> Path:
    """Write a minimal test scaffold YAML to tmp_path."""
    scaffold = {
        "name": name,
        "reason": "Test SDK integration",
        "files": [{"path": "docs/test.md"}],
        "executor": {
            "system_prompt": "You are a doc validator.",
            "user_prompt": "Check the docs.",
            "tools": ["read_file"],
        },
        "judges": [
            {
                "name": "accuracy",
                "system_prompt": 'Evaluate accuracy. Return JSON: {"verdict": "PASS", "confidence": 0.9, "reason": "..."}',
            }
        ],
    }
    scaffold.update(overrides)
    scaffold_path = tmp_path / f"{name}.test.yaml"
    scaffold_path.write_text(yaml.dump(scaffold))
    return scaffold_path


def _write_docs(tmp_path: Path) -> None:
    """Write minimal docs structure for scaffold file resolution."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "test.md").write_text("# Test Document\n\nThis is a test.")


class TestLoaderAlwaysUsesSdk:
    def test_load_scaffold_produces_sdk_executor(self, tmp_path):
        """Loader always produces SdkExecutorWrapper for executor."""
        from dokumen.loader import load_scaffold

        _write_docs(tmp_path)
        scaffold_path = _write_scaffold(tmp_path)

        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
        )

        assert test_obj.executor.__class__.__name__ == "SdkExecutorWrapper"
        assert test_obj.executor.agent_type == AgentType.EXECUTOR
        assert test_obj.executor.id == "test-sdk-executor"

    def test_load_scaffold_produces_sdk_judge(self, tmp_path):
        """Loader always produces SdkJudgeWrapper for judges."""
        from dokumen.loader import load_scaffold

        _write_docs(tmp_path)
        scaffold_path = _write_scaffold(tmp_path)

        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
        )

        assert len(test_obj.judges) == 1
        judge = test_obj.judges[0]
        assert judge.__class__.__name__ == "SdkJudgeWrapper"
        assert judge.agent_type == AgentType.JUDGE
        assert judge.id == "accuracy"

    def test_load_scaffold_preserves_test_name(self, tmp_path):
        """SDK path preserves test name from scaffold."""
        from dokumen.loader import load_scaffold

        _write_docs(tmp_path)
        scaffold_path = _write_scaffold(tmp_path, name="my-cool-test")

        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
        )

        assert test_obj.id == "my-cool-test"

    def test_load_scaffold_multiple_judges(self, tmp_path):
        """SDK path handles multiple judges."""
        from dokumen.loader import load_scaffold

        _write_docs(tmp_path)
        scaffold_path = _write_scaffold(
            tmp_path,
            judges=[
                {
                    "name": "accuracy",
                    "system_prompt": "Evaluate accuracy. Return JSON.",
                },
                {
                    "name": "completeness",
                    "system_prompt": "Evaluate completeness. Return JSON.",
                },
            ],
        )

        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
        )

        assert len(test_obj.judges) == 2
        assert test_obj.judges[0].id == "accuracy"
        assert test_obj.judges[1].id == "completeness"
        for judge in test_obj.judges:
            assert judge.__class__.__name__ == "SdkJudgeWrapper"

    def test_load_scaffold_with_multiple_tools(self, tmp_path):
        """SDK path resolves multiple executor tools."""
        from dokumen.loader import load_scaffold

        _write_docs(tmp_path)
        scaffold_path = _write_scaffold(
            tmp_path,
            executor={
                "system_prompt": "You validate docs.",
                "user_prompt": "Check everything.",
                "tools": ["read_file", "glob", "search_file_content"],
            },
        )

        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
        )

        assert test_obj.executor.__class__.__name__ == "SdkExecutorWrapper"

    def test_load_scaffold_with_shell_tool(self, tmp_path):
        """SDK path resolves run_shell_command to Bash."""
        from dokumen.loader import load_scaffold

        _write_docs(tmp_path)
        scaffold_path = _write_scaffold(
            tmp_path,
            executor={
                "system_prompt": "You validate docs.",
                "user_prompt": "Run commands.",
                "tools": ["read_file", "run_shell_command"],
            },
        )

        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
        )

        assert test_obj.executor.__class__.__name__ == "SdkExecutorWrapper"

    def test_experimental_config_param_ignored(self, tmp_path):
        """experimental_config parameter is accepted but ignored (backward compat)."""
        from dokumen.loader import load_scaffold

        _write_docs(tmp_path)
        scaffold_path = _write_scaffold(tmp_path)

        # Passing experimental_config should not cause errors
        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
            experimental_config=None,
        )

        assert test_obj.executor.__class__.__name__ == "SdkExecutorWrapper"


class TestNoExperimentalConfig:
    def test_config_no_experimental_field(self):
        """DokumenConfig no longer has experimental field."""
        from dokumen.config import DokumenConfig

        config = DokumenConfig(provider={"name": "anthropic", "model": "test"})
        assert not hasattr(config, 'experimental')
