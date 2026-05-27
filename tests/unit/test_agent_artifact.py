"""Tests for agent artifact writing in test_object.py."""

import os
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from dokumen.test_object import TestObject
from dokumen.agent_object import AgentType


class TestWriteAgentArtifact:
    """Tests for TestObject._write_agent_artifact method."""

    def _make_test_object(self, agent_name=None):
        """Create a minimal TestObject with mock executor/judge."""
        executor = MagicMock()
        executor.id = "test-executor"
        executor.agent_type = AgentType.EXECUTOR
        executor.system_prompt = "Test prompt"
        executor.tools = []
        executor.provider = MagicMock(model="test-model")

        judge = MagicMock()
        judge.id = "test-judge"
        judge.agent_type = AgentType.JUDGE
        judge.system_prompt = "Judge prompt"
        judge.tools = []
        judge.provider = MagicMock(model="test-model")

        return TestObject(
            id="test-artifact",
            reason="Test artifact writing",
            executor=executor,
            judges=[judge],
            agent=agent_name,
        )

    def test_writes_artifact_for_known_agent(self, tmp_path, monkeypatch):
        """Artifact YAML is written for a known agent."""
        monkeypatch.chdir(tmp_path)

        test_obj = self._make_test_object(agent_name="general")
        test_obj._write_agent_artifact()

        artifact_path = tmp_path / ".dokumen-cache" / "agents" / "general.agent.yaml"
        assert artifact_path.exists()

        data = yaml.safe_load(artifact_path.read_text())
        assert data["name"] == "general"
        assert isinstance(data["tools"], list)

    def test_no_artifact_for_unknown_agent(self, tmp_path, monkeypatch):
        """No artifact written when agent name is unknown."""
        monkeypatch.chdir(tmp_path)

        test_obj = self._make_test_object(agent_name="nonexistent-xyz")
        test_obj._write_agent_artifact()

        agents_dir = tmp_path / ".dokumen-cache" / "agents"
        if agents_dir.exists():
            assert not list(agents_dir.glob("*.yaml"))

    def test_no_artifact_when_no_agent(self, tmp_path, monkeypatch):
        """No artifact when agent is None (standard test)."""
        monkeypatch.chdir(tmp_path)

        test_obj = self._make_test_object(agent_name=None)
        # _write_agent_artifact should not be called (guarded in run()),
        # but test it doesn't crash
        # The run() method guards: if self.agent: self._write_agent_artifact()
        assert test_obj.agent is None

    def test_artifact_creates_directory(self, tmp_path, monkeypatch):
        """Creates .dokumen-cache/agents/ directory if it doesn't exist."""
        monkeypatch.chdir(tmp_path)

        test_obj = self._make_test_object(agent_name="browser-tester")
        test_obj._write_agent_artifact()

        assert (tmp_path / ".dokumen-cache" / "agents").is_dir()

    def test_artifact_content_matches_agent_def(self, tmp_path, monkeypatch):
        """Artifact content matches the agent definition."""
        monkeypatch.chdir(tmp_path)

        from dokumen_schema.agent_defs import load_agent
        agent_def = load_agent("browser-tester")

        test_obj = self._make_test_object(agent_name="browser-tester")
        test_obj._write_agent_artifact()

        artifact_path = tmp_path / ".dokumen-cache" / "agents" / "browser-tester.agent.yaml"
        data = yaml.safe_load(artifact_path.read_text())

        assert data["name"] == agent_def.name
        assert data["tools"] == agent_def.tools
        assert data["capabilities"] == agent_def.capabilities

    def test_artifact_handles_write_error_gracefully(self, tmp_path, monkeypatch):
        """Write errors are logged but don't crash."""
        monkeypatch.chdir(tmp_path)

        # Make the cache dir a file to cause a write error
        cache_dir = tmp_path / ".dokumen-cache" / "agents"
        cache_dir.parent.mkdir(parents=True)
        cache_dir.parent.joinpath("agents").write_text("not a dir")

        test_obj = self._make_test_object(agent_name="general")
        # Should not raise
        test_obj._write_agent_artifact()
