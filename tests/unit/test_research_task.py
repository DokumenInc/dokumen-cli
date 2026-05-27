"""Tests for research task features: auto-injection, report extraction, artifacts."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestResearchWebSearchAutoInjection:
    """Tests for web_search auto-injection in research tests."""

    def _write_research_scaffold(self, tmp_path, tools=None, use_researcher_agent=True):
        """Helper to create a minimal research test scaffold YAML."""
        scaffold = {
            "name": "test-research",
            "reason": "Test research auto-injection",
            "executor": {
                "user_prompt": "Research something",
            },
            "judges": [
                {
                    "name": "accuracy",
                    "system_prompt": "Judge accuracy.\n\nReturn JSON: {\"verdict\": \"PASS\", \"confidence\": 0.9, \"reason\": \"ok\"}",
                }
            ],
        }
        if use_researcher_agent:
            scaffold["executor"]["agent"] = "researcher"
        if tools is not None:
            scaffold["executor"]["tools"] = tools

        yaml_path = tmp_path / "test-research.test.yaml"
        yaml_path.write_text(yaml.dump(scaffold))
        return str(yaml_path)

    @patch("dokumen.loader.load_executor_prompt")
    @patch("dokumen.scaffold.validate_scaffold")
    def test_web_search_auto_injected_for_research(self, mock_validate, mock_load_prompt, tmp_path):
        """web_search is auto-injected for researcher agent tests."""
        from dokumen.loader import load_scaffold

        mock_validate.return_value = MagicMock(valid=True, errors=[])
        mock_load_prompt.return_value = "Research prompt"

        yaml_path = self._write_research_scaffold(tmp_path, tools=["read_file"])
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        tool_names = [t.name for t in test_obj.executor.tools]
        # SDK maps web_search to WebSearch
        assert "WebSearch" in tool_names or "web_search" in tool_names

    @patch("dokumen.loader.load_executor_prompt")
    @patch("dokumen.scaffold.validate_scaffold")
    def test_web_search_not_duplicated_if_already_present(self, mock_validate, mock_load_prompt, tmp_path):
        """web_search is NOT duplicated if already in tools list."""
        from dokumen.loader import load_scaffold

        mock_validate.return_value = MagicMock(valid=True, errors=[])
        mock_load_prompt.return_value = "Research prompt"

        yaml_path = self._write_research_scaffold(tmp_path, tools=["read_file", "web_search"])
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        tool_names = [t.name for t in test_obj.executor.tools]
        # SDK maps web_search to WebSearch
        assert tool_names.count("WebSearch") == 1 or tool_names.count("web_search") == 1

    @patch("dokumen.loader.load_executor_prompt")
    @patch("dokumen.scaffold.validate_scaffold")
    def test_web_search_not_injected_for_non_research(self, mock_validate, mock_load_prompt, tmp_path):
        """web_search is NOT auto-injected for non-researcher agent tests."""
        from dokumen.loader import load_scaffold

        mock_validate.return_value = MagicMock(valid=True, errors=[])
        mock_load_prompt.return_value = "Standard prompt"

        # Standard test with doc-validator agent (not researcher)
        scaffold = {
            "name": "test-standard",
            "reason": "Standard test",
            "executor": {
                "agent": "doc-validator",
                "user_prompt": "Validate something",
                "tools": ["read_file"],
            },
            "judges": [
                {
                    "name": "accuracy",
                    "system_prompt": "Judge.\n\nReturn JSON: {\"verdict\": \"PASS\", \"confidence\": 0.9, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = tmp_path / "test-standard.test.yaml"
        yaml_path.write_text(yaml.dump(scaffold))

        test_obj = load_scaffold(str(yaml_path), project_root=str(tmp_path))
        tool_names = [t.name for t in test_obj.executor.tools]
        assert "web_search" not in tool_names


class TestResearchJudgeAutoInjection:
    """Tests for automatic sources + verdict judge injection."""

    def _write_research_scaffold(self, tmp_path, judges=None):
        """Helper to create a research scaffold with configurable judges."""
        scaffold = {
            "name": "test-research-judges",
            "reason": "Test judge auto-injection",
            "executor": {
                "agent": "researcher",
                "user_prompt": "Research something",
                "tools": ["web_search"],
            },
            "judges": judges or [
                {
                    "name": "accuracy",
                    "system_prompt": "Judge accuracy.\n\nReturn JSON: {\"verdict\": \"PASS\", \"confidence\": 0.9, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = tmp_path / "test-research-judges.test.yaml"
        yaml_path.write_text(yaml.dump(scaffold))
        return str(yaml_path)

    @patch("dokumen.loader.load_executor_prompt")
    @patch("dokumen.scaffold.validate_scaffold")
    def test_sources_and_verdict_judges_auto_injected(self, mock_validate, mock_load_prompt, tmp_path):
        """sources and verdict judges are auto-injected for research tests."""
        from dokumen.loader import load_scaffold

        mock_validate.return_value = MagicMock(valid=True, errors=[])
        mock_load_prompt.return_value = "Research prompt"

        yaml_path = self._write_research_scaffold(tmp_path)
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        judge_ids = [j.id for j in test_obj.judges]
        assert "accuracy" in judge_ids  # Custom judge kept
        assert "sources" in judge_ids  # Auto-injected
        assert "verdict" in judge_ids  # Auto-injected

    @patch("dokumen.loader.load_executor_prompt")
    @patch("dokumen.scaffold.validate_scaffold")
    def test_custom_judges_kept_alongside_auto_injected(self, mock_validate, mock_load_prompt, tmp_path):
        """Custom judges appear before auto-injected judges."""
        from dokumen.loader import load_scaffold

        mock_validate.return_value = MagicMock(valid=True, errors=[])
        mock_load_prompt.return_value = "Research prompt"

        judges = [
            {
                "name": "accuracy",
                "system_prompt": "Judge accuracy.\n\nReturn JSON: {\"verdict\": \"PASS\", \"confidence\": 0.9, \"reason\": \"ok\"}",
            },
            {
                "name": "completeness",
                "system_prompt": "Judge completeness.\n\nReturn JSON: {\"verdict\": \"PASS\", \"confidence\": 0.8, \"reason\": \"ok\"}",
            },
        ]
        yaml_path = self._write_research_scaffold(tmp_path, judges=judges)
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        judge_ids = [j.id for j in test_obj.judges]
        # Custom judges first, then auto-injected
        assert judge_ids.index("accuracy") < judge_ids.index("sources")
        assert judge_ids.index("completeness") < judge_ids.index("sources")
        assert judge_ids.index("sources") < judge_ids.index("verdict")

    @patch("dokumen.loader.load_executor_prompt")
    @patch("dokumen.scaffold.validate_scaffold")
    def test_no_duplicate_if_sources_already_defined(self, mock_validate, mock_load_prompt, tmp_path):
        """If user already defined a 'sources' judge, no duplicate is injected."""
        from dokumen.loader import load_scaffold

        mock_validate.return_value = MagicMock(valid=True, errors=[])
        mock_load_prompt.return_value = "Research prompt"

        judges = [
            {
                "name": "sources",
                "system_prompt": "My custom sources judge.\n\nReturn JSON: {\"verdict\": \"PASS\", \"confidence\": 0.9, \"reason\": \"ok\"}",
            },
        ]
        yaml_path = self._write_research_scaffold(tmp_path, judges=judges)
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        judge_ids = [j.id for j in test_obj.judges]
        assert judge_ids.count("sources") == 1
        # verdict should still be auto-injected
        assert "verdict" in judge_ids

    @patch("dokumen.loader.load_executor_prompt")
    @patch("dokumen.scaffold.validate_scaffold")
    def test_no_duplicate_if_verdict_already_defined(self, mock_validate, mock_load_prompt, tmp_path):
        """If user already defined a 'verdict' judge, no duplicate is injected."""
        from dokumen.loader import load_scaffold

        mock_validate.return_value = MagicMock(valid=True, errors=[])
        mock_load_prompt.return_value = "Research prompt"

        judges = [
            {
                "name": "verdict",
                "system_prompt": "My custom verdict judge.\n\nReturn JSON: {\"verdict\": \"PASS\", \"confidence\": 0.9, \"reason\": \"ok\"}",
            },
        ]
        yaml_path = self._write_research_scaffold(tmp_path, judges=judges)
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        judge_ids = [j.id for j in test_obj.judges]
        assert judge_ids.count("verdict") == 1
        # sources should still be auto-injected
        assert "sources" in judge_ids


class TestTestObjectExecutorAgent:
    """Tests for executor.agent field on TestObject."""

    @patch("dokumen.loader.load_executor_prompt")
    @patch("dokumen.scaffold.validate_scaffold")
    def test_researcher_agent_set_on_test_object(self, mock_validate, mock_load_prompt, tmp_path):
        """executor.agent from scaffold is available on TestObject."""
        from dokumen.loader import load_scaffold

        mock_validate.return_value = MagicMock(valid=True, errors=[])
        mock_load_prompt.return_value = "Research prompt"

        scaffold = {
            "name": "test-research-type",
            "reason": "Test agent passing",
            "executor": {
                "agent": "researcher",
                "user_prompt": "Research something",
                "tools": ["web_search"],
            },
            "judges": [
                {
                    "name": "accuracy",
                    "system_prompt": "Judge.\n\nReturn JSON: {\"verdict\": \"PASS\", \"confidence\": 0.9, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = tmp_path / "test-research-type.test.yaml"
        yaml_path.write_text(yaml.dump(scaffold))

        test_obj = load_scaffold(str(yaml_path), project_root=str(tmp_path))
        assert test_obj.executor.agent == "researcher"

    @patch("dokumen.loader.load_executor_prompt")
    @patch("dokumen.scaffold.validate_scaffold")
    def test_doc_validator_agent_for_standard_tests(self, mock_validate, mock_load_prompt, tmp_path):
        """doc-validator agent is set for standard tests."""
        from dokumen.loader import load_scaffold

        mock_validate.return_value = MagicMock(valid=True, errors=[])
        mock_load_prompt.return_value = "Standard prompt"

        scaffold = {
            "name": "test-standard",
            "reason": "Standard test",
            "executor": {
                "agent": "doc-validator",
                "user_prompt": "Validate something",
                "tools": ["read_file"],
            },
            "judges": [
                {
                    "name": "accuracy",
                    "system_prompt": "Judge.\n\nReturn JSON: {\"verdict\": \"PASS\", \"confidence\": 0.9, \"reason\": \"ok\"}",
                }
            ],
        }
        yaml_path = tmp_path / "test-standard.test.yaml"
        yaml_path.write_text(yaml.dump(scaffold))

        test_obj = load_scaffold(str(yaml_path), project_root=str(tmp_path))
        assert test_obj.executor.agent == "doc-validator"


class TestExtractReportMarkdown:
    """Tests for _extract_report_markdown helper."""

    def test_extract_with_json_code_fence(self):
        """Extracts markdown before a ```json verdict block."""
        from dokumen.test_object import _extract_report_markdown

        response = """### Executive Summary
Great research with solid sources.

### Key Findings
- Finding 1
- Finding 2

```json
{"verdict": "PASS", "confidence": 0.9, "reason": "Good research"}
```"""

        report = _extract_report_markdown(response)
        assert "### Executive Summary" in report
        assert "### Key Findings" in report
        assert "Finding 1" in report
        assert '"verdict"' not in report

    def test_extract_without_json_block(self):
        """Returns full response if no JSON verdict block found."""
        from dokumen.test_object import _extract_report_markdown

        response = """### Executive Summary
Great research, no JSON block here.

### Key Findings
- Finding 1"""

        report = _extract_report_markdown(response)
        assert report == response

    def test_extract_with_inline_json(self):
        """Extracts markdown before inline JSON at end of response."""
        from dokumen.test_object import _extract_report_markdown

        response = """### Executive Summary
Good research overall.

{"verdict": "PASS", "confidence": 0.85, "reason": "Solid work"}"""

        report = _extract_report_markdown(response)
        assert "### Executive Summary" in report
        assert '"verdict"' not in report

    def test_extract_empty_response(self):
        """Returns empty string for empty response."""
        from dokumen.test_object import _extract_report_markdown

        assert _extract_report_markdown("") == ""
        assert _extract_report_markdown(None) == ""


class TestReportArtifactSchema:
    """Tests for ReportArtifact output schema."""

    def test_report_artifact_model_validation(self):
        """ReportArtifact pydantic model validates correctly."""
        from dokumen.output_schemas import ReportArtifact

        artifact = ReportArtifact(
            type="report",
            path="reports/test-id/report.md",
            filename="report.md",
            size_bytes=1234,
            content="# Report\nSome content",
        )
        assert artifact.type == "report"
        assert artifact.path == "reports/test-id/report.md"
        assert artifact.filename == "report.md"
        assert artifact.size_bytes == 1234
        assert artifact.content == "# Report\nSome content"

    def test_report_artifact_optional_fields(self):
        """ReportArtifact works with minimal fields."""
        from dokumen.output_schemas import ReportArtifact

        artifact = ReportArtifact(
            path="reports/test-id/report.md",
            filename="report.md",
        )
        assert artifact.type == "report"
        assert artifact.size_bytes is None
        assert artifact.content is None

    def test_test_output_result_has_report_artifacts(self):
        """TestOutputResult has report_artifacts field."""
        from dokumen.output_schemas import TestOutputResult

        result = TestOutputResult(
            name="test-research",
            status="passed",
            duration_ms=5000,
            report_artifacts=None,
        )
        assert result.report_artifacts is None


class TestOutputWriterReportArtifacts:
    """Tests for report artifacts serialization in output writer."""

    def test_output_writer_includes_report_artifacts(self, tmp_path):
        """results.json includes report_artifacts when present."""
        from dokumen.cli.output import OutputWriter

        # Create mock results
        mock_results = MagicMock()
        mock_results.duration = 5.0
        mock_results.total_tests = 1
        mock_results.passed = 1
        mock_results.failed = 0

        # Mock test result with report_artifacts
        mock_tr = MagicMock()
        mock_tr.test_id = "test-research"
        mock_tr.passed = True
        mock_tr.duration = 5.0
        mock_tr.failure_reasons = []
        mock_tr.executor_output = None
        mock_tr.explore_output = None
        mock_tr.explore_tool_calls = None
        mock_tr.executor_input_tokens = 100
        mock_tr.executor_output_tokens = 200
        mock_tr.judge_input_tokens = 50
        mock_tr.judge_output_tokens = 100
        mock_tr.explore_input_tokens = 0
        mock_tr.explore_output_tokens = 0
        mock_tr.executor_model = "claude-sonnet"
        mock_tr.judge_model = "claude-sonnet"
        mock_tr.explore_model = None
        mock_tr.judge_prompts = None
        mock_tr.browser_artifacts = None
        mock_tr.executor_tools = ["web_search"]
        mock_tr.files = []
        mock_tr.judge_results = []
        mock_tr.report_artifacts = [
            {
                "type": "report",
                "path": "reports/test-research/report.md",
                "filename": "report.md",
                "size_bytes": 500,
                "content": "# Research Report\nFindings here.",
            }
        ]

        mock_results.test_results = [mock_tr]

        writer = OutputWriter(cache_dir=str(tmp_path))
        output_path = writer.write_results_json(mock_results, coverage_stats={})

        with open(output_path) as f:
            data = json.load(f)

        test = data["tests"][0]
        assert test["report_artifacts"] is not None
        assert len(test["report_artifacts"]) == 1
        assert test["report_artifacts"][0]["type"] == "report"
        assert test["report_artifacts"][0]["content"] == "# Research Report\nFindings here."

    def test_output_writer_no_report_artifacts_when_absent(self, tmp_path):
        """results.json has null report_artifacts when not present."""
        from dokumen.cli.output import OutputWriter

        mock_results = MagicMock()
        mock_results.duration = 5.0
        mock_results.total_tests = 1
        mock_results.passed = 1
        mock_results.failed = 0

        mock_tr = MagicMock()
        mock_tr.test_id = "test-standard"
        mock_tr.passed = True
        mock_tr.duration = 5.0
        mock_tr.failure_reasons = []
        mock_tr.executor_output = None
        mock_tr.explore_output = None
        mock_tr.explore_tool_calls = None
        mock_tr.executor_input_tokens = 0
        mock_tr.executor_output_tokens = 0
        mock_tr.judge_input_tokens = 0
        mock_tr.judge_output_tokens = 0
        mock_tr.explore_input_tokens = 0
        mock_tr.explore_output_tokens = 0
        mock_tr.executor_model = None
        mock_tr.judge_model = None
        mock_tr.explore_model = None
        mock_tr.judge_prompts = None
        mock_tr.browser_artifacts = None
        mock_tr.executor_tools = []
        mock_tr.files = []
        mock_tr.judge_results = []
        mock_tr.report_artifacts = None

        mock_results.test_results = [mock_tr]

        writer = OutputWriter(cache_dir=str(tmp_path))
        output_path = writer.write_results_json(mock_results, coverage_stats={})

        with open(output_path) as f:
            data = json.load(f)

        test = data["tests"][0]
        assert test["report_artifacts"] is None


class TestTestResultReportArtifacts:
    """Tests for report_artifacts on TestResult dataclass."""

    def test_report_artifacts_field_exists(self):
        """TestResult has report_artifacts field."""
        from dokumen.test_object import TestResult
        from datetime import datetime

        result = TestResult(
            test_id="test-research",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=5.0,
            timestamp=datetime.now(),
        )
        assert result.report_artifacts is None

    def test_report_artifacts_in_to_dict(self):
        """TestResult.to_dict() includes report_artifacts."""
        from dokumen.test_object import TestResult
        from datetime import datetime

        result = TestResult(
            test_id="test-research",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=5.0,
            timestamp=datetime.now(),
            report_artifacts=[
                {
                    "type": "report",
                    "path": "reports/test-research/report.md",
                    "filename": "report.md",
                    "size_bytes": 500,
                    "content": "# Report",
                }
            ],
        )
        d = result.to_dict()
        assert "report_artifacts" in d
        assert d["report_artifacts"][0]["type"] == "report"
