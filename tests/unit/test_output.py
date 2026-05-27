"""
Tests for output module - Phase 5.

TDD tests for writing .dokumen-cache/ output files:
- results.json (spec format)
- junit.xml (GitLab CI)
- coverage.json (spec format)
- debug/{test-name}-{timestamp}.json
"""
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
import xml.etree.ElementTree as ET

import pytest

from dokumen.cli.output import OutputWriter


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_suite_results():
    """Factory for TestSuiteResults mock."""
    def _make(total=1, passed=1, failed=0, error=0, skipped=0, test_results=None):
        mock = MagicMock()
        mock.total_tests = total
        mock.passed = passed
        mock.failed = failed
        mock.error = error
        mock.skipped = skipped
        mock.duration = 45.0
        mock.test_results = test_results or []
        mock.cached_results = 0
        return mock
    return _make


@pytest.fixture
def mock_test_result():
    """Factory for TestResult mock with judge results."""
    def _make(test_id="test-1", passed=True, duration=15.0, judge_results=None, failure_reasons=None):
        mock = MagicMock()
        mock.test_id = test_id
        mock.passed = passed
        mock.duration = duration
        mock.failure_reasons = failure_reasons or ([] if passed else ["Assertion failed"])
        mock.executor_output = MagicMock()
        mock.executor_output.final_response = "Executor output text"
        mock.executor_output.tool_calls = []
        # Per PHASE0-CLI-SPEC: ExecutorOutput includes prompts for debug traces
        mock.executor_output.system_prompt = ""
        mock.executor_output.user_prompt = ""
        # original_user_prompt is the prompt before explore context injection
        mock.executor_output.original_user_prompt = ""
        mock.judge_results = judge_results or []
        mock.timestamp = datetime.now()
        mock.files = []
        mock.explore_output = None
        mock.explore_status = None
        mock.explore_tool_calls = None
        mock.executor_model = "claude-sonnet-4-5-20250929"
        mock.judge_model = "claude-sonnet-4-5-20250929"
        mock.explore_model = None
        return mock
    return _make


@pytest.fixture
def mock_judge_result():
    """Factory for JudgeResult mock."""
    def _make(judge_id="accuracy", passed=True, reasoning="Correctly evaluated", assertion_text=None, reason=None):
        mock = MagicMock()
        mock.judge_id = judge_id
        mock.passed = passed
        mock.failure_reason = None if passed else reasoning
        mock.reason = reason or reasoning
        mock.response = f"Full response text.\n{reasoning}"
        mock.confidence = 0.95
        # Per PHASE0-CLI-SPEC: assertion should be the question text, not judge_id
        mock.assertion_text = assertion_text or f"Evaluate: {judge_id}"
        return mock
    return _make


@pytest.fixture
def output_writer(tmp_path):
    """OutputWriter with temp cache directory."""
    cache_dir = tmp_path / ".dokumen-cache"
    return OutputWriter(cache_dir=str(cache_dir))


@pytest.fixture
def mock_coverage_stats():
    """Factory for coverage stats dict."""
    def _make(total=10, passed=7, failed=1, files_detail=None):
        detail = files_detail or {
            "docs/a.md": {"status": "passed", "test_ids": ["test-1"]},
            "docs/b.md": {"status": "failed", "test_ids": ["test-2"]},
            "docs/c.md": {"status": "uncovered", "test_ids": []}
        }
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "percentage": (passed / total * 100) if total > 0 else 0.0,
            "covered_files": [f for f, d in detail.items() if d["status"] == "passed"],
            "failed_files": [f for f, d in detail.items() if d["status"] == "failed"],
            "uncovered_files": [f for f, d in detail.items() if d["status"] == "uncovered"],
            "files_detail": detail
        }
    return _make


# =============================================================================
# Results JSON Tests
# =============================================================================

class TestResultsJsonWriter:
    """Tests for results.json output."""

    def test_writes_to_dokumen_cache_directory(self, output_writer, mock_suite_results):
        """Results file is written to .dokumen-cache/results.json."""
        results = mock_suite_results(total=1, passed=1, test_results=[])

        path = output_writer.write_results_json(results)

        assert path.exists()
        assert path.name == "results.json"
        assert ".dokumen-cache" in str(path)

    def test_creates_directory_if_not_exists(self, tmp_path, mock_suite_results):
        """Creates .dokumen-cache directory if missing."""
        cache_dir = tmp_path / "nonexistent" / ".dokumen-cache"
        writer = OutputWriter(cache_dir=str(cache_dir))
        results = mock_suite_results(total=0, passed=0, test_results=[])

        path = writer.write_results_json(results)

        assert cache_dir.exists()
        assert path.exists()

    def test_includes_timestamp_in_iso_format(self, output_writer, mock_suite_results):
        """Timestamp is ISO 8601 format with Z suffix."""
        results = mock_suite_results(total=0, passed=0, test_results=[])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert "timestamp" in data
        assert data["timestamp"].endswith("Z")
        # Should parse as ISO datetime
        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

    def test_duration_is_in_milliseconds(self, output_writer, mock_suite_results):
        """Duration is integer milliseconds."""
        results = mock_suite_results(total=0, passed=0, test_results=[])
        results.duration = 45.5  # 45.5 seconds

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert data["duration_ms"] == 45500
        assert isinstance(data["duration_ms"], int)

    def test_test_status_is_passed_or_failed_string(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Test status is 'passed' or 'failed' string, not boolean."""
        test_pass = mock_test_result("test-1", passed=True)
        test_fail = mock_test_result("test-2", passed=False)
        results = mock_suite_results(
            total=2, passed=1, failed=1,
            test_results=[test_pass, test_fail]
        )

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert data["tests"][0]["status"] == "passed"
        assert data["tests"][1]["status"] == "failed"

    def test_test_duration_is_in_milliseconds(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Individual test duration is in milliseconds."""
        test = mock_test_result("test-1", passed=True, duration=15.75)
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert data["tests"][0]["duration_ms"] == 15750
        assert isinstance(data["tests"][0]["duration_ms"], int)

    def test_includes_assertions_with_reasoning(
        self, output_writer, mock_suite_results, mock_test_result, mock_judge_result
    ):
        """Tests include assertions array with reasoning."""
        judge = mock_judge_result("accuracy", passed=True, reasoning="Correctly identified")
        test = mock_test_result("test-1", passed=True, judge_results=[judge])
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert len(data["tests"][0]["assertions"]) == 1
        assertion = data["tests"][0]["assertions"][0]
        assert "assertion" in assertion
        assert "passed" in assertion
        assert "reasoning" in assertion
        assert assertion["passed"] is True

    def test_failed_assertion_includes_reasoning(
        self, output_writer, mock_suite_results, mock_test_result, mock_judge_result
    ):
        """Failed assertions include the failure reasoning."""
        judge = mock_judge_result("accuracy", passed=False, reasoning="Did not match expected output")
        test = mock_test_result("test-1", passed=False, judge_results=[judge])
        results = mock_suite_results(total=1, passed=0, failed=1, test_results=[test])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assertion = data["tests"][0]["assertions"][0]
        assert assertion["passed"] is False
        assert "Did not match expected output" in assertion["reasoning"]

    def test_assertion_uses_text_not_judge_id(
        self, output_writer, mock_suite_results, mock_test_result, mock_judge_result
    ):
        """Assertion field uses assertion_text, not judge_id.

        Per PHASE0-CLI-SPEC: assertion should be the question like
        "Did the executor approve the laptop refund?", not "accuracy".
        """
        judge = mock_judge_result(
            judge_id="accuracy",
            passed=True,
            assertion_text="Did the executor correctly validate the refund policy?"
        )
        test = mock_test_result("test-1", passed=True, judge_results=[judge])
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assertion = data["tests"][0]["assertions"][0]
        # Should use assertion_text, not judge_id
        assert assertion["assertion"] == "Did the executor correctly validate the refund policy?"
        assert "accuracy" not in assertion["assertion"]

    def test_assertion_uses_parsed_reason_not_full_response(
        self, output_writer, mock_suite_results, mock_test_result, mock_judge_result
    ):
        """Assertion reasoning uses the parsed verdict reason, not the full response."""
        judge = mock_judge_result(
            "completeness",
            passed=True,
            reason="Sites scanned: 15/15. All requirements met.",
        )
        judge.response = "I checked everything.\n```json\n{\"verdict\": \"PASS\", \"confidence\": 0.95, \"reason\": \"Sites scanned: 15/15. All requirements met.\"}\n```"
        test = mock_test_result("test-1", passed=True, judge_results=[judge])
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assertion = data["tests"][0]["assertions"][0]
        assert assertion["reasoning"] == "Sites scanned: 15/15. All requirements met."
        assert assertion["confidence"] == 0.95
        assert "```json" not in assertion["reasoning"]

    def test_summary_matches_spec_schema(self, output_writer, mock_suite_results):
        """Summary has exact spec fields: total, passed, failed, skipped, error."""
        results = mock_suite_results(total=3, passed=2, failed=1, skipped=0, test_results=[])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert data["summary"] == {
            "total": 3,
            "passed": 2,
            "failed": 1,
            "skipped": 0,
            "error": 0
        }

    def test_handles_empty_test_results(self, output_writer, mock_suite_results):
        """Handles empty test results list gracefully."""
        results = mock_suite_results(total=0, passed=0, failed=0, test_results=[])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert data["tests"] == []
        assert data["summary"]["total"] == 0

    def test_includes_executor_output(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Results include executor_output field with final response."""
        test = mock_test_result("test-1", passed=True)
        test.executor_output.final_response = "Based on the documentation, the refund policy allows returns within 30 days."
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert "executor_output" in data["tests"][0]
        assert data["tests"][0]["executor_output"] == "Based on the documentation, the refund policy allows returns within 30 days."

    def test_executor_output_is_null_when_missing(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """executor_output is null when no executor output available."""
        test = mock_test_result("test-1", passed=True)
        test.executor_output = None  # No executor output
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert "executor_output" in data["tests"][0]
        assert data["tests"][0]["executor_output"] is None

    def test_executor_output_is_null_when_empty_string(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """executor_output is null when final_response is empty string."""
        test = mock_test_result("test-1", passed=True)
        test.executor_output.final_response = ""  # Empty response
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert data["tests"][0]["executor_output"] is None

    def test_explore_status_included_in_results(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """explore_status is included in results.json when set."""
        test = mock_test_result("test-1", passed=True)
        test.explore_status = "pass"
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert data["tests"][0]["explore_status"] == "pass"

    def test_explore_status_null_when_not_set(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """explore_status is null when explore phase didn't run."""
        test = mock_test_result("test-1", passed=True)
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert data["tests"][0]["explore_status"] is None


# =============================================================================
# JUnit XML Tests
# =============================================================================

class TestJunitXmlWriter:
    """Tests for junit.xml output."""

    def test_writes_junit_xml_to_cache(self, output_writer, mock_suite_results):
        """JUnit XML is written to .dokumen-cache/junit.xml."""
        results = mock_suite_results(total=0, passed=0, test_results=[])

        path = output_writer.write_junit_xml(results)

        assert path.exists()
        assert path.name == "junit.xml"

    def test_valid_xml_structure(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Output is valid XML with testsuite/testcase structure."""
        test = mock_test_result("test-1", passed=True)
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_junit_xml(results)

        tree = ET.parse(output_writer.cache_dir / "junit.xml")
        root = tree.getroot()

        assert root.tag == "testsuite"
        assert root.find("testcase") is not None

    def test_testsuite_has_required_attributes(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Testsuite has name, tests, failures, time attributes."""
        test = mock_test_result("test-1", passed=True)
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_junit_xml(results)

        tree = ET.parse(output_writer.cache_dir / "junit.xml")
        root = tree.getroot()

        assert "name" in root.attrib
        assert "tests" in root.attrib
        assert "failures" in root.attrib
        assert "time" in root.attrib

    def test_testcase_has_name_and_time(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Each testcase has name and time attributes."""
        test = mock_test_result("my-test", passed=True, duration=12.5)
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_junit_xml(results)

        tree = ET.parse(output_writer.cache_dir / "junit.xml")
        testcase = tree.find(".//testcase")

        assert testcase.attrib["name"] == "my-test"
        assert float(testcase.attrib["time"]) == 12.5

    def test_includes_failure_messages(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Failed tests include failure element with message."""
        test = mock_test_result("test-1", passed=False, failure_reasons=["Assertion X failed"])
        results = mock_suite_results(total=1, passed=0, failed=1, test_results=[test])

        output_writer.write_junit_xml(results)

        tree = ET.parse(output_writer.cache_dir / "junit.xml")
        failure = tree.find(".//failure")

        assert failure is not None
        assert "message" in failure.attrib
        assert "Assertion X failed" in failure.attrib["message"]


# =============================================================================
# Coverage JSON Tests
# =============================================================================

class TestCoverageJsonWriter:
    """Tests for coverage.json output."""

    def test_writes_coverage_json_to_cache(self, output_writer, mock_coverage_stats):
        """Coverage JSON is written to .dokumen-cache/coverage.json."""
        coverage_stats = mock_coverage_stats()

        path = output_writer.write_coverage_json(coverage_stats)

        assert path.exists()
        assert path.name == "coverage.json"

    def test_includes_timestamp(self, output_writer, mock_coverage_stats):
        """Coverage JSON includes timestamp."""
        coverage_stats = mock_coverage_stats()

        output_writer.write_coverage_json(coverage_stats)

        with open(output_writer.cache_dir / "coverage.json") as f:
            data = json.load(f)

        assert "timestamp" in data
        assert data["timestamp"].endswith("Z")

    def test_summary_has_total_covered_percentage(self, output_writer, mock_coverage_stats):
        """Summary has total_files, covered_files, percentage."""
        coverage_stats = mock_coverage_stats(total=10, passed=7)

        output_writer.write_coverage_json(coverage_stats)

        with open(output_writer.cache_dir / "coverage.json") as f:
            data = json.load(f)

        assert data["summary"]["total_files"] == 10
        assert data["summary"]["covered_files"] == 7
        assert data["summary"]["percentage"] == 70.0

    def test_files_array_has_path_covered_tests(self, output_writer, mock_coverage_stats):
        """Files array includes path, covered boolean, tests array."""
        files_detail = {
            "docs/a.md": {"status": "passed", "test_ids": ["test-1", "test-2"]},
            "docs/b.md": {"status": "uncovered", "test_ids": []}
        }
        coverage_stats = mock_coverage_stats(total=2, passed=1, files_detail=files_detail)

        output_writer.write_coverage_json(coverage_stats)

        with open(output_writer.cache_dir / "coverage.json") as f:
            data = json.load(f)

        assert len(data["files"]) == 2

        file_a = next((f for f in data["files"] if f["path"] == "docs/a.md"), None)
        assert file_a is not None
        assert file_a["covered"] is True
        assert "test-1" in file_a["tests"]
        assert "test-2" in file_a["tests"]

        file_b = next((f for f in data["files"] if f["path"] == "docs/b.md"), None)
        assert file_b is not None
        assert file_b["covered"] is False
        assert file_b["tests"] == []

    def test_handles_empty_coverage(self, output_writer):
        """Handles empty coverage data gracefully."""
        coverage_stats = {
            "total": 0,
            "passed": 0,
            "percentage": 0.0,
            "files_detail": {}
        }

        output_writer.write_coverage_json(coverage_stats)

        with open(output_writer.cache_dir / "coverage.json") as f:
            data = json.load(f)

        assert data["files"] == []
        assert data["summary"]["total_files"] == 0

    def test_failed_tests_still_mark_file_as_covered(self, output_writer, mock_coverage_stats):
        """Files with failed tests are still marked as 'covered'.

        Per PHASE0-CLI-SPEC: 'covered' means the file was tested (passed OR failed),
        not that all tests passed. Only files with status='uncovered' should have
        covered=False.
        """
        files_detail = {
            "docs/passing.md": {"status": "passed", "test_ids": ["test-1"]},
            "docs/failing.md": {"status": "failed", "test_ids": ["test-2"]},
            "docs/untested.md": {"status": "uncovered", "test_ids": []}
        }
        coverage_stats = mock_coverage_stats(total=3, passed=1, failed=1, files_detail=files_detail)

        output_writer.write_coverage_json(coverage_stats)

        with open(output_writer.cache_dir / "coverage.json") as f:
            data = json.load(f)

        file_passing = next((f for f in data["files"] if f["path"] == "docs/passing.md"), None)
        file_failing = next((f for f in data["files"] if f["path"] == "docs/failing.md"), None)
        file_untested = next((f for f in data["files"] if f["path"] == "docs/untested.md"), None)

        # Passed tests → covered
        assert file_passing["covered"] is True

        # Failed tests → still covered (file was tested)
        assert file_failing["covered"] is True, "Files with failed tests should still be marked as covered"

        # Uncovered → not covered
        assert file_untested["covered"] is False


# =============================================================================
# Debug Trace Tests
# =============================================================================

class TestDebugTraceWriter:
    """Tests for debug trace output."""

    def test_writes_only_when_debug_flag_set(
        self, output_writer, mock_suite_results, mock_test_result, mock_coverage_stats
    ):
        """Debug traces only written when debug_enabled=True."""
        test = mock_test_result("test-1", passed=True)
        results = mock_suite_results(total=1, passed=1, test_results=[test])
        coverage = mock_coverage_stats()

        # Without debug
        output_writer.write_all(results, coverage, debug_enabled=False)
        debug_dir = Path(output_writer.cache_dir) / "debug"

        assert not debug_dir.exists() or len(list(debug_dir.glob("*.json"))) == 0

    def test_writes_to_debug_subdirectory(
        self, output_writer, mock_suite_results, mock_test_result, mock_coverage_stats
    ):
        """Debug traces written to .dokumen-cache/debug/."""
        test = mock_test_result("test-1", passed=True)
        results = mock_suite_results(total=1, passed=1, test_results=[test])
        coverage = mock_coverage_stats()

        output_writer.write_all(results, coverage, debug_enabled=True)

        debug_dir = Path(output_writer.cache_dir) / "debug"
        assert debug_dir.exists()
        assert len(list(debug_dir.glob("*.json"))) > 0

    def test_filename_includes_test_name(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Debug filename includes test name."""
        test = mock_test_result("my-special-test", passed=True)
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_debug_traces(results)

        debug_files = list((Path(output_writer.cache_dir) / "debug").glob("*.json"))
        assert len(debug_files) == 1
        assert "my-special-test" in debug_files[0].name

    def test_debug_trace_has_required_fields(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Debug trace includes test_name, started_at, completed_at, executor, judge."""
        test = mock_test_result("test-1", passed=True)
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_debug_traces(results)

        debug_files = list((Path(output_writer.cache_dir) / "debug").glob("*.json"))
        with open(debug_files[0]) as f:
            data = json.load(f)

        assert "test_name" in data
        assert "started_at" in data
        assert "completed_at" in data
        assert "executor" in data
        assert "judge" in data

    def test_debug_trace_includes_executor_prompts(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Debug trace executor includes system_prompt and user_prompt.

        Per PHASE0-CLI-SPEC: Debug traces must include executor's system_prompt and user_prompt.
        """
        test = mock_test_result("test-1", passed=True)
        # Set up executor_output with prompts
        test.executor_output.system_prompt = "You are a documentation tester."
        test.executor_output.user_prompt = "Read the docs and validate them."
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_debug_traces(results)

        debug_files = list((Path(output_writer.cache_dir) / "debug").glob("*.json"))
        with open(debug_files[0]) as f:
            data = json.load(f)

        assert data["executor"]["system_prompt"] == "You are a documentation tester."
        assert data["executor"]["user_prompt"] == "Read the docs and validate them."


# =============================================================================
# Integration Tests
# =============================================================================

class TestOutputWriterIntegration:
    """Integration tests for complete output flow."""

    def test_write_all_creates_all_files(
        self, output_writer, mock_suite_results, mock_test_result, mock_coverage_stats
    ):
        """write_all() creates results.json, junit.xml, coverage.json."""
        test = mock_test_result("test-1", passed=True)
        results = mock_suite_results(total=1, passed=1, test_results=[test])
        coverage = mock_coverage_stats()

        output_writer.write_all(results, coverage, debug_enabled=False)

        assert (Path(output_writer.cache_dir) / "results.json").exists()
        assert (Path(output_writer.cache_dir) / "junit.xml").exists()
        assert (Path(output_writer.cache_dir) / "coverage.json").exists()

    def test_write_all_with_debug_creates_debug_traces(
        self, output_writer, mock_suite_results, mock_test_result, mock_coverage_stats
    ):
        """write_all() with debug=True also creates debug traces."""
        test = mock_test_result("test-1", passed=True)
        results = mock_suite_results(total=1, passed=1, test_results=[test])
        coverage = mock_coverage_stats()

        output_writer.write_all(results, coverage, debug_enabled=True)

        debug_dir = Path(output_writer.cache_dir) / "debug"
        assert debug_dir.exists()
        assert len(list(debug_dir.glob("*.json"))) == 1

    def test_write_all_on_failure_still_creates_files(
        self, output_writer, mock_suite_results, mock_test_result, mock_coverage_stats
    ):
        """All files created even when tests fail."""
        test = mock_test_result("test-1", passed=False)
        results = mock_suite_results(total=1, passed=0, failed=1, test_results=[test])
        coverage = mock_coverage_stats()

        output_writer.write_all(results, coverage, debug_enabled=False)

        assert (Path(output_writer.cache_dir) / "results.json").exists()
        assert (Path(output_writer.cache_dir) / "junit.xml").exists()
        assert (Path(output_writer.cache_dir) / "coverage.json").exists()


class TestOutputArtifactsInResultsJson:
    """Tests for output_artifacts serialization in results.json."""

    def test_output_artifacts_serialized_in_results_json(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Output artifacts from executor/judge are included in results.json."""
        test = mock_test_result("test-output-folder", passed=True)
        test.output_artifacts = [
            {
                "filename": "calculation.py",
                "path": "test-output-folder/calculation.py",
                "size_bytes": 512,
                "content_type": "text/x-python",
                "content": "print('hello')",
            },
            {
                "filename": "report.md",
                "path": "test-output-folder/report.md",
                "size_bytes": 1024,
                "content_type": "text/markdown",
                "content": "# Report\nSome content",
            },
        ]
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        path = output_writer.write_results_json(results)

        with open(path) as f:
            data = json.load(f)

        artifacts = data["tests"][0]["output_artifacts"]
        assert artifacts is not None
        assert len(artifacts) == 2
        assert artifacts[0]["filename"] == "calculation.py"
        assert artifacts[0]["content_type"] == "text/x-python"
        assert artifacts[0]["content"] == "print('hello')"
        assert artifacts[1]["filename"] == "report.md"
        assert artifacts[1]["size_bytes"] == 1024

    def test_output_artifacts_source_field_preserved(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Source field from unified collection is preserved in results.json."""
        test = mock_test_result("test-unified", passed=True)
        test.output_artifacts = [
            {
                "filename": "video.webm",
                "path": "recordings/videos/video.webm",
                "size_bytes": 1000,
                "content_type": "video/webm",
                "content": None,
                "source": "browser",
            },
            {
                "filename": "report.md",
                "path": "report.md",
                "size_bytes": 500,
                "content_type": "text/markdown",
                "content": "# Report",
                "source": "report",
            },
            {
                "filename": "analysis.py",
                "path": "analysis.py",
                "size_bytes": 200,
                "content_type": "text/x-python",
                "content": "print('ok')",
                "source": "output",
            },
        ]
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        path = output_writer.write_results_json(results)

        with open(path) as f:
            data = json.load(f)

        artifacts = data["tests"][0]["output_artifacts"]
        assert len(artifacts) == 3
        assert artifacts[0]["source"] == "browser"
        assert artifacts[1]["source"] == "report"
        assert artifacts[2]["source"] == "output"

    def test_output_artifacts_null_when_none(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Output artifacts are null in results.json when not set."""
        test = mock_test_result("test-no-output", passed=True)
        test.output_artifacts = None
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        path = output_writer.write_results_json(results)

        with open(path) as f:
            data = json.load(f)

        assert data["tests"][0]["output_artifacts"] is None


class TestDictStrStrOrNone:
    """Tests for _dict_str_str_or_none helper."""

    def test_valid_dict_returns_dict(self):
        from dokumen.cli.output import _dict_str_str_or_none

        d = {"accuracy": "claude-opus-4-6", "format": "claude-haiku-4-5-20251001"}
        assert _dict_str_str_or_none(d) == d

    def test_empty_dict_returns_dict(self):
        from dokumen.cli.output import _dict_str_str_or_none

        assert _dict_str_str_or_none({}) == {}

    def test_none_returns_none(self):
        from dokumen.cli.output import _dict_str_str_or_none

        assert _dict_str_str_or_none(None) is None

    def test_mock_returns_none(self):
        from dokumen.cli.output import _dict_str_str_or_none

        assert _dict_str_str_or_none(MagicMock()) is None

    def test_non_string_values_returns_none(self):
        from dokumen.cli.output import _dict_str_str_or_none

        assert _dict_str_str_or_none({"key": 123}) is None

    def test_non_string_keys_returns_none(self):
        from dokumen.cli.output import _dict_str_str_or_none

        assert _dict_str_str_or_none({1: "value"}) is None

    def test_string_returns_none(self):
        from dokumen.cli.output import _dict_str_str_or_none

        assert _dict_str_str_or_none("not a dict") is None


class TestJudgeModelsInResultsJson:
    """Tests for judge_models field in results.json output."""

    def test_judge_models_included_in_results_json(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """judge_models dict is included in results.json when set."""
        test = mock_test_result("test-models", passed=True)
        test.judge_models = {"accuracy": "claude-opus-4-6", "format": "claude-haiku-4-5-20251001"}
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        path = output_writer.write_results_json(results)

        with open(path) as f:
            data = json.load(f)

        assert data["tests"][0]["judge_models"] == {
            "accuracy": "claude-opus-4-6",
            "format": "claude-haiku-4-5-20251001",
        }

    def test_judge_models_null_when_none(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """judge_models is null in results.json when not set."""
        test = mock_test_result("test-no-models", passed=True)
        test.judge_models = None
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        path = output_writer.write_results_json(results)

        with open(path) as f:
            data = json.load(f)

        assert data["tests"][0]["judge_models"] is None


class TestScaffoldYamlInResultsJson:
    """Tests for scaffold_yaml field in results.json output."""

    def test_scaffold_yaml_included_in_results_json(
        self, output_writer, mock_suite_results, mock_test_result, tmp_path
    ):
        """scaffold_yaml field is populated when source_path points to a readable file."""
        yaml_path = tmp_path / "my-test.test.yaml"
        yaml_path.write_text("name: my-test\nreason: test reason\n")

        tr = mock_test_result(test_id="scaffold-test", passed=True)
        tr.source_path = str(yaml_path)
        results = mock_suite_results(total=1, passed=1, test_results=[tr])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert data["tests"][0]["scaffold_yaml"] == "name: my-test\nreason: test reason\n"

    def test_scaffold_yaml_none_when_no_source_path(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """scaffold_yaml is None when source_path is not set."""
        tr = mock_test_result(test_id="no-source", passed=True)
        tr.source_path = None
        results = mock_suite_results(total=1, passed=1, test_results=[tr])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert data["tests"][0]["scaffold_yaml"] is None

    def test_scaffold_yaml_none_when_file_missing(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """scaffold_yaml is None when source_path file doesn't exist."""
        tr = mock_test_result(test_id="missing-file", passed=True)
        tr.source_path = "/nonexistent/path/to/test.yaml"
        results = mock_suite_results(total=1, passed=1, test_results=[tr])

        output_writer.write_results_json(results)

        with open(output_writer.cache_dir / "results.json") as f:
            data = json.load(f)

        assert data["tests"][0]["scaffold_yaml"] is None


# =============================================================================
# Explore Trace Tests
# =============================================================================

class TestExploreTraceWriter:
    """Tests for per-test explore trace files in .dokumen-cache/explore/."""

    def test_write_explore_traces_creates_file(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Explore trace file is created at explore/{test-id}.json with correct structure."""
        test = mock_test_result("biomass-news-research", passed=True)
        test.explore_status = "pass"
        test.explore_model = "claude-haiku-4-5-20251001"
        test.explore_output = "Found 3 relevant files"
        test.explore_tool_calls = [
            {"tool": "glob", "command": "docs/**/*.md", "output": "docs/a.md\ndocs/b.md"},
            {"tool": "read_file", "command": "docs/a.md", "output": "# API Reference\nContent here"},
        ]
        test.explore_input_tokens = 1234
        test.explore_output_tokens = 567
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_explore_traces(results)

        trace_path = Path(output_writer.cache_dir) / "explore" / "biomass-news-research.json"
        assert trace_path.exists()

        with open(trace_path) as f:
            data = json.load(f)

        assert data["test_id"] == "biomass-news-research"
        assert data["explore_status"] == "pass"
        assert data["explore_model"] == "claude-haiku-4-5-20251001"
        assert data["explore_output"] == "Found 3 relevant files"
        assert len(data["tool_calls"]) == 2
        assert data["tool_calls"][0]["tool"] == "glob"
        assert data["tool_calls"][0]["command"] == "docs/**/*.md"
        assert data["tool_calls"][1]["tool"] == "read_file"
        assert data["tokens"]["input"] == 1234
        assert data["tokens"]["output"] == 567

    def test_write_explore_traces_full_output(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Explore trace tool call output is NOT truncated (contrast with results.json 500-char limit)."""
        long_output = "x" * 2000  # Well over 500 chars
        test = mock_test_result("long-output-test", passed=True)
        test.explore_tool_calls = [
            {"tool": "read_file", "command": "docs/big.md", "output": long_output},
        ]
        test.explore_input_tokens = 100
        test.explore_output_tokens = 50
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_explore_traces(results)

        trace_path = Path(output_writer.cache_dir) / "explore" / "long-output-test.json"
        with open(trace_path) as f:
            data = json.load(f)

        # Full 2000 chars, NOT truncated to 500
        assert len(data["tool_calls"][0]["output"]) == 2000

    def test_write_explore_traces_skips_no_explore(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Test without explore data produces no file."""
        test = mock_test_result("no-explore-test", passed=True)
        test.explore_tool_calls = None
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_explore_traces(results)

        explore_dir = Path(output_writer.cache_dir) / "explore"
        assert not (explore_dir / "no-explore-test.json").exists()

    def test_write_explore_traces_skips_empty_list(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Test with empty explore_tool_calls list produces no file."""
        test = mock_test_result("empty-explore-test", passed=True)
        test.explore_tool_calls = []
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_explore_traces(results)

        explore_dir = Path(output_writer.cache_dir) / "explore"
        assert not (explore_dir / "empty-explore-test.json").exists()

    def test_write_all_includes_explore(
        self, output_writer, mock_suite_results, mock_test_result, mock_coverage_stats
    ):
        """write_all() creates explore directory with trace files."""
        test = mock_test_result("explore-in-all", passed=True)
        test.explore_tool_calls = [
            {"tool": "glob", "command": "*.md", "output": "README.md"},
        ]
        test.explore_input_tokens = 10
        test.explore_output_tokens = 5
        results = mock_suite_results(total=1, passed=1, test_results=[test])
        coverage = mock_coverage_stats()

        output_writer.write_all(results, coverage, debug_enabled=False)

        explore_dir = Path(output_writer.cache_dir) / "explore"
        assert explore_dir.exists()
        assert (explore_dir / "explore-in-all.json").exists()

    def test_write_explore_traces_filters_non_dict_items(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """Non-dict items in explore_tool_calls are silently filtered out."""
        test = mock_test_result("filter-test", passed=True)
        test.explore_tool_calls = [
            {"tool": "glob", "command": "*.md", "output": "README.md"},
            "not-a-dict",
            None,
            42,
        ]
        test.explore_input_tokens = 10
        test.explore_output_tokens = 5
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_explore_traces(results)

        trace_path = Path(output_writer.cache_dir) / "explore" / "filter-test.json"
        with open(trace_path) as f:
            data = json.load(f)

        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool"] == "glob"

    def test_write_explore_traces_sanitizes_path_unsafe_test_id(
        self, output_writer, mock_suite_results, mock_test_result
    ):
        """test_id with path separators is sanitized to prevent path traversal."""
        test = mock_test_result("sub/dir/../test", passed=True)
        test.explore_tool_calls = [
            {"tool": "glob", "command": "*.md", "output": "a.md"},
        ]
        test.explore_input_tokens = 0
        test.explore_output_tokens = 0
        results = mock_suite_results(total=1, passed=1, test_results=[test])

        output_writer.write_explore_traces(results)

        explore_dir = Path(output_writer.cache_dir) / "explore"
        # File should be flat in explore/ dir, not in a subdirectory
        trace_path = explore_dir / "sub_dir___test.json"
        assert trace_path.exists()
        # No file should escape explore/ directory
        assert not (explore_dir / "sub").exists()
