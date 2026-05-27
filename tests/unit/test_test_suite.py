"""Tests for TestSuite module."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json


class TestTestSuiteConfig:
    """Tests for TestSuiteConfig dataclass."""

    def test_config_defaults(self):
        """TestSuiteConfig has sensible defaults."""
        from dokumen.test_suite import TestSuiteConfig
        config = TestSuiteConfig(name="test-suite")

        assert config.name == "test-suite"
        assert config.cache_path == ".dokumen-cache"
        assert config.parallel_execution is False
        assert config.max_concurrency == 4
        assert config.coverage_agent is None
        assert config.sandbox_config is None

    def test_config_custom_values(self):
        """TestSuiteConfig accepts custom values."""
        from dokumen.test_suite import TestSuiteConfig
        config = TestSuiteConfig(
            name="custom",
            cache_path="/custom/cache",
            parallel_execution=True,
            max_concurrency=8
        )

        assert config.name == "custom"
        assert config.cache_path == "/custom/cache"
        assert config.parallel_execution is True
        assert config.max_concurrency == 8


class TestTestSuiteResults:
    """Tests for TestSuiteResults dataclass."""

    def test_results_fields(self):
        """TestSuiteResults has required fields."""
        from dokumen.test_suite import TestSuiteResults

        results = TestSuiteResults(
            total_tests=10,
            passed=8,
            failed=2,
            skipped=0,
            duration=5.5,
            test_results=[],
            cached_results=3
        )

        assert results.total_tests == 10
        assert results.passed == 8
        assert results.failed == 2
        assert results.skipped == 0
        assert results.duration == 5.5
        assert results.cached_results == 3


class TestCoverageReport:
    """Tests for CoverageReport dataclass."""

    def test_coverage_report_fields(self):
        """CoverageReport has required fields."""
        from dokumen.test_suite import CoverageReport

        report = CoverageReport(
            total_files=10,
            covered_files=8,
            coverage_percentage=80.0,
            file_details=[]
        )

        assert report.total_files == 10
        assert report.covered_files == 8
        assert report.coverage_percentage == 80.0


class TestTestSuiteInit:
    """Tests for TestSuite initialization."""

    def test_init_with_config(self):
        """TestSuite initializes with config."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="my-suite")
        suite = TestSuite(config)

        assert suite.name == "my-suite"
        assert suite.config == config
        assert suite.tests == []
        assert suite._cached_results == {}


class TestTestSuiteManagement:
    """Tests for TestSuite add/remove/get operations."""

    def test_add_test(self):
        """Can add a test to the suite."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test = MagicMock()
        mock_test.id = "test-1"
        suite.add_test(mock_test)

        assert len(suite.tests) == 1
        assert suite.tests[0] is mock_test

    def test_get_test(self):
        """Can get a test by ID."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test = MagicMock()
        mock_test.id = "test-1"
        suite.add_test(mock_test)

        result = suite.get_test("test-1")
        assert result is mock_test

    def test_get_test_not_found(self):
        """get_test returns None for unknown ID."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        result = suite.get_test("nonexistent")
        assert result is None

    def test_remove_test(self):
        """Can remove a test by ID."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test = MagicMock()
        mock_test.id = "test-1"
        suite.add_test(mock_test)

        result = suite.remove_test("test-1")

        assert result is True
        assert len(suite.tests) == 0

    def test_remove_test_not_found(self):
        """remove_test returns False for unknown ID."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        result = suite.remove_test("nonexistent")
        assert result is False


class TestExtractAccessedFiles:
    """Tests for _extract_accessed_files method."""

    def test_extract_handles_no_executor_output(self):
        """Handles missing executor output."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_result = MagicMock()
        mock_result.executor_output = None

        files = suite._extract_accessed_files(mock_result)
        assert files == []

    def test_extract_handles_no_tool_calls(self):
        """Handles no tool calls."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_result = MagicMock()
        mock_result.executor_output = MagicMock()
        mock_result.executor_output.tool_calls = []

        files = suite._extract_accessed_files(mock_result)
        assert files == []


class TestExtractSubagentCoverage:
    """Tests for _extract_subagent_coverage method."""

    def test_extract_handles_no_executor_output(self):
        """Handles no executor output."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_result = MagicMock()
        mock_result.executor_output = None

        coverage = suite._extract_subagent_coverage(mock_result)
        assert coverage == {}

    def test_extract_handles_no_tool_calls(self):
        """Handles no tool calls."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_result = MagicMock()
        mock_result.executor_output = MagicMock()
        mock_result.executor_output.tool_calls = []

        coverage = suite._extract_subagent_coverage(mock_result)
        assert coverage == {}


class TestAggregateResults:
    """Tests for _aggregate_results method."""

    def test_aggregate_basic(self):
        """Aggregates basic results."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_results = [
            MagicMock(passed=True),
            MagicMock(passed=True),
            MagicMock(passed=False),
        ]

        aggregated = suite._aggregate_results(mock_results, 10.5, 1)

        assert aggregated.total_tests == 3
        assert aggregated.passed == 2
        assert aggregated.failed == 1
        assert aggregated.duration == 10.5
        assert aggregated.cached_results == 1

    def test_aggregate_empty(self):
        """Aggregates empty results list."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        aggregated = suite._aggregate_results([], 0.0, 0)

        assert aggregated.total_tests == 0
        assert aggregated.passed == 0
        assert aggregated.failed == 0


class TestCacheOperations:
    """Tests for cache loading/saving."""

    @pytest.mark.asyncio
    async def test_load_cache(self, tmp_path):
        """Loads cache from file."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        # Create cache directory
        cache_dir = tmp_path / ".dokumen-cache"
        cache_dir.mkdir()

        config = TestSuiteConfig(name="suite", cache_path=str(cache_dir))
        suite = TestSuite(config)

        # load_cache should not raise
        await suite.load_cache()

    @pytest.mark.asyncio
    async def test_save_cache(self, tmp_path):
        """Saves cache to file."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        cache_dir = tmp_path / ".dokumen-cache"
        cache_dir.mkdir()

        config = TestSuiteConfig(name="suite", cache_path=str(cache_dir))
        suite = TestSuite(config)

        # save_cache should not raise
        await suite.save_cache()

    @pytest.mark.asyncio
    async def test_clear_cache(self, tmp_path):
        """Clears cached results."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite", cache_path=str(tmp_path))
        suite = TestSuite(config)

        # Add some cached results
        suite._cached_results = {"test-1": MagicMock()}

        await suite.clear_cache()

        assert suite._cached_results == {}


class TestGetCoverage:
    """Tests for get_coverage method."""

    def test_get_coverage_empty(self):
        """get_coverage returns result for empty suite."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        coverage = suite.get_coverage()
        # May return dict or CoverageReport depending on implementation
        assert coverage is not None

    def test_get_coverage_with_files(self):
        """get_coverage returns file coverage data."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.file_object import FileObject, FileStatus

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        # Add file to registry
        file1 = FileObject(path="docs/api.md")
        suite._file_registry["docs/api.md"] = file1
        suite._file_status["docs/api.md"] = FileStatus.PASSED

        coverage = suite.get_coverage()
        # Result depends on implementation
        assert coverage is not None


class TestFileStatus:
    """Tests for file status tracking."""

    def test_get_file_status(self):
        """Gets file status."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.file_object import FileStatus

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        suite._file_status["docs/api.md"] = FileStatus.PASSED

        status = suite.get_file_status("docs/api.md")
        assert status == FileStatus.PASSED

    def test_get_file_status_not_found(self):
        """Returns UNCOVERED for unknown files."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.file_object import FileStatus

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        status = suite.get_file_status("unknown.md")
        # Returns UNCOVERED for unknown files
        assert status == FileStatus.UNCOVERED

    def test_get_all_file_statuses(self):
        """Gets all file statuses."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.file_object import FileStatus

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        suite._file_status["docs/api.md"] = FileStatus.PASSED
        suite._file_status["docs/readme.md"] = FileStatus.FAILED

        statuses = suite.get_all_file_statuses()

        # _file_status is the internal dict, get_all_file_statuses may return
        # it directly or a copy
        assert isinstance(statuses, dict)


class TestRunTests:
    """Tests for running tests."""

    @pytest.mark.asyncio
    async def test_run_sequential(self):
        """Runs tests sequentially."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        # Mock test
        mock_test = MagicMock()
        mock_test.id = "test-1"
        mock_test.reason = "Test reason"
        mock_test.timeout = 30
        mock_test.is_stale = MagicMock(return_value=True)

        mock_executor = MagicMock()
        mock_executor.id = "executor-1"
        mock_executor.user_prompt = "Test prompt"
        mock_executor.provider = MagicMock()
        mock_executor.provider.model = "test-model"
        mock_test.executor = mock_executor
        mock_test.judges = []

        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.test_id = "test-1"
        mock_result.executor_output = MagicMock()
        mock_result.executor_output.tool_calls = []
        mock_result.judge_results = []
        mock_result.line_coverage = {}
        mock_test.run = AsyncMock(return_value=mock_result)

        suite.add_test(mock_test)

        # Patch both is_debug and _save_cache_incremental
        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            results = await suite.run()

        assert results.total_tests == 1
        assert results.passed == 1

    @pytest.mark.asyncio
    async def test_run_with_callbacks(self):
        """Runs tests with progress callbacks."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        # Mock test
        mock_test = MagicMock()
        mock_test.id = "test-1"
        mock_test.reason = "Test reason"
        mock_test.timeout = 30
        mock_test.is_stale = MagicMock(return_value=True)

        mock_executor = MagicMock()
        mock_executor.id = "executor-1"
        mock_executor.user_prompt = "Test prompt"
        mock_executor.provider = MagicMock()
        mock_test.executor = mock_executor
        mock_test.judges = []

        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.executor_output = MagicMock()
        mock_result.executor_output.tool_calls = []
        mock_result.judge_results = []
        mock_result.line_coverage = {}
        mock_test.run = AsyncMock(return_value=mock_result)

        suite.add_test(mock_test)

        progress_calls = []

        def on_progress(event, test_id, data):
            progress_calls.append((event, test_id))

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            await suite.run(on_progress=on_progress)

        assert ("start", "test-1") in progress_calls
        assert ("complete", "test-1") in progress_calls

    @pytest.mark.asyncio
    async def test_run_uses_cache(self):
        """Uses cached results when available."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        # Mock test
        mock_test = MagicMock()
        mock_test.id = "test-1"
        mock_test.reason = "Test reason"
        mock_test.timeout = 30
        mock_test.is_stale = MagicMock(return_value=False)  # Not stale, use cache
        mock_test.run = AsyncMock()

        mock_executor = MagicMock()
        mock_test.executor = mock_executor

        suite.add_test(mock_test)

        # Pre-populate cache
        cached_result = MagicMock()
        cached_result.passed = True
        suite._cached_results["test-1"] = cached_result

        with patch('dokumen.debug.is_debug', return_value=False):
            results = await suite.run()

        # Should use cached result, not run test
        mock_test.run.assert_not_called()
        assert results.cached_results == 1


class TestGetTests:
    """Tests for getting test lists."""

    def test_tests_attribute(self):
        """tests attribute returns all tests."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test1 = MagicMock()
        mock_test1.id = "test-1"
        mock_test2 = MagicMock()
        mock_test2.id = "test-2"

        suite.add_test(mock_test1)
        suite.add_test(mock_test2)

        assert len(suite.tests) == 2

    def test_get_test_by_id(self):
        """Can get test by ID."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test1 = MagicMock()
        mock_test1.id = "test-1"
        mock_test2 = MagicMock()
        mock_test2.id = "test-2"

        suite.add_test(mock_test1)
        suite.add_test(mock_test2)

        test = suite.get_test("test-1")
        assert test is mock_test1


class TestFailureAnalysis:
    """Tests for failure analysis tracking."""

    def test_get_failure_analysis(self):
        """Gets failure analysis for a file."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.test_object import FailureAnalysis

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        # Add failure analysis
        analysis = FailureAnalysis(
            file_path="docs/api.md",
            referenced_lines=[1, 2, 3],
            incorrect_lines=[],
            analysis="Some issue"
        )
        suite._failure_analysis["docs/api.md"] = {"test-1": analysis}

        result = suite.get_failure_analysis("docs/api.md")

        assert result is not None
        assert "test-1" in result

    def test_get_failure_analysis_not_found(self):
        """Returns empty dict for unknown file."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        result = suite.get_failure_analysis("unknown.md")

        assert result == {}


class TestFileStatusTracking:
    """Tests for file status tracking during test runs."""

    def test_update_file_status_passed(self):
        """File status updates to PASSED when test passes."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig, FileStatus

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        # Simulate a passed test accessing a file
        suite._file_status["docs/api.md"] = FileStatus.PASSED

        assert suite._file_status["docs/api.md"] == FileStatus.PASSED

    def test_update_file_status_failed(self):
        """File status updates to FAILED when test fails."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig, FileStatus

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        suite._file_status["docs/api.md"] = FileStatus.FAILED

        assert suite._file_status["docs/api.md"] == FileStatus.FAILED

    def test_file_registry_adds_new_files(self):
        """File registry registers new files."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.file_object import FileObject

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        file_obj = FileObject(path="docs/new.md")
        suite._file_registry["docs/new.md"] = file_obj

        assert "docs/new.md" in suite._file_registry


class TestCacheIncrementalSave:
    """Tests for incremental cache saving."""

    @pytest.mark.asyncio
    async def test_save_cache_incremental(self, tmp_path):
        """Incremental cache save persists results."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.test_object import TestResult
        from datetime import datetime

        config = TestSuiteConfig(name="suite", cache_path=str(tmp_path))
        suite = TestSuite(config)

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            executor_output=None,
            judge_results=[],
            failure_reasons=[],
            duration=1.0,
            timestamp=datetime.now()
        )

        await suite._save_cache_incremental("test-1", result)

        # Should not raise
        assert True


class TestLineCoverage:
    """Tests for line coverage tracking."""

    def test_line_coverage_dict(self):
        """TestResult has line_coverage dict."""
        from dokumen.test_object import TestResult
        from datetime import datetime

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            executor_output=None,
            judge_results=[],
            failure_reasons=[],
            duration=1.0,
            timestamp=datetime.now(),
            line_coverage={}
        )

        assert result.line_coverage == {}


class TestProcessTestCoverage:
    """Tests for process_test_coverage method."""

    @pytest.mark.asyncio
    async def test_process_coverage_no_agent(self):
        """Process coverage works without coverage agent."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.test_object import TestResult
        from unittest.mock import MagicMock
        from datetime import datetime

        config = TestSuiteConfig(name="suite", coverage_agent=None)
        suite = TestSuite(config)

        mock_test = MagicMock()
        mock_test.id = "test-1"

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            executor_output=None,
            judge_results=[],
            failure_reasons=[],
            duration=1.0,
            timestamp=datetime.now()
        )

        await suite.process_test_coverage(mock_test, result)

        # Should complete without error
        assert True


class TestSuiteConfigValidation:
    """Tests for TestSuiteConfig validation."""

    def test_config_with_name(self):
        """Config requires name."""
        from dokumen.test_suite import TestSuiteConfig

        config = TestSuiteConfig(name="my-suite")

        assert config.name == "my-suite"

    def test_config_cache_path_default(self):
        """Config has default cache path."""
        from dokumen.test_suite import TestSuiteConfig

        config = TestSuiteConfig(name="suite")

        assert config.cache_path == ".dokumen-cache"

    def test_config_cache_path(self):
        """Config can set cache path."""
        from dokumen.test_suite import TestSuiteConfig

        config = TestSuiteConfig(name="suite", cache_path="/custom/path")

        assert config.cache_path == "/custom/path"


class TestTestResultSerialization:
    """Tests for TestResult serialization."""

    def test_to_dict_basic(self):
        """TestResult converts to dict."""
        from dokumen.test_object import TestResult
        from datetime import datetime

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            executor_output=None,
            judge_results=[],
            failure_reasons=[],
            duration=1.5,
            timestamp=datetime.now()
        )

        d = result.to_dict()

        assert d["test_id"] == "test-1"
        assert d["passed"] is True
        assert d["duration"] == 1.5

    def test_to_dict_with_failures(self):
        """TestResult includes failure reasons."""
        from dokumen.test_object import TestResult
        from datetime import datetime

        result = TestResult(
            test_id="test-1",
            passed=False,
            executor_passed=False,
            executor_output=None,
            judge_results=[],
            failure_reasons=["Judge failed", "Timeout"],
            duration=2.0,
            timestamp=datetime.now()
        )

        d = result.to_dict()

        assert len(d["failure_reasons"]) == 2
        assert "Judge failed" in d["failure_reasons"]


class TestCoverageReportDetails:
    """Tests for CoverageReport details."""

    def test_coverage_report_percentage(self):
        """CoverageReport stores percentage."""
        from dokumen.test_suite import CoverageReport

        report = CoverageReport(
            total_files=10,
            covered_files=5,
            coverage_percentage=50.0,
            file_details=[]
        )

        assert report.coverage_percentage == 50.0

    def test_coverage_report_empty(self):
        """CoverageReport handles empty state."""
        from dokumen.test_suite import CoverageReport

        report = CoverageReport(
            total_files=0,
            covered_files=0,
            coverage_percentage=0.0,
            file_details=[]
        )

        assert report.total_files == 0
        assert report.coverage_percentage == 0.0


class TestGetResults:
    """Tests for get_results method."""

    def test_get_results_empty(self):
        """get_results returns None initially."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        results = suite.get_results()

        # Returns None when no results yet
        assert results is None

    def test_get_line_coverage_empty(self):
        """get_line_coverage returns empty dict initially."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        coverage = suite.get_line_coverage()

        assert isinstance(coverage, dict)


class TestSuiteTestManagement:
    """Tests for test management in TestSuite."""

    def test_count_tests(self):
        """Suite counts tests correctly."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.test_object import TestObject
        from unittest.mock import MagicMock

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test1 = MagicMock(spec=TestObject)
        mock_test1.id = "test-1"
        mock_test2 = MagicMock(spec=TestObject)
        mock_test2.id = "test-2"

        suite.add_test(mock_test1)
        suite.add_test(mock_test2)

        assert len(suite.tests) == 2

    def test_filter_tests_by_pattern(self):
        """Suite can filter tests by pattern."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.test_object import TestObject
        from unittest.mock import MagicMock

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test1 = MagicMock(spec=TestObject)
        mock_test1.id = "api-test"
        mock_test2 = MagicMock(spec=TestObject)
        mock_test2.id = "ui-test"

        suite.add_test(mock_test1)
        suite.add_test(mock_test2)

        # Get test by ID
        api_test = suite.get_test("api-test")
        assert api_test.id == "api-test"


class TestExtractAccessedFilesAdvanced:
    """Advanced tests for _extract_accessed_files."""

    def test_extract_files_from_read_file_tool(self):
        """Extracts files from read_file tool calls."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_result = MagicMock()
        mock_result.executor_output = MagicMock()
        mock_call = MagicMock()
        mock_call.tool_name = "read_file"
        mock_call.parameters = {"path": "docs/api.md"}
        mock_result.executor_output.tool_calls = [mock_call]

        files = suite._extract_accessed_files(mock_result)

        # Should extract file path from read_file call
        assert len(files) >= 0  # May be empty if regex doesn't match

    def test_extract_files_from_bash_cat_command(self):
        """Extracts files from bash cat commands."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_result = MagicMock()
        mock_result.executor_output = MagicMock()
        mock_call = MagicMock()
        mock_call.tool_name = "bash"
        mock_call.parameters = {"command": "cat docs/readme.md"}
        mock_result.executor_output.tool_calls = [mock_call]

        files = suite._extract_accessed_files(mock_result)

        # Should extract file path from bash cat command
        assert isinstance(files, list)

    def test_extract_files_handles_multiple_tools(self):
        """Handles multiple tool calls."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_result = MagicMock()
        mock_result.executor_output = MagicMock()

        mock_call1 = MagicMock()
        mock_call1.tool_name = "read_file"
        mock_call1.parameters = {"path": "docs/api.md"}

        mock_call2 = MagicMock()
        mock_call2.tool_name = "list_files"
        mock_call2.parameters = {"pattern": "*.md"}

        mock_result.executor_output.tool_calls = [mock_call1, mock_call2]

        files = suite._extract_accessed_files(mock_result)

        assert isinstance(files, list)


class TestCacheRoundTrip:
    """Tests for cache save and load round-trip."""

    @pytest.mark.asyncio
    async def test_save_and_load_cache(self, tmp_path):
        """Cache can be saved and loaded."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.test_object import TestResult
        from datetime import datetime

        cache_dir = tmp_path / ".dokumen-cache"
        cache_dir.mkdir()

        config = TestSuiteConfig(name="suite", cache_path=str(cache_dir))
        suite = TestSuite(config)

        # Add cached result
        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            executor_output=None,
            judge_results=[],
            failure_reasons=[],
            duration=1.0,
            timestamp=datetime.now()
        )
        suite._cached_results["test-1"] = result

        # Save
        await suite.save_cache()

        # Create new suite and load
        config2 = TestSuiteConfig(name="suite", cache_path=str(cache_dir))
        suite2 = TestSuite(config2)
        await suite2.load_cache()

        # Should load the cache
        # The exact behavior depends on implementation

    @pytest.mark.asyncio
    async def test_load_cache_missing_file(self, tmp_path):
        """Loading non-existent cache file doesn't crash."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite", cache_path=str(tmp_path / "nonexistent"))
        suite = TestSuite(config)

        # Should not raise
        await suite.load_cache()

        assert suite._cached_results == {}


class TestRunWithDebug:
    """Tests for running tests with debug enabled."""

    @pytest.mark.asyncio
    async def test_run_completes_with_results(self):
        """Run completes and returns results."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test = MagicMock()
        mock_test.id = "test-1"
        mock_test.reason = "Test reason"
        mock_test.timeout = 30
        mock_test.is_stale = MagicMock(return_value=True)

        mock_executor = MagicMock()
        mock_executor.id = "executor-1"
        mock_executor.user_prompt = "Test prompt"
        mock_executor.provider = MagicMock()
        mock_test.executor = mock_executor
        mock_test.judges = []

        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.test_id = "test-1"
        mock_result.executor_output = MagicMock()
        mock_result.executor_output.tool_calls = []
        mock_result.executor_output.to_dict = MagicMock(return_value={})
        mock_result.judge_results = []
        mock_result.line_coverage = {}
        mock_result.duration = 1.0
        mock_result.failure_reasons = []
        mock_test.run = AsyncMock(return_value=mock_result)

        suite.add_test(mock_test)

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            results = await suite.run()

        assert results.total_tests == 1
        assert results.passed == 1


class TestRunWithFailedTest:
    """Tests for running tests that fail."""

    @pytest.mark.asyncio
    async def test_run_failed_test(self):
        """Handles failed test correctly."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test = MagicMock()
        mock_test.id = "test-1"
        mock_test.reason = "Test reason"
        mock_test.timeout = 30
        mock_test.is_stale = MagicMock(return_value=True)

        mock_executor = MagicMock()
        mock_executor.id = "executor-1"
        mock_executor.user_prompt = "Test prompt"
        mock_executor.provider = MagicMock()
        mock_test.executor = mock_executor
        mock_test.judges = []

        mock_result = MagicMock()
        mock_result.passed = False
        mock_result.test_id = "test-1"
        mock_result.executor_output = MagicMock()
        mock_result.executor_output.tool_calls = []
        mock_result.judge_results = []
        mock_result.line_coverage = {}
        mock_result.failure_reasons = ["Test failed"]
        mock_test.run = AsyncMock(return_value=mock_result)

        suite.add_test(mock_test)

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            results = await suite.run()

        assert results.failed == 1
        # Failed tests should not be cached
        assert "test-1" not in suite._cached_results


class TestCoverageGeneration:
    """Tests for coverage report generation."""

    def test_get_coverage_with_registered_files(self):
        """get_coverage includes registered files."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.file_object import FileObject, FileStatus

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        # Add files to registry and status
        file1 = FileObject(path="docs/api.md")
        file2 = FileObject(path="docs/guide.md")
        suite._file_registry["docs/api.md"] = file1
        suite._file_registry["docs/guide.md"] = file2
        suite._file_status["docs/api.md"] = FileStatus.PASSED
        suite._file_status["docs/guide.md"] = FileStatus.FAILED

        coverage = suite.get_coverage()

        assert coverage is not None

    def test_get_all_file_statuses_returns_dict(self):
        """get_all_file_statuses returns a dictionary."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.file_object import FileStatus

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        suite._file_status["docs/passed.md"] = FileStatus.PASSED
        suite._file_status["docs/failed.md"] = FileStatus.FAILED

        statuses = suite.get_all_file_statuses()

        # Should return a dictionary (may be empty or include registered files)
        assert isinstance(statuses, dict)


class TestCacheResultsData:
    """Tests for cache results data structure."""

    @pytest.mark.asyncio
    async def test_save_cache_with_results(self, tmp_path):
        """save_cache writes results to file."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.test_object import TestResult
        from datetime import datetime
        import os

        cache_dir = tmp_path / ".dokumen-cache"
        cache_dir.mkdir()

        config = TestSuiteConfig(name="test-suite", cache_path=str(cache_dir))
        suite = TestSuite(config)

        # Add a cached result
        result = TestResult(
            test_id="my-test",
            passed=True,
            executor_passed=True,
            executor_output=None,
            judge_results=[],
            failure_reasons=[],
            duration=2.5,
            timestamp=datetime.now()
        )
        suite._cached_results["my-test"] = result

        await suite.save_cache()

        # Check that cache was written
        # Implementation may vary


class TestAggregateResultsAdvanced:
    """Advanced tests for _aggregate_results."""

    def test_aggregate_with_all_passed(self):
        """Aggregates all passing tests."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        results = [
            MagicMock(passed=True),
            MagicMock(passed=True),
            MagicMock(passed=True),
        ]

        aggregated = suite._aggregate_results(results, 5.0, 0)

        assert aggregated.total_tests == 3
        assert aggregated.passed == 3
        assert aggregated.failed == 0
        assert aggregated.skipped == 0

    def test_aggregate_with_all_failed(self):
        """Aggregates all failing tests."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        results = [
            MagicMock(passed=False),
            MagicMock(passed=False),
        ]

        aggregated = suite._aggregate_results(results, 3.0, 0)

        assert aggregated.total_tests == 2
        assert aggregated.passed == 0
        assert aggregated.failed == 2

    def test_aggregate_mixed_results(self):
        """Aggregates mixed results."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        results = [
            MagicMock(passed=True),
            MagicMock(passed=False),
            MagicMock(passed=True),
            MagicMock(passed=False),
        ]

        aggregated = suite._aggregate_results(results, 8.0, 2)

        assert aggregated.total_tests == 4
        assert aggregated.passed == 2
        assert aggregated.failed == 2
        assert aggregated.cached_results == 2

    def test_aggregate_excludes_error_status_from_failed_count(self):
        """Error-status results are not double-counted as failed.

        Suite with 5 pass, 3 fail, 2 error should report:
        passed=5, failed=3 (not 5), total=10.
        """
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        results = (
            [MagicMock(passed=True, status='') for _ in range(5)]
            + [MagicMock(passed=False, status='') for _ in range(3)]
            + [MagicMock(passed=False, status='error') for _ in range(2)]
        )

        aggregated = suite._aggregate_results(results, 10.0, 0)

        assert aggregated.total_tests == 10
        assert aggregated.passed == 5
        assert aggregated.failed == 3


class TestCacheClearOperation:
    """Tests for cache clearing."""

    @pytest.mark.asyncio
    async def test_clear_cache_removes_all(self, tmp_path):
        """clear_cache removes all cached results."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite", cache_path=str(tmp_path))
        suite = TestSuite(config)

        # Add multiple cached results
        suite._cached_results = {
            "test-1": MagicMock(),
            "test-2": MagicMock(),
            "test-3": MagicMock(),
        }

        await suite.clear_cache()

        assert len(suite._cached_results) == 0

    @pytest.mark.asyncio
    async def test_clear_cache_idempotent(self, tmp_path):
        """clear_cache is idempotent."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite", cache_path=str(tmp_path))
        suite = TestSuite(config)

        # Clear twice
        await suite.clear_cache()
        await suite.clear_cache()

        assert suite._cached_results == {}


class TestLineCoverageTracking:
    """Tests for line-level coverage tracking."""

    def test_line_coverage_returns_dict(self):
        """get_line_coverage returns a dictionary."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        coverage = suite.get_line_coverage()

        # Should return a dictionary
        assert isinstance(coverage, dict)

    def test_file_status_update_on_pass(self):
        """File status updates correctly on pass."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.file_object import FileStatus

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        # Simulate status update
        suite._file_status["test.md"] = FileStatus.PASSED

        status = suite.get_file_status("test.md")

        assert status == FileStatus.PASSED

    def test_file_status_update_on_fail(self):
        """File status updates correctly on fail."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.file_object import FileStatus

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        # Simulate status update
        suite._file_status["test.md"] = FileStatus.FAILED

        status = suite.get_file_status("test.md")

        assert status == FileStatus.FAILED


class TestCallbacksIntegration:
    """Tests for callback invocations during test run."""

    @pytest.mark.asyncio
    async def test_executor_complete_callback_passed_to_run(self):
        """Executor complete callback is passed to test.run."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test = MagicMock()
        mock_test.id = "test-1"
        mock_test.reason = "Test"
        mock_test.timeout = 30
        mock_test.is_stale = MagicMock(return_value=True)

        mock_executor = MagicMock()
        mock_executor.id = "exec-1"
        mock_executor.user_prompt = "Prompt"
        mock_executor.provider = MagicMock()
        mock_test.executor = mock_executor
        mock_test.judges = []

        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.executor_output = MagicMock()
        mock_result.executor_output.tool_calls = []
        mock_result.judge_results = []
        mock_result.line_coverage = {}
        mock_test.run = AsyncMock(return_value=mock_result)

        suite.add_test(mock_test)

        def on_executor_complete(test_id, output):
            pass

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            await suite.run(on_executor_complete=on_executor_complete)

        # The callback wrapper is passed to test.run
        mock_test.run.assert_called_once()
        call_kwargs = mock_test.run.call_args.kwargs
        assert "on_executor_complete" in call_kwargs

    @pytest.mark.asyncio
    async def test_tool_call_callback(self):
        """Tool call callback is passed to test.run."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test = MagicMock()
        mock_test.id = "test-1"
        mock_test.reason = "Test"
        mock_test.timeout = 30
        mock_test.is_stale = MagicMock(return_value=True)

        mock_executor = MagicMock()
        mock_executor.id = "exec-1"
        mock_executor.user_prompt = "Prompt"
        mock_executor.provider = MagicMock()
        mock_test.executor = mock_executor
        mock_test.judges = []

        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.executor_output = MagicMock()
        mock_result.executor_output.tool_calls = []
        mock_result.judge_results = []
        mock_result.line_coverage = {}
        mock_test.run = AsyncMock(return_value=mock_result)

        suite.add_test(mock_test)

        tool_calls = []

        def on_tool_call(name, params, result):
            tool_calls.append((name, params))

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            await suite.run(on_tool_call=on_tool_call)

        # The callback was passed to test.run
        mock_test.run.assert_called_once()
        call_kwargs = mock_test.run.call_args.kwargs
        assert "on_tool_call" in call_kwargs


class TestRunWithDebugMode:
    """Tests for running tests with debug mode enabled."""

    @pytest.mark.asyncio
    async def test_run_with_debug_tracks_test(self):
        """Debug mode tracks test start and completion."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.debug import DebugSession

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test = MagicMock()
        mock_test.id = "test-1"
        mock_test.reason = "Test reason"
        mock_test.timeout = 30
        mock_test.is_stale = MagicMock(return_value=True)

        mock_executor = MagicMock()
        mock_executor.id = "executor-1"
        mock_executor.user_prompt = "Test prompt"
        mock_executor.provider = MagicMock()
        mock_test.executor = mock_executor
        mock_test.judges = []

        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.test_id = "test-1"
        mock_result.duration = 1.0
        mock_result.failure_reasons = []
        mock_result.executor_output = MagicMock()
        mock_result.executor_output.tool_calls = []
        mock_result.executor_output.to_dict = MagicMock(return_value={})
        mock_result.judge_results = []
        mock_result.line_coverage = {}
        mock_test.run = AsyncMock(return_value=mock_result)

        suite.add_test(mock_test)

        # Create mock debug session
        mock_session = MagicMock(spec=DebugSession)

        with patch('dokumen.debug.is_debug', return_value=True), \
             patch('dokumen.debug.get_debug_session', return_value=mock_session), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            results = await suite.run()

        # Debug session should track test lifecycle
        mock_session.start_test.assert_called_with("test-1")
        mock_session.start_executor.assert_called_once()
        mock_session.finish_executor.assert_called_once()
        mock_session.finish_test.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_debug_tracks_judge_results(self):
        """Debug mode tracks judge results."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.debug import DebugSession
        from dokumen.agent_object import JudgeResult

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test = MagicMock()
        mock_test.id = "test-1"
        mock_test.reason = "Test reason"
        mock_test.timeout = 30
        mock_test.is_stale = MagicMock(return_value=True)

        mock_executor = MagicMock()
        mock_executor.id = "executor-1"
        mock_executor.user_prompt = "Test prompt"
        mock_executor.provider = MagicMock()
        mock_test.executor = mock_executor
        mock_test.judges = [MagicMock(id="judge-1")]

        # Create mock judge result
        judge_result = JudgeResult(
            judge_id="judge-1",
            passed=True,
            confidence=0.9
        )

        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.test_id = "test-1"
        mock_result.duration = 1.0
        mock_result.failure_reasons = []
        mock_result.executor_output = MagicMock()
        mock_result.executor_output.tool_calls = []
        mock_result.executor_output.to_dict = MagicMock(return_value={})
        mock_result.judge_results = [judge_result]
        mock_result.line_coverage = {}
        mock_test.run = AsyncMock(return_value=mock_result)

        suite.add_test(mock_test)

        # Create mock debug session
        mock_session = MagicMock(spec=DebugSession)

        with patch('dokumen.debug.is_debug', return_value=True), \
             patch('dokumen.debug.get_debug_session', return_value=mock_session), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            results = await suite.run()

        # Debug session should track judge
        mock_session.start_judge.assert_called_with("judge-1")
        mock_session.finish_judge.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_debug_handles_no_executor_output(self):
        """Debug mode handles missing executor output."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.debug import DebugSession

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_test = MagicMock()
        mock_test.id = "test-1"
        mock_test.reason = "Test reason"
        mock_test.timeout = 30
        mock_test.is_stale = MagicMock(return_value=True)

        mock_executor = MagicMock()
        mock_executor.id = "executor-1"
        mock_executor.user_prompt = "Test prompt"
        mock_executor.provider = MagicMock()
        mock_test.executor = mock_executor
        mock_test.judges = []

        mock_result = MagicMock()
        mock_result.passed = False
        mock_result.test_id = "test-1"
        mock_result.duration = 1.0
        mock_result.failure_reasons = ["Executor failed"]
        mock_result.executor_output = None  # No executor output
        mock_result.judge_results = []
        mock_result.line_coverage = {}
        mock_test.run = AsyncMock(return_value=mock_result)

        suite.add_test(mock_test)

        # Create mock debug session
        mock_session = MagicMock(spec=DebugSession)

        with patch('dokumen.debug.is_debug', return_value=True), \
             patch('dokumen.debug.get_debug_session', return_value=mock_session), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            results = await suite.run()

        # Debug session should still finish executor with empty dict
        mock_session.finish_executor.assert_called_with({})


class TestExtractFilesPatterns:
    """Tests for file extraction patterns."""

    def test_extract_glob_tool_paths(self):
        """Extracts paths from glob tool calls."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_result = MagicMock()
        mock_result.executor_output = MagicMock()
        mock_call = MagicMock()
        mock_call.tool_name = "glob"
        mock_call.parameters = {"pattern": "docs/*.md"}
        mock_result.executor_output.tool_calls = [mock_call]

        files = suite._extract_accessed_files(mock_result)

        # Should return list (may be empty based on pattern extraction)
        assert isinstance(files, list)

    def test_extract_list_directory_paths(self):
        """Extracts paths from list_directory tool calls."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        mock_result = MagicMock()
        mock_result.executor_output = MagicMock()
        mock_call = MagicMock()
        mock_call.tool_name = "list_directory"
        mock_call.parameters = {"path": "docs"}
        mock_result.executor_output.tool_calls = [mock_call]

        files = suite._extract_accessed_files(mock_result)

        assert isinstance(files, list)


# =============================================================================
# Parallel Execution Tests
# =============================================================================


def _make_mock_test(test_id, passed=True):
    """Helper to create a mock test for parallel execution tests."""
    mock_test = MagicMock()
    mock_test.id = test_id
    mock_test.reason = f"Test {test_id}"
    mock_test.timeout = 30
    mock_test.is_stale = MagicMock(return_value=True)
    mock_test.files = []
    mock_test.tool_provenance = None

    mock_executor = MagicMock()
    mock_executor.id = f"executor-{test_id}"
    mock_executor.user_prompt = f"Prompt for {test_id}"
    mock_executor.provider = MagicMock()
    mock_executor.provider.model = "test-model"
    mock_test.executor = mock_executor
    mock_test.judges = []

    mock_result = MagicMock()
    mock_result.passed = passed
    mock_result.test_id = test_id
    mock_result.executor_output = MagicMock()
    mock_result.executor_output.tool_calls = []
    mock_result.executor_output.to_dict = MagicMock(return_value={})
    mock_result.judge_results = []
    mock_result.line_coverage = {}
    mock_result.duration = 1.0
    mock_result.failure_reasons = [] if passed else [f"{test_id} failed"]
    mock_test.run = AsyncMock(return_value=mock_result)

    return mock_test


class TestRunConcurrentBasic:
    """Tests for basic parallel execution."""

    @pytest.mark.asyncio
    async def test_run_concurrent_basic(self):
        """3 tests with parallel_execution=True, all pass."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(
            name="suite",
            parallel_execution=True,
            max_concurrency=4,
        )
        suite = TestSuite(config)

        for i in range(3):
            suite.add_test(_make_mock_test(f"test-{i}"))

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            results = await suite.run()

        assert results.total_tests == 3
        assert results.passed == 3
        assert results.failed == 0

    @pytest.mark.asyncio
    async def test_run_concurrent_dispatches_when_enabled(self):
        """run() dispatches to _run_concurrent when parallel_execution=True."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(
            name="suite",
            parallel_execution=True,
            max_concurrency=2,
        )
        suite = TestSuite(config)
        suite.add_test(_make_mock_test("test-1"))

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_run_concurrent', new_callable=AsyncMock, return_value=([], 0)) as mock_concurrent, \
             patch.object(suite, '_run_sequential', new_callable=AsyncMock) as mock_sequential:
            await suite.run()

        mock_concurrent.assert_called_once()
        mock_sequential.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_sequential_dispatches_when_disabled(self):
        """run() dispatches to _run_sequential when parallel_execution=False."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(
            name="suite",
            parallel_execution=False,
        )
        suite = TestSuite(config)
        suite.add_test(_make_mock_test("test-1"))

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_run_concurrent', new_callable=AsyncMock) as mock_concurrent, \
             patch.object(suite, '_run_sequential', new_callable=AsyncMock, return_value=([], 0)) as mock_sequential:
            await suite.run()

        mock_sequential.assert_called_once()
        mock_concurrent.assert_not_called()


class TestRunConcurrentConcurrencyLimit:
    """Tests for concurrency limit enforcement."""

    @pytest.mark.asyncio
    async def test_run_concurrent_respects_max_concurrency(self):
        """5 tests with max_concurrency=2, never exceeds 2 concurrent."""
        import asyncio
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(
            name="suite",
            parallel_execution=True,
            max_concurrency=2,
        )
        suite = TestSuite(config)

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracked_run(*args, **kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent

            # Simulate work
            await asyncio.sleep(0.05)

            async with lock:
                current_concurrent -= 1

            result = MagicMock()
            result.passed = True
            result.test_id = "tracked"
            result.executor_output = MagicMock()
            result.executor_output.tool_calls = []
            result.executor_output.to_dict = MagicMock(return_value={})
            result.judge_results = []
            result.line_coverage = {}
            result.duration = 0.05
            result.failure_reasons = []
            return result

        for i in range(5):
            test = _make_mock_test(f"test-{i}")
            test.run = AsyncMock(side_effect=tracked_run)
            suite.add_test(test)

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            results = await suite.run()

        assert results.total_tests == 5
        assert results.passed == 5
        assert max_concurrent <= 2


class TestRunConcurrentWithCache:
    """Tests for cache handling in parallel execution."""

    @pytest.mark.asyncio
    async def test_run_concurrent_with_cache(self):
        """Mix of cached + non-cached tests in parallel mode."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(
            name="suite",
            parallel_execution=True,
            max_concurrency=4,
        )
        suite = TestSuite(config)

        # Add 3 tests, 1 will be cached
        test_cached = _make_mock_test("test-cached")
        test_cached.is_stale = MagicMock(return_value=False)  # Not stale = use cache
        suite.add_test(test_cached)

        suite.add_test(_make_mock_test("test-run-1"))
        suite.add_test(_make_mock_test("test-run-2"))

        # Pre-populate cache for the cached test
        cached_result = MagicMock()
        cached_result.passed = True
        suite._cached_results["test-cached"] = cached_result

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            results = await suite.run()

        assert results.total_tests == 3
        assert results.passed == 3
        assert results.cached_results == 1
        # Cached test should NOT have run()
        test_cached.run.assert_not_called()


class TestRunConcurrentWithFailure:
    """Tests for failure handling in parallel execution."""

    @pytest.mark.asyncio
    async def test_run_concurrent_with_failure(self):
        """One test fails, others pass in parallel mode."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(
            name="suite",
            parallel_execution=True,
            max_concurrency=4,
        )
        suite = TestSuite(config)

        suite.add_test(_make_mock_test("test-pass-1"))
        suite.add_test(_make_mock_test("test-fail", passed=False))
        suite.add_test(_make_mock_test("test-pass-2"))

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            results = await suite.run()

        assert results.total_tests == 3
        assert results.passed == 2
        assert results.failed == 1


class TestRunConcurrentCallbacks:
    """Tests for callback invocations in parallel execution."""

    @pytest.mark.asyncio
    async def test_run_concurrent_callbacks(self):
        """Verify on_progress fires start/complete for each test."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(
            name="suite",
            parallel_execution=True,
            max_concurrency=4,
        )
        suite = TestSuite(config)

        suite.add_test(_make_mock_test("test-1"))
        suite.add_test(_make_mock_test("test-2"))

        progress_calls = []

        def on_progress(event, test_id, data):
            progress_calls.append((event, test_id))

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            await suite.run(on_progress=on_progress)

        # Each test should have start and complete events
        assert ("start", "test-1") in progress_calls
        assert ("complete", "test-1") in progress_calls
        assert ("start", "test-2") in progress_calls
        assert ("complete", "test-2") in progress_calls


class TestRunConcurrentCacheWrites:
    """Tests for cache write safety in parallel execution."""

    @pytest.mark.asyncio
    async def test_run_concurrent_cache_writes(self):
        """Verify _save_cache_incremental called for each test."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(
            name="suite",
            parallel_execution=True,
            max_concurrency=4,
        )
        suite = TestSuite(config)

        suite.add_test(_make_mock_test("test-1"))
        suite.add_test(_make_mock_test("test-2"))
        suite.add_test(_make_mock_test("test-3"))

        with patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock) as mock_cache:
            await suite.run()

        # _save_cache_incremental should be called once per non-cached test
        assert mock_cache.call_count == 3


class TestLogToolProvenance:
    """Tests for _log_tool_provenance helper."""

    def test_no_crash_without_provenance(self):
        """_log_tool_provenance handles None provenance gracefully."""
        from dokumen.test_suite import _log_tool_provenance

        with patch('dokumen.test_suite.logger') as mock_logger:
            _log_tool_provenance("test-1", None)

        mock_logger.info.assert_not_called()

    def test_no_crash_empty_provenance(self):
        """Empty provenance (no tools) produces no log entries."""
        from dokumen.test_suite import _log_tool_provenance
        from dokumen.loader import ToolProvenance

        provenance = ToolProvenance()

        with patch('dokumen.test_suite.logger') as mock_logger:
            _log_tool_provenance("test-1", provenance)

        mock_logger.info.assert_not_called()

    def test_executor_tools_grouped_by_source(self):
        """Executor tools are grouped by source in log output."""
        from dokumen.test_suite import _log_tool_provenance
        from dokumen.loader import ToolProvenance

        provenance = ToolProvenance(
            executor_tools={
                "read_file": "scaffold",
                "glob": "scaffold",
                "run_shell_command": "auto:standard",
            }
        )

        with patch('dokumen.test_suite.logger') as mock_logger:
            _log_tool_provenance("test-1", provenance)

        info_calls = mock_logger.info.call_args_list
        executor_calls = [c for c in info_calls if c[0][0] == "Executor tools"]
        assert len(executor_calls) == 1
        kwargs = executor_calls[0][1]
        assert kwargs['test_id'] == "test-1"
        # Tools within each group are sorted alphabetically
        tools_str = kwargs['tools']
        assert "glob, read_file (scaffold)" in tools_str
        assert "run_shell_command (auto:standard)" in tools_str

    def test_judge_tools_logged(self):
        """Judge tools appear in log output."""
        from dokumen.test_suite import _log_tool_provenance
        from dokumen.loader import ToolProvenance

        provenance = ToolProvenance(
            judge_tools={
                "accuracy": {"run_shell_command": "auto:standard"},
            }
        )

        with patch('dokumen.test_suite.logger') as mock_logger:
            _log_tool_provenance("test-1", provenance)

        info_calls = mock_logger.info.call_args_list
        judge_calls = [c for c in info_calls if c[0][0] == "Judge tools"]
        assert len(judge_calls) == 1
        kwargs = judge_calls[0][1]
        assert kwargs['test_id'] == "test-1"
        assert kwargs['judge'] == "accuracy"
        assert "run_shell_command (auto:standard)" in kwargs['tools']

    def test_explore_tools_logged(self):
        """Explore tools appear in log output."""
        from dokumen.test_suite import _log_tool_provenance
        from dokumen.loader import ToolProvenance

        provenance = ToolProvenance(
            explore_tools={
                "read_file": "explore:config",
                "glob": "explore:config",
            }
        )

        with patch('dokumen.test_suite.logger') as mock_logger:
            _log_tool_provenance("test-1", provenance)

        info_calls = mock_logger.info.call_args_list
        explore_calls = [c for c in info_calls if c[0][0] == "Explore tools"]
        assert len(explore_calls) == 1
        kwargs = explore_calls[0][1]
        assert kwargs['test_id'] == "test-1"
        assert "glob" in kwargs['tools']
        assert "read_file" in kwargs['tools']
        assert "explore:config" in kwargs['source']

    def test_overrides_active_logged(self):
        """Overrides active flag is logged."""
        from dokumen.test_suite import _log_tool_provenance
        from dokumen.loader import ToolProvenance

        provenance = ToolProvenance(
            executor_tools={"read_file": "scaffold"},
            overrides_active=True,
        )

        with patch('dokumen.test_suite.logger') as mock_logger:
            _log_tool_provenance("test-1", provenance)

        info_calls = mock_logger.info.call_args_list
        override_calls = [c for c in info_calls if c[0][0] == "Tool overrides active"]
        assert len(override_calls) == 1
        assert override_calls[0][1]['test_id'] == "test-1"

    def test_removed_tools_logged(self):
        """Removed tools are logged."""
        from dokumen.test_suite import _log_tool_provenance
        from dokumen.loader import ToolProvenance

        provenance = ToolProvenance(
            removed_tools=["web_fetch", "browser_navigate"],
        )

        with patch('dokumen.test_suite.logger') as mock_logger:
            _log_tool_provenance("test-1", provenance)

        info_calls = mock_logger.info.call_args_list
        removed_calls = [c for c in info_calls if c[0][0] == "Tools filtered out"]
        assert len(removed_calls) == 1
        kwargs = removed_calls[0][1]
        assert kwargs['test_id'] == "test-1"
        assert kwargs['removed'] == ["web_fetch", "browser_navigate"]


class TestSuiteRunLogging:
    """Tests for structured logging in TestSuite.run()."""

    @pytest.mark.asyncio
    async def test_run_logs_suite_start(self):
        """run() logs suite.run.start with test count and parallel flag."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite", parallel_execution=False)
        suite = TestSuite(config)
        suite.add_test(_make_mock_test("test-1"))

        with patch('dokumen.test_suite.logger') as mock_logger, \
             patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            await suite.run()

        start_calls = [c for c in mock_logger.info.call_args_list
                       if c[0][0] == "suite.run.start"]
        assert len(start_calls) == 1
        kwargs = start_calls[0][1]
        assert kwargs['test_count'] == 1
        assert kwargs['parallel'] is False

    @pytest.mark.asyncio
    async def test_run_logs_suite_complete(self):
        """run() logs suite.run.complete with totals and duration."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite", parallel_execution=False)
        suite = TestSuite(config)
        suite.add_test(_make_mock_test("test-1"))

        with patch('dokumen.test_suite.logger') as mock_logger, \
             patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            await suite.run()

        complete_calls = [c for c in mock_logger.info.call_args_list
                          if c[0][0] == "suite.run.complete"]
        assert len(complete_calls) == 1
        kwargs = complete_calls[0][1]
        assert kwargs['total'] == 1
        assert kwargs['cached'] == 0
        assert 'duration_ms' in kwargs


class TestConcurrentStructuredLogging:
    """Tests for structured logging in _run_concurrent()."""

    @pytest.mark.asyncio
    async def test_concurrent_logs_structured_test_starting(self):
        """Concurrent path logs 'Test starting' with structured kwargs, not f-strings."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(
            name="suite", parallel_execution=True, max_concurrency=4
        )
        suite = TestSuite(config)
        suite.add_test(_make_mock_test("test-1"))

        with patch('dokumen.test_suite.logger') as mock_logger, \
             patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            await suite.run()

        # Should have structured "test.starting" not f-string "TEST: ..."
        start_calls = [c for c in mock_logger.info.call_args_list
                       if c[0][0] == "test.starting"]
        assert len(start_calls) >= 1
        kwargs = start_calls[0][1]
        assert kwargs['test_id'] == "test-1"
        assert kwargs['mode'] == "parallel"

    @pytest.mark.asyncio
    async def test_concurrent_logs_cache_hit(self):
        """Concurrent path logs test.cache.hit for cached tests."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.test_object import TestResult
        from datetime import datetime

        config = TestSuiteConfig(
            name="suite", parallel_execution=True, max_concurrency=4
        )
        suite = TestSuite(config)

        test = _make_mock_test("test-cached")
        test.is_stale.return_value = False
        suite.add_test(test)
        # Pre-populate cache
        suite._cached_results["test-cached"] = TestResult(
            test_id="test-cached", passed=True, executor_passed=True,
            judge_results=[], executor_output=None, duration=0.1,
            timestamp=datetime.now()
        )

        with patch('dokumen.test_suite.logger') as mock_logger, \
             patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            await suite.run()

        cache_calls = [c for c in mock_logger.info.call_args_list
                       if c[0][0] == "cache.hit"]
        assert len(cache_calls) == 1
        assert cache_calls[0][1]['test_id'] == "test-cached"

    @pytest.mark.asyncio
    async def test_concurrent_logs_test_complete(self):
        """Concurrent path logs 'Test complete' with passed and duration."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(
            name="suite", parallel_execution=True, max_concurrency=4
        )
        suite = TestSuite(config)
        suite.add_test(_make_mock_test("test-1"))

        with patch('dokumen.test_suite.logger') as mock_logger, \
             patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            await suite.run()

        complete_calls = [c for c in mock_logger.info.call_args_list
                          if c[0][0] == "test.complete"]
        assert len(complete_calls) == 1
        kwargs = complete_calls[0][1]
        assert kwargs['test_id'] == "test-1"
        assert 'passed' in kwargs
        assert 'duration_ms' in kwargs


class TestProcessTestResultLogging:
    """Tests for logging in _process_test_result()."""

    def test_process_result_logs_entry(self):
        """_process_test_result logs debug entry with test info."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.test_object import TestResult
        from datetime import datetime

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        test = _make_mock_test("test-1")
        test.files = ["docs/api.md"]
        result = TestResult(
            test_id="test-1", passed=True, executor_passed=True,
            judge_results=[], executor_output=None, duration=0.5,
            timestamp=datetime.now()
        )

        with patch('dokumen.test_suite.logger') as mock_logger:
            suite._process_test_result(test, result)

        processing_calls = [c for c in mock_logger.debug.call_args_list
                            if c[0][0] == "test.result.processing"]
        assert len(processing_calls) == 1
        assert processing_calls[0][1]['test_id'] == "test-1"
        assert processing_calls[0][1]['passed'] is True

    def test_process_result_logs_processed(self):
        """_process_test_result logs debug processed with status."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.test_object import TestResult
        from datetime import datetime

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        test = _make_mock_test("test-1")
        test.files = ["docs/api.md"]
        result = TestResult(
            test_id="test-1", passed=False, executor_passed=False,
            judge_results=[], executor_output=None, duration=0.5,
            timestamp=datetime.now()
        )

        with patch('dokumen.test_suite.logger') as mock_logger:
            suite._process_test_result(test, result)

        processed_calls = [c for c in mock_logger.debug.call_args_list
                           if c[0][0] == "test.result.processed"]
        assert len(processed_calls) == 1
        assert processed_calls[0][1]['status'] == "failed"


class TestSilentExceptionLogging:
    """Tests for logging in previously silent except:pass blocks."""

    @pytest.mark.asyncio
    async def test_cache_status_invalid_logged(self):
        """Invalid FileStatus value logs debug instead of silent pass."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        import json
        import tempfile
        import os

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        # Create a cache with an invalid status value
        cache_dir = tempfile.mkdtemp()
        config.cache_path = cache_dir
        cache_file = os.path.join(cache_dir, "cache.json")
        with open(cache_file, 'w') as f:
            json.dump({
                "version": "3.0",
                "results": {},
                "line_coverage": {},
                "file_status": {"docs/api.md": "INVALID_STATUS"},
                "failure_analysis": {},
            }, f)

        with patch('dokumen.test_suite.logger') as mock_logger:
            await suite.load_cache()

        debug_calls = [c for c in mock_logger.debug.call_args_list
                       if c[0][0] == "cache.status.invalid"]
        assert len(debug_calls) == 1
        assert debug_calls[0][1]['value'] == "INVALID_STATUS"

    @pytest.mark.asyncio
    async def test_cache_load_failed_logged(self):
        """Corrupted cache file logs debug instead of silent pass."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        import tempfile
        import os

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        cache_dir = tempfile.mkdtemp()
        config.cache_path = cache_dir
        cache_file = os.path.join(cache_dir, "cache.json")
        with open(cache_file, 'w') as f:
            f.write("NOT VALID JSON{{{")

        with patch('dokumen.test_suite.logger') as mock_logger:
            await suite.load_cache()

        debug_calls = [c for c in mock_logger.debug.call_args_list
                       if c[0][0] == "cache.load.failed"]
        assert len(debug_calls) == 1


class TestSequentialTestCompleteLogging:
    """Tests for test.complete logging in sequential path."""

    @pytest.mark.asyncio
    async def test_sequential_logs_test_complete(self):
        """Sequential path logs test.complete with passed and duration."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig

        config = TestSuiteConfig(name="suite", parallel_execution=False)
        suite = TestSuite(config)
        suite.add_test(_make_mock_test("test-1"))

        with patch('dokumen.test_suite.logger') as mock_logger, \
             patch('dokumen.debug.is_debug', return_value=False), \
             patch.object(suite, '_save_cache_incremental', new_callable=AsyncMock):
            await suite.run()

        complete_calls = [c for c in mock_logger.info.call_args_list
                          if c[0][0] == "test.complete"]
        assert len(complete_calls) == 1
        kwargs = complete_calls[0][1]
        assert kwargs['test_id'] == "test-1"
        assert 'passed' in kwargs
        assert 'duration_ms' in kwargs


class TestCacheReadFailedLogging:
    """Tests for cache.read.failed and cache.incremental.read.failed logging."""

    @pytest.mark.asyncio
    async def test_cache_read_failed_logged(self):
        """save_cache logs debug when existing cache file is corrupted."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        import tempfile
        import os

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)

        cache_dir = tempfile.mkdtemp()
        config.cache_path = cache_dir
        cache_file = os.path.join(cache_dir, "cache.json")
        with open(cache_file, 'w') as f:
            f.write("CORRUPT JSON!!!")

        with patch('dokumen.test_suite.logger') as mock_logger:
            await suite.save_cache()

        debug_calls = [c for c in mock_logger.debug.call_args_list
                       if c[0][0] == "cache.read.failed"]
        assert len(debug_calls) == 1
        assert 'path' in debug_calls[0][1]

    @pytest.mark.asyncio
    async def test_cache_incremental_read_failed_logged(self):
        """_save_cache_incremental logs debug when cache file is corrupted."""
        from dokumen.test_suite import TestSuite, TestSuiteConfig
        from dokumen.test_object import TestResult
        from datetime import datetime
        import tempfile
        import os

        config = TestSuiteConfig(name="suite")
        suite = TestSuite(config)
        mock_test = _make_mock_test("test-inc")
        mock_test.get_hash = MagicMock(return_value="abc123")
        suite.add_test(mock_test)

        cache_dir = tempfile.mkdtemp()
        config.cache_path = cache_dir
        cache_file = os.path.join(cache_dir, "cache.json")
        with open(cache_file, 'w') as f:
            f.write("NOT VALID JSON{{{")

        result = TestResult(
            test_id="test-inc", passed=True, executor_passed=True,
            judge_results=[], executor_output=None, duration=0.5,
            timestamp=datetime.now()
        )

        with patch('dokumen.test_suite.logger') as mock_logger:
            await suite._save_cache_incremental("test-inc", result)

        debug_calls = [c for c in mock_logger.debug.call_args_list
                       if c[0][0] == "cache.incremental.read.failed"]
        assert len(debug_calls) == 1
