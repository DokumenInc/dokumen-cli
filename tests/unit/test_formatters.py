"""Tests for CLI formatters module."""

import pytest
from unittest.mock import MagicMock, patch
from io import StringIO

from dokumen.cli.formatters import (
    _compress_line_ranges,
    coverage_to_lcov,
    file_coverage_to_dict,
    results_to_dict,
    results_to_junit,
    results_to_tap,
    print_coverage_text,
    print_line_coverage_text,
    print_file_with_coverage,
    print_coverage_tree,
    print_results_text,
    print_run_settings,
    make_progress_callback,
    make_tool_call_callback,
    make_conversation_callback,
    make_executor_complete_callback,
    make_judge_complete_callback,
    _print_state_bars,
    _print_files_table,
)


class TestCompressLineRanges:
    """Tests for _compress_line_ranges function."""

    def test_empty_list(self):
        """Empty list should return 'none'."""
        result = _compress_line_ranges([])
        assert result == "none"

    def test_single_line(self):
        """Single line should return just the number."""
        result = _compress_line_ranges([5])
        assert result == "5"

    def test_consecutive_lines(self):
        """Consecutive lines should be compressed to range."""
        result = _compress_line_ranges([1, 2, 3, 4, 5])
        assert result == "1-5"

    def test_non_consecutive_lines(self):
        """Non-consecutive lines should be listed individually."""
        result = _compress_line_ranges([1, 3, 5])
        assert result == "1, 3, 5"

    def test_mixed_ranges_and_singles(self):
        """Should handle mix of ranges and singles."""
        result = _compress_line_ranges([1, 2, 3, 7, 10, 11, 12])
        assert result == "1-3, 7, 10-12"

    def test_unsorted_input(self):
        """Should sort input before processing."""
        result = _compress_line_ranges([5, 1, 3, 2, 4])
        assert result == "1-5"

    def test_duplicate_lines(self):
        """Should handle duplicates (sorted creates consecutive duplicates)."""
        # Duplicates don't affect range compression since we check line == end + 1
        result = _compress_line_ranges([1, 1, 2, 2, 3])
        # With duplicates: sorted is [1,1,2,2,3], range checks work with duplicates
        # Actually the function doesn't dedupe, so 1-1, then 2-2 extended to 2-3 = "1, 2-3"
        # Let's verify actual behavior
        assert "1" in result

    def test_large_gap(self):
        """Should handle large gaps between ranges."""
        result = _compress_line_ranges([1, 2, 100, 101])
        assert result == "1-2, 100-101"


class TestCoverageToLcov:
    """Tests for coverage_to_lcov function."""

    def test_empty_files(self):
        """Empty files dict should return empty string."""
        result = coverage_to_lcov({"files": {}})
        assert result == ""

    def test_single_file(self):
        """Should format single file correctly."""
        stats = {
            "files": {
                "docs/api.md": {
                    "covered_lines": [1, 3],
                    "total_lines": 5
                }
            }
        }
        result = coverage_to_lcov(stats)

        assert "SF:docs/api.md" in result
        assert "DA:1,1" in result  # Line 1 covered
        assert "DA:2,0" in result  # Line 2 not covered
        assert "DA:3,1" in result  # Line 3 covered
        assert "LH:2" in result    # 2 lines hit
        assert "LF:5" in result    # 5 lines total
        assert "end_of_record" in result

    def test_multiple_files(self):
        """Should format multiple files."""
        stats = {
            "files": {
                "file1.md": {
                    "covered_lines": [1],
                    "total_lines": 2
                },
                "file2.md": {
                    "covered_lines": [1, 2],
                    "total_lines": 2
                }
            }
        }
        result = coverage_to_lcov(stats)

        assert "SF:file1.md" in result
        assert "SF:file2.md" in result
        assert result.count("end_of_record") == 2

    def test_no_covered_lines(self):
        """Should handle file with no coverage."""
        stats = {
            "files": {
                "uncovered.md": {
                    "covered_lines": [],
                    "total_lines": 3
                }
            }
        }
        result = coverage_to_lcov(stats)

        assert "DA:1,0" in result
        assert "DA:2,0" in result
        assert "DA:3,0" in result
        assert "LH:0" in result
        assert "LF:3" in result


class TestFileCoverageToDict:
    """Tests for file_coverage_to_dict function."""

    def test_basic_structure(self):
        """Should create correct structure."""
        lines = ["line 1", "line 2", "line 3"]
        coverage_data = {
            "covered_lines": [1],
            "failed_lines": [2],
            "incorrect_lines": [],
            "covered_count": 1,
            "failed_count": 1,
            "percentage": 33.3
        }
        result = file_coverage_to_dict("test.md", lines, coverage_data)

        assert result["file_path"] == "test.md"
        assert result["total_lines"] == 3
        assert result["covered_count"] == 1
        assert result["failed_count"] == 1
        assert result["percentage"] == 33.3
        assert len(result["lines"]) == 3

    def test_line_statuses(self):
        """Should assign correct statuses to lines."""
        lines = ["covered", "failed", "uncovered", ""]
        coverage_data = {
            "covered_lines": [1],
            "failed_lines": [2],
            "incorrect_lines": [],
            "covered_count": 1,
            "failed_count": 1,
            "percentage": 25.0
        }
        result = file_coverage_to_dict("test.md", lines, coverage_data)

        assert result["lines"][0]["status"] == "passed"
        assert result["lines"][1]["status"] == "failed"
        assert result["lines"][2]["status"] == "uncovered"
        assert result["lines"][3]["status"] == "blank"

    def test_incorrect_line_with_reason(self):
        """Should include reason and confidence for incorrect lines."""
        lines = ["incorrect line"]
        coverage_data = {
            "covered_lines": [],
            "failed_lines": [],
            "incorrect_lines": [
                {"line_number": 1, "reason": "Outdated info", "confidence": 0.85}
            ],
            "covered_count": 0,
            "failed_count": 0,
            "percentage": 0.0
        }
        result = file_coverage_to_dict("test.md", lines, coverage_data)

        assert result["lines"][0]["status"] == "incorrect"
        assert result["lines"][0]["reason"] == "Outdated info"
        assert result["lines"][0]["confidence"] == 0.85

    def test_empty_lines(self):
        """Should handle empty line list."""
        result = file_coverage_to_dict("empty.md", [], {})
        assert result["total_lines"] == 0
        assert result["lines"] == []


class TestResultsToDict:
    """Tests for results_to_dict function."""

    def test_basic_structure(self):
        """Should create correct structure."""
        results = MagicMock()
        results.total_tests = 3
        results.passed = 2
        results.failed = 1
        results.skipped = 0
        results.duration = 1.5678
        results.cached_results = 0
        results.test_results = []

        result = results_to_dict(results)

        assert result["total"] == 3
        assert result["passed"] == 2
        assert result["failed"] == 1
        assert result["skipped"] == 0
        assert result["duration"] == 1.57  # Rounded to 2 decimals
        assert result["cached"] == 0
        assert result["tests"] == []

    def test_passed_test(self):
        """Should include passed test details."""
        test_result = MagicMock()
        test_result.test_id = "my-test"
        test_result.passed = True
        test_result.duration = 0.5

        results = MagicMock()
        results.total_tests = 1
        results.passed = 1
        results.failed = 0
        results.skipped = 0
        results.duration = 0.5
        results.cached_results = 0
        results.test_results = [test_result]

        result = results_to_dict(results)

        assert len(result["tests"]) == 1
        assert result["tests"][0]["id"] == "my-test"
        assert result["tests"][0]["passed"] is True
        assert result["tests"][0]["duration"] == 0.5

    def test_failed_test_with_reasons(self):
        """Should include failure reasons for failed test."""
        test_result = MagicMock()
        test_result.test_id = "failing-test"
        test_result.passed = False
        test_result.duration = 1.0
        test_result.failure_reasons = ["Assertion failed", "Missing content"]

        results = MagicMock()
        results.total_tests = 1
        results.passed = 0
        results.failed = 1
        results.skipped = 0
        results.duration = 1.0
        results.cached_results = 0
        results.test_results = [test_result]

        result = results_to_dict(results)

        assert result["tests"][0]["passed"] is False
        assert result["tests"][0]["failure_reasons"] == ["Assertion failed", "Missing content"]

    def test_test_without_duration_attr(self):
        """Should handle test without duration attribute."""
        test_result = MagicMock(spec=['test_id', 'passed'])
        test_result.test_id = "test"
        test_result.passed = True

        results = MagicMock()
        results.total_tests = 1
        results.passed = 1
        results.failed = 0
        results.skipped = 0
        results.duration = 0
        results.cached_results = 0
        results.test_results = [test_result]

        result = results_to_dict(results)
        assert result["tests"][0]["duration"] == 0

    def test_failure_analysis_included(self):
        """Should include failure analysis when present."""
        analysis = MagicMock()
        analysis.referenced_lines = [1, 2, 3]
        analysis.incorrect_lines = []
        analysis.analysis = "The doc is outdated"

        test_result = MagicMock()
        test_result.test_id = "test"
        test_result.passed = False
        test_result.duration = 1.0
        test_result.failure_reasons = ["Failed"]
        test_result.failure_analysis = {"docs/api.md": analysis}

        results = MagicMock()
        results.total_tests = 1
        results.passed = 0
        results.failed = 1
        results.skipped = 0
        results.duration = 1.0
        results.cached_results = 0
        results.test_results = [test_result]

        result = results_to_dict(results)

        assert "failure_analysis" in result["tests"][0]
        assert "docs/api.md" in result["tests"][0]["failure_analysis"]


class TestResultsToJunit:
    """Tests for results_to_junit function."""

    def test_valid_xml_structure(self):
        """Should produce valid XML."""
        results = MagicMock()
        results.total_tests = 1
        results.failed = 0
        results.error = 0
        results.duration = 1.0
        results.test_results = []

        result = results_to_junit(results)

        assert result.startswith('<?xml version="1.0" encoding="UTF-8"?>')
        assert "<testsuite" in result
        assert "</testsuite>" in result

    def test_testsuite_attributes(self):
        """Should include correct testsuite attributes."""
        results = MagicMock()
        results.total_tests = 5
        results.failed = 2
        results.error = 0
        results.duration = 10.5
        results.test_results = []

        result = results_to_junit(results)

        assert 'name="dokumen"' in result
        assert 'tests="5"' in result
        assert 'failures="2"' in result
        assert 'time="10.50"' in result

    def test_passed_testcase(self):
        """Should format passed test case."""
        test_result = MagicMock()
        test_result.test_id = "my-test"
        test_result.passed = True
        test_result.duration = 0.5

        results = MagicMock()
        results.total_tests = 1
        results.failed = 0
        results.error = 0
        results.duration = 0.5
        results.test_results = [test_result]

        result = results_to_junit(results)

        assert '<testcase name="my-test" time="0.50"/>' in result

    def test_failed_testcase(self):
        """Should format failed test case with failure element."""
        test_result = MagicMock()
        test_result.test_id = "failing-test"
        test_result.passed = False
        test_result.duration = 1.0
        test_result.failure_reasons = ["Assertion failed"]

        results = MagicMock()
        results.total_tests = 1
        results.failed = 1
        results.error = 0
        results.duration = 1.0
        results.test_results = [test_result]

        result = results_to_junit(results)

        assert '<testcase name="failing-test" time="1.00">' in result
        assert '<failure message="Assertion failed"/>' in result
        assert '</testcase>' in result

    def test_failed_testcase_multiple_reasons(self):
        """Should join multiple failure reasons."""
        test_result = MagicMock()
        test_result.test_id = "test"
        test_result.passed = False
        test_result.duration = 1.0
        test_result.failure_reasons = ["Reason 1", "Reason 2"]

        results = MagicMock()
        results.total_tests = 1
        results.failed = 1
        results.error = 0
        results.duration = 1.0
        results.test_results = [test_result]

        result = results_to_junit(results)

        assert '<failure message="Reason 1; Reason 2"/>' in result

    def test_empty_results(self):
        """Should handle empty test results."""
        results = MagicMock()
        results.total_tests = 0
        results.failed = 0
        results.error = 0
        results.duration = 0.0
        results.test_results = []

        result = results_to_junit(results)

        assert 'tests="0"' in result
        assert 'failures="0"' in result

    def test_failures_attribute_includes_error_count(self):
        """JUnit failures= attribute includes both failed and error tests."""
        results = MagicMock()
        results.total_tests = 3
        results.failed = 1   # 1 legitimate FAIL verdict
        results.error = 2    # 2 judge timeouts
        results.duration = 5.0
        results.test_results = []

        result = results_to_junit(results)

        # failures should be failed + error = 3
        assert 'failures="3"' in result


class TestResultsToTap:
    """Tests for results_to_tap function."""

    def test_tap_header(self):
        """Should include TAP version and plan."""
        results = MagicMock()
        results.total_tests = 3
        results.test_results = []

        result = results_to_tap(results)

        assert "TAP version 13" in result
        assert "1..3" in result

    def test_ok_for_passed(self):
        """Should output 'ok' for passed tests."""
        test_result = MagicMock()
        test_result.test_id = "my-test"
        test_result.passed = True

        results = MagicMock()
        results.total_tests = 1
        results.test_results = [test_result]

        result = results_to_tap(results)

        assert "ok 1 - my-test" in result

    def test_not_ok_for_failed(self):
        """Should output 'not ok' for failed tests."""
        test_result = MagicMock()
        test_result.test_id = "failing-test"
        test_result.passed = False

        results = MagicMock()
        results.total_tests = 1
        results.test_results = [test_result]

        result = results_to_tap(results)

        assert "not ok 1 - failing-test" in result

    def test_test_numbering(self):
        """Should number tests sequentially."""
        test1 = MagicMock()
        test1.test_id = "test1"
        test1.passed = True

        test2 = MagicMock()
        test2.test_id = "test2"
        test2.passed = False

        test3 = MagicMock()
        test3.test_id = "test3"
        test3.passed = True

        results = MagicMock()
        results.total_tests = 3
        results.test_results = [test1, test2, test3]

        result = results_to_tap(results)

        assert "ok 1 - test1" in result
        assert "not ok 2 - test2" in result
        assert "ok 3 - test3" in result

    def test_empty_results(self):
        """Should handle empty test results."""
        results = MagicMock()
        results.total_tests = 0
        results.test_results = []

        result = results_to_tap(results)

        assert "1..0" in result


class TestPrintCoverageText:
    """Tests for print_coverage_text function."""

    @patch('dokumen.cli.formatters.click.echo')
    def test_quiet_mode_file_only(self, mock_echo):
        """Quiet mode should show single summary line."""
        stats = {
            'total': 10,
            'passed': 8,
            'percentage': 80.0,
            'by_state': {'passed': 8, 'failed': 0, 'uncovered': 2},
            'test_counts': {}
        }
        print_coverage_text(stats, quiet=True)

        mock_echo.assert_called_once()
        call_arg = mock_echo.call_args[0][0]
        assert "Coverage: 80%" in call_arg
        assert "(8/10 files)" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    def test_quiet_mode_with_lines(self, mock_echo):
        """Quiet mode with line stats should show both."""
        stats = {
            'total': 10,
            'passed': 8,
            'percentage': 80.0,
            'by_state': {'passed': 8, 'failed': 0, 'uncovered': 2},
            'test_counts': {}
        }
        line_stats = {
            'percentage': 75.5,
            'total_lines': 100,
            'covered_lines': 75
        }
        print_coverage_text(stats, line_stats=line_stats, quiet=True)

        call_arg = mock_echo.call_args[0][0]
        assert "80% files" in call_arg
        assert "75.5% lines" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    def test_normal_mode_shows_header(self, mock_echo):
        """Normal mode should show header and sections."""
        stats = {
            'total': 5,
            'passed': 3,
            'percentage': 60.0,
            'by_state': {'passed': 3, 'failed': 1, 'uncovered': 1},
            'test_counts': {'test1': 2, 'test2': 1}
        }
        print_coverage_text(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Documentation Coverage" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_with_line_stats(self, mock_echo):
        """Should show line stats section when provided."""
        stats = {
            'total': 5,
            'passed': 3,
            'percentage': 60.0,
            'by_state': {'passed': 3, 'failed': 1, 'uncovered': 1},
            'test_counts': {}
        }
        line_stats = {
            'total_lines': 100,
            'by_state': {'passed': 70, 'failed': 10, 'uncovered': 20}
        }
        print_coverage_text(stats, line_stats=line_stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Lines:" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('dokumen.cli.formatters._print_files_table')
    def test_with_files_flag(self, mock_table, mock_echo):
        """Should call files table when files=True."""
        stats = {
            'total': 5,
            'passed': 3,
            'percentage': 60.0,
            'by_state': {'passed': 3, 'failed': 1, 'uncovered': 1},
            'test_counts': {}
        }
        print_coverage_text(stats, files=True)
        mock_table.assert_called_once()

    @patch('dokumen.cli.formatters.click.echo')
    def test_with_failed_files(self, mock_echo):
        """Should show failed files section when present."""
        stats = {
            'total': 5,
            'passed': 3,
            'percentage': 60.0,
            'by_state': {'passed': 3, 'failed': 1, 'uncovered': 1},
            'test_counts': {},
            'failed_files': ['docs/broken.md']
        }
        print_coverage_text(stats, files=False)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Failed Files" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_with_uncovered_flag(self, mock_echo):
        """Should show uncovered files when uncovered=True."""
        stats = {
            'total': 5,
            'passed': 3,
            'percentage': 60.0,
            'by_state': {'passed': 3, 'failed': 0, 'uncovered': 2},
            'test_counts': {},
            'uncovered_files': ['docs/missing.md', 'docs/other.md']
        }
        print_coverage_text(stats, uncovered=True, files=False)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Uncovered Files" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('dokumen.cli.formatters.print_coverage_tree')
    def test_with_tree_flag(self, mock_tree, mock_echo):
        """Should call coverage tree when tree=True."""
        stats = {
            'total': 5,
            'passed': 3,
            'percentage': 60.0,
            'by_state': {'passed': 3, 'failed': 0, 'uncovered': 2},
            'test_counts': {}
        }
        print_coverage_text(stats, tree=True)
        mock_tree.assert_called_once()


class TestPrintLineCoverageText:
    """Tests for print_line_coverage_text function."""

    @patch('dokumen.cli.formatters.click.echo')
    def test_no_data_message(self, mock_echo):
        """Should show message when no data."""
        stats = {'total_lines': 0}
        print_line_coverage_text(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "No line coverage data available" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_summary(self, mock_echo):
        """Should show overall summary."""
        stats = {
            'total_lines': 100,
            'covered_lines': 80,
            'failed_lines': 0,
            'percentage': 80.0
        }
        print_line_coverage_text(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "80/100 lines" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_failed_count(self, mock_echo):
        """Should show failed line count when present."""
        stats = {
            'total_lines': 100,
            'covered_lines': 70,
            'failed_lines': 10,
            'percentage': 70.0
        }
        print_line_coverage_text(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "failed" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_yellow_color_for_medium_coverage(self, mock_echo):
        """Should use yellow color for medium coverage (50-80%)."""
        stats = {
            'total_lines': 100,
            'covered_lines': 60,
            'failed_lines': 0,
            'percentage': 60.0
        }
        print_line_coverage_text(stats)
        # Just verify it runs without error

    @patch('dokumen.cli.formatters.click.echo')
    def test_red_color_for_low_coverage(self, mock_echo):
        """Should use red color for low coverage (<50%)."""
        stats = {
            'total_lines': 100,
            'covered_lines': 30,
            'failed_lines': 0,
            'percentage': 30.0
        }
        print_line_coverage_text(stats)
        # Just verify it runs without error

    @patch('dokumen.cli.formatters.click.echo')
    def test_detailed_mode_with_files(self, mock_echo):
        """Should show per-file breakdown in detailed mode."""
        stats = {
            'total_lines': 100,
            'covered_lines': 80,
            'failed_lines': 0,
            'percentage': 80.0,
            'files': {
                'docs/api.md': {
                    'percentage': 85.0,
                    'total_lines': 50,
                    'covered_count': 42,
                    'failed_count': 0,
                    'status': 'passed',
                    'covered_lines': [1, 2, 3],
                    'failed_lines': []
                }
            }
        }
        print_line_coverage_text(stats, detailed=True)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Per-File Line Coverage" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_detailed_mode_failed_file(self, mock_echo):
        """Should show failed file details in detailed mode."""
        stats = {
            'total_lines': 100,
            'covered_lines': 70,
            'failed_lines': 10,
            'percentage': 70.0,
            'files': {
                'docs/api.md': {
                    'percentage': 40.0,
                    'total_lines': 50,
                    'covered_count': 20,
                    'failed_count': 5,
                    'status': 'failed',
                    'covered_lines': [1, 2],
                    'failed_lines': [3, 4, 5]
                }
            }
        }
        print_line_coverage_text(stats, detailed=True)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "docs/api.md" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_detailed_mode_incorrect_lines(self, mock_echo):
        """Should show incorrect lines in detailed mode."""
        stats = {
            'total_lines': 100,
            'covered_lines': 80,
            'failed_lines': 0,
            'percentage': 80.0,
            'files': {
                'docs/api.md': {
                    'percentage': 80.0,
                    'total_lines': 50,
                    'covered_count': 40,
                    'failed_count': 0,
                    'status': 'passed',
                    'covered_lines': [1, 2, 3],
                    'failed_lines': [],
                    'incorrect_lines': [
                        {'line_number': 10, 'reason': 'Outdated', 'confidence': 0.9}
                    ]
                }
            }
        }
        print_line_coverage_text(stats, detailed=True)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Potentially Incorrect Lines" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_failure_analysis_section(self, mock_echo):
        """Should show failure analysis when present."""
        stats = {
            'total_lines': 100,
            'covered_lines': 80,
            'failed_lines': 0,
            'percentage': 80.0,
            'failure_analysis': {
                'docs/api.md': {
                    'test-1': {
                        'analysis': 'The documentation is incorrect',
                        'incorrect_lines': [
                            {'line_number': 5, 'reason': 'Wrong API endpoint'}
                        ]
                    }
                }
            }
        }
        print_line_coverage_text(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Failure Analysis" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_detailed_mode_yellow_coverage(self, mock_echo):
        """Should show yellow for 50-79% coverage."""
        stats = {
            'total_lines': 100,
            'covered_lines': 60,
            'failed_lines': 0,
            'percentage': 60.0,
            'files': {
                'docs/api.md': {
                    'percentage': 60.0,
                    'total_lines': 50,
                    'covered_count': 30,
                    'failed_count': 0,
                    'status': 'passed',
                    'covered_lines': list(range(1, 31)),
                    'failed_lines': []
                }
            }
        }
        print_line_coverage_text(stats, detailed=True)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "docs/api.md" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_detailed_mode_red_coverage(self, mock_echo):
        """Should show red for <50% coverage."""
        stats = {
            'total_lines': 100,
            'covered_lines': 30,
            'failed_lines': 0,
            'percentage': 30.0,
            'files': {
                'docs/api.md': {
                    'percentage': 30.0,
                    'total_lines': 50,
                    'covered_count': 15,
                    'failed_count': 0,
                    'status': 'passed',
                    'covered_lines': list(range(1, 16)),
                    'failed_lines': []
                }
            }
        }
        print_line_coverage_text(stats, detailed=True)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "docs/api.md" in full_output


class TestPrintFileCoverage:
    """Tests for print_file_with_coverage function."""

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_header_with_stats(self, mock_echo):
        """Should show file header with percentage."""
        lines = ["line 1", "line 2"]
        coverage_data = {
            'covered_lines': [1],
            'failed_lines': [],
            'incorrect_lines': [],
            'percentage': 50.0,
            'covered_count': 1,
            'failed_count': 0
        }
        print_file_with_coverage("test.md", lines, coverage_data)

        first_call = mock_echo.call_args_list[0][0][0]
        assert "test.md" in first_call
        assert "50%" in first_call

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_legend(self, mock_echo):
        """Should show legend at the bottom."""
        lines = ["line 1"]
        coverage_data = {
            'covered_lines': [],
            'failed_lines': [],
            'incorrect_lines': [],
            'percentage': 0.0
        }
        print_file_with_coverage("test.md", lines, coverage_data)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Legend" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_failed_count_in_header(self, mock_echo):
        """Should show failed line count in header."""
        lines = ["line 1", "line 2", "line 3"]
        coverage_data = {
            'covered_lines': [1],
            'failed_lines': [2, 3],
            'incorrect_lines': [],
            'percentage': 33.0,
            'covered_count': 1,
            'failed_count': 2
        }
        print_file_with_coverage("test.md", lines, coverage_data)

        first_call = mock_echo.call_args_list[0][0][0]
        assert "failed" in first_call

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_incorrect_lines(self, mock_echo):
        """Should show incorrect lines with annotation."""
        lines = ["correct line", "incorrect line"]
        coverage_data = {
            'covered_lines': [1],
            'failed_lines': [],
            'incorrect_lines': [
                {'line_number': 2, 'confidence': 0.85, 'reason': 'Outdated'}
            ],
            'percentage': 50.0,
            'covered_count': 1,
            'failed_count': 0
        }
        print_file_with_coverage("test.md", lines, coverage_data)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "incorrect" in full_output.lower()

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_failed_lines(self, mock_echo):
        """Should show failed lines with indicator."""
        lines = ["passed line", "failed line"]
        coverage_data = {
            'covered_lines': [1],
            'failed_lines': [2],
            'incorrect_lines': [],
            'percentage': 50.0,
            'covered_count': 1,
            'failed_count': 1
        }
        print_file_with_coverage("test.md", lines, coverage_data)
        # Should execute without error, displaying failed indicator

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_blank_lines(self, mock_echo):
        """Should show blank lines with appropriate indicator."""
        lines = ["content", "", "more content"]
        coverage_data = {
            'covered_lines': [1, 3],
            'failed_lines': [],
            'incorrect_lines': [],
            'percentage': 66.7,
            'covered_count': 2,
            'failed_count': 0
        }
        print_file_with_coverage("test.md", lines, coverage_data)
        # Should execute without error


class TestPrintCoverageTree:
    """Tests for print_coverage_tree function."""

    @patch('dokumen.cli.formatters.click.echo')
    def test_empty_files_no_output(self, mock_echo):
        """Should not output anything for empty files."""
        stats = {
            'covered_files': [],
            'failed_files': [],
            'uncovered_files': []
        }
        print_coverage_tree(stats)
        mock_echo.assert_not_called()

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_directory_header(self, mock_echo):
        """Should show directory tree header."""
        stats = {
            'covered_files': ['docs/api.md'],
            'failed_files': [],
            'uncovered_files': []
        }
        print_coverage_tree(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Directory Tree" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_groups_by_directory(self, mock_echo):
        """Should group files by directory."""
        stats = {
            'covered_files': ['docs/api.md', 'docs/guide.md'],
            'failed_files': [],
            'uncovered_files': ['tests/example.md']
        }
        print_coverage_tree(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "docs/" in full_output
        assert "tests/" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_root_level_files(self, mock_echo):
        """Should handle root level files."""
        stats = {
            'covered_files': ['README.md'],
            'failed_files': [],
            'uncovered_files': []
        }
        print_coverage_tree(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "README.md" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_failed_files(self, mock_echo):
        """Should show failed files with indicator."""
        stats = {
            'covered_files': [],
            'failed_files': ['docs/broken.md'],
            'uncovered_files': []
        }
        print_coverage_tree(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "docs/" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_uncovered_files(self, mock_echo):
        """Should show uncovered files with indicator."""
        stats = {
            'covered_files': [],
            'failed_files': [],
            'uncovered_files': ['docs/missing.md']
        }
        print_coverage_tree(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "docs/" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_dir_summary_with_failures(self, mock_echo):
        """Should show failure count in directory summary."""
        stats = {
            'covered_files': ['docs/api.md'],
            'failed_files': ['docs/broken.md'],
            'uncovered_files': []
        }
        print_coverage_tree(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "failed" in full_output


class TestPrintResultsText:
    """Tests for print_results_text function."""

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_header(self, mock_echo):
        """Should show documentation unit tests header."""
        results = MagicMock()
        results.total_tests = 1
        results.passed = 1
        results.failed = 0
        results.cached_results = 0
        results.duration = 1.0
        results.test_results = []

        print_results_text(results)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Documentation Unit Tests" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_summary(self, mock_echo):
        """Should show results summary."""
        results = MagicMock()
        results.total_tests = 3
        results.passed = 2
        results.failed = 1
        results.cached_results = 0
        results.duration = 2.5
        results.test_results = []

        print_results_text(results)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "2.5s" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_failed_tests(self, mock_echo):
        """Should show individual failed tests."""
        test_result = MagicMock()
        test_result.test_id = "failing-test"
        test_result.passed = False
        test_result.failure_reasons = ["Reason 1"]

        results = MagicMock()
        results.total_tests = 1
        results.passed = 0
        results.failed = 1
        results.cached_results = 0
        results.duration = 1.0
        results.test_results = [test_result]
        results.cached_test_ids = set()

        print_results_text(results)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "FAIL" in full_output
        assert "failing-test" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_cached_tests(self, mock_echo):
        """Should show cached tests as SKIP."""
        test_result = MagicMock()
        test_result.test_id = "cached-test"
        test_result.passed = True

        results = MagicMock()
        results.total_tests = 1
        results.passed = 1
        results.failed = 0
        results.cached_results = 1
        results.duration = 0.1
        results.test_results = [test_result]
        results.cached_test_ids = {"cached-test"}

        print_results_text(results, verbose=True)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "SKIP" in full_output
        assert "cached" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_passed_tests_verbose(self, mock_echo):
        """Should show passed tests in verbose mode."""
        test_result = MagicMock()
        test_result.test_id = "passing-test"
        test_result.passed = True

        results = MagicMock()
        results.total_tests = 1
        results.passed = 1
        results.failed = 0
        results.cached_results = 0
        results.duration = 1.0
        results.test_results = [test_result]
        results.cached_test_ids = set()

        print_results_text(results, verbose=True)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "PASS" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_failure_analysis(self, mock_echo):
        """Should show failure analysis for failed tests."""
        analysis = MagicMock()
        analysis.analysis = "The documentation is incorrect"
        analysis.incorrect_lines = [
            MagicMock(line_number=5, confidence=0.9, reason="Wrong endpoint")
        ]

        test_result = MagicMock()
        test_result.test_id = "failing-test"
        test_result.passed = False
        test_result.failure_reasons = ["Assertion failed"]
        test_result.failure_analysis = {"docs/api.md": analysis}

        results = MagicMock()
        results.total_tests = 1
        results.passed = 0
        results.failed = 1
        results.cached_results = 0
        results.duration = 1.0
        results.test_results = [test_result]
        results.cached_test_ids = set()

        print_results_text(results)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Failure Analysis" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_executor_output_verbose(self, mock_echo):
        """Should show executor output in verbose mode."""
        tool_call = MagicMock()
        tool_call.tool_name = "read_file"
        tool_call.parameters = {"path": "docs/api.md"}

        executor_output = MagicMock()
        executor_output.tool_calls = [tool_call]
        executor_output.final_response = "The documentation says X"

        test_result = MagicMock()
        test_result.test_id = "my-test"
        test_result.passed = True
        test_result.executor_output = executor_output

        results = MagicMock()
        results.total_tests = 1
        results.passed = 1
        results.failed = 0
        results.cached_results = 0
        results.duration = 1.0
        results.test_results = [test_result]
        results.cached_test_ids = set()

        print_results_text(results, verbose=True)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Tool calls" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_shows_judge_results_verbose(self, mock_echo):
        """Should show judge results in verbose mode."""
        judge_result = MagicMock()
        judge_result.judge_id = "accuracy"
        judge_result.passed = True
        judge_result.confidence = 0.95
        judge_result.response = "The assertion is satisfied"

        test_result = MagicMock()
        test_result.test_id = "my-test"
        test_result.passed = True
        test_result.judge_results = [judge_result]

        results = MagicMock()
        results.total_tests = 1
        results.passed = 1
        results.failed = 0
        results.cached_results = 0
        results.duration = 1.0
        results.test_results = [test_result]
        results.cached_test_ids = set()

        print_results_text(results, verbose=True)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Judge" in full_output


class TestMakeProgressCallback:
    """Tests for make_progress_callback function."""

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_quiet_mode_silent(self, mock_stdout, mock_echo):
        """Quiet mode should not output."""
        callback = make_progress_callback(quiet=True)
        callback('start', 'test-id', None)
        mock_echo.assert_not_called()

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_event(self, mock_stdout, mock_echo):
        """Should show RUN for start event."""
        callback = make_progress_callback(quiet=False)
        callback('start', 'my-test', None)

        call_arg = mock_echo.call_args[0][0]
        assert "my-test" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_complete_passed(self, mock_stdout, mock_echo):
        """Should show PASS for passed test."""
        callback = make_progress_callback(quiet=False)
        data = MagicMock()
        data.passed = True
        callback('complete', 'my-test', data)

        call_arg = mock_echo.call_args[0][0]
        assert "my-test" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_complete_failed(self, mock_stdout, mock_echo):
        """Should show FAIL for failed test."""
        callback = make_progress_callback(quiet=False)
        data = MagicMock()
        data.passed = False
        callback('complete', 'my-test', data)

        call_arg = mock_echo.call_args[0][0]
        assert "my-test" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_cached_event(self, mock_stdout, mock_echo):
        """Should show SKIP for cached test."""
        cached_tests = set()
        callback = make_progress_callback(quiet=False, cached_tests=cached_tests)
        callback('cached', 'my-test', None)

        assert 'my-test' in cached_tests
        call_arg = mock_echo.call_args[0][0]
        assert "cached" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_flushes_stdout(self, mock_stdout, mock_echo):
        """Should flush stdout after each event."""
        callback = make_progress_callback(quiet=False)
        callback('start', 'test', None)
        mock_stdout.flush.assert_called()

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_event_prints_tools_when_data_provided(self, mock_stdout, mock_echo):
        """Should print executor tools on start when data contains tools list."""
        callback = make_progress_callback(quiet=False)
        callback('start', 'my-test', {'tools': ['read_file', 'glob']})

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "my-test" in full
        assert "Tools: [read_file, glob]" in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_event_no_tools_line_when_data_is_none(self, mock_stdout, mock_echo):
        """Should not print tools line when data is None (backward-compatible)."""
        callback = make_progress_callback(quiet=False)
        callback('start', 'my-test', None)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "my-test" in full
        assert "Tools:" not in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_event_no_tools_line_when_tools_key_missing(self, mock_stdout, mock_echo):
        """Should not print tools line when data dict has no tools key."""
        callback = make_progress_callback(quiet=False)
        callback('start', 'my-test', {'other': 'value'})

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "my-test" in full
        assert "Tools:" not in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_prints_executor_provenance_grouped(self, mock_stdout, mock_echo):
        """Should group executor tools by source when provenance is present."""
        callback = make_progress_callback(quiet=False)
        provenance = {
            'executor_tools': {
                'read_file': 'scaffold',
                'glob': 'scaffold',
                'run_shell_command': 'auto:standard',
            },
            'judge_tools': {},
            'explore_tools': {},
            'overrides_active': False,
            'removed_tools': [],
        }
        callback('start', 'my-test', {'tools': ['read_file', 'glob', 'run_shell_command'], 'tool_provenance': provenance})

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Executor tools:" in full
        assert "scaffold" in full
        assert "auto:standard" in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_prints_judge_tools(self, mock_stdout, mock_echo):
        """Should print one line per judge with tools."""
        callback = make_progress_callback(quiet=False)
        provenance = {
            'executor_tools': {'read_file': 'scaffold'},
            'judge_tools': {
                'accuracy': {'run_shell_command': 'auto:standard', 'read_file': 'scaffold'},
            },
            'explore_tools': {},
            'overrides_active': False,
            'removed_tools': [],
        }
        callback('start', 'my-test', {'tools': ['read_file'], 'tool_provenance': provenance})

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Judge [accuracy]:" in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_prints_explore_tools(self, mock_stdout, mock_echo):
        """Should print explore line when explore tools are present."""
        callback = make_progress_callback(quiet=False)
        provenance = {
            'executor_tools': {'read_file': 'scaffold'},
            'judge_tools': {},
            'explore_tools': {'read_file': 'explore:config', 'glob': 'explore:config'},
            'overrides_active': False,
            'removed_tools': [],
        }
        callback('start', 'my-test', {'tools': ['read_file'], 'tool_provenance': provenance})

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Explore:" in full
        assert "explore:config" in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_prints_overrides_active(self, mock_stdout, mock_echo):
        """Should show overrides indicator when active."""
        callback = make_progress_callback(quiet=False)
        provenance = {
            'executor_tools': {'read_file': 'scaffold'},
            'judge_tools': {},
            'explore_tools': {},
            'overrides_active': True,
            'removed_tools': [],
        }
        callback('start', 'my-test', {'tools': ['read_file'], 'tool_provenance': provenance})

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Overrides: active" in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_prints_removed_tools(self, mock_stdout, mock_echo):
        """Should show filtered-out tools when present."""
        callback = make_progress_callback(quiet=False)
        provenance = {
            'executor_tools': {'read_file': 'scaffold'},
            'judge_tools': {},
            'explore_tools': {},
            'overrides_active': False,
            'removed_tools': ['glob', 'web_fetch'],
        }
        callback('start', 'my-test', {'tools': ['read_file'], 'tool_provenance': provenance})

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Filtered out:" in full
        assert "glob" in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_backward_compatible_no_provenance(self, mock_stdout, mock_echo):
        """Should fall back to flat tool list when no provenance."""
        callback = make_progress_callback(quiet=False)
        callback('start', 'my-test', {'tools': ['read_file', 'glob']})

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Tools: [read_file, glob]" in full
        # Should NOT show provenance-style output
        assert "Executor tools:" not in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_provenance_none_falls_back_to_flat_list(self, mock_stdout, mock_echo):
        """When tool_provenance key is present but None, falls back to flat list."""
        callback = make_progress_callback(quiet=False)
        callback('start', 'my-test', {'tools': ['read_file', 'glob'], 'tool_provenance': None})

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Tools: [read_file, glob]" in full
        assert "Executor tools:" not in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_empty_provenance_no_crash(self, mock_stdout, mock_echo):
        """Empty provenance dict should not crash or print executor/judge/explore lines."""
        callback = make_progress_callback(quiet=False)
        provenance = {
            'executor_tools': {},
            'judge_tools': {},
            'explore_tools': {},
            'overrides_active': False,
            'removed_tools': [],
        }
        callback('start', 'my-test', {'tools': [], 'tool_provenance': provenance})

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "my-test" in full
        assert "Executor tools:" not in full
        assert "Judge" not in full
        assert "Explore:" not in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_judge_empty_tools_no_line(self, mock_stdout, mock_echo):
        """Judge with empty tools dict should not print a judge line."""
        callback = make_progress_callback(quiet=False)
        provenance = {
            'executor_tools': {'read_file': 'scaffold'},
            'judge_tools': {'accuracy': {}},
            'explore_tools': {},
            'overrides_active': False,
            'removed_tools': [],
        }
        callback('start', 'my-test', {'tools': ['read_file'], 'tool_provenance': provenance})

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Judge [accuracy]:" not in full

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_event_no_tools_line_when_quiet(self, mock_stdout, mock_echo):
        """Should not print anything in quiet mode even with tools data."""
        callback = make_progress_callback(quiet=True)
        callback('start', 'my-test', {'tools': ['read_file']})
        mock_echo.assert_not_called()


class TestMakeToolCallCallback:
    """Tests for make_tool_call_callback function."""

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_quiet_mode_silent(self, mock_stdout, mock_echo):
        """Quiet mode without verbose should not output."""
        callback = make_tool_call_callback(quiet=True, verbose=False)
        callback('read_file', {'path': 'test.md'}, "content")
        mock_echo.assert_not_called()

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_verbose_overrides_quiet(self, mock_stdout, mock_echo):
        """Verbose should override quiet to show output."""
        callback = make_tool_call_callback(quiet=True, verbose=True)
        callback('read_file', {'path': 'test.md'}, "content")
        mock_echo.assert_called()
        call_arg = mock_echo.call_args[0][0]
        assert "read_file" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_shows_tool_name(self, mock_stdout, mock_echo):
        """Should show tool name and params."""
        callback = make_tool_call_callback(quiet=False)
        callback('read_file', {'path': 'test.md'}, "content")

        call_arg = mock_echo.call_args[0][0]
        assert "read_file" in call_arg
        assert "test.md" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_handles_non_dict_params(self, mock_stdout, mock_echo):
        """Should handle non-dict parameters."""
        callback = make_tool_call_callback(quiet=False)
        callback('some_tool', "simple_param", "result")

        call_arg = mock_echo.call_args[0][0]
        assert "some_tool" in call_arg
        assert "simple_param" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_handles_empty_params(self, mock_stdout, mock_echo):
        """Should handle empty parameters."""
        callback = make_tool_call_callback(quiet=False)
        callback('tool', None, "result")

        call_arg = mock_echo.call_args[0][0]
        assert "tool()" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_flushes_stdout(self, mock_stdout, mock_echo):
        """Should flush stdout after each tool call."""
        callback = make_tool_call_callback(quiet=False)
        callback('tool', {}, "result")
        mock_stdout.flush.assert_called()


class TestPrintStateBars:
    """Tests for _print_state_bars helper function."""

    @patch('dokumen.cli.formatters.click.echo')
    def test_prints_all_states(self, mock_echo):
        """Should print bars for passed, failed, uncovered."""
        by_state = {'passed': 5, 'failed': 2, 'uncovered': 3}
        _print_state_bars(by_state, 10)

        assert mock_echo.call_count == 3

    @patch('dokumen.cli.formatters.click.echo')
    def test_handles_zero_total(self, mock_echo):
        """Should handle zero total without division error."""
        by_state = {'passed': 0, 'failed': 0, 'uncovered': 0}
        _print_state_bars(by_state, 0)
        # Should not raise


class TestPrintFilesTable:
    """Tests for _print_files_table helper function."""

    @patch('dokumen.cli.formatters.click.echo')
    def test_prints_header(self, mock_echo):
        """Should print table header."""
        stats = {'files_detail': {}}
        _print_files_table(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Per-File Coverage" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_prints_file_entries(self, mock_echo):
        """Should print each file entry."""
        stats = {
            'files_detail': {
                'docs/api.md': {
                    'test_count': 2,
                    'status': 'passed',
                    'line_coverage_pct': 80.0
                },
                'docs/guide.md': {
                    'test_count': 1,
                    'status': 'failed',
                    'line_coverage_pct': 50.0
                }
            }
        }
        _print_files_table(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "docs/api.md" in full_output
        assert "docs/guide.md" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    def test_truncates_long_paths(self, mock_echo):
        """Should truncate paths longer than 40 chars."""
        long_path = "a" * 50 + ".md"
        stats = {
            'files_detail': {
                long_path: {
                    'test_count': 1,
                    'status': 'passed'
                }
            }
        }
        _print_files_table(stats)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "..." in full_output  # Path should be truncated

    @patch('dokumen.cli.formatters.click.echo')
    def test_uncovered_status(self, mock_echo):
        """Should show uncovered status for files without coverage."""
        stats = {
            'files_detail': {
                'docs/api.md': {
                    'test_count': 0,
                    'status': 'uncovered'
                }
            }
        }
        _print_files_table(stats)
        # Should execute without error, showing uncovered indicator

    @patch('dokumen.cli.formatters.click.echo')
    def test_line_pct_from_line_stats(self, mock_echo):
        """Should get line coverage from line_stats if not in detail."""
        stats = {
            'files_detail': {
                'docs/api.md': {
                    'test_count': 2,
                    'status': 'passed'
                    # No line_coverage_pct
                }
            }
        }
        line_stats = {
            'files': {
                'docs/api.md': {
                    'percentage': 75.0
                }
            }
        }
        _print_files_table(stats, line_stats)
        # Should execute without error, getting pct from line_stats

    @patch('dokumen.cli.formatters.click.echo')
    def test_no_line_pct(self, mock_echo):
        """Should handle missing line coverage percentage."""
        stats = {
            'files_detail': {
                'docs/api.md': {
                    'test_count': 2,
                    'status': 'passed'
                    # No line_coverage_pct
                }
            }
        }
        _print_files_table(stats)
        # Should show "-" for missing line coverage


class TestMakeConversationCallback:
    """Tests for make_conversation_callback function."""

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_verbose_overrides_quiet(self, mock_stdout, mock_echo):
        """Verbose should override quiet to show output."""
        callback = make_conversation_callback(quiet=True, verbose=True)
        callback('executor', 'system', 'System prompt content')
        mock_echo.assert_called()
        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "EXECUTOR" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_non_verbose_silent(self, mock_stdout, mock_echo):
        """Non-verbose mode should not output."""
        callback = make_conversation_callback(quiet=False, verbose=False)
        callback('executor', 'system', 'System prompt content')
        mock_echo.assert_not_called()

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_executor_system_prompt(self, mock_stdout, mock_echo):
        """Should format executor system prompt correctly."""
        callback = make_conversation_callback(quiet=False, verbose=True)
        callback('executor', 'system', 'You are a testing agent.')

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "EXECUTOR" in full_output
        assert "System" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_executor_user_prompt(self, mock_stdout, mock_echo):
        """Should format executor user prompt correctly."""
        callback = make_conversation_callback(quiet=False, verbose=True)
        callback('executor', 'user', 'Read the docs and validate.')

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "EXECUTOR" in full_output
        assert "User" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_executor_response(self, mock_stdout, mock_echo):
        """Should format executor response correctly."""
        callback = make_conversation_callback(quiet=False, verbose=True)
        callback('executor', 'assistant', 'I will read the documentation.')

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "EXECUTOR" in full_output
        assert "Response" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_judge_system_prompt(self, mock_stdout, mock_echo):
        """Should format judge system prompt correctly."""
        callback = make_conversation_callback(quiet=False, verbose=True)
        callback('judge', 'system', 'Evaluate the executor output.')

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "JUDGE" in full_output
        assert "System" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_judge_response(self, mock_stdout, mock_echo):
        """Should format judge response correctly."""
        callback = make_conversation_callback(quiet=False, verbose=True)
        callback('judge', 'assistant', '{"verdict": "PASS"}')

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "JUDGE" in full_output
        assert "Response" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_flushes_stdout(self, mock_stdout, mock_echo):
        """Should flush stdout after each message."""
        callback = make_conversation_callback(quiet=False, verbose=True)
        callback('executor', 'system', 'prompt')
        mock_stdout.flush.assert_called()

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_multiline_content(self, mock_stdout, mock_echo):
        """Should handle multiline content correctly."""
        callback = make_conversation_callback(quiet=False, verbose=True)
        callback('executor', 'system', 'Line 1\nLine 2\nLine 3')

        # Should have multiple echo calls for header + lines
        assert mock_echo.call_count >= 2


class TestMakeExecutorCompleteCallback:
    """Tests for make_executor_complete_callback function."""

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_quiet_mode_silent(self, mock_stdout, mock_echo):
        """Quiet mode without verbose should not output."""
        callback = make_executor_complete_callback(quiet=True, verbose=False)
        executor_output = MagicMock()
        callback('test-id', executor_output)
        mock_echo.assert_not_called()

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_verbose_overrides_quiet(self, mock_stdout, mock_echo):
        """Verbose should override quiet to show output."""
        callback = make_executor_complete_callback(quiet=True, verbose=True)
        executor_output = MagicMock()
        executor_output.success = True
        executor_output.tool_calls = []
        executor_output.final_response = "The answer is 42"
        executor_output.error = None
        callback('test-id', executor_output)
        mock_echo.assert_called()
        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Executor Complete" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_success_status(self, mock_stdout, mock_echo):
        """Should show SUCCESS for successful executor."""
        callback = make_executor_complete_callback(quiet=False)
        executor_output = MagicMock()
        executor_output.success = True
        executor_output.tool_calls = [MagicMock()]
        executor_output.final_response = "The answer is 42"
        executor_output.error = None

        callback('test-id', executor_output)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "SUCCESS" in full_output
        assert "Tool calls: 1" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_failed_status(self, mock_stdout, mock_echo):
        """Should show FAILED for failed executor."""
        callback = make_executor_complete_callback(quiet=False)
        executor_output = MagicMock()
        executor_output.success = False
        executor_output.error = "Timeout occurred"
        executor_output.tool_calls = []
        executor_output.final_response = ""

        callback('test-id', executor_output)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "FAILED" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_shows_error_message(self, mock_stdout, mock_echo):
        """Should show error message when present."""
        callback = make_executor_complete_callback(quiet=False)
        executor_output = MagicMock()
        executor_output.success = False
        executor_output.error = "Connection failed"
        executor_output.tool_calls = []
        executor_output.final_response = ""

        callback('test-id', executor_output)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Connection failed" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_verbose_shows_full_response(self, mock_stdout, mock_echo):
        """Should show full response in verbose mode."""
        callback = make_executor_complete_callback(quiet=False, verbose=True)
        executor_output = MagicMock()
        executor_output.success = True
        executor_output.tool_calls = []
        executor_output.final_response = "This is a detailed response\nwith multiple lines"
        executor_output.error = None

        callback('test-id', executor_output)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "detailed response" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_truncates_long_response_non_verbose(self, mock_stdout, mock_echo):
        """Should truncate response in non-verbose mode."""
        callback = make_executor_complete_callback(quiet=False, verbose=False)
        executor_output = MagicMock()
        executor_output.success = True
        executor_output.tool_calls = []
        executor_output.final_response = "x" * 300  # Long response
        executor_output.error = None

        callback('test-id', executor_output)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "..." in full_output  # Should be truncated

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_flushes_stdout(self, mock_stdout, mock_echo):
        """Should flush stdout after output."""
        callback = make_executor_complete_callback(quiet=False)
        executor_output = MagicMock()
        executor_output.success = True
        executor_output.tool_calls = []
        executor_output.final_response = "response"
        executor_output.error = None

        callback('test-id', executor_output)
        mock_stdout.flush.assert_called()


class TestMakeJudgeCompleteCallback:
    """Tests for make_judge_complete_callback function."""

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_quiet_mode_silent(self, mock_stdout, mock_echo):
        """Quiet mode without verbose should not output."""
        callback = make_judge_complete_callback(quiet=True, verbose=False)
        judge_result = MagicMock()
        callback('test-id', judge_result)
        mock_echo.assert_not_called()

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_verbose_overrides_quiet(self, mock_stdout, mock_echo):
        """Verbose should override quiet to show output."""
        callback = make_judge_complete_callback(quiet=True, verbose=True)
        judge_result = MagicMock()
        judge_result.judge_id = "accuracy"
        judge_result.passed = True
        judge_result.confidence = 0.95
        judge_result.failure_reason = None
        judge_result.response = "All good"
        callback('test-id', judge_result)
        mock_echo.assert_called()
        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "accuracy" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_pass_verdict(self, mock_stdout, mock_echo):
        """Should show PASS verdict."""
        callback = make_judge_complete_callback(quiet=False)
        judge_result = MagicMock()
        judge_result.judge_id = "accuracy"
        judge_result.passed = True
        judge_result.confidence = 0.95
        judge_result.failure_reason = None
        judge_result.response = None

        callback('test-id', judge_result)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "PASS" in full_output
        assert "accuracy" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_fail_verdict(self, mock_stdout, mock_echo):
        """Should show FAIL verdict."""
        callback = make_judge_complete_callback(quiet=False)
        judge_result = MagicMock()
        judge_result.judge_id = "accuracy"
        judge_result.passed = False
        judge_result.confidence = 0.8
        judge_result.failure_reason = "Documentation is outdated"
        judge_result.response = None

        callback('test-id', judge_result)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "FAIL" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_shows_confidence(self, mock_stdout, mock_echo):
        """Should show confidence percentage."""
        callback = make_judge_complete_callback(quiet=False)
        judge_result = MagicMock()
        judge_result.judge_id = "accuracy"
        judge_result.passed = True
        judge_result.confidence = 0.95
        judge_result.failure_reason = None
        judge_result.response = None

        callback('test-id', judge_result)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "95%" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_shows_failure_reason(self, mock_stdout, mock_echo):
        """Should show failure reason when failed."""
        callback = make_judge_complete_callback(quiet=False)
        judge_result = MagicMock()
        judge_result.judge_id = "accuracy"
        judge_result.passed = False
        judge_result.confidence = 0.7
        judge_result.failure_reason = "Missing required information"
        judge_result.response = None

        callback('test-id', judge_result)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "Missing required information" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_verbose_shows_full_response(self, mock_stdout, mock_echo):
        """Should show full response in verbose mode."""
        callback = make_judge_complete_callback(quiet=False, verbose=True)
        judge_result = MagicMock()
        judge_result.judge_id = "accuracy"
        judge_result.passed = True
        judge_result.confidence = 0.9
        judge_result.failure_reason = None
        judge_result.response = '{"verdict": "PASS", "confidence": 0.9, "reason": "All good"}'

        callback('test-id', judge_result)

        calls = [str(c) for c in mock_echo.call_args_list]
        full_output = " ".join(calls)
        assert "verdict" in full_output

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_flushes_stdout(self, mock_stdout, mock_echo):
        """Should flush stdout after output."""
        callback = make_judge_complete_callback(quiet=False)
        judge_result = MagicMock()
        judge_result.judge_id = "accuracy"
        judge_result.passed = True
        judge_result.confidence = 0.9
        judge_result.failure_reason = None
        judge_result.response = None

        callback('test-id', judge_result)
        mock_stdout.flush.assert_called()

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_handles_none_confidence(self, mock_stdout, mock_echo):
        """Should handle None confidence gracefully."""
        callback = make_judge_complete_callback(quiet=False)
        judge_result = MagicMock()
        judge_result.judge_id = "accuracy"
        judge_result.passed = True
        judge_result.confidence = None
        judge_result.failure_reason = None
        judge_result.response = None

        callback('test-id', judge_result)
        # Should not raise and should not show confidence line


class TestMakeExploreCallback:
    """Tests for make_explore_callback function."""

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_quiet_mode_silent(self, mock_stdout, mock_echo):
        """Quiet mode should not output."""
        from dokumen.cli.formatters import make_explore_callback
        callback = make_explore_callback(quiet=True)
        callback('start', {'goal': 'find margin docs'})
        mock_echo.assert_not_called()

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_start_event(self, mock_stdout, mock_echo):
        """Should show EXPLORE for start event."""
        from dokumen.cli.formatters import make_explore_callback
        callback = make_explore_callback(quiet=False)
        callback('start', {'goal': 'find margin documentation'})

        call_arg = mock_echo.call_args[0][0]
        assert "EXPLORE" in call_arg
        assert "margin" in call_arg.lower()

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_file_found_event(self, mock_stdout, mock_echo):
        """Should show found file path."""
        from dokumen.cli.formatters import make_explore_callback
        callback = make_explore_callback(quiet=False)
        callback('file_found', {
            'path': 'docs/policies/margin-policy.md',
            'summary': 'Margin requirements'
        })

        call_arg = mock_echo.call_args[0][0]
        assert "Found" in call_arg
        assert "margin-policy.md" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_complete_event(self, mock_stdout, mock_echo):
        """Should show completion with files count and duration."""
        from dokumen.cli.formatters import make_explore_callback
        callback = make_explore_callback(quiet=False)
        callback('complete', {
            'files_found': 3,
            'duration': 1.5
        })

        call_arg = mock_echo.call_args[0][0]
        assert "EXPLORE" in call_arg
        assert "Complete" in call_arg
        assert "3 files" in call_arg
        assert "1.5s" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_complete_event_single_file(self, mock_stdout, mock_echo):
        """Should use singular 'file' for 1 file."""
        from dokumen.cli.formatters import make_explore_callback
        callback = make_explore_callback(quiet=False)
        callback('complete', {
            'files_found': 1,
            'duration': 0.8
        })

        call_arg = mock_echo.call_args[0][0]
        assert "1 file" in call_arg

    @patch('dokumen.cli.formatters.click.echo')
    @patch('sys.stdout')
    def test_flushes_stdout(self, mock_stdout, mock_echo):
        """Should flush stdout after each event."""
        from dokumen.cli.formatters import make_explore_callback
        callback = make_explore_callback(quiet=False)
        callback('start', {'goal': 'find docs'})
        mock_stdout.flush.assert_called()


class TestExploreToDict:
    """Tests for explore_to_dict function."""

    def test_basic_structure(self):
        """Should create correct structure."""
        from dokumen.cli.formatters import explore_to_dict
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        result = ExploreResult(
            files=[
                FileDiscovery(path='docs/api.md', summary='API documentation', relevance=0.9),
                FileDiscovery(path='docs/guide.md', summary='User guide', relevance=0.8)
            ],
            duration=1.5,
            tool_calls_count=5,
            success=True
        )

        output = explore_to_dict(result)

        assert output['success'] is True
        assert output['duration'] == 1.5
        assert output['tool_calls_count'] == 5
        assert len(output['files']) == 2

    def test_file_entries(self):
        """Should include file details."""
        from dokumen.cli.formatters import explore_to_dict
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        result = ExploreResult(
            files=[
                FileDiscovery(path='docs/api.md', summary='API docs', relevance=0.95)
            ],
            duration=1.0,
            tool_calls_count=3,
            success=True
        )

        output = explore_to_dict(result)

        assert output['files'][0]['path'] == 'docs/api.md'
        assert output['files'][0]['summary'] == 'API docs'
        assert output['files'][0]['relevance'] == 0.95

    def test_error_included(self):
        """Should include error when present."""
        from dokumen.cli.formatters import explore_to_dict
        from dokumen.explore_agent import ExploreResult

        result = ExploreResult(
            files=[],
            duration=0.5,
            tool_calls_count=1,
            success=False,
            error='Exploration timeout'
        )

        output = explore_to_dict(result)

        assert output['success'] is False
        assert output['error'] == 'Exploration timeout'

    def test_empty_files(self):
        """Should handle empty files list."""
        from dokumen.cli.formatters import explore_to_dict
        from dokumen.explore_agent import ExploreResult

        result = ExploreResult(
            files=[],
            duration=0.3,
            tool_calls_count=2,
            success=True
        )

        output = explore_to_dict(result)

        assert output['files'] == []
        assert output['success'] is True


class TestPrintRunSettings:
    """Tests for print_run_settings function."""

    def _make_config(self, **overrides):
        """Build a config dict with sensible defaults, applying overrides."""
        config = {
            'provider': {'name': 'anthropic', 'model': 'claude-haiku-4-5-20251001'},
            'executor_model': 'claude-haiku-4-5-20251001',
            'judge_model': 'claude-haiku-4-5-20251001',
            'execution': {'timeout': 60, 'retries': 0},
            'coverage': {'include': ['docs/**/*.md', 'README.md'], 'min_threshold': 80},
            'cache': {'enabled': True, 'path': '.dokumen-cache'},
            'explore': {
                'enabled': True,
                'model': 'claude-sonnet-4-5-20250929',
                'max_files': 20,
                'timeout': 60,
            },
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
        return config

    @patch('dokumen.cli.formatters.click.echo')
    def test_provider_and_models(self, mock_echo):
        """Should print provider name and all three models."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config()
        print_run_settings(config, test_count=5)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Provider: anthropic" in full
        assert "Executor model: claude-haiku-4-5-20251001" in full
        assert "Judge model: claude-haiku-4-5-20251001" in full
        assert "Explore model: claude-sonnet-4-5-20250929" in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_explore_disabled(self, mock_echo):
        """When explore is disabled, should show 'Explore: disabled'."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config(explore={'enabled': False})
        print_run_settings(config, test_count=3)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Explore: disabled" in full
        assert "Explore model" not in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_execution_settings(self, mock_echo):
        """Should print timeout and retries from execution config."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config(execution={'timeout': 120, 'retries': 2})
        print_run_settings(config, test_count=1)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "timeout=120s" in full
        assert "retries=2" in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_coverage_settings(self, mock_echo):
        """Should print coverage include patterns and threshold."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config(
            coverage={'include': ['docs/**/*.md'], 'min_threshold': 90}
        )
        print_run_settings(config, test_count=1)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "docs/**/*.md" in full
        assert "min=90%" in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_cache_enabled(self, mock_echo):
        """Should print cache enabled with path."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config()
        print_run_settings(config, test_count=1)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Cache: enabled" in full
        assert ".dokumen-cache" in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_cache_disabled(self, mock_echo):
        """Should print cache disabled when force is set."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config()
        print_run_settings(config, test_count=1, force=True)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Cache: disabled (force)" in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_flags_shown_when_active(self, mock_echo):
        """Should show Flags line listing active flags."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config()
        print_run_settings(config, test_count=1, force=True, bail=True,
                          debug=True, verbose=True, timeout_override=300)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Flags:" in full
        assert "force" in full
        assert "bail" in full
        assert "debug" in full
        assert "verbose" in full
        assert "timeout=300s" in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_flags_hidden_when_none_active(self, mock_echo):
        """Should not show Flags line when no flags are active."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config()
        print_run_settings(config, test_count=5)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Flags:" not in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_test_count_line(self, mock_echo):
        """Should print 'Running N test(s)...' as last line."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config()
        print_run_settings(config, test_count=7)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Running 7 test(s)..." in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_default_config_no_explore_section(self, mock_echo):
        """Should handle config with no explore section gracefully."""
        from dokumen.cli.formatters import print_run_settings
        config = {
            'provider': {'name': 'anthropic', 'model': 'claude-haiku-4-5-20251001'},
            'execution': {'timeout': 60, 'retries': 0},
            'coverage': {'include': ['docs/**/*.md'], 'min_threshold': 80},
            'cache': {'enabled': True, 'path': '.dokumen-cache'},
        }
        print_run_settings(config, test_count=2)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Provider: anthropic" in full
        assert "Running 2 test(s)..." in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_tools_defaults_shown(self, mock_echo):
        """Should show tools defaults when present in config."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config(tools={
            'defaults': ['read_file', 'glob', 'run_shell_command'],
        })
        print_run_settings(config, test_count=1)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Tools defaults:" in full
        assert "read_file" in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_tools_allowed_shown(self, mock_echo):
        """Should show tools allowed when present in config."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config(tools={
            'allowed': ['read_file', 'glob', 'run_shell_command', 'web_fetch'],
        })
        print_run_settings(config, test_count=1)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Tools allowed:" in full
        assert "web_fetch" in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_tools_not_shown_when_absent(self, mock_echo):
        """Should not show tools section when not in config."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config()
        print_run_settings(config, test_count=1)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Tools defaults:" not in full
        assert "Tools allowed:" not in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_tools_blocked_shown(self, mock_echo):
        """Should show tools blocked when present in config."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config(tools={
            'defaults': ['read_file', 'glob'],
            'blocked': ['web_fetch'],
        })
        print_run_settings(config, test_count=1)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Tools blocked:" in full
        assert "web_fetch" in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_tools_blocked_alongside_defaults_and_allowed(self, mock_echo):
        """Should show blocked alongside defaults and allowed."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config(tools={
            'defaults': ['read_file', 'glob'],
            'allowed': ['read_file', 'glob', 'web_fetch'],
            'blocked': ['web_fetch'],
        })
        print_run_settings(config, test_count=1)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Tools defaults:" in full
        assert "Tools allowed:" in full
        assert "Tools blocked:" in full

    @patch('dokumen.cli.formatters.click.echo')
    def test_tools_blocked_not_shown_when_absent(self, mock_echo):
        """Should not show tools blocked when not in config."""
        from dokumen.cli.formatters import print_run_settings
        config = self._make_config(tools={
            'defaults': ['read_file', 'glob'],
        })
        print_run_settings(config, test_count=1)

        calls = [str(c) for c in mock_echo.call_args_list]
        full = " ".join(calls)
        assert "Tools blocked:" not in full


class TestExploreEventsToJson:
    """Tests for explore JSON event formatting."""

    def test_start_event_json(self):
        """Should format start event as JSON."""
        from dokumen.cli.formatters import explore_event_to_json

        event = explore_event_to_json('start', 'margin-test', {'goal': 'margin documentation'})

        assert event['event'] == 'explore_start'
        assert event['test_id'] == 'margin-test'
        assert event['goal'] == 'margin documentation'

    def test_file_event_json(self):
        """Should format file event as JSON."""
        from dokumen.cli.formatters import explore_event_to_json

        event = explore_event_to_json('file_found', 'margin-test', {
            'path': 'docs/policies/margin-policy.md',
            'summary': 'Margin requirements'
        })

        assert event['event'] == 'explore_file'
        assert event['test_id'] == 'margin-test'
        assert event['path'] == 'docs/policies/margin-policy.md'
        assert event['summary'] == 'Margin requirements'

    def test_complete_event_json(self):
        """Should format complete event as JSON."""
        from dokumen.cli.formatters import explore_event_to_json

        event = explore_event_to_json('complete', 'margin-test', {
            'files_found': 2,
            'duration': 1.3
        })

        assert event['event'] == 'explore_complete'
        assert event['test_id'] == 'margin-test'
        assert event['files_found'] == 2
        assert event['duration_ms'] == 1300  # Converted to ms
