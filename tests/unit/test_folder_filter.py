"""Tests for Test Folders Support functionality in CLI.

Tests folder path normalization, validation, and matching.
"""

import pytest
from unittest.mock import MagicMock


# =============================================================================
# Normalization Tests
# =============================================================================


class TestNormalizeFolderPath:
    """Tests for normalize_folder_path function."""

    def test_strips_tests_prefix(self):
        """normalize_folder_path strips 'tests/' prefix."""
        from dokumen.cli.helpers import normalize_folder_path

        result = normalize_folder_path("tests/api/auth/login.test.yaml")
        assert result == "api/auth"

    def test_handles_root_level(self):
        """normalize_folder_path returns empty string for root level tests."""
        from dokumen.cli.helpers import normalize_folder_path

        result = normalize_folder_path("tests/smoke.test.yaml")
        assert result == ""

    def test_normalizes_backslashes(self):
        """normalize_folder_path converts backslashes to forward slashes."""
        from dokumen.cli.helpers import normalize_folder_path

        result = normalize_folder_path("tests\\api\\auth\\test.yaml")
        assert result == "api/auth"

    def test_collapses_multiple_slashes(self):
        """normalize_folder_path collapses multiple slashes."""
        from dokumen.cli.helpers import normalize_folder_path

        result = normalize_folder_path("tests//api//auth//test.yaml")
        assert result == "api/auth"

    def test_handles_absolute_with_tests(self):
        """normalize_folder_path handles absolute path with /tests/ segment."""
        from dokumen.cli.helpers import normalize_folder_path

        result = normalize_folder_path("/home/user/project/tests/api/test.yaml")
        assert result == "api"

    def test_handles_windows_with_tests(self):
        """normalize_folder_path handles Windows path with tests segment."""
        from dokumen.cli.helpers import normalize_folder_path

        result = normalize_folder_path("C:\\Users\\project\\tests\\api\\test.yaml")
        assert result == "api"

    def test_rejects_absolute_without_tests(self):
        """normalize_folder_path raises ValueError for absolute path without /tests/."""
        from dokumen.cli.helpers import normalize_folder_path

        with pytest.raises(ValueError, match="Cannot determine tests root"):
            normalize_folder_path("/home/user/api/test.yaml")

    def test_strips_result_slashes(self):
        """normalize_folder_path result has no leading/trailing slashes."""
        from dokumen.cli.helpers import normalize_folder_path

        result = normalize_folder_path("tests/api/auth/test.yaml")
        assert not result.startswith("/")
        assert not result.endswith("/")

    def test_normalizes_dot_segments(self):
        """normalize_folder_path normalizes . segments."""
        from dokumen.cli.helpers import normalize_folder_path

        result = normalize_folder_path("tests/./api/./test.yaml")
        assert result == "api"

    def test_normalizes_dotdot_segments(self):
        """normalize_folder_path normalizes .. segments."""
        from dokumen.cli.helpers import normalize_folder_path

        result = normalize_folder_path("tests/api/../auth/test.yaml")
        assert result == "auth"


# =============================================================================
# Matching Tests
# =============================================================================


class TestIsInFolder:
    """Tests for is_in_folder function."""

    def test_exact_match(self):
        """is_in_folder returns True for exact folder match."""
        from dokumen.cli.helpers import is_in_folder

        assert is_in_folder("api/auth", "api/auth") is True

    def test_nested_match(self):
        """is_in_folder returns True for nested folder."""
        from dokumen.cli.helpers import is_in_folder

        assert is_in_folder("api/auth/v2", "api/auth") is True

    def test_boundary_safe(self):
        """is_in_folder does not match folders with shared prefix."""
        from dokumen.cli.helpers import is_in_folder

        # "api" should NOT match "api-v2"
        assert is_in_folder("api-v2/test", "api") is False

    def test_empty_matches_all(self):
        """is_in_folder with empty target matches all tests."""
        from dokumen.cli.helpers import is_in_folder

        assert is_in_folder("api/auth", "") is True
        assert is_in_folder("integration/checkout", "") is True
        assert is_in_folder("", "") is True

    def test_dot_means_root_only(self):
        """is_in_folder with '.' target matches only root-level tests."""
        from dokumen.cli.helpers import is_in_folder

        assert is_in_folder("", ".") is True
        assert is_in_folder("api/auth", ".") is False
        assert is_in_folder("integration", ".") is False


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateFolderPath:
    """Tests for validate_folder_path function."""

    def test_preserves_empty_string(self):
        """validate_folder_path preserves empty string."""
        from dokumen.cli.helpers import validate_folder_path

        result = validate_folder_path("")
        assert result == ""

    def test_preserves_dot_sentinel(self):
        """validate_folder_path preserves '.' sentinel."""
        from dokumen.cli.helpers import validate_folder_path

        result = validate_folder_path(".")
        assert result == "."

    def test_normalizes_dot_slash_to_dot(self):
        """validate_folder_path normalizes './' to '.'."""
        from dokumen.cli.helpers import validate_folder_path

        result = validate_folder_path("./")
        assert result == "."

    def test_normalizes_dot_dot_slash_to_dot(self):
        """validate_folder_path normalizes './.' to '.'."""
        from dokumen.cli.helpers import validate_folder_path

        result = validate_folder_path("./.")
        assert result == "."

    def test_collapses_multiple_slashes(self):
        """validate_folder_path collapses multiple slashes."""
        from dokumen.cli.helpers import validate_folder_path

        result = validate_folder_path("api//auth")
        assert result == "api/auth"

    def test_rejects_parent_traversal(self):
        """validate_folder_path rejects .. segments."""
        from dokumen.cli.helpers import validate_folder_path

        with pytest.raises(ValueError, match="parent traversal"):
            validate_folder_path("api/../auth")

    def test_rejects_absolute_unix(self):
        """validate_folder_path rejects Unix absolute paths."""
        from dokumen.cli.helpers import validate_folder_path

        with pytest.raises(ValueError, match="absolute paths"):
            validate_folder_path("/api/auth")

    def test_rejects_windows_drive(self):
        """validate_folder_path rejects Windows drive paths."""
        from dokumen.cli.helpers import validate_folder_path

        with pytest.raises(ValueError, match="absolute paths"):
            validate_folder_path("C:\\api\\auth")

    def test_rejects_unc(self):
        """validate_folder_path rejects UNC paths."""
        from dokumen.cli.helpers import validate_folder_path

        with pytest.raises(ValueError, match="absolute paths"):
            validate_folder_path("//server/share/api")

    def test_allows_dots_in_names(self):
        """validate_folder_path allows dots in folder names."""
        from dokumen.cli.helpers import validate_folder_path

        result = validate_folder_path("api.v2/auth.test")
        assert result == "api.v2/auth.test"


# =============================================================================
# Filter Tests - filter_tests with folder parameter
# =============================================================================


class TestFilterTestsWithFolder:
    """Tests for filter_tests function with folder parameter."""

    def test_filter_tests_by_folder(self):
        """filter_tests filters by folder path."""
        from dokumen.cli.helpers import filter_tests

        # Create mock tests with source_path attribute
        test1 = MagicMock()
        test1.id = "login"
        test1.reason = ""
        test1.files = []
        test1.source_path = "tests/api/auth/login.test.yaml"

        test2 = MagicMock()
        test2.id = "smoke"
        test2.reason = ""
        test2.files = []
        test2.source_path = "tests/smoke.test.yaml"

        tests = [test1, test2]

        # Filter by api/auth folder
        result = filter_tests(tests, folder="api/auth")

        assert len(result) == 1
        assert result[0].id == "login"

    def test_filter_tests_by_folder_nested(self):
        """filter_tests includes tests in nested subfolders."""
        from dokumen.cli.helpers import filter_tests

        test1 = MagicMock()
        test1.id = "login"
        test1.reason = ""
        test1.files = []
        test1.source_path = "tests/api/auth/login.test.yaml"

        test2 = MagicMock()
        test2.id = "v2-login"
        test2.reason = ""
        test2.files = []
        test2.source_path = "tests/api/auth/v2/login.test.yaml"

        tests = [test1, test2]

        # Filter by api folder should include both
        result = filter_tests(tests, folder="api")

        assert len(result) == 2

    def test_filter_tests_by_folder_empty_returns_all(self):
        """filter_tests with empty folder returns all tests."""
        from dokumen.cli.helpers import filter_tests

        test1 = MagicMock()
        test1.id = "test1"
        test1.reason = ""
        test1.files = []
        test1.source_path = "tests/api/test1.test.yaml"

        test2 = MagicMock()
        test2.id = "test2"
        test2.reason = ""
        test2.files = []
        test2.source_path = "tests/test2.test.yaml"

        tests = [test1, test2]

        result = filter_tests(tests, folder="")

        assert len(result) == 2

    def test_filter_tests_by_folder_dot_root_only(self):
        """filter_tests with '.' returns only root-level tests."""
        from dokumen.cli.helpers import filter_tests

        test1 = MagicMock()
        test1.id = "api-test"
        test1.reason = ""
        test1.files = []
        test1.source_path = "tests/api/test.test.yaml"

        test2 = MagicMock()
        test2.id = "root-test"
        test2.reason = ""
        test2.files = []
        test2.source_path = "tests/smoke.test.yaml"

        tests = [test1, test2]

        result = filter_tests(tests, folder=".")

        assert len(result) == 1
        assert result[0].id == "root-test"

    def test_filter_tests_no_folder_returns_all(self):
        """filter_tests with folder=None returns all tests."""
        from dokumen.cli.helpers import filter_tests

        test1 = MagicMock()
        test1.id = "test1"
        test1.reason = ""
        test1.files = []
        test1.source_path = "tests/api/test1.test.yaml"

        test2 = MagicMock()
        test2.id = "test2"
        test2.reason = ""
        test2.files = []
        test2.source_path = "tests/test2.test.yaml"

        tests = [test1, test2]

        result = filter_tests(tests, folder=None)

        assert len(result) == 2

    def test_filter_tests_skips_invalid_source_path(self):
        """filter_tests skips tests with invalid source paths."""
        from dokumen.cli.helpers import filter_tests

        # Test with a valid path
        test1 = MagicMock()
        test1.id = "valid"
        test1.reason = ""
        test1.files = []
        test1.source_path = "tests/api/test.test.yaml"

        # Test with an invalid absolute path (no /tests/ segment)
        test2 = MagicMock()
        test2.id = "invalid"
        test2.reason = ""
        test2.files = []
        test2.source_path = "/absolute/path/without/tests/segment.yaml"

        tests = [test1, test2]

        # Filter by api folder - invalid test should be skipped (not cause error)
        result = filter_tests(tests, folder="api")

        # Only valid test should be returned
        assert len(result) == 1
        assert result[0].id == "valid"

    def test_filter_tests_no_source_path_included_for_root(self):
        """filter_tests includes tests without source_path when targeting root."""
        from dokumen.cli.helpers import filter_tests

        # Test with source_path
        test1 = MagicMock()
        test1.id = "with-path"
        test1.reason = ""
        test1.files = []
        test1.source_path = "tests/api/test.test.yaml"

        # Test without source_path attribute (legacy test object)
        test2 = MagicMock(spec=["id", "reason", "files"])  # No source_path
        test2.id = "no-path"
        test2.reason = ""
        test2.files = []

        tests = [test1, test2]

        # Filter with folder="." (root only) - test without source_path should be included
        result = filter_tests(tests, folder=".")

        # The test without source_path should be included when targeting root
        assert len(result) == 1
        assert result[0].id == "no-path"

    def test_filter_tests_no_source_path_included_for_all(self):
        """filter_tests includes tests without source_path when targeting all tests."""
        from dokumen.cli.helpers import filter_tests

        # Test with source_path
        test1 = MagicMock()
        test1.id = "with-path"
        test1.reason = ""
        test1.files = []
        test1.source_path = "tests/api/test.test.yaml"

        # Test without source_path attribute
        test2 = MagicMock(spec=["id", "reason", "files"])
        test2.id = "no-path"
        test2.reason = ""
        test2.files = []

        tests = [test1, test2]

        # Filter with folder="" (all) - both should be included
        result = filter_tests(tests, folder="")

        assert len(result) == 2

    def test_filter_tests_no_source_path_excluded_for_specific_folder(self):
        """filter_tests excludes tests without source_path when targeting specific folder."""
        from dokumen.cli.helpers import filter_tests

        # Test with source_path
        test1 = MagicMock()
        test1.id = "with-path"
        test1.reason = ""
        test1.files = []
        test1.source_path = "tests/api/test.test.yaml"

        # Test without source_path attribute
        test2 = MagicMock(spec=["id", "reason", "files"])
        test2.id = "no-path"
        test2.reason = ""
        test2.files = []

        tests = [test1, test2]

        # Filter by api folder - test without source_path should NOT be included
        result = filter_tests(tests, folder="api")

        assert len(result) == 1
        assert result[0].id == "with-path"
