"""Tests for file_object module."""

import pytest
from pathlib import Path


class TestNormalizePath:
    """Tests for normalize_path function."""

    def test_forward_slashes(self):
        """Forward slashes pass through."""
        from dokumen.file_object import normalize_path
        assert normalize_path("docs/api.md") == "docs/api.md"

    def test_backslashes_converted(self):
        """Backslashes converted to forward slashes."""
        from dokumen.file_object import normalize_path
        result = normalize_path("docs\\api\\v2\\auth.md")
        assert "/" in result
        assert "\\" not in result

    def test_leading_dot_slash_removed(self):
        """Leading ./ is removed."""
        from dokumen.file_object import normalize_path
        result = normalize_path("./docs/api.md")
        assert not result.startswith("./")
        assert "api.md" in result

    def test_normalizes_parent_refs(self):
        """Parent references are normalized."""
        from dokumen.file_object import normalize_path
        result = normalize_path("docs/../docs/api.md")
        assert "api.md" in result


class TestFileStatus:
    """Tests for FileStatus enum."""

    def test_uncovered_value(self):
        """UNCOVERED has correct value."""
        from dokumen.file_object import FileStatus
        assert FileStatus.UNCOVERED.value == "uncovered"

    def test_passed_value(self):
        """PASSED has correct value."""
        from dokumen.file_object import FileStatus
        assert FileStatus.PASSED.value == "passed"

    def test_failed_value(self):
        """FAILED has correct value."""
        from dokumen.file_object import FileStatus
        assert FileStatus.FAILED.value == "failed"


class TestIncorrectLine:
    """Tests for IncorrectLine dataclass."""

    def test_fields(self):
        """IncorrectLine has correct fields."""
        from dokumen.file_object import IncorrectLine
        line = IncorrectLine(
            line_number=10,
            reason="Outdated info",
            test_id="test-1",
            confidence=0.85
        )
        assert line.line_number == 10
        assert line.reason == "Outdated info"
        assert line.test_id == "test-1"
        assert line.confidence == 0.85

    def test_default_confidence(self):
        """Confidence defaults to 0.0."""
        from dokumen.file_object import IncorrectLine
        line = IncorrectLine(
            line_number=5,
            reason="Test reason",
            test_id="test-2"
        )
        assert line.confidence == 0.0


class TestLineCoverage:
    """Tests for LineCoverage dataclass."""

    def test_basic_creation(self):
        """LineCoverage can be created."""
        from dokumen.file_object import LineCoverage
        cov = LineCoverage(
            file_path="docs/api.md",
            total_lines=100,
            covered_lines={1, 2, 3}
        )
        assert cov.file_path == "docs/api.md"
        assert cov.total_lines == 100
        assert cov.covered_lines == {1, 2, 3}

    def test_path_normalized_on_creation(self):
        """File path is normalized on creation."""
        from dokumen.file_object import LineCoverage
        cov = LineCoverage(
            file_path="docs\\api\\v2.md",
            total_lines=50
        )
        assert "\\" not in cov.file_path

    def test_covered_count(self):
        """covered_count returns number of covered lines."""
        from dokumen.file_object import LineCoverage
        cov = LineCoverage(
            file_path="test.md",
            total_lines=100,
            covered_lines={1, 2, 3, 4, 5}
        )
        assert cov.covered_count == 5

    def test_failed_count(self):
        """failed_count returns number of failed lines."""
        from dokumen.file_object import LineCoverage
        cov = LineCoverage(
            file_path="test.md",
            total_lines=100,
            failed_lines={10, 11, 12}
        )
        assert cov.failed_count == 3

    def test_incorrect_count(self):
        """incorrect_count returns number of incorrect lines."""
        from dokumen.file_object import LineCoverage, IncorrectLine
        cov = LineCoverage(
            file_path="test.md",
            total_lines=100,
            incorrect_lines=[
                IncorrectLine(line_number=5, reason="Bad", test_id="t1"),
                IncorrectLine(line_number=6, reason="Wrong", test_id="t1"),
            ]
        )
        assert cov.incorrect_count == 2

    def test_coverage_percentage(self):
        """coverage_percentage calculates correctly."""
        from dokumen.file_object import LineCoverage
        cov = LineCoverage(
            file_path="test.md",
            total_lines=100,
            covered_lines=set(range(1, 51))  # 50 lines
        )
        assert cov.coverage_percentage == 50.0

    def test_coverage_percentage_zero_lines(self):
        """coverage_percentage returns 0.0 for empty file."""
        from dokumen.file_object import LineCoverage
        cov = LineCoverage(
            file_path="test.md",
            total_lines=0
        )
        assert cov.coverage_percentage == 0.0

    def test_touched_lines(self):
        """touched_lines combines covered and failed."""
        from dokumen.file_object import LineCoverage
        cov = LineCoverage(
            file_path="test.md",
            total_lines=100,
            covered_lines={1, 2, 3},
            failed_lines={4, 5, 6}
        )
        assert cov.touched_lines == {1, 2, 3, 4, 5, 6}

    def test_touched_percentage(self):
        """touched_percentage calculates correctly."""
        from dokumen.file_object import LineCoverage
        cov = LineCoverage(
            file_path="test.md",
            total_lines=100,
            covered_lines={1, 2, 3},
            failed_lines={4, 5, 6, 7}
        )
        assert abs(cov.touched_percentage - 7.0) < 0.01

    def test_touched_percentage_zero_lines(self):
        """touched_percentage returns 0.0 for empty file."""
        from dokumen.file_object import LineCoverage
        cov = LineCoverage(
            file_path="test.md",
            total_lines=0
        )
        assert cov.touched_percentage == 0.0

    def test_merge_coverage(self):
        """merge combines two LineCoverages."""
        from dokumen.file_object import LineCoverage
        cov1 = LineCoverage(
            file_path="test.md",
            total_lines=100,
            covered_lines={1, 2, 3}
        )
        cov2 = LineCoverage(
            file_path="test.md",
            total_lines=100,
            covered_lines={3, 4, 5}
        )
        merged = cov1.merge(cov2)
        assert merged.covered_lines == {1, 2, 3, 4, 5}

    def test_merge_different_files_raises(self):
        """merge raises for different files."""
        from dokumen.file_object import LineCoverage
        cov1 = LineCoverage(file_path="a.md", total_lines=10)
        cov2 = LineCoverage(file_path="b.md", total_lines=10)
        with pytest.raises(ValueError):
            cov1.merge(cov2)

    def test_merge_combines_source_test_ids(self):
        """merge combines source_test_ids."""
        from dokumen.file_object import LineCoverage
        cov1 = LineCoverage(
            file_path="test.md",
            total_lines=100,
            source_test_ids={1: {"test-a"}}
        )
        cov2 = LineCoverage(
            file_path="test.md",
            total_lines=100,
            source_test_ids={1: {"test-b"}, 2: {"test-c"}}
        )
        merged = cov1.merge(cov2)
        assert merged.source_test_ids[1] == {"test-a", "test-b"}
        assert merged.source_test_ids[2] == {"test-c"}

    def test_merge_combines_failed_test_ids(self):
        """merge combines failed_test_ids."""
        from dokumen.file_object import LineCoverage
        cov1 = LineCoverage(
            file_path="test.md",
            total_lines=100,
            failed_test_ids={5: {"test-x"}}
        )
        cov2 = LineCoverage(
            file_path="test.md",
            total_lines=100,
            failed_test_ids={5: {"test-y"}, 6: {"test-z"}}
        )
        merged = cov1.merge(cov2)
        assert merged.failed_test_ids[5] == {"test-x", "test-y"}
        assert merged.failed_test_ids[6] == {"test-z"}

    def test_merge_uses_max_total_lines(self):
        """merge uses max of total_lines."""
        from dokumen.file_object import LineCoverage
        cov1 = LineCoverage(file_path="test.md", total_lines=50)
        cov2 = LineCoverage(file_path="test.md", total_lines=100)
        merged = cov1.merge(cov2)
        assert merged.total_lines == 100

    def test_merge_combines_incorrect_lines(self):
        """merge concatenates incorrect_lines."""
        from dokumen.file_object import LineCoverage, IncorrectLine
        cov1 = LineCoverage(
            file_path="test.md",
            total_lines=100,
            incorrect_lines=[IncorrectLine(1, "A", "t1")]
        )
        cov2 = LineCoverage(
            file_path="test.md",
            total_lines=100,
            incorrect_lines=[IncorrectLine(2, "B", "t2")]
        )
        merged = cov1.merge(cov2)
        assert len(merged.incorrect_lines) == 2

    def test_to_dict(self):
        """to_dict serializes correctly."""
        from dokumen.file_object import LineCoverage, IncorrectLine
        cov = LineCoverage(
            file_path="test.md",
            total_lines=100,
            covered_lines={1, 2, 3},
            failed_lines={4, 5},
            incorrect_lines=[IncorrectLine(6, "Bad", "t1", 0.9)],
            source_test_ids={1: {"ta"}},
            failed_test_ids={4: {"tb"}}
        )
        d = cov.to_dict()
        assert d["file_path"] == "test.md"
        assert d["total_lines"] == 100
        assert d["covered_lines"] == [1, 2, 3]
        assert d["failed_lines"] == [4, 5]
        assert len(d["incorrect_lines"]) == 1
        assert d["incorrect_lines"][0]["line_number"] == 6

    def test_from_dict(self):
        """from_dict deserializes correctly."""
        from dokumen.file_object import LineCoverage
        data = {
            "file_path": "test.md",
            "total_lines": 100,
            "covered_lines": [1, 2, 3],
            "failed_lines": [4, 5],
            "incorrect_lines": [
                {"line_number": 6, "reason": "Bad", "test_id": "t1", "confidence": 0.9}
            ],
            "source_test_ids": {"1": ["ta"]},
            "failed_test_ids": {"4": ["tb"]}
        }
        cov = LineCoverage.from_dict(data)
        assert cov.file_path == "test.md"
        assert cov.total_lines == 100
        assert cov.covered_lines == {1, 2, 3}
        assert cov.failed_lines == {4, 5}
        assert cov.incorrect_lines[0].confidence == 0.9
        assert 1 in cov.source_test_ids
        assert 4 in cov.failed_test_ids

    def test_from_dict_minimal(self):
        """from_dict works with minimal data."""
        from dokumen.file_object import LineCoverage
        data = {
            "file_path": "test.md",
            "total_lines": 50
        }
        cov = LineCoverage.from_dict(data)
        assert cov.file_path == "test.md"
        assert cov.covered_lines == set()


class TestFileMetrics:
    """Tests for FileMetrics dataclass."""

    def test_defaults(self):
        """FileMetrics has correct defaults."""
        from dokumen.file_object import FileMetrics
        metrics = FileMetrics()
        assert metrics.ref_count == 0
        assert metrics.pass_count == 0
        assert metrics.line_coverage is None

    def test_coverage_property(self):
        """coverage property calculates correctly."""
        from dokumen.file_object import FileMetrics
        metrics = FileMetrics(ref_count=10, pass_count=8)
        assert metrics.coverage == 0.8

    def test_coverage_zero_refs(self):
        """coverage returns 0.0 when no refs."""
        from dokumen.file_object import FileMetrics
        metrics = FileMetrics(ref_count=0, pass_count=0)
        assert metrics.coverage == 0.0

    def test_line_coverage_percentage_none(self):
        """line_coverage_percentage returns 0.0 when no line coverage."""
        from dokumen.file_object import FileMetrics
        metrics = FileMetrics()
        assert metrics.line_coverage_percentage == 0.0

    def test_line_coverage_percentage_with_data(self):
        """line_coverage_percentage uses line coverage data."""
        from dokumen.file_object import FileMetrics, LineCoverage
        line_cov = LineCoverage(
            file_path="test.md",
            total_lines=100,
            covered_lines=set(range(1, 51))
        )
        metrics = FileMetrics(line_coverage=line_cov)
        assert metrics.line_coverage_percentage == 50.0


class TestFileObject:
    """Tests for FileObject dataclass."""

    def test_creation(self):
        """FileObject can be created."""
        from dokumen.file_object import FileObject
        fo = FileObject(path="docs/api.md")
        assert fo.path == "docs/api.md"

    def test_path_normalized(self):
        """Path is normalized on creation."""
        from dokumen.file_object import FileObject
        fo = FileObject(path="docs\\api\\v2.md")
        assert "\\" not in fo.path

    def test_default_metrics(self):
        """FileObject has default metrics."""
        from dokumen.file_object import FileObject
        fo = FileObject(path="test.md")
        assert fo.metrics.ref_count == 0
        assert fo.metrics.pass_count == 0

    def test_get_metrics(self):
        """get_metrics returns metrics."""
        from dokumen.file_object import FileObject, FileMetrics
        fo = FileObject(path="test.md")
        fo.metrics = FileMetrics(ref_count=5, pass_count=3)
        metrics = fo.get_metrics()
        assert metrics.ref_count == 5
        assert metrics.pass_count == 3

    def test_increment_ref_count(self):
        """increment_ref_count increases ref_count."""
        from dokumen.file_object import FileObject
        fo = FileObject(path="test.md")
        fo.increment_ref_count()
        fo.increment_ref_count()
        assert fo.metrics.ref_count == 2

    def test_increment_pass_count(self):
        """increment_pass_count increases pass_count."""
        from dokumen.file_object import FileObject
        fo = FileObject(path="test.md")
        fo.increment_pass_count()
        fo.increment_pass_count()
        fo.increment_pass_count()
        assert fo.metrics.pass_count == 3

    def test_hash(self):
        """FileObject is hashable."""
        from dokumen.file_object import FileObject
        fo = FileObject(path="test.md")
        assert hash(fo) == hash("test.md")

    def test_equality(self):
        """FileObjects are equal by path."""
        from dokumen.file_object import FileObject
        fo1 = FileObject(path="test.md")
        fo2 = FileObject(path="test.md")
        fo3 = FileObject(path="other.md")
        assert fo1 == fo2
        assert fo1 != fo3

    def test_equality_non_fileobject(self):
        """FileObject not equal to non-FileObject."""
        from dokumen.file_object import FileObject
        fo = FileObject(path="test.md")
        # Returns False or NotImplemented for non-FileObject comparison
        result = fo == "test.md"
        assert result is False or result is NotImplemented

    @pytest.mark.asyncio
    async def test_read(self, tmp_path):
        """read() returns file content."""
        from dokumen.file_object import FileObject
        test_file = tmp_path / "test.md"
        test_file.write_text("Hello World")

        fo = FileObject(path=str(test_file))
        content = await fo.read()
        assert content == "Hello World"

    @pytest.mark.asyncio
    async def test_write(self, tmp_path):
        """write() writes content to file."""
        from dokumen.file_object import FileObject
        test_file = tmp_path / "test.md"

        fo = FileObject(path=str(test_file))
        await fo.write("New content")

        assert test_file.read_text() == "New content"

    @pytest.mark.asyncio
    async def test_read_not_found(self, tmp_path):
        """read() raises FileNotFoundError for missing file."""
        from dokumen.file_object import FileObject
        fo = FileObject(path=str(tmp_path / "nonexistent.md"))

        with pytest.raises(FileNotFoundError):
            await fo.read()
