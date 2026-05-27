"""
File Object module for the Documentation Unit Test Framework.

Provides abstraction layer for files being tested (documentation, specs, etc.)
and tracks usage metrics for test coverage reporting.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import aiofiles
import os


def normalize_path(path: str) -> str:
    """Normalize a file path for cross-platform consistency.

    Converts backslashes to forward slashes, removes leading ./ or .\\,
    and normalizes the path structure.

    Args:
        path: The file path to normalize

    Returns:
        Normalized path with forward slashes
    """
    # Normalize OS-specific separators
    normalized = os.path.normpath(path)
    # Convert to forward slashes for consistency
    normalized = normalized.replace('\\', '/')
    # Remove leading ./
    if normalized.startswith('./'):
        normalized = normalized[2:]
    return normalized


class FileStatus(Enum):
    """Status of a file based on test results."""
    UNCOVERED = "uncovered"  # No tests reference this file
    PASSED = "passed"        # All tests that reference this file pass
    FAILED = "failed"        # At least one test that references this file failed


@dataclass
class IncorrectLine:
    """Information about a line identified as potentially incorrect."""
    line_number: int
    reason: str
    test_id: str
    confidence: float = 0.0


@dataclass
class LineCoverage:
    """Line-level coverage data for a single file.

    Tracks which specific lines of a documentation file were exercised
    during test execution, as inferred by the coverage agent.
    """
    file_path: str
    total_lines: int
    covered_lines: Set[int] = field(default_factory=set)  # Lines from passing tests
    failed_lines: Set[int] = field(default_factory=set)   # Lines from failing tests
    incorrect_lines: List[IncorrectLine] = field(default_factory=list)  # Lines flagged as incorrect
    source_test_ids: Dict[int, Set[str]] = field(default_factory=dict)  # line -> test IDs (passing)
    failed_test_ids: Dict[int, Set[str]] = field(default_factory=dict)  # line -> test IDs (failing)

    def __post_init__(self):
        """Normalize file_path on creation for cross-platform consistency."""
        self.file_path = normalize_path(self.file_path)

    @property
    def covered_count(self) -> int:
        """Number of covered lines (from passing tests)."""
        return len(self.covered_lines)

    @property
    def failed_count(self) -> int:
        """Number of lines from failing tests."""
        return len(self.failed_lines)

    @property
    def incorrect_count(self) -> int:
        """Number of lines flagged as incorrect."""
        return len(self.incorrect_lines)

    @property
    def coverage_percentage(self) -> float:
        """Line coverage as percentage (passing tests only)."""
        if self.total_lines == 0:
            return 0.0
        return (len(self.covered_lines) / self.total_lines) * 100

    @property
    def touched_lines(self) -> Set[int]:
        """All lines that were touched (covered + failed)."""
        return self.covered_lines | self.failed_lines

    @property
    def touched_percentage(self) -> float:
        """Percentage of lines touched by any test."""
        if self.total_lines == 0:
            return 0.0
        return (len(self.touched_lines) / self.total_lines) * 100

    def merge(self, other: 'LineCoverage') -> 'LineCoverage':
        """Merge with another LineCoverage (union of all lines).

        Args:
            other: Another LineCoverage for the same file

        Returns:
            New LineCoverage with merged data

        Raises:
            ValueError: If file paths don't match
        """
        if self.file_path != other.file_path:
            raise ValueError("Cannot merge coverage for different files")

        merged_covered = self.covered_lines | other.covered_lines
        merged_failed = self.failed_lines | other.failed_lines
        merged_incorrect = self.incorrect_lines + other.incorrect_lines
        merged_sources: Dict[int, Set[str]] = {}
        merged_failed_sources: Dict[int, Set[str]] = {}

        # Merge source_test_ids (passing)
        for line, tests in self.source_test_ids.items():
            merged_sources[line] = tests.copy()
        for line, tests in other.source_test_ids.items():
            if line in merged_sources:
                merged_sources[line] |= tests
            else:
                merged_sources[line] = tests.copy()

        # Merge failed_test_ids
        for line, tests in self.failed_test_ids.items():
            merged_failed_sources[line] = tests.copy()
        for line, tests in other.failed_test_ids.items():
            if line in merged_failed_sources:
                merged_failed_sources[line] |= tests
            else:
                merged_failed_sources[line] = tests.copy()

        return LineCoverage(
            file_path=self.file_path,
            total_lines=max(self.total_lines, other.total_lines),
            covered_lines=merged_covered,
            failed_lines=merged_failed,
            incorrect_lines=merged_incorrect,
            source_test_ids=merged_sources,
            failed_test_ids=merged_failed_sources
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for caching.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "file_path": self.file_path,
            "total_lines": self.total_lines,
            "covered_lines": sorted(list(self.covered_lines)),
            "failed_lines": sorted(list(self.failed_lines)),
            "incorrect_lines": [
                {
                    "line_number": il.line_number,
                    "reason": il.reason,
                    "test_id": il.test_id,
                    "confidence": il.confidence
                }
                for il in self.incorrect_lines
            ],
            "source_test_ids": {
                str(k): sorted(list(v)) for k, v in self.source_test_ids.items()
            },
            "failed_test_ids": {
                str(k): sorted(list(v)) for k, v in self.failed_test_ids.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LineCoverage':
        """Deserialize from dictionary.

        Args:
            data: Dictionary from to_dict() or cache

        Returns:
            LineCoverage instance
        """
        incorrect_lines = [
            IncorrectLine(
                line_number=il["line_number"],
                reason=il["reason"],
                test_id=il["test_id"],
                confidence=il.get("confidence", 0.0)
            )
            for il in data.get("incorrect_lines", [])
        ]
        return cls(
            file_path=data["file_path"],
            total_lines=data["total_lines"],
            covered_lines=set(data.get("covered_lines", [])),
            failed_lines=set(data.get("failed_lines", [])),
            incorrect_lines=incorrect_lines,
            source_test_ids={
                int(k): set(v) for k, v in data.get("source_test_ids", {}).items()
            },
            failed_test_ids={
                int(k): set(v) for k, v in data.get("failed_test_ids", {}).items()
            }
        )


@dataclass
class FileMetrics:
    """Metrics tracked for files."""
    ref_count: int = 0       # Number of times referenced in test suite
    pass_count: int = 0      # Number of cached passing results
    line_coverage: Optional[LineCoverage] = None  # Line-level coverage data

    @property
    def coverage(self) -> float:
        """Computed coverage: pass_count / ref_count."""
        if self.ref_count == 0:
            return 0.0
        return self.pass_count / self.ref_count

    @property
    def line_coverage_percentage(self) -> float:
        """Line coverage percentage (0.0 if no line coverage data)."""
        if self.line_coverage is None:
            return 0.0
        return self.line_coverage.coverage_percentage


@dataclass
class FileObject:
    """Represents a file in the test framework."""
    path: str
    metrics: FileMetrics = field(default_factory=FileMetrics)
    _content_cache: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """Normalize path on creation for cross-platform consistency."""
        self.path = normalize_path(self.path)

    async def read(self) -> str:
        """Read and return the file content.

        Returns:
            str: The file content as a string

        Raises:
            FileNotFoundError: If the file does not exist
        """
        async with aiofiles.open(self.path, 'r') as f:
            return await f.read()

    async def write(self, content: str) -> None:
        """Write content to the file, creating it if it doesn't exist.

        Args:
            content: The content to write

        Raises:
            IOError: If the write operation fails
        """
        async with aiofiles.open(self.path, 'w') as f:
            await f.write(content)

    def get_metrics(self) -> FileMetrics:
        """Return the metrics for this file."""
        return self.metrics

    def increment_ref_count(self) -> None:
        """Increment reference count when test references this file."""
        self.metrics.ref_count += 1

    def increment_pass_count(self) -> None:
        """Increment pass count when a test using this file passes."""
        self.metrics.pass_count += 1

    def __hash__(self) -> int:
        """Make FileObject hashable by path."""
        return hash(self.path)

    def __eq__(self, other: object) -> bool:
        """Equality based on path."""
        if not isinstance(other, FileObject):
            return NotImplemented
        return self.path == other.path
