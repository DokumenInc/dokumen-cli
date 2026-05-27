"""
Shared helper utilities for CLI commands.
"""
import asyncio
import fnmatch
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml

logger = logging.getLogger(__name__)

from dokumen.config import DEFAULT_FAST_MODEL, _preprocess_yaml

# =============================================================================
# Exit Codes
# =============================================================================

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_CONFIG_ERROR = 2
EXIT_RUNTIME_ERROR = 3
EXIT_INVALID_ARGS = 4


# =============================================================================
# Async Bridge
# =============================================================================

def run_async(coro):
    """Run async coroutine in sync context."""
    return asyncio.run(coro)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "version": "1.0",
    "provider": {"name": "anthropic", "model": DEFAULT_FAST_MODEL},
    "coverage": {
        "include": ["docs/**/*.md", "README.md"],
        "exclude": [],
        "min_threshold": 80
    },
    "execution": {"timeout": 60, "retries": 0, "parallel": False, "max_workers": 4},
    "cache": {"enabled": True, "path": ".dokumen-cache"},
    "analyzers": {
        "enabled": True,
        "output_path": ".dokumen-analysis",
        "agents": []  # Empty means use built-in analyzers
    },
    "sandbox": {
        "type": "whitelist",  # none, whitelist, subprocess, docker, virtual_fs
        "allowed_commands": ["dokumen"],
        "timeout": 30,
        "max_memory_mb": 512,
        "docker_image": "python:3.11-slim"
    },
    "history": {
        "enabled": True,
        "path": ".history"
    },
    "scaffold_prompts": {
        "enabled": True,
        "base_path": "prompts/scaffold-generator",
        "categories": {}
    }
}


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from file with defaults."""
    config = DEFAULT_CONFIG.copy()

    path = Path(config_path) if config_path else Path("dokumen.yaml")
    if path.exists():
        try:
            with open(path) as f:
                file_config = yaml.safe_load(_preprocess_yaml(f.read())) or {}
                config = deep_merge(config, file_config)
        except IOError as e:
            logger.error("Config file read failed", extra={"path": str(path), "error": str(e)})
        except yaml.YAMLError as e:
            logger.error("Config file YAML parse failed", extra={"path": str(path), "error": str(e)})

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# =============================================================================
# Path Normalization
# =============================================================================

# Import normalize_path from file_object for cross-platform path consistency
from ..file_object import normalize_path


# =============================================================================
# Folder Path Helpers (for test folder filtering)
# =============================================================================


def normalize_folder_path(file_path: str) -> str:
    """Extract canonical folder_path from a TEST FILE path (not folder-only input).

    Input: Full path to a test file (e.g., tests/api/auth/login.test.yaml)
    Output: Canonical folder path (e.g., api/auth)

    Handles both relative and absolute paths by finding the 'tests/' segment.
    Raises ValueError if absolute path has no /tests/ segment.

    Also normalizes . and .. segments in the path for defensive handling.

    Note: For relative paths containing /tests/ (like foo/tests/api/test.yaml),
    we use the last /tests/ segment. This handles monorepo structures where
    tests may be in subdirectories.
    """
    # Normalize separators and collapse duplicate slashes
    path = file_path.replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")

    # Check if absolute path (before any modifications)
    is_absolute = (
        path.startswith("/")
        or (len(path) > 1 and path[1] == ":")  # Windows drive
        or file_path.replace("\\", "/").startswith("//")  # UNC (check original)
    )

    # Normalize . and .. segments (defensive - handles non-canonical source paths)
    segments = path.split("/")
    normalized_segments: List[str] = []
    for seg in segments:
        if seg == "." or seg == "":
            continue  # Skip current dir and empty segments
        elif seg == "..":
            if normalized_segments and normalized_segments[-1] != "..":
                normalized_segments.pop()  # Go up one level
            elif not is_absolute:
                normalized_segments.append(seg)  # Keep .. for relative paths
        else:
            normalized_segments.append(seg)
    path = "/".join(normalized_segments)

    # Find /tests/ segment
    if "/tests/" in path:
        path = path.split("/tests/")[-1]  # Take everything after last /tests/
    elif path.startswith("tests/"):
        path = path[6:]
    elif is_absolute:
        # Absolute path without /tests/ - reject
        raise ValueError(f"Cannot determine tests root from absolute path: {file_path}")
    # else: relative path, assume already relative to tests root

    # Get directory part (everything before last /)
    if "/" in path:
        folder = path.rsplit("/", 1)[0]
        # Ensure no leading/trailing slashes in result
        return folder.strip("/")
    return ""  # Root level


def validate_folder_path(folder_path: str) -> str:
    """Validate and normalize folder path. Raises ValueError if invalid.

    Special values:
    - "" (empty string) = all tests
    - "." or "./" = root-level only (normalized to ".")
    """
    # Handle empty string early
    if folder_path == "":
        return ""

    # Normalize separators and collapse duplicate slashes
    path = folder_path.replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")

    # Reject absolute paths (Unix or Windows drive/UNC)
    if (
        path.startswith("/")
        or re.match(r"^[A-Za-z]:", path)
        or folder_path.replace("\\", "/").startswith("//")
    ):
        raise ValueError("Invalid folder path: absolute paths not allowed")

    # Split into segments and validate each
    segments = [s for s in path.split("/") if s]  # Remove empty segments

    for segment in segments:
        if segment == "..":
            raise ValueError("Invalid folder path: parent traversal not allowed")

    # Filter out "." segments
    filtered = [s for s in segments if s != "."]

    # If all segments were ".", this is root-only (e.g., ".", "./", "./.")
    if not filtered and any(s == "." for s in segments):
        return "."

    # Rebuild normalized path (no leading/trailing slash)
    return "/".join(filtered) if filtered else ""


def is_in_folder(test_folder_path: str, target_folder: str) -> bool:
    """Check if test belongs to target folder or any subfolder.

    Special cases:
    - target_folder="" means ALL tests (root = entire tree)
    - target_folder="." means root-level only (no subfolders)
    """
    if target_folder == "":
        return True  # Root folder = all tests
    if target_folder == ".":
        return test_folder_path == ""  # Root-level only
    return test_folder_path == target_folder or test_folder_path.startswith(
        target_folder + "/"
    )


# =============================================================================
# Coverage Statistics
# =============================================================================

def get_coverage_stats(tests_dir: str = "tests", config: dict = None) -> dict:
    """
    Calculate coverage statistics with file status tracking.

    Returns dict with:
        - total, passed, failed, percentage (legacy)
        - by_state: {passed, failed, uncovered} counts
        - test_counts: {file_path: num_tests}
        - covered_files, failed_files, uncovered_files
        - files_detail: {file_path: {test_count, test_ids, status, line_coverage_pct}}
    """
    from ..scaffold import discover_scaffolds

    config = config or {}

    # Get all doc files from config patterns
    all_doc_files = set(discover_doc_files(config))

    # Get test counts per file
    test_counts = get_test_counts_per_file(tests_dir)

    # Get test IDs per file (for coverage.json output)
    test_ids_map = get_test_ids_per_file(tests_dir)

    # Load file status from cache
    cache_path = config.get('cache', {}).get('path', '.dokumen-cache')
    file_status = get_file_status_from_cache(cache_path)

    # Load basic line coverage for per-file percentages (without by_state)
    basic_line_stats = get_line_coverage_stats(cache_path)
    line_files = basic_line_stats.get('files', {})

    # Categorize files into 3 states
    covered_files = []
    failed_files = []
    uncovered_files = []
    files_detail = {}

    for file_path in sorted(all_doc_files):
        file_test_count = test_counts.get(file_path, 0)
        cache_status = file_status.get(file_path)

        # Get line coverage percentage if available
        line_pct = None
        if file_path in line_files:
            line_pct = line_files[file_path].get('percentage', 0.0)

        # Determine state: passed/failed from cache, otherwise uncovered
        if cache_status == "passed":
            state = "passed"
            covered_files.append(file_path)
        elif cache_status == "failed":
            state = "failed"
            failed_files.append(file_path)
        else:
            state = "uncovered"
            uncovered_files.append(file_path)

        files_detail[file_path] = {
            "test_count": file_test_count,
            "test_ids": test_ids_map.get(file_path, []),
            "status": state,
            "line_coverage_pct": line_pct
        }

    total = len(all_doc_files)
    by_state = {
        "passed": len(covered_files),
        "failed": len(failed_files),
        "uncovered": len(uncovered_files)
    }

    # Legacy percentage (covered / total)
    covered_count = len(covered_files)
    percentage = (covered_count / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "passed": covered_count,
        "failed": len(failed_files),
        "percentage": percentage,
        "by_state": by_state,
        "test_counts": test_counts,
        "covered_files": covered_files,
        "failed_files": failed_files,
        "uncovered_files": uncovered_files,
        "files_detail": files_detail
    }


def get_file_status_from_cache(cache_path: str = ".dokumen-cache", tests_dir: str = "tests") -> Dict[str, str]:
    """
    Get file status from cache.

    If file_status is not in cache, derives it from test results and scaffolds.

    Args:
        cache_path: Path to cache directory
        tests_dir: Path to tests directory (for scaffold lookup)

    Returns:
        Dictionary mapping file path to status string ("passed", "failed", "uncovered")
        All paths are normalized for consistent comparison.
    """
    import json
    import os

    cache_file = os.path.join(cache_path, "cache.json")

    if not os.path.exists(cache_file):
        return {}

    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # First try to get file_status directly from cache
        raw_status = data.get("file_status", {})
        if raw_status:
            return {normalize_path(path): status for path, status in raw_status.items()}

        # If file_status is empty, derive it from test results and scaffolds
        results = data.get("results", {})
        if not results:
            return {}

        # Load test scaffolds to map tests to files
        from ..scaffold import discover_scaffolds, load_scaffold_yaml
        scaffold_paths = discover_scaffolds(tests_dir)
        test_to_files = {}
        for scaffold_path in scaffold_paths:
            try:
                scaffold_data = load_scaffold_yaml(scaffold_path)
                test_name = scaffold_data.get("name", "")
                files = scaffold_data.get("files", [])
                test_to_files[test_name] = [normalize_path(f.get("path", "") if isinstance(f, dict) else f) for f in files]
            except Exception:
                continue  # Skip invalid scaffolds

        # Derive file status from test results
        derived_status = {}
        for test_name, test_result in results.items():
            files = test_to_files.get(test_name, [])
            test_passed = test_result.get("passed", False)

            for file_path in files:
                if test_passed:
                    # Mark as passed only if not already failed
                    if derived_status.get(file_path) != "failed":
                        derived_status[file_path] = "passed"
                else:
                    # Failed tests always mark files as failed
                    derived_status[file_path] = "failed"

        return derived_status

    except (json.JSONDecodeError, IOError):
        return {}


def discover_doc_files(config: dict) -> List[str]:
    """Find all documentation files matching config patterns."""
    import glob as glob_module

    include_patterns = config.get("coverage", {}).get("include", ["docs/**/*.md", "README.md"])
    exclude_patterns = config.get("coverage", {}).get("exclude", [])

    files = []
    for pattern in include_patterns:
        # Use glob module which handles both relative and absolute patterns
        # and supports ** for recursive matching
        try:
            matched = glob_module.glob(pattern, recursive=True)
            for path_str in matched:
                # Normalize path for consistent comparison across platforms
                path_str = normalize_path(path_str)
                if not any(fnmatch.fnmatch(path_str, exc) for exc in exclude_patterns):
                    if path_str not in files:
                        files.append(path_str)
        except Exception:
            # Skip invalid patterns
            continue

    return files


def get_uncovered_files(config: dict = None) -> List[str]:
    """Get list of uncovered files."""
    stats = get_coverage_stats(config=config)
    return stats.get('uncovered_files', [])


def filter_stats_by_path(stats: dict, path: str) -> dict:
    """
    Filter coverage stats to only include files matching the given path.

    Args:
        stats: Coverage stats from get_coverage_stats()
        path: File path or directory prefix to filter by

    Returns:
        Filtered stats dict with recalculated totals
    """
    if not path:
        return stats

    # Normalize path separators
    path = path.replace("\\", "/")

    # Filter files_detail to only include matching files
    files_detail = stats.get('files_detail', {})
    filtered_detail = {}
    for file_path, detail in files_detail.items():
        # Match exact path or directory prefix
        if file_path == path or file_path.startswith(path.rstrip('/') + '/'):
            filtered_detail[file_path] = detail

    if not filtered_detail:
        # No files matched - return empty stats
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "percentage": 0.0,
            "by_state": {"passed": 0, "failed": 0, "uncovered": 0},
            "test_counts": {},
            "covered_files": [],
            "failed_files": [],
            "uncovered_files": [],
            "files_detail": {}
        }

    # Recalculate stats from filtered files
    covered_files = []
    failed_files = []
    uncovered_files = []
    test_counts = {}

    for file_path, detail in filtered_detail.items():
        status = detail.get('status', 'uncovered')
        test_count = detail.get('test_count', 0)
        if test_count > 0:
            test_counts[file_path] = test_count

        if status == 'passed':
            covered_files.append(file_path)
        elif status == 'failed':
            failed_files.append(file_path)
        else:
            uncovered_files.append(file_path)

    total = len(filtered_detail)
    covered_count = len(covered_files)
    percentage = (covered_count / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "passed": covered_count,
        "failed": len(failed_files),
        "percentage": percentage,
        "by_state": {
            "passed": covered_count,
            "failed": len(failed_files),
            "uncovered": len(uncovered_files)
        },
        "test_counts": test_counts,
        "covered_files": covered_files,
        "failed_files": failed_files,
        "uncovered_files": uncovered_files,
        "files_detail": filtered_detail
    }


def filter_line_stats_by_path(line_stats: dict, path: str) -> dict:
    """
    Filter line coverage stats to only include files matching the given path.

    Args:
        line_stats: Line stats from get_line_coverage_stats()
        path: File path or directory prefix to filter by

    Returns:
        Filtered line stats dict with recalculated totals
    """
    if not path or not line_stats:
        return line_stats

    # Normalize path separators
    path = path.replace("\\", "/")

    # Filter files
    files = line_stats.get('files', {})
    filtered_files = {}
    for file_path, data in files.items():
        if file_path == path or file_path.startswith(path.rstrip('/') + '/'):
            filtered_files[file_path] = data

    if not filtered_files:
        return {
            "total_lines": 0,
            "covered_lines": 0,
            "failed_lines": 0,
            "percentage": 0.0,
            "by_state": {"passed": 0, "failed": 0, "uncovered": 0},
            "files": {}
        }

    # Recalculate totals from filtered files
    total_lines = 0
    covered_lines = 0
    failed_lines = 0
    lines_by_state = {"passed": 0, "failed": 0, "uncovered": 0}

    for file_path, data in filtered_files.items():
        file_total = data.get('total_lines', 0)
        file_covered = len(data.get('covered_lines', []))
        file_failed = len(data.get('failed_lines', []))
        status = data.get('status', 'uncovered')

        total_lines += file_total
        covered_lines += file_covered
        failed_lines += file_failed

        # Accumulate lines by state
        if status == "passed":
            lines_by_state["passed"] += file_total
        elif status == "failed":
            lines_by_state["failed"] += file_failed
            lines_by_state["passed"] += file_covered
            lines_by_state["uncovered"] += file_total - file_covered - file_failed
        else:
            lines_by_state["uncovered"] += file_total

    return {
        "total_lines": total_lines,
        "covered_lines": covered_lines,
        "failed_lines": failed_lines,
        "percentage": (covered_lines / total_lines * 100) if total_lines > 0 else 0.0,
        "by_state": lines_by_state,
        "files": filtered_files,
        "failure_analysis": line_stats.get('failure_analysis', {})
    }


def get_test_counts_per_file(tests_dir: str = "tests") -> Dict[str, int]:
    """
    Count how many tests reference each file.

    Parses executor and judge prompts to find file references.

    Args:
        tests_dir: Directory containing test scaffolds

    Returns:
        Dictionary mapping file paths to number of tests referencing them
    """
    from ..scaffold import discover_scaffolds, extract_file_references_from_prompts

    test_counts: Dict[str, int] = {}
    scaffolds = discover_scaffolds(tests_dir)

    for scaffold_path in scaffolds:
        try:
            with open(scaffold_path) as f:
                scaffold_data = yaml.safe_load(f) or {}

            # Extract file references from prompts
            file_refs = extract_file_references_from_prompts(scaffold_data)
            for file_path in file_refs:
                # Normalize path for consistent comparison
                file_path = normalize_path(file_path)
                test_counts[file_path] = test_counts.get(file_path, 0) + 1
        except (IOError, yaml.YAMLError):
            continue

    return test_counts


def get_test_ids_per_file(tests_dir: str = "tests") -> Dict[str, List[str]]:
    """
    Get list of test IDs (names) that reference each file.

    Parses scaffolds to find which tests reference each file.
    Includes both explicit 'files' field and file paths found in prompts.

    Args:
        tests_dir: Directory containing test scaffolds

    Returns:
        Dictionary mapping file paths to list of test IDs referencing them
    """
    from ..scaffold import discover_scaffolds, extract_file_references_from_prompts

    test_ids: Dict[str, List[str]] = {}
    scaffolds = discover_scaffolds(tests_dir)

    for scaffold_path in scaffolds:
        try:
            with open(scaffold_path) as f:
                scaffold_data = yaml.safe_load(f) or {}

            # Get test name from scaffold
            test_name = scaffold_data.get("name", "")
            if not test_name:
                continue

            # Collect all file references
            all_file_refs = set()

            # 1. Extract explicit 'files' field from scaffold
            files_list = scaffold_data.get("files", [])
            for f in files_list:
                if isinstance(f, dict):
                    path = f.get("path", "")
                elif isinstance(f, str):
                    path = f
                else:
                    continue
                if path:
                    all_file_refs.add(normalize_path(path))

            # 2. Extract file references from prompts (fallback/additional)
            prompt_refs = extract_file_references_from_prompts(scaffold_data)
            for file_path in prompt_refs:
                all_file_refs.add(normalize_path(file_path))

            # Add test name to each file's list
            for file_path in all_file_refs:
                if file_path not in test_ids:
                    test_ids[file_path] = []
                if test_name not in test_ids[file_path]:
                    test_ids[file_path].append(test_name)
        except (IOError, yaml.YAMLError):
            continue

    return test_ids


def get_failure_analysis_from_cache(cache_path: str = ".dokumen-cache") -> Dict[str, Any]:
    """
    Get failure analysis from cache.

    Args:
        cache_path: Path to cache directory

    Returns:
        Dictionary with failure analysis data:
        {
            "file_path": {
                "test_id": {
                    "referenced_lines": [1, 2, 3],
                    "incorrect_lines": [
                        {"line_number": 5, "reason": "...", "confidence": 0.8}
                    ],
                    "analysis": "Overall analysis..."
                }
            }
        }
    """
    import json
    import os

    cache_file = os.path.join(cache_path, "cache.json")

    if not os.path.exists(cache_file):
        return {}

    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        return data.get("failure_analysis", {})
    except (json.JSONDecodeError, IOError):
        return {}


def get_line_coverage_stats(cache_path: str = ".dokumen-cache",
                            file_states: Dict[str, str] = None,
                            all_doc_files: List[str] = None) -> Dict[str, Any]:
    """
    Get line-level coverage statistics from cache, including failure information.

    Args:
        cache_path: Path to cache directory
        file_states: Optional dict mapping file_path to state (passed/failed/uncovered)
        all_doc_files: Optional list of all doc files (for computing uncovered line counts)

    Returns:
        Dictionary with line coverage data:
        {
            "total_lines": int,
            "covered_lines": int,
            "failed_lines": int,
            "percentage": float,
            "by_state": {
                "passed": int,    # Lines in covered_lines from passing tests
                "failed": int,    # Lines in failed_lines from failing tests
                "uncovered": int  # Lines in files with no tests or not run
            },
            "files": {
                "path": {
                    "total_lines": int,
                    "covered_lines": [list of ints],
                    "failed_lines": [list of ints],
                    "incorrect_lines": [{"line_number": int, "reason": str, "confidence": float}],
                    "covered_count": int,
                    "failed_count": int,
                    "percentage": float,
                    "source_tests": {line: [test_ids]},
                    "failed_tests": {line: [test_ids]},
                    "status": "passed" | "failed" | "uncovered"
                }
            }
        }
    """
    import json
    import os

    cache_file = os.path.join(cache_path, "cache.json")

    empty_result = {
        "total_lines": 0,
        "covered_lines": 0,
        "failed_lines": 0,
        "percentage": 0.0,
        "by_state": {"passed": 0, "failed": 0, "uncovered": 0},
        "files": {}
    }

    # Try to load cache
    cache_data = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Normalize paths from cache for consistent comparison
    raw_line_coverage = cache_data.get("line_coverage", {})
    line_coverage = {normalize_path(path): cov for path, cov in raw_line_coverage.items()}
    raw_file_status = cache_data.get("file_status", {})
    cached_file_status = {normalize_path(path): status for path, status in raw_file_status.items()}
    raw_failure_analysis = cache_data.get("failure_analysis", {})
    failure_analysis = {normalize_path(path): analysis for path, analysis in raw_failure_analysis.items()}
    # Files where coverage inference was attempted (even if it failed)
    coverage_attempted = set(normalize_path(p) for p in cache_data.get("coverage_attempted", []))

    total_lines = 0
    covered_lines = 0
    failed_lines = 0
    lines_by_state = {"passed": 0, "failed": 0, "uncovered": 0}
    files_data = {}

    # Process files from cache
    # Only include files that are tracked (in all_doc_files) when that list is provided
    all_doc_files_set = set(all_doc_files) if all_doc_files else None
    for file_path, coverage in line_coverage.items():
        # Skip non-tracked files when all_doc_files is provided
        if all_doc_files_set and file_path not in all_doc_files_set:
            continue

        file_total = coverage.get("total_lines", 0)
        file_covered = coverage.get("covered_lines", [])
        file_failed = coverage.get("failed_lines", [])
        file_incorrect = coverage.get("incorrect_lines", [])
        file_covered_count = len(file_covered)
        file_failed_count = len(file_failed)

        total_lines += file_total
        covered_lines += file_covered_count
        failed_lines += file_failed_count

        # Get status from provided file_states only - ensures consistency with file coverage
        # Files not in file_states are treated as "uncovered" to match file coverage semantics
        if file_states:
            status = file_states.get(file_path, "uncovered")
        else:
            status = cached_file_status.get(file_path, "uncovered")

        # Accumulate lines by state based on actual line coverage
        # passed = lines that were actually read/covered
        # failed = lines in failed tests
        # uncovered = remaining lines
        lines_by_state["passed"] += file_covered_count
        lines_by_state["failed"] += file_failed_count
        uncovered_in_file = file_total - file_covered_count - file_failed_count
        lines_by_state["uncovered"] += max(0, uncovered_in_file)

        files_data[file_path] = {
            "total_lines": file_total,
            "covered_lines": file_covered,
            "failed_lines": file_failed,
            "incorrect_lines": file_incorrect,
            "covered_count": file_covered_count,
            "failed_count": file_failed_count,
            "percentage": (file_covered_count / file_total * 100) if file_total > 0 else 0.0,
            "source_tests": coverage.get("source_test_ids", {}),
            "failed_tests": coverage.get("failed_test_ids", {}),
            "status": status
        }

    # Process files not in cache but in all_doc_files
    # These files have no line coverage data, so all lines are uncovered
    if all_doc_files and file_states:
        for file_path in all_doc_files:
            if file_path not in files_data:
                status = file_states.get(file_path, "uncovered")
                # Count lines in the file
                file_total = _count_file_lines(file_path)
                if file_total > 0:
                    total_lines += file_total
                    # No coverage data = all lines uncovered
                    lines_by_state["uncovered"] += file_total
                    files_data[file_path] = {
                        "total_lines": file_total,
                        "covered_lines": [],
                        "failed_lines": [],
                        "incorrect_lines": [],
                        "covered_count": 0,
                        "failed_count": 0,
                        "percentage": 0.0,
                        "source_tests": {},
                        "failed_tests": {},
                        "status": status
                    }

    # Incorporate failure_analysis data into files_data
    # This adds referenced_lines as failed_lines and incorrect_lines from failure analysis
    for file_path, analyses in failure_analysis.items():
        # Skip non-tracked files when all_doc_files is provided
        if all_doc_files_set and file_path not in all_doc_files_set:
            continue

        if file_path not in files_data:
            # Create entry for file not yet in files_data
            file_total = _count_file_lines(file_path)
            if file_total > 0:
                files_data[file_path] = {
                    "total_lines": file_total,
                    "covered_lines": [],
                    "failed_lines": [],
                    "incorrect_lines": [],
                    "covered_count": 0,
                    "failed_count": 0,
                    "percentage": 0.0,
                    "source_tests": {},
                    "failed_tests": {},
                    "status": cached_file_status.get(file_path, "failed")
                }

        if file_path in files_data:
            # Merge referenced_lines from all test analyses as failed_lines
            all_referenced = set(files_data[file_path].get("failed_lines", []))
            all_incorrect = list(files_data[file_path].get("incorrect_lines", []))

            for test_id, analysis in analyses.items():
                # Add referenced_lines as failed_lines
                referenced = analysis.get("referenced_lines", [])
                all_referenced.update(referenced)

                # Add incorrect_lines
                incorrect = analysis.get("incorrect_lines", [])
                for item in incorrect:
                    # Avoid duplicates by line number
                    if not any(i.get("line_number") == item.get("line_number") for i in all_incorrect):
                        all_incorrect.append(item)

            files_data[file_path]["failed_lines"] = sorted(all_referenced)
            files_data[file_path]["failed_count"] = len(all_referenced)
            files_data[file_path]["incorrect_lines"] = all_incorrect

    # For files with "passed" status but no line_coverage data, mark all lines as covered
    # ONLY if coverage was NOT attempted (meaning coverage agent was disabled)
    # If coverage was attempted but failed (e.g., token limit), don't assume 100% coverage
    for file_path, data in files_data.items():
        if data.get("status") == "passed" and not data.get("covered_lines"):
            # Only apply fallback if coverage was NOT attempted for this file
            if file_path not in coverage_attempted:
                file_total = data.get("total_lines", 0)
                if file_total > 0:
                    # Mark all lines as covered (fallback when coverage agent was disabled)
                    all_lines = list(range(1, file_total + 1))
                    data["covered_lines"] = all_lines
                    data["covered_count"] = file_total
                    data["percentage"] = 100.0
                    data["fallback_applied"] = True  # Mark that this is fallback data
                    # Update overall covered_lines count and move from uncovered to passed
                    covered_lines += file_total
                    lines_by_state["passed"] += file_total
                    lines_by_state["uncovered"] -= file_total

    # For files with "failed" status but no line_coverage data, mark all lines as failed
    # ONLY if coverage was NOT attempted (meaning coverage agent was disabled)
    for file_path, data in files_data.items():
        if data.get("status") == "failed" and not data.get("failed_lines"):
            # Only apply fallback if coverage was NOT attempted for this file
            if file_path not in coverage_attempted:
                file_total = data.get("total_lines", 0)
                if file_total > 0:
                    # Mark all lines as failed (fallback when coverage agent was disabled)
                    all_lines = list(range(1, file_total + 1))
                    data["failed_lines"] = all_lines
                    data["failed_count"] = file_total
                    data["fallback_applied"] = True  # Mark that this is fallback data
                    # Update overall failed_lines count and move from uncovered to failed
                    failed_lines += file_total
                    lines_by_state["failed"] += file_total
                    lines_by_state["uncovered"] -= file_total

    return {
        "total_lines": total_lines,
        "covered_lines": covered_lines,
        "failed_lines": failed_lines,
        "percentage": (covered_lines / total_lines * 100) if total_lines > 0 else 0.0,
        "by_state": lines_by_state,
        "files": files_data,
        "failure_analysis": failure_analysis
    }


def _count_file_lines(file_path: str) -> int:
    """Count lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except (IOError, UnicodeDecodeError):
        return 0


# =============================================================================
# Test Filtering
# =============================================================================

def filter_tests(all_tests, test_ids=None, grep=None, for_file=None, folder=None):
    """Filter tests based on criteria.

    Args:
        all_tests: List of test objects
        test_ids: Optional list of test IDs to filter by
        grep: Optional pattern to match test ID or reason
        for_file: Optional file path to filter tests covering that file
        folder: Optional folder path to filter by. "" = all tests, "." = root-level only

    Returns:
        Filtered list of tests
    """
    result = list(all_tests)

    if test_ids:
        result = [t for t in result if t.id in test_ids]

    if grep:
        result = [t for t in result if fnmatch.fnmatch(t.id, grep) or grep in t.reason]

    if for_file:
        result = [t for t in result if for_file in [f.path for f in t.files]]

    if folder is not None:  # "" = all tests, "." = root-only, but None = no filter
        validated_folder = validate_folder_path(folder)
        filtered = []
        for t in result:
            # Get source_path from test object
            source_path = getattr(t, 'source_path', None)
            if source_path:
                try:
                    test_folder = normalize_folder_path(source_path)
                    if is_in_folder(test_folder, validated_folder):
                        filtered.append(t)
                except ValueError:
                    # Skip tests with invalid paths
                    continue
            else:
                # If no source_path, include only when targeting root
                if validated_folder in ("", "."):
                    filtered.append(t)
        result = filtered

    return result


def filter_scaffold_paths(
    scaffold_paths: List[str],
    test_names: List[str],
    grep_pattern: Optional[str]
) -> List[str]:
    """
    Filter scaffold paths by test name or grep pattern.

    Test names are matched against the 'name' field in the YAML.
    Grep patterns use fnmatch for wildcard matching.
    Test names can also be fnmatch patterns.

    Args:
        scaffold_paths: List of paths to scaffold YAML files
        test_names: List of test names to match (can include fnmatch patterns)
        grep_pattern: Grep pattern to match test names

    Returns:
        List of scaffold paths that match the criteria
    """
    from ..scaffold import get_scaffold_name

    if not test_names and not grep_pattern:
        return scaffold_paths

    filtered = []
    for path in scaffold_paths:
        # Extract test name from YAML
        name = get_scaffold_name(path)
        if name is None:
            continue  # Skip unparseable files

        matched = False

        # Match by test names (exact match or fnmatch pattern)
        if test_names:
            if name in test_names:
                matched = True
            else:
                # Try fnmatch patterns for each test name
                for t in test_names:
                    if fnmatch.fnmatch(name, t):
                        matched = True
                        break

        # Match by grep pattern
        if grep_pattern and not matched:
            if fnmatch.fnmatch(name, grep_pattern):
                matched = True

        if matched:
            filtered.append(path)

    return filtered
