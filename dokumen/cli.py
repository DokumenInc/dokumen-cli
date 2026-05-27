"""
CLI for the Dokumen Documentation Unit Test Framework.

This module re-exports from dokumen.cli for backward compatibility.
The implementation has been modularized into dokumen/cli/ directory.
"""
# Re-export everything from the new modular CLI
from .cli import (
    # Main CLI
    cli,
    DokumenGroup,
    BANNER,
    # Exit codes
    EXIT_SUCCESS,
    EXIT_FAILURE,
    EXIT_CONFIG_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_INVALID_ARGS,
    # Helpers
    run_async,
    DEFAULT_CONFIG,
    load_config,
    deep_merge,
    get_coverage_stats,
    discover_doc_files,
    get_uncovered_files,
    filter_tests,
    # Backward-compatible private names
    _load_config,
    _deep_merge,
    _discover_doc_files,
    _filter_tests,
    _run_tests,
    # Test helpers
    run_test_suite,
    run_test_by_id,
    find_tests_for_file,
    get_failed_tests,
)

# Entry point
if __name__ == '__main__':
    cli()
