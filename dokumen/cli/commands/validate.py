"""
Validate command for dokumen CLI.

Validates configuration and test scaffold files without executing tests.
"""
import glob as glob_mod
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from dokumen_schema import check_ci_compatibility
from dokumen_schema.suggestions import get_suggested_fix

from ...config import ConfigError, load_config as load_dokumen_config
from ...test_scaffold import validate_scaffold_file
from ...scaffold import discover_scaffolds, load_scaffold_yaml
from ..helpers import EXIT_SUCCESS, EXIT_CONFIG_ERROR, filter_scaffold_paths
from ...logging_config import get_logger

logger = get_logger(__name__)


def extract_yaml_snippet(file_path: str, search_term: str, context_lines: int = 2) -> Optional[str]:
    """Extract a snippet from a YAML file around a search term.

    Args:
        file_path: Path to the YAML file.
        search_term: Term to search for in the file.
        context_lines: Number of context lines before/after match.

    Returns:
        String snippet with line numbers or None if not found.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return None

        content = path.read_text()
        lines = content.split('\n')

        # Find line containing the search term
        match_idx = None
        for idx, line in enumerate(lines):
            if search_term in line:
                match_idx = idx
                break

        if match_idx is None:
            return None

        # Extract context
        start_idx = max(0, match_idx - context_lines)
        end_idx = min(len(lines), match_idx + context_lines + 1)

        snippet_lines = []
        for idx in range(start_idx, end_idx):
            line_num = idx + 1  # 1-indexed
            marker = "→ " if idx == match_idx else "  "
            snippet_lines.append(f"  {line_num:3d} {marker}{lines[idx]}")

        return '\n'.join(snippet_lines)
    except Exception:
        return None


def validate_pdf_constraints(scaffold_path: str, base_dir: Path) -> List[str]:
    """Validate PDF file constraints for a test scaffold.

    Checks:
    - PDF files do not exceed 4.5MB size limit
    - Test does not reference more than 5 PDFs

    Args:
        scaffold_path: Path to test scaffold file
        base_dir: Base directory for resolving file paths

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []

    try:
        # Load scaffold to get file references
        scaffold = load_scaffold_yaml(scaffold_path)
        files = scaffold.get("files", [])

        pdf_files = []

        for file_ref in files:
            path = file_ref.get("path", "")
            if path.lower().endswith(".pdf"):
                pdf_files.append(path)

                # Check size if file exists
                full_path = base_dir / path
                if full_path.exists():
                    size = full_path.stat().st_size
                    size_mb = size / 1024 / 1024

                    if size > 4.5 * 1024 * 1024:
                        warnings.append(
                            f"PDF exceeds 4.5MB limit: {path} ({size_mb:.1f}MB)"
                        )
                        logger.warning(f"PDF validation: oversized file", extra={
                            "file": path,
                            "size_mb": size_mb,
                            "limit_mb": 4.5
                        })

        # Check count
        if len(pdf_files) > 5:
            warnings.append(
                f"Test references {len(pdf_files)} PDF files (limit: 5 per test)"
            )
            logger.warning(f"PDF validation: too many PDFs", extra={
                "count": len(pdf_files),
                "limit": 5,
                "scaffold": scaffold_path
            })

    except Exception as e:
        logger.debug(f"PDF validation error (non-fatal)", extra={
            "scaffold": scaffold_path,
            "error": str(e)
        })
        # Don't fail validation if PDF check fails - just skip it

    return warnings


@click.command()
@click.argument("tests", nargs=-1)
@click.option("--grep", "-g", help="Filter tests by pattern")
@click.option("--config-only", is_flag=True, help="Validate configuration only, skip test files")
@click.option("--json", "json_output", is_flag=True, help="Output results as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed error context and suggested fixes")
@click.pass_context
def validate(ctx, tests: tuple, grep: Optional[str], config_only: bool, json_output: bool, verbose: bool):
    """Validate configuration and test scaffolds.

    Checks that dokumen.yaml is valid and all test scaffold files
    have correct syntax and required fields.

    Supports filtering by test name, grep pattern, or DOKUMEN_TESTS env var.

    Examples:

        dokumen validate              Validate all config and tests

        dokumen validate --config-only    Validate config only

        dokumen validate tests/my.test.yaml    Validate specific file

        dokumen validate my-test          Validate test by name

        dokumen validate --grep "api-*"   Validate tests matching pattern

        dokumen validate --verbose        Show detailed error context
    """
    config_path = ctx.obj.get("config_path") if ctx.obj else None

    errors: List[Dict[str, Any]] = []  # Now stores structured error info
    warnings: List[str] = []
    validated_files: List[str] = []

    # Read DOKUMEN_TESTS env var (CLI args override env vars)
    env_tests = os.environ.get('DOKUMEN_TESTS', '')
    if env_tests and not tests:  # Only use env var if no CLI tests specified
        tests = tuple(t.strip() for t in env_tests.split(',') if t.strip())
        click.echo(f"Filtering by DOKUMEN_TESTS: {len(tests)} test(s) selected")
        logger.info("DOKUMEN_TESTS filtering active", extra={"tests": list(tests), "count": len(tests)})

    # Step 1: Validate config
    try:
        config = load_dokumen_config(config_path)
    except ConfigError as e:
        error_info = {
            "message": f"Configuration error: {e}",
            "file": config_path or "dokumen.yaml",
            "suggestion": get_suggested_fix(str(e)),
        }
        errors.append(error_info)
        _output_results(json_output, verbose, False, errors, warnings, validated_files)
        ctx.exit(EXIT_CONFIG_ERROR)
        return

    # Step 2: If config-only, we're done
    if config_only:
        _output_results(json_output, verbose, True, errors, warnings, validated_files)
        ctx.exit(EXIT_SUCCESS)
        return

    # Step 3: Determine scaffold paths to validate
    # Separate file paths from test names
    file_paths = []
    test_names = []
    for t in tests:
        if '/' in t or '\\' in t or t.endswith('.yaml') or t.endswith('.yml'):
            file_paths.append(t)
        else:
            test_names.append(t)

    # If only file paths specified (no test names or grep), use those directly
    if file_paths and not test_names and not grep:
        scaffold_paths = file_paths
    else:
        # Discover all scaffolds
        all_scaffold_paths = discover_scaffolds("tests")

        # Filter by test names or grep if specified
        if test_names or grep:
            scaffold_paths = filter_scaffold_paths(all_scaffold_paths, test_names, grep)
            # Add any explicitly specified file paths
            for fp in file_paths:
                if fp not in scaffold_paths:
                    scaffold_paths.append(fp)
        else:
            scaffold_paths = all_scaffold_paths

    # Check if no scaffolds matched the filter
    if (test_names or grep) and not scaffold_paths:
        if test_names:
            error_info = {
                "message": f"No tests match: {', '.join(test_names)}",
                "file": None,
                "suggestion": "Check the test name spelling or use --grep for pattern matching.",
            }
        elif grep:
            error_info = {
                "message": f"No tests match pattern: {grep}",
                "file": None,
                "suggestion": "Try a different pattern or run 'dokumen list' to see available tests.",
            }
        else:
            error_info = {
                "message": "No tests match the specified criteria",
                "file": None,
                "suggestion": None,
            }
        errors.append(error_info)
        _output_results(json_output, verbose, False, errors, warnings, validated_files)
        ctx.exit(EXIT_CONFIG_ERROR)
        return

    # Gather existing files for CI compat file existence checks
    existing_files = _gather_existing_files()

    for scaffold_path in scaffold_paths:
        # Check if file exists
        if not Path(scaffold_path).exists():
            errors.append({
                "message": f"File not found: {scaffold_path}",
                "file": scaffold_path,
                "suggestion": "Verify the file path or create the missing test scaffold.",
            })
            continue

        validated_files.append(scaffold_path)
        valid, file_errors, file_warnings = validate_scaffold_file(scaffold_path)

        if not valid:
            for err in file_errors:
                error_msg = f"{scaffold_path}: {err}"
                error_info: Dict[str, Any] = {
                    "message": error_msg,
                    "file": scaffold_path,
                    "suggestion": get_suggested_fix(err),
                }

                # In verbose mode, try to extract relevant snippet
                if verbose:
                    # Try to find relevant search term from error
                    if "Referenced file not found:" in err:
                        # Extract the file path from the error
                        parts = err.split("Referenced file not found:")
                        if len(parts) > 1:
                            ref_path = parts[1].strip()
                            snippet = extract_yaml_snippet(scaffold_path, ref_path)
                            if snippet:
                                error_info["snippet"] = snippet
                    elif "Unknown tool:" in err:
                        parts = err.split("Unknown tool:")
                        if len(parts) > 1:
                            tool_name = parts[1].strip().strip("'\"")
                            snippet = extract_yaml_snippet(scaffold_path, tool_name)
                            if snippet:
                                error_info["snippet"] = snippet
                    elif "Hardcoded path" in err:
                        # Extract path from error
                        if "'" in err:
                            path_start = err.index("'") + 1
                            path_end = err.index("'", path_start)
                            hardcoded_path = err[path_start:path_end]
                            snippet = extract_yaml_snippet(scaffold_path, hardcoded_path)
                            if snippet:
                                error_info["snippet"] = snippet

                errors.append(error_info)

        for warn in file_warnings:
            warnings.append(f"{scaffold_path}: {warn}")

        # Check PDF constraints
        pdf_warnings = validate_pdf_constraints(scaffold_path, Path.cwd())
        for pdf_warn in pdf_warnings:
            warnings.append(f"{scaffold_path}: {pdf_warn}")

        # CI compatibility checks
        try:
            scaffold_data = load_scaffold_yaml(scaffold_path)
            ci_result = check_ci_compatibility(
                scaffold_data,
                allowed_tools=config.tools.allowed,
                existing_files=existing_files,
            )
            logger.info("CI compat check", extra={
                "scaffold": scaffold_path,
                "ci_compatible": ci_result.ci_compatible,
                "ci_error_count": len(ci_result.ci_errors),
            })

            for ci_err in ci_result.ci_errors:
                error_msg = f"{scaffold_path}: [CI] {ci_err}"
                error_info = {
                    "message": error_msg,
                    "file": scaffold_path,
                    "suggestion": get_suggested_fix(ci_err),
                    "ci": True,
                }
                errors.append(error_info)

            for ci_warn in ci_result.ci_warnings:
                warnings.append(f"{scaffold_path}: [CI] {ci_warn}")
        except Exception as e:
            logger.debug("CI compat check failed (non-fatal)", extra={
                "scaffold": scaffold_path,
                "error": str(e),
            })

    # Output results
    has_errors = len(errors) > 0
    _output_results(json_output, verbose, not has_errors, errors, warnings, validated_files)

    if has_errors:
        ctx.exit(EXIT_CONFIG_ERROR)
    else:
        ctx.exit(EXIT_SUCCESS)


def _gather_existing_files() -> List[str]:
    """Gather list of existing files in the repository for CI compat checks.

    Scans the current working directory for all files, returning relative paths.

    Returns:
        List of relative file paths that exist in the repo.
    """
    existing: List[str] = []
    cwd = Path.cwd()
    try:
        for path in cwd.rglob("*"):
            if path.is_file():
                try:
                    rel = str(path.relative_to(cwd))
                    # Skip hidden dirs and common non-content paths
                    if not any(part.startswith('.') for part in Path(rel).parts):
                        existing.append(rel)
                except ValueError:
                    continue
        logger.debug("Gathered existing files", extra={"count": len(existing)})
    except Exception as e:
        logger.debug("Failed to gather existing files", extra={"error": str(e)})
    return existing


def _output_results(
    json_output: bool,
    verbose: bool,
    valid: bool,
    errors: List[Dict[str, Any]],
    warnings: List[str],
    validated_files: List[str]
):
    """Output validation results in text or JSON format.

    Args:
        json_output: Whether to output as JSON.
        verbose: Whether to show detailed context.
        valid: Whether validation passed.
        errors: List of error dicts with message, file, suggestion.
        warnings: List of warning messages.
        validated_files: List of validated file paths.
    """
    if json_output:
        # Convert error dicts to include all available info
        if verbose:
            result = {
                "valid": valid,
                "errors": errors,  # Keep full structure
                "warnings": warnings,
                "files_validated": validated_files
            }
        else:
            # Simpler format for non-verbose JSON
            result = {
                "valid": valid,
                "errors": [e["message"] for e in errors],
                "warnings": warnings,
                "files_validated": validated_files
            }
        click.echo(json.dumps(result, indent=2))
    else:
        if valid:
            if validated_files:
                click.echo(click.style("✓ Validation passed", fg="green"))
                click.echo(f"  Validated {len(validated_files)} file(s)")
            else:
                click.echo(click.style("✓ Configuration valid", fg="green"))
                click.echo("  No test scaffolds found in tests/")
        else:
            click.echo(click.style("✗ Validation failed", fg="red"))

        if errors:
            error_count = len(errors)
            click.echo(f"\n{error_count} error{'s' if error_count != 1 else ''} found:")

            for err in errors:
                msg = err["message"]
                click.echo(f"\n  • {msg}")

                # Show snippet in verbose mode
                if verbose and err.get("snippet"):
                    click.echo(click.style("\n    Context:", fg="cyan"))
                    click.echo(err["snippet"])

                # Show suggestion in verbose mode
                if verbose and err.get("suggestion"):
                    click.echo(click.style(f"\n    💡 Fix: ", fg="yellow") + err["suggestion"])

        if warnings:
            click.echo("\nWarnings:")
            for warn in warnings:
                click.echo(f"  ⚠ {warn}")
