"""
Test scaffold models for dokumen-cli.

Thin wrapper around dokumen_schema (shared validation package).
Adds filesystem I/O operations on top of the pure validation.
"""

from pathlib import Path
from typing import List

import yaml

# Re-export all models and constants from the shared package
from dokumen_schema import (
    VALID_EXECUTOR_TOOLS,
    BrowserScaffoldConfig,
    DockerMount,
    ExecutorConfig,
    FileRef,
    JudgeConfig,
    SandboxConfig,
    TestScaffold,
    validate_test_data,
)


class ScaffoldError(Exception):
    """Scaffold-related errors."""

    pass


def load_scaffold(path: str) -> TestScaffold:
    """
    Load and validate a test scaffold from a YAML file.

    Args:
        path: Path to the .test.yaml file.

    Returns:
        Validated TestScaffold instance.

    Raises:
        ScaffoldError: If the file is not found, invalid YAML, or validation fails.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise ScaffoldError(f"Scaffold file not found: {path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ScaffoldError(f"Invalid YAML in scaffold file: {e}")

    if raw_data is None:
        raise ScaffoldError(f"Empty scaffold file: {path}")

    try:
        return TestScaffold(**raw_data)
    except Exception as e:
        error_msg = str(e)
        raise ScaffoldError(f"Scaffold validation failed: {error_msg}")


def validate_scaffold_file(path: str) -> tuple[bool, List[str], List[str]]:
    """
    Validate a scaffold file and return errors and warnings.

    Wraps the shared validate_test_data() function and adds filesystem I/O
    checks (file existence for referenced docs).

    Args:
        path: Path to the .test.yaml file.

    Returns:
        Tuple of (valid, errors, warnings).
        - valid: True if the scaffold is valid (may still have warnings)
        - errors: List of error messages (validation failures)
        - warnings: List of warning messages (non-critical issues)
    """
    errors: List[str] = []
    validation_warnings: List[str] = []

    file_path = Path(path)

    if not file_path.exists():
        errors.append(f"Scaffold file not found: {path}")
        return False, errors, validation_warnings

    try:
        with open(file_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML: {e}")
        return False, errors, validation_warnings

    if raw_data is None:
        errors.append(f"Empty scaffold file: {path}")
        return False, errors, validation_warnings

    # Use shared pure validation (schema + semantic checks)
    result = validate_test_data(raw_data)
    if not result.valid:
        return False, result.errors, result.warnings

    # Additional I/O checks (not in shared package)
    # Check that referenced doc files exist on disk
    scaffold = TestScaffold(**raw_data)
    for file_ref in scaffold.files:
        if not Path(file_ref.path).exists():
            errors.append(f"Referenced file not found: {file_ref.path}")

    if errors:
        return False, errors, validation_warnings

    return True, errors, result.warnings
