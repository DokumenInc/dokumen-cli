"""
Coverage report parser for CLI analyzer integration.

Parses pytest/coverage.py JSON reports and maps coverage data
to CLI commands for prioritization in the cli-problems analyzer.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CommandCoverage:
    """Coverage information for a CLI command."""
    command: str            # e.g., "dokumen run"
    module_path: str        # e.g., "dokumen/cli/commands/run.py"
    percent_covered: float  # 0.0 to 100.0
    covered_lines: int
    total_lines: int


def parse_coverage_json(coverage_path: str) -> Dict[str, dict]:
    """
    Parse a coverage JSON file (supports both coverage.json and htmlcov/status.json).

    Args:
        coverage_path: Path to coverage.json or status.json

    Returns:
        Dict with file paths as keys, coverage data as values

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is invalid JSON
        ValueError: If JSON structure is unexpected
    """
    with open(coverage_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'files' not in data:
        raise ValueError("Invalid coverage JSON: missing 'files' key")

    # Check if this is htmlcov/status.json format
    first_key = next(iter(data['files']), None)
    if first_key and 'index' in data['files'].get(first_key, {}):
        # Convert htmlcov/status.json format to standard format
        return _convert_status_json(data['files'])

    return data['files']


def _convert_status_json(files: Dict[str, dict]) -> Dict[str, dict]:
    """
    Convert htmlcov/status.json format to standard coverage.json format.

    status.json format:
        "z_hash_filename_py": {
            "index": {
                "file": "dokumen\\cli\\commands\\run.py",
                "nums": {"n_statements": 77, "n_missing": 6, ...}
            }
        }

    Converts to:
        "dokumen/cli/commands/run.py": {
            "summary": {"percent_covered": 92.2, "covered_lines": 71, "num_statements": 77}
        }
    """
    result = {}
    for entry in files.values():
        index = entry.get('index', {})
        file_path = index.get('file', '')
        nums = index.get('nums', {})

        if not file_path:
            continue

        # Normalize path separators
        file_path = file_path.replace('\\', '/')

        n_statements = nums.get('n_statements', 0)
        n_missing = nums.get('n_missing', 0)
        covered = n_statements - n_missing
        percent = (covered / n_statements * 100) if n_statements > 0 else 0.0

        result[file_path] = {
            'summary': {
                'percent_covered': percent,
                'covered_lines': covered,
                'num_statements': n_statements,
                'missing_lines': n_missing,
            }
        }

    return result


def map_module_to_command(module_path: str) -> Optional[str]:
    """
    Map a CLI module path to its command name.

    Args:
        module_path: e.g., "dokumen/cli/commands/run.py"

    Returns:
        Command name like "dokumen run" or None if not a command module
    """
    # Normalize path separators
    module_path = module_path.replace('\\', '/')

    # Check if it's a CLI command module
    if '/cli/commands/' not in module_path:
        return None

    # Extract filename without extension
    filename = Path(module_path).stem

    # Special case mappings
    command_map = {
        'list_cmd': 'list',
        '__init__': None,  # Skip init files
    }

    if filename in command_map:
        cmd = command_map[filename]
        return f"dokumen {cmd}" if cmd else None

    return f"dokumen {filename}"


def extract_cli_coverage(
    coverage_data: Dict[str, dict],
    threshold: float = 70.0
) -> Tuple[List[CommandCoverage], List[CommandCoverage]]:
    """
    Extract CLI command coverage from parsed coverage data.

    Args:
        coverage_data: Parsed coverage file data
        threshold: Coverage percentage threshold (commands below this are "low")

    Returns:
        Tuple of (all_commands, low_coverage_commands)
    """
    all_commands: List[CommandCoverage] = []

    for file_path, file_data in coverage_data.items():
        command = map_module_to_command(file_path)
        if not command:
            continue

        summary = file_data.get('summary', {})
        percent = summary.get('percent_covered', 0.0)
        covered = summary.get('covered_lines', 0)
        total = summary.get('num_statements', 0)

        all_commands.append(CommandCoverage(
            command=command,
            module_path=file_path,
            percent_covered=percent,
            covered_lines=covered,
            total_lines=total
        ))

    # Sort by coverage (lowest first)
    all_commands.sort(key=lambda c: c.percent_covered)

    # Filter low-coverage commands
    low_coverage = [c for c in all_commands if c.percent_covered < threshold]

    return all_commands, low_coverage


def format_coverage_prompt_injection(
    low_coverage_commands: List[CommandCoverage],
    threshold: float = 70.0
) -> str:
    """
    Format coverage data for injection into the system prompt.

    Args:
        low_coverage_commands: List of commands with low coverage
        threshold: The threshold used to determine "low coverage"

    Returns:
        Formatted string to inject into system prompt
    """
    if not low_coverage_commands:
        return ""

    lines = [
        "",
        "PRIORITY TESTING - LOW CODE COVERAGE:",
        f"The following commands have less than {threshold:.0f}% code coverage.",
        "PRIORITIZE testing these commands first, as they have less test coverage:",
        ""
    ]

    for cmd in low_coverage_commands:
        lines.append(
            f"  - {cmd.command}: {cmd.percent_covered:.1f}% coverage "
            f"({cmd.covered_lines}/{cmd.total_lines} lines)"
        )

    lines.append("")
    lines.append(
        "Focus on edge cases, error conditions, and untested code paths "
        "for these commands."
    )
    lines.append("")

    return "\n".join(lines)
