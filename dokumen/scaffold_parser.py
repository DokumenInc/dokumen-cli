"""
Scaffold parsing utilities for the Documentation Unit Test Framework.

Handles YAML scaffold parsing, prompt variable substitution, browser/viewport
config parsing, model normalization, and file path extraction.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import re

from dokumen_schema.constants import KNOWN_MODEL_ALIASES

from .logging_config import get_logger
from .test_object import BrowserConfig

logger = get_logger(__name__)


def substitute_prompt_variables(prompt: str, variables: Dict[str, str]) -> str:
    """
    Substitute variables in a prompt string.

    Variables are specified as {variable_name} and replaced with their values.

    Args:
        prompt: The prompt string containing variable placeholders
        variables: Dict mapping variable names to their values

    Returns:
        Prompt with variables substituted
    """
    if not prompt or not variables:
        return prompt or ""

    result = prompt
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", str(value))

    return result


def find_project_root(start_path: str) -> str:
    """
    Find the project root by searching upward for dokumen.yaml.

    Args:
        start_path: Path to start searching from

    Returns:
        Project root directory, or start_path's directory if not found
    """
    current = Path(start_path).resolve()
    if current.is_file():
        current = current.parent

    while current != current.parent:
        if (current / "dokumen.yaml").exists():
            return str(current)
        current = current.parent

    # Fallback to start path's directory
    return str(Path(start_path).parent if Path(start_path).is_file() else start_path)


def normalize_raw_model(raw: Any) -> Optional[str]:
    """Normalize a raw model value: strip, empty->None, resolve aliases.

    Returns None for non-string, empty, whitespace-only, or invalid format values.
    Valid model strings must be <=200 chars and match [a-zA-Z0-9._-]+.
    """
    if not raw or not isinstance(raw, str):
        return None
    v = raw.strip()
    if not v:
        return None
    if len(v) > 200 or not re.match(r'^[a-zA-Z0-9._-]+$', v):
        return None
    return KNOWN_MODEL_ALIASES.get(v, v)


def parse_max_iterations(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Parse a max_iterations value, returning default if None.

    Raises ValueError for invalid non-None values.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid max_iterations value: {value}") from exc


def default_executor_iterations(tool_names: List[str]) -> int:
    """Determine default max iterations based on tool types.

    Browser, research, and code tools get higher iteration limits.
    """
    from .playwright_tools import BROWSER_TOOLS
    from .tools_object import CODE_TOOLS

    if any(name in BROWSER_TOOLS for name in tool_names):
        return 100
    if "web_search" in tool_names:
        return 100
    if any(name in CODE_TOOLS for name in tool_names):
        return 100
    return 100


def parse_browser_config(raw: Any) -> Optional[BrowserConfig]:
    """Parse browser configuration from scaffold data.

    Args:
        raw: Raw browser config dict from scaffold YAML

    Returns:
        BrowserConfig or None if not configured
    """
    if not raw or not isinstance(raw, dict):
        return None

    viewport_size = parse_viewport_size(raw.get("viewport") or raw.get("viewport_size"))
    return BrowserConfig(
        headless=raw.get("headless"),
        save_video=raw.get("save_video"),
        viewport_size=viewport_size
    )


def parse_viewport_size(value: Any) -> Optional[str]:
    """Parse viewport size from various formats.

    Supports: string "1280x720", list [1280, 720], dict {"width": 1280, "height": 720}

    Returns:
        Viewport string like "1280x720" or None
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]}x{value[1]}"
    if isinstance(value, dict):
        width = value.get("width")
        height = value.get("height")
        if width is None or height is None:
            return None
        return f"{width}x{height}"
    return None


def extract_test_name(scaffold_path: str) -> str:
    """Extract test name from a scaffold file path.

    Tries to parse the YAML to get the 'name' field. Falls back to
    deriving the name from the filename (strip .test.yaml suffix).

    Args:
        scaffold_path: Path to the scaffold YAML file

    Returns:
        Test name string
    """
    import yaml

    # Try to extract name from YAML content
    try:
        with open(scaffold_path, 'r') as f:
            data = yaml.safe_load(f)
        name_val = data.get('name') if isinstance(data, dict) else None
        if isinstance(name_val, str) and name_val:
            return name_val
    except (OSError, yaml.YAMLError):
        pass

    # Fall back to filename-based extraction
    path = Path(scaffold_path)
    name = path.name
    # Strip .test.yaml or .test.yml suffix
    for suffix in ['.test.yaml', '.test.yml']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def extract_file_paths(files_data: list) -> List[str]:
    """Extract file paths from scaffold files section.

    Handles both dict format ({"path": "..."}) and plain string format.

    Args:
        files_data: List of file entries from scaffold YAML

    Returns:
        List of file path strings
    """
    file_paths = []
    for f in files_data:
        if isinstance(f, dict):
            file_paths.append(f.get('path', ''))
        elif isinstance(f, str):
            file_paths.append(f)
    return file_paths
