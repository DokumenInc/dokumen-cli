"""Load agent config from backend API for tool merging.

When DOKUMEN_AGENT_ID is set (by the pipeline trigger in sandbox mode),
fetches the agent's tool list from the backend API and makes it available
for merging into scaffold tools during test execution.

This replaces the need for the separate sandbox Docker image's setup_agent.py
by integrating agent loading directly into the CLI.
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Map DB agent tool names to CLI tool names.
# DB tools use the MCP/API naming convention; the CLI uses different names.
AGENT_TOOL_NAME_MAP = {
    "list_files": "list_directory",
    "search_files": "search_file_content",
    "glob_files": "glob",
    # These are the same in both systems:
    # read_file, run_shell_command, web_fetch, web_search, glob
    # code_read_file, code_glob, code_search, code_list_directory
}

# Cache to avoid repeated API calls within the same CLI run
_cached_config: Optional[dict] = None
_cache_loaded: bool = False
_cached_skills: Optional[list] = None
_skills_loaded: bool = False


def _derive_api_base_url(mcp_endpoint: str) -> str:
    """Derive the API base URL from the MCP endpoint.

    MCP endpoint: https://api.dokumen.app/api/mcp/stream/mcp
    API base:     https://api.dokumen.app/api

    Args:
        mcp_endpoint: The DOKUMEN_MCP_ENDPOINT value.

    Returns:
        The API base URL (e.g. https://api.dokumen.app/api).
    """
    return mcp_endpoint.split("/api/mcp/")[0] + "/api"


def load_agent_config() -> Optional[dict]:
    """Fetch agent config from the backend API.

    Reads DOKUMEN_AGENT_ID and DOKUMEN_MCP_ENDPOINT from environment.
    Returns None (non-blocking) if env vars are missing or API call fails.

    Returns:
        Dict with agent_id, name, tools, model, max_steps — or None.
    """
    agent_id = os.environ.get("DOKUMEN_AGENT_ID", "").strip()
    if not agent_id:
        return None

    mcp_endpoint = os.environ.get("DOKUMEN_MCP_ENDPOINT", "").strip()
    if not mcp_endpoint:
        logger.warning(
            "DOKUMEN_AGENT_ID set but DOKUMEN_MCP_ENDPOINT missing",
            extra={"agent_id": agent_id},
        )
        return None

    gitlab_token = os.environ.get("GITLAB_TOKEN", "").strip()
    if not gitlab_token:
        logger.warning(
            "DOKUMEN_AGENT_ID set but GITLAB_TOKEN missing",
            extra={"agent_id": agent_id},
        )
        return None

    api_base = _derive_api_base_url(mcp_endpoint)
    url = f"{api_base}/agents/{agent_id}"

    logger.info(
        "Fetching agent config from backend",
        extra={"agent_id": agent_id, "url": url},
    )

    try:
        # Authenticate via session to get project context (PAT alone
        # doesn't carry project_id, causing "Company not found").
        session = requests.Session()
        auth_resp = session.post(
            f"{api_base}/auth/validate",
            json={"pat": gitlab_token},
            headers={"X-Requested-With": "XMLHttpRequest"},
            timeout=15,
        )
        auth_resp.raise_for_status()
        logger.info(
            "Agent loader authenticated",
            extra={"agent_id": agent_id, "username": auth_resp.json().get("user", {}).get("username")},
        )

        # Fetch agent config using session cookie
        resp = session.get(
            url,
            headers={"X-Requested-With": "XMLHttpRequest"},
            timeout=30,
        )
        resp.raise_for_status()
        agent = resp.json()
    except Exception as e:
        logger.error(
            "Failed to fetch agent config",
            extra={"agent_id": agent_id, "error": str(e)},
        )
        return None

    config = {
        "agent_id": agent_id,
        "name": agent.get("name", "unknown"),
        "tools": agent.get("tools", []) or [],
        "model": agent.get("model"),
        "max_steps": agent.get("max_steps", 25),
        "system_prompt": agent.get("system_prompt"),
    }

    logger.info(
        "Agent config loaded",
        extra={
            "agent_id": agent_id,
            "agent_name": config["name"],
            "tool_count": len(config["tools"]),
        },
    )

    return config


def _map_tool_names(tools: list[str]) -> list[str]:
    """Map DB agent tool names to CLI tool names.

    Args:
        tools: Tool names from the agent API.

    Returns:
        Tool names translated to CLI conventions.
    """
    mapped = []
    for tool in tools:
        cli_name = AGENT_TOOL_NAME_MAP.get(tool, tool)
        mapped.append(cli_name)
        if cli_name != tool:
            logger.debug(
                "Agent tool name mapped",
                extra={"db_name": tool, "cli_name": cli_name},
            )
    return mapped


def get_agent_tools() -> list[str]:
    """Get agent tools for merging into scaffold tools.

    Caches the result after the first call to avoid repeated API calls
    within a single CLI run. Tool names are mapped from DB conventions
    to CLI conventions.

    Returns:
        List of CLI tool names from the agent config, or empty list.
    """
    global _cached_config, _cache_loaded

    if _cache_loaded:
        if _cached_config is None:
            return []
        return _map_tool_names(list(_cached_config.get("tools", [])))

    _cached_config = load_agent_config()
    _cache_loaded = True

    if _cached_config is None:
        return []
    return _map_tool_names(list(_cached_config.get("tools", [])))


def _get_authenticated_session() -> Optional[requests.Session]:
    """Create an authenticated requests session for API calls.

    Reads DOKUMEN_MCP_ENDPOINT and GITLAB_TOKEN from environment.
    Returns None if env vars are missing or auth fails.

    Returns:
        Authenticated requests.Session or None.
    """
    mcp_endpoint = os.environ.get("DOKUMEN_MCP_ENDPOINT", "").strip()
    gitlab_token = os.environ.get("GITLAB_TOKEN", "").strip()

    if not mcp_endpoint or not gitlab_token:
        logger.debug(
            "Cannot create auth session: missing env vars",
            extra={"has_endpoint": bool(mcp_endpoint), "has_token": bool(gitlab_token)},
        )
        return None

    api_base = _derive_api_base_url(mcp_endpoint)

    try:
        session = requests.Session()
        auth_resp = session.post(
            f"{api_base}/auth/validate",
            json={"pat": gitlab_token},
            headers={"X-Requested-With": "XMLHttpRequest"},
            timeout=15,
        )
        auth_resp.raise_for_status()
        logger.info(
            "Authenticated session created",
            extra={"username": auth_resp.json().get("user", {}).get("username")},
        )
        return session
    except Exception as e:
        logger.error(
            "Failed to create authenticated session",
            extra={"error": str(e)},
        )
        return None


def get_agent_skills() -> list[dict]:
    """Get skills assigned to the agent from the backend API.

    Fetches skills via GET /agents/{agent_id}/skills. Each dict
    contains name, content, and description.

    Caches the result after the first call to avoid repeated API calls
    within a single CLI run.

    Returns:
        List of skill dicts with name, content, description — or empty list.
    """
    global _cached_skills, _skills_loaded, _cached_config, _cache_loaded

    if _skills_loaded:
        return _cached_skills or []

    _skills_loaded = True

    # Ensure agent config is loaded first
    if not _cache_loaded:
        _cached_config = load_agent_config()
        _cache_loaded = True

    if _cached_config is None:
        _cached_skills = []
        return []

    agent_id = _cached_config.get("agent_id", "")
    if not agent_id:
        _cached_skills = []
        return []

    mcp_endpoint = os.environ.get("DOKUMEN_MCP_ENDPOINT", "").strip()
    if not mcp_endpoint:
        _cached_skills = []
        return []

    api_base = _derive_api_base_url(mcp_endpoint)
    url = f"{api_base}/agents/{agent_id}/skills"

    logger.info(
        "Fetching agent skills from backend",
        extra={"agent_id": agent_id, "url": url},
    )

    session = _get_authenticated_session()
    if session is None:
        _cached_skills = []
        return []

    try:
        resp = session.get(
            url,
            headers={"X-Requested-With": "XMLHttpRequest"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(
            "Failed to fetch agent skills",
            extra={"agent_id": agent_id, "error": str(e)},
        )
        _cached_skills = []
        return []

    skills_list = data.get("skills", [])
    _cached_skills = [
        {
            "name": s.get("name", ""),
            "content": s.get("content", ""),
            "description": s.get("description", ""),
        }
        for s in skills_list
    ]

    logger.info(
        "Agent skills loaded",
        extra={
            "agent_id": agent_id,
            "skill_count": len(_cached_skills),
            "skill_names": [s["name"] for s in _cached_skills],
        },
    )

    return _cached_skills


def _reset_cache() -> None:
    """Reset the cache (for testing)."""
    global _cached_config, _cache_loaded, _cached_skills, _skills_loaded
    _cached_config = None
    _cache_loaded = False
    _cached_skills = None
    _skills_loaded = False
