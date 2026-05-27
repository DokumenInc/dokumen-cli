"""Resolve dokumen:// project names to git proxy URLs via the backend API."""

import logging

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.dokumen.app"


def resolve_project(
    project: str,
    pat: str,
    api_url: str = DEFAULT_API_URL,
) -> dict:
    """Resolve a project name to git proxy information.

    Calls POST /api/workspace/resolve with the PAT and project name.

    Args:
        project: Project name from the dokumen:// URL.
        pat: GitLab Personal Access Token.
        api_url: Base URL of the Dokumen API.

    Returns:
        Dict with project_id, username, git_proxy_url, gitlab_url.

    Raises:
        RuntimeError: If resolution fails.
    """
    url = f"{api_url.rstrip('/')}/api/workspace/resolve"
    logger.info("Resolving project", extra={"project": project, "api_url": api_url})

    try:
        response = httpx.post(
            url,
            json={"pat": pat, "project": project},
            timeout=30.0,
            verify=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to resolve project '{project}': {e}")

    if response.status_code == 401:
        raise RuntimeError("Authentication failed: invalid PAT")

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to resolve project '{project}': "
            f"HTTP {response.status_code} — {response.text}"
        )

    data = response.json()
    logger.info(
        "Project resolved",
        extra={
            "project_id": data["project_id"],
            "username": data["username"],
            "git_proxy_url": data["git_proxy_url"],
        },
    )
    return data
