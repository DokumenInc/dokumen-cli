"""Git remote helper for dokumen:// URLs.

Implements the git-remote-helpers protocol to let git understand
dokumen:// URLs. When git encounters `dokumen://project`, it invokes
`git-remote-dokumen` which resolves the project via the Dokumen API
and delegates to git-remote-https pointing at the git proxy.

Usage:
    git clone dokumen://my-project
    git push / git pull  (via dokumen:// remote)

Environment:
    DOKUMEN_PAT — GitLab Personal Access Token (required)
    DOKUMEN_API_URL — Override default API URL (optional)

Protocol:
    Git sends commands on stdin, we respond on stdout.
    We resolve the project URL, set up credentials, then exec into
    git-remote-https which handles the smart HTTP protocol.
"""

import logging
import os
import stat
import subprocess
import sys
import tempfile

from dokumen.workspace.credentials import get_pat, mask_pat
from dokumen.workspace.resolver import resolve_project
from dokumen.workspace.url_parser import parse_url

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.dokumen.app"

FILTER_CONFIG = [
    ("filter.dokumen.clean", "dokumen-filter --clean"),
    ("filter.dokumen.smudge", "dokumen-filter --smudge"),
    ("filter.dokumen.required", "true"),
]


def _ensure_filter_configured():
    """Ensure dokumen clean/smudge/required are all configured in global git config.

    Fail-closed: if config cannot be written, exit(1) with remediation steps.
    """
    needs_update = False
    for key, value in FILTER_CONFIG:
        result = subprocess.run(
            ["git", "config", "--global", key],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or result.stdout.strip() != value:
            needs_update = True
            break

    if not needs_update:
        logger.debug("git_filter.already_configured")
        return

    try:
        for key, value in FILTER_CONFIG:
            subprocess.run(
                ["git", "config", "--global", "--replace-all", key, value],
                check=True,
            )
        logger.info("git_filter.configured")
    except subprocess.CalledProcessError:
        print(
            "Error: Could not configure git filter in global config.\n"
            "Run manually:\n"
            "  git config --global filter.dokumen.clean 'dokumen-filter --clean'\n"
            "  git config --global filter.dokumen.smudge 'dokumen-filter --smudge'\n"
            "  git config --global filter.dokumen.required true",
            file=sys.stderr,
        )
        sys.exit(1)


def handle_capabilities(output):
    """Respond to the 'capabilities' command.

    We advertise 'connect' which tells git we can establish
    a bidirectional channel to the remote.
    """
    output.write(b"connect\n")
    output.write(b"\n")


def handle_connect(project: str, api_url: str) -> tuple[str, str]:
    """Resolve project and return (proxy_url, pat).

    Args:
        project: Project name from the dokumen:// URL.
        api_url: Dokumen API base URL.

    Returns:
        Tuple of (git_proxy_url, pat).
    """
    pat = get_pat()
    logger.info("Resolving project", extra={"project": project, "pat": mask_pat(pat)})

    result = resolve_project(project=project, pat=pat, api_url=api_url)
    return result["git_proxy_url"], pat


def main() -> None:
    """Entry point for git-remote-dokumen.

    Called by git when it encounters a dokumen:// URL.
    argv[1] is the remote name, argv[2] is the URL.

    Strategy: resolve the dokumen:// URL to an HTTPS proxy URL,
    then exec into git-remote-https to handle the smart HTTP protocol.
    """
    if len(sys.argv) < 3:
        print("Usage: git-remote-dokumen <remote> <url>", file=sys.stderr)
        sys.exit(1)

    remote_name = sys.argv[1]
    url = sys.argv[2]

    # Configure logging
    log_level = os.environ.get("DOKUMEN_LOG_LEVEL", "WARNING")
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.WARNING))

    logger.debug("git-remote-dokumen invoked", extra={"remote": remote_name, "url": url})

    try:
        parsed = parse_url(url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    api_url = os.environ.get("DOKUMEN_API_URL", DEFAULT_API_URL)
    if parsed.api_host:
        api_url = f"https://{parsed.api_host}"

    # Resolve the project to get the proxy URL
    try:
        pat = get_pat()
        result = resolve_project(project=parsed.project, pat=pat, api_url=api_url)
        proxy_url = result["git_proxy_url"]
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    logger.info(
        "Resolved proxy URL",
        extra={"proxy_url": proxy_url, "project": parsed.project},
    )

    # Embed credentials in the proxy URL for git-remote-https.
    # Format: https://oauth2:<PAT>@host/path
    from urllib.parse import urlparse, urlunparse

    parsed_proxy = urlparse(proxy_url)
    authed_url = urlunparse(
        parsed_proxy._replace(netloc=f"oauth2:{pat}@{parsed_proxy.netloc}")
    )
    os.environ["GIT_TERMINAL_PROMPT"] = "0"

    # Ensure clean/smudge filter is configured before delegating to git
    _ensure_filter_configured()

    # Exec into git-remote-https with the authenticated proxy URL.
    # git-remote-https reads the same commands from stdin that we would,
    # and handles the smart HTTP protocol (info/refs, upload-pack, etc.)
    logger.info(
        "Delegating to git-remote-https",
        extra={"proxy_url": proxy_url},
    )

    os.execvp("git-remote-https", ["git-remote-https", remote_name, authed_url])


def _create_askpass_script(pat: str) -> str:
    """Create a temporary script that provides the PAT as password.

    Git calls GIT_ASKPASS with a prompt string. We always return the PAT.
    """
    script_content = f"""#!/bin/sh
echo "{pat}"
"""
    fd, path = tempfile.mkstemp(prefix="dokumen-askpass-", suffix=".sh")
    with os.fdopen(fd, "w") as f:
        f.write(script_content)
    os.chmod(path, stat.S_IRWXU)
    return path


if __name__ == "__main__":
    main()
