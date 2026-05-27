"""Parse dokumen:// URLs into components for the git remote helper."""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

SCHEME = "dokumen://"


@dataclass
class DokumenURL:
    """Parsed dokumen:// URL."""

    project: str
    api_host: Optional[str] = None

    def proxy_base_url(self, default_api_url: str) -> str:
        """Build the git proxy base URL.

        If api_host was specified in the URL, use it.
        Otherwise fall back to the default API URL.
        """
        if self.api_host:
            base = f"https://{self.api_host}"
        else:
            base = default_api_url.rstrip("/")
        return f"{base}/api/git-proxy"


def parse_url(url: str) -> DokumenURL:
    """Parse a dokumen:// URL into its components.

    Formats:
        dokumen://project-name
        dokumen://api-host/project-name
        dokumen://api-host:port/project-name
    """
    if not url.startswith(SCHEME):
        raise ValueError(f"URL must start with dokumen:// (got: {url})")

    remainder = url[len(SCHEME) :]

    if not remainder or remainder == "/":
        raise ValueError(f"URL must include a project name (got: {url})")

    logger.debug("Parsing URL", extra={"url": url, "remainder": remainder})

    if "/" in remainder:
        # dokumen://host/project or dokumen://host:port/project
        host, project = remainder.split("/", 1)
        if not project:
            raise ValueError(f"URL must include a project name (got: {url})")
        logger.debug("Parsed URL", extra={"api_host": host, "project": project})
        return DokumenURL(project=project, api_host=host)
    else:
        # dokumen://project
        logger.debug("Parsed URL", extra={"project": remainder})
        return DokumenURL(project=remainder)
