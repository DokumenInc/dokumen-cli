"""PAT credential handling for git-remote-dokumen."""

import logging
import os

logger = logging.getLogger(__name__)

ENV_VAR = "DOKUMEN_PAT"


def get_pat() -> str:
    """Get the PAT from the DOKUMEN_PAT environment variable.

    Raises:
        RuntimeError: If DOKUMEN_PAT is not set or empty.
    """
    pat = os.environ.get(ENV_VAR, "").strip()
    if not pat:
        raise RuntimeError(
            f"{ENV_VAR} environment variable is not set. "
            f"Set it with: export {ENV_VAR}=glpat-xxx"
        )
    logger.debug("PAT retrieved", extra={"pat": mask_pat(pat)})
    return pat


def mask_pat(pat: str) -> str:
    """Mask a PAT for safe logging, showing only first 8 chars."""
    if len(pat) <= 8:
        return "********"
    return pat[:8] + "********"
