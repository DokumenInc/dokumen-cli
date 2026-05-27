"""Dokumen content filter — clean/smudge for git.

Implements bidirectional content transformation via git's clean/smudge
filter mechanism. The clean filter adds the `: DOKUMEN\\n` prefix before
content is stored in git objects. The smudge filter strips it on checkout.

Usage as console_scripts entry point:
    dokumen-filter --clean   < file > cleaned
    dokumen-filter --smudge  < file > smudged

Usage as module:
    python -m dokumen.workspace.filter --clean
    python -m dokumen.workspace.filter --smudge
"""

import logging
import sys

logger = logging.getLogger(__name__)

DOKUMEN_PREFIX = b": DOKUMEN\n"


def _is_binary(data: bytes) -> bool:
    """Detect binary content by checking for null bytes in first 8KB."""
    return b"\x00" in data[:8192]


def smudge(data: bytes) -> bytes:
    """Strip DOKUMEN prefix on checkout (GitLab → working tree)."""
    if _is_binary(data):
        return data
    if data.startswith(DOKUMEN_PREFIX):
        return data[len(DOKUMEN_PREFIX) :]
    return data


def clean(data: bytes) -> bytes:
    """Add DOKUMEN prefix on stage (working tree → git objects)."""
    if _is_binary(data):
        return data
    if data.startswith(DOKUMEN_PREFIX):
        return data  # idempotent
    return DOKUMEN_PREFIX + data


def main():
    """Entry point for dokumen-filter command."""
    if len(sys.argv) < 2 or sys.argv[1] not in ("--clean", "--smudge"):
        print("Usage: dokumen-filter --clean|--smudge", file=sys.stderr)
        sys.exit(1)

    data = sys.stdin.buffer.read()
    fn = clean if sys.argv[1] == "--clean" else smudge
    sys.stdout.buffer.write(fn(data))


if __name__ == "__main__":
    main()
