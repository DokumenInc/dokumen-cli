"""Integration test: git clean/smudge filter with real git operations.

Requires `dokumen-filter` on PATH (install via `pip install -e dokumen-cli/`).
"""

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo with dokumen filter configured locally."""
    repo = tmp_path / "repo"
    repo.mkdir()

    # Check dokumen-filter is available
    if not shutil.which("dokumen-filter"):
        pytest.skip("dokumen-filter not on PATH — install with: pip install -e dokumen-cli/")

    def run(cmd, **kwargs):
        return subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
            **kwargs,
        )

    # Init repo
    run(["git", "init"])
    run(["git", "config", "user.name", "Test User"])
    run(["git", "config", "user.email", "test@example.com"])

    # Configure filter in LOCAL repo config (not global)
    run(["git", "config", "filter.dokumen.clean", "dokumen-filter --clean"])
    run(["git", "config", "filter.dokumen.smudge", "dokumen-filter --smudge"])
    run(["git", "config", "filter.dokumen.required", "true"])

    # Create .gitattributes with path-scoped patterns (matches production config)
    attrs = repo / ".gitattributes"
    attrs.write_text(
        "docs/**/*.md filter=dokumen\n"
        "docs/**/*.yaml filter=dokumen\n"
        "docs/**/*.yml filter=dokumen\n"
        "docs/**/*.json filter=dokumen\n"
        "docs/**/*.toml filter=dokumen\n"
        "docs/**/*.txt filter=dokumen\n"
        "docs/**/*.rst filter=dokumen\n"
    )
    run(["git", "add", ".gitattributes"])
    run(["git", "commit", "-m", "add gitattributes"])

    # Create docs/ directory for path-scoped tests
    (repo / "docs").mkdir()

    return repo, run


class TestFilterIntegration:
    """Test clean/smudge filter with real git operations."""

    def test_real_git_add_checkout(self, git_repo):
        """Full roundtrip: write → add → commit → delete → checkout.

        Verifies:
        - Index content has `: DOKUMEN\\n` prefix (clean filter applied)
        - Working tree content has no prefix (smudge filter applied)
        """
        repo, run = git_repo

        # Write a markdown file inside docs/ (path-scoped filter)
        test_file = repo / "docs" / "test.md"
        test_file.write_text("Hello world\n")

        # Stage it
        run(["git", "add", "docs/test.md"])

        # Verify index content has the prefix
        result = run(["git", "show", ":docs/test.md"])
        assert result.stdout.startswith(": DOKUMEN\n"), (
            f"Index content should start with ': DOKUMEN\\n', got: {result.stdout[:30]!r}"
        )
        assert ": DOKUMEN\nHello world\n" == result.stdout

        # Commit
        run(["git", "commit", "-m", "add test.md"])

        # Delete working tree file and checkout
        test_file.unlink()
        run(["git", "checkout", "--", "docs/test.md"])

        # Verify working tree has NO prefix (smudge stripped it)
        content = test_file.read_text()
        assert content == "Hello world\n", (
            f"Working tree should have no prefix, got: {content[:30]!r}"
        )

    def test_root_md_file_unaffected(self, git_repo):
        """Markdown files outside docs/ are NOT filtered (path-scoped)."""
        repo, run = git_repo

        # Write a .md file at repo root (outside docs/)
        root_md = repo / "README.md"
        root_md.write_text("# Root README\n")
        run(["git", "add", "README.md"])

        # Verify index content has NO prefix (not in docs/)
        result = run(["git", "show", ":README.md"])
        assert result.stdout == "# Root README\n"

    def test_non_doc_extension_unaffected(self, git_repo):
        """Non-matching extensions inside docs/ are not filtered."""
        repo, run = git_repo

        # Write a .py file inside docs/ (not a matched extension)
        py_file = repo / "docs" / "script.py"
        py_file.write_text("print('hello')\n")
        run(["git", "add", "docs/script.py"])

        # Verify index content has NO prefix
        result = run(["git", "show", ":docs/script.py"])
        assert result.stdout == "print('hello')\n"

    def test_multiline_roundtrip(self, git_repo):
        """Multiline content survives roundtrip through clean/smudge."""
        repo, run = git_repo

        content = "# Title\n\nParagraph one.\n\nParagraph two.\n"
        test_file = repo / "docs" / "doc.md"
        test_file.write_text(content)

        run(["git", "add", "docs/doc.md"])
        run(["git", "commit", "-m", "add doc.md"])

        # Delete and checkout
        test_file.unlink()
        run(["git", "checkout", "--", "docs/doc.md"])

        assert test_file.read_text() == content

    def test_yaml_in_docs_filtered(self, git_repo):
        """YAML files inside docs/ are filtered."""
        repo, run = git_repo

        test_file = repo / "docs" / "config.yaml"
        test_file.write_text("version: 1.0\n")

        run(["git", "add", "docs/config.yaml"])

        result = run(["git", "show", ":docs/config.yaml"])
        assert result.stdout.startswith(": DOKUMEN\n")
