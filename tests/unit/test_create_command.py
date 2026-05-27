"""
Tests for dokumen create command.

TDD tests for the create command, including:
- Basic execution (create test scaffolds)
- Output formats (yaml, json)
- Stdin mode for backend integration
- Dry-run mode
"""
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner


def extract_json_from_output(output: str) -> dict:
    """Extract JSON from CLI output, ignoring log lines."""
    # Find the first '{' which starts JSON
    start_idx = output.find('{')
    if start_idx == -1:
        return json.loads(output)

    # Try to parse from the start position
    decoder = json.JSONDecoder()
    try:
        result, end_idx = decoder.raw_decode(output[start_idx:])
        return result
    except json.JSONDecodeError:
        return json.loads(output)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def runner():
    """Click CLI test runner with separate stderr."""
    return CliRunner()


@pytest.fixture
def mock_create_result():
    """Factory for creating mock CreateResult objects."""
    def _make(
        success: bool = True,
        name: str = "verify-policy",
        scaffold_yaml: str = None,
        scaffold_dict: dict = None,
        discovered_files: list = None,
        error: str = None,
    ):
        result = MagicMock()
        result.success = success
        result.name = name
        result.scaffold_yaml = scaffold_yaml or """name: verify-policy
reason: Verify policy documentation

files:
  - path: docs/policy.md

executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Verify the policy document.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: Verify accuracy.

timeout: 120
"""
        result.scaffold_dict = scaffold_dict or {
            "name": "verify-policy",
            "reason": "Verify policy documentation",
            "files": [{"path": "docs/policy.md"}],
            "executor": {
                "system_prompt": "@prompts/documentation-validation.txt",
                "user_prompt": "Verify the policy document.",
                "tools": ["read_file"],
            },
            "judges": [{"name": "accuracy", "system_prompt": "Verify accuracy."}],
            "timeout": 120,
        }
        result.discovered_files = discovered_files or ["docs/policy.md"]
        result.duration = 2.5
        result.error = error
        result.test_type = "standard"
        result.to_dict = MagicMock(return_value={
            "success": success,
            "name": name,
            "scaffold_yaml": result.scaffold_yaml,
            "scaffold_dict": result.scaffold_dict,
            "discovered_files": result.discovered_files,
            "duration": 2.5,
            "error": error,
        })
        return result
    return _make


@pytest.fixture
def project_with_docs(tmp_path: Path) -> Path:
    """Create a project with documentation."""
    # Create docs
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "policy.md").write_text("# Policy\n\nThe margin requirement is 25%.")

    # Create config
    config = tmp_path / "dokumen.yaml"
    config.write_text("""
version: "1.0"
provider:
  name: anthropic
  model: claude-sonnet-4-5-20250929
""")

    # Create tests dir
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    return tmp_path


# =============================================================================
# Create Command Exists Tests
# =============================================================================


class TestCreateCommandExists:
    """Tests that the create command is properly registered."""

    def test_create_command_registered(self, runner):
        """Create command should be registered in CLI."""
        from dokumen.cli import cli

        result = runner.invoke(cli, ["create", "--help"])

        assert result.exit_code == 0
        assert "goal" in result.output.lower() or "Generate" in result.output

    def test_create_requires_goal(self, runner, project_with_docs):
        """Create command should require --goal option."""
        from dokumen.cli import cli

        os.chdir(project_with_docs)
        result = runner.invoke(cli, ["create"])

        # Should fail without --goal
        assert result.exit_code != 0
        combined = (result.output or "") + (result.stderr or "")
        assert "goal" in combined.lower() or "required" in combined.lower()


# =============================================================================
# Basic Create Execution Tests
# =============================================================================


class TestCreateBasicExecution:
    """Tests for basic create command execution."""

    def test_create_generates_scaffold(self, runner, project_with_docs, mock_create_result):
        """Create command should generate a test scaffold."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()

            # Setup mock create agent
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.create = AsyncMock(return_value=mock_create_result())

            os.chdir(project_with_docs)
            result = runner.invoke(cli, ["create", "--goal", "Verify policy"])

            assert result.exit_code == 0
            # Output should contain YAML scaffold
            assert "name:" in result.output or "verify-policy" in result.output

    def test_create_with_files_option(self, runner, project_with_docs, mock_create_result):
        """Create command should accept --files option."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.create = AsyncMock(return_value=mock_create_result())

            os.chdir(project_with_docs)
            result = runner.invoke(
                cli,
                ["create", "--goal", "Verify policy", "--files", "docs/policy.md"]
            )

            assert result.exit_code == 0
            # Verify files were passed to agent
            mock_agent.create.assert_called_once()
            call_kwargs = mock_agent.create.call_args
            assert "docs/policy.md" in (call_kwargs.kwargs.get("files") or [])

    def test_create_with_existing_tests(self, runner, project_with_docs, mock_create_result):
        """Create command should accept --existing-tests option."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.create = AsyncMock(return_value=mock_create_result())

            os.chdir(project_with_docs)
            result = runner.invoke(
                cli,
                ["create", "--goal", "Verify policy", "--existing-tests", "test-a,test-b"]
            )

            assert result.exit_code == 0
            # Verify existing_tests were passed
            call_kwargs = mock_agent.create.call_args
            existing = call_kwargs.kwargs.get("existing_tests") or []
            assert "test-a" in existing
            assert "test-b" in existing


# =============================================================================
# Output Format Tests
# =============================================================================


class TestCreateOutputFormats:
    """Tests for different output formats."""

    def test_create_yaml_output(self, runner, project_with_docs, mock_create_result):
        """Create command should output YAML by default."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.create = AsyncMock(return_value=mock_create_result())

            os.chdir(project_with_docs)
            result = runner.invoke(cli, ["create", "--goal", "Verify policy"])

            assert result.exit_code == 0
            # Should contain YAML structure
            assert "name:" in result.output
            assert "executor:" in result.output

    def test_create_json_output(self, runner, project_with_docs, mock_create_result):
        """Create command should support JSON output format."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.create = AsyncMock(return_value=mock_create_result())

            os.chdir(project_with_docs)
            result = runner.invoke(
                cli,
                ["create", "--goal", "Verify policy", "--format", "json"]
            )

            assert result.exit_code == 0
            # Output should be valid JSON
            output = extract_json_from_output(result.output)
            assert output["success"] is True
            assert output["name"] == "verify-policy"


# =============================================================================
# Dry-Run Mode Tests
# =============================================================================


class TestCreateDryRun:
    """Tests for dry-run mode."""

    def test_create_dry_run_does_not_write(self, runner, project_with_docs, mock_create_result):
        """Create --dry-run should not write files."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.create = AsyncMock(return_value=mock_create_result())

            os.chdir(project_with_docs)
            result = runner.invoke(
                cli,
                ["create", "--goal", "Verify policy", "--dry-run"]
            )

            assert result.exit_code == 0
            # Should output scaffold but not write
            assert "name:" in result.output

            # No file should be created
            tests_dir = project_with_docs / "tests"
            yaml_files = list(tests_dir.glob("*.test.yaml"))
            assert len(yaml_files) == 0


# =============================================================================
# Stdin Mode Tests
# =============================================================================


class TestCreateStdinMode:
    """Tests for stdin mode (backend integration)."""

    def test_create_stdin_mode_reads_json(self, runner, project_with_docs, mock_create_result):
        """Create --stdin should read JSON from stdin."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.create = AsyncMock(return_value=mock_create_result())

            os.chdir(project_with_docs)

            # Provide JSON input
            input_json = json.dumps({
                "goal": "Verify policy",
                "files": ["docs/policy.md"],
                "existing_tests": ["test-a"]
            })

            result = runner.invoke(
                cli,
                ["create", "--stdin"],
                input=input_json
            )

            assert result.exit_code == 0
            # Verify agent was called with correct args
            mock_agent.create.assert_called_once()
            call_kwargs = mock_agent.create.call_args
            assert call_kwargs.kwargs.get("goal") == "Verify policy"

    def test_create_stdin_mode_outputs_ndjson(self, runner, project_with_docs, mock_create_result):
        """Create --stdin should output NDJSON events."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.create = AsyncMock(return_value=mock_create_result())

            os.chdir(project_with_docs)

            input_json = json.dumps({"goal": "Verify policy"})

            result = runner.invoke(
                cli,
                ["create", "--stdin"],
                input=input_json
            )

            assert result.exit_code == 0
            # Output should contain NDJSON events
            lines = result.output.strip().split("\n")
            # At least one line should be valid JSON
            found_json = False
            for line in lines:
                try:
                    event = json.loads(line)
                    if "event" in event:
                        found_json = True
                        break
                except json.JSONDecodeError:
                    continue
            assert found_json or "done" in result.output.lower()


# =============================================================================
# Output File Tests
# =============================================================================


class TestCreateOutputFile:
    """Tests for --output file option."""

    def test_create_writes_to_output_file(self, runner, project_with_docs, mock_create_result):
        """Create --output should write scaffold to file."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.create = AsyncMock(return_value=mock_create_result())

            os.chdir(project_with_docs)
            output_path = project_with_docs / "tests" / "new-test.test.yaml"

            result = runner.invoke(
                cli,
                ["create", "--goal", "Verify policy", "--output", str(output_path)]
            )

            assert result.exit_code == 0
            # File should be created
            assert output_path.exists()
            content = output_path.read_text()
            assert "name:" in content


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestCreateErrorHandling:
    """Tests for error handling in create command."""

    def test_create_handles_agent_error(self, runner, project_with_docs, mock_create_result):
        """Create command should handle agent errors gracefully."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            # Return failed result
            failed_result = mock_create_result(
                success=False,
                error="API rate limit exceeded"
            )
            mock_agent.create = AsyncMock(return_value=failed_result)

            os.chdir(project_with_docs)
            result = runner.invoke(cli, ["create", "--goal", "Verify policy"])

            # Should exit with non-zero code
            assert result.exit_code != 0
            combined = (result.output or "") + (result.stderr or "")
            assert "error" in combined.lower() or "failed" in combined.lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestCreateIntegration:
    """Integration tests for create command."""

    def test_create_discovers_existing_tests(self, runner, project_with_docs, mock_create_result):
        """Create should auto-discover existing tests for conflict avoidance."""
        from dokumen.cli import cli

        # Create an existing test
        tests_dir = project_with_docs / "tests"
        (tests_dir / "verify-policy.test.yaml").write_text("""
name: verify-policy
reason: Existing test
executor:
  system_prompt: test
  user_prompt: test
  tools:
    - read_file
judges:
  - name: test
    system_prompt: test
""")

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            # Return result with conflict-avoiding name
            new_result = mock_create_result(name="verify-policy-2")
            mock_agent.create = AsyncMock(return_value=new_result)

            os.chdir(project_with_docs)
            result = runner.invoke(cli, ["create", "--goal", "Verify policy"])

            assert result.exit_code == 0
            # Verify existing tests were discovered and passed
            call_kwargs = mock_agent.create.call_args
            existing = call_kwargs.kwargs.get("existing_tests") or []
            assert "verify-policy" in existing


# =============================================================================
# Browser Type Tests
# =============================================================================


class TestCreateBrowserType:
    """Tests for --type browser CLI option."""

    def test_type_browser_cli_option(self, runner, project_with_docs, mock_create_result):
        """Create --type browser should pass test_type='browser' to agent."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.create = AsyncMock(return_value=mock_create_result())

            os.chdir(project_with_docs)
            result = runner.invoke(
                cli,
                ["create", "--goal", "Test login flow", "--type", "browser"]
            )

            assert result.exit_code == 0
            call_kwargs = mock_agent.create.call_args
            assert call_kwargs.kwargs.get("test_type") == "browser"

    def test_type_defaults_standard(self, runner, project_with_docs, mock_create_result):
        """Create without --type should default to 'standard'."""
        from dokumen.cli import cli

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.create = AsyncMock(return_value=mock_create_result())

            os.chdir(project_with_docs)
            result = runner.invoke(
                cli,
                ["create", "--goal", "Verify policy"]
            )

            assert result.exit_code == 0
            call_kwargs = mock_agent.create.call_args
            assert call_kwargs.kwargs.get("test_type") == "standard"

    def test_stdin_type_browser(self, runner, project_with_docs, mock_create_result):
        """Create --stdin with type: browser should pass test_type='browser'."""
        from dokumen.cli.commands.create import _run_stdin_session, _emit_event
        import asyncio

        with patch('dokumen.create_agent.CreateAgent') as mock_agent_class, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.tools_object.BUILTIN_TOOLS', {}):

            mock_provider.return_value = MagicMock()
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            from dokumen.create_agent import CreateResult
            create_result = CreateResult(
                success=True,
                scaffold_yaml="name: test\n",
                scaffold_dict={"name": "test"},
                name="test",
                discovered_files=[],
                duration=1.0,
                test_type="browser",
            )
            mock_agent.create = AsyncMock(return_value=create_result)

            os.chdir(project_with_docs)

            input_json = json.dumps({
                "goal": "Test login flow",
                "type": "browser",
            })

            # Test through _run_stdin_session directly via stdin mock
            with patch('sys.stdin') as mock_stdin:
                mock_stdin.read.return_value = input_json

                asyncio.run(_run_stdin_session(timeout=120.0, config=None))

            # Verify agent was called with test_type='browser'
            call_kwargs = mock_agent.create.call_args
            assert call_kwargs.kwargs.get("test_type") == "browser"

    def test_stdin_invalid_type_emits_error(self, runner, project_with_docs, capsys):
        """Create --stdin with invalid type should emit error event."""
        from dokumen.cli.commands.create import _run_stdin_session
        import asyncio

        os.chdir(project_with_docs)

        input_json = json.dumps({
            "goal": "Test something",
            "type": "invalid",
        })

        with patch('sys.stdin') as mock_stdin:
            mock_stdin.read.return_value = input_json

            asyncio.run(_run_stdin_session(timeout=120.0, config=None))

        captured = capsys.readouterr()
        # Should contain an error about invalid type
        assert "invalid" in captured.out.lower() or "error" in captured.out.lower()
