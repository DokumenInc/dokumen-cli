"""
Unit tests for test scaffold parsing (*.test.yaml).

TDD: These tests are written first, before implementation.
They should fail initially until test_scaffold.py is implemented.
"""
from pathlib import Path

import pytest


class TestFileRefModel:
    """Tests for the FileRef Pydantic model."""

    def test_file_ref_exists(self):
        """FileRef class should exist."""
        from dokumen.test_scaffold import FileRef
        assert FileRef is not None

    def test_file_ref_valid_path(self):
        """FileRef should accept valid path."""
        from dokumen.test_scaffold import FileRef

        ref = FileRef(path="docs/api/auth.md")
        assert ref.path == "docs/api/auth.md"

    def test_file_ref_requires_path(self):
        """FileRef should require path field."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import FileRef

        with pytest.raises(ValidationError):
            FileRef()


class TestExecutorConfigModel:
    """Tests for the ExecutorConfig Pydantic model."""

    def test_executor_config_exists(self):
        """ExecutorConfig class should exist."""
        from dokumen.test_scaffold import ExecutorConfig
        assert ExecutorConfig is not None

    def test_executor_config_valid(self):
        """Valid ExecutorConfig should parse correctly."""
        from dokumen.test_scaffold import ExecutorConfig

        config = ExecutorConfig(
            system_prompt="@prompts/documentation-validation.txt",
            user_prompt="Read the file.",
            tools=["read_file"]
        )
        assert config.system_prompt == "@prompts/documentation-validation.txt"
        assert config.user_prompt == "Read the file."
        assert config.tools == ["read_file"]

    def test_executor_config_valid_general_prompt(self):
        """ExecutorConfig with @prompts/general.txt should be valid."""
        from dokumen.test_scaffold import ExecutorConfig

        config = ExecutorConfig(
            system_prompt="@prompts/general.txt",
            user_prompt="Do the task.",
            tools=["read_file"]
        )
        assert config.system_prompt == "@prompts/general.txt"

    def test_executor_config_system_prompt_optional(self):
        """ExecutorConfig system_prompt is optional (agent provides default)."""
        from dokumen.test_scaffold import ExecutorConfig

        config = ExecutorConfig(
            user_prompt="Read the file.",
            tools=["read_file"]
        )
        assert config.system_prompt is None

    def test_executor_config_inline_text_accepted(self):
        """ExecutorConfig accepts inline system_prompt text (for agent overrides)."""
        from dokumen.test_scaffold import ExecutorConfig

        config = ExecutorConfig(
            system_prompt="You are a validator.",
            user_prompt="Read the file.",
            tools=["read_file"]
        )
        assert config.system_prompt == "You are a validator."

    def test_executor_config_unknown_prompt_rejected(self):
        """ExecutorConfig rejects unknown prompt names."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import ExecutorConfig

        with pytest.raises(ValidationError) as exc_info:
            ExecutorConfig(
                system_prompt="@prompts/unknown-prompt.txt",
                user_prompt="Read the file.",
                tools=["read_file"]
            )
        assert "unknown" in str(exc_info.value).lower() or "valid options" in str(exc_info.value).lower()

    def test_executor_config_requires_user_prompt(self):
        """ExecutorConfig requires user_prompt."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import ExecutorConfig

        with pytest.raises(ValidationError) as exc_info:
            ExecutorConfig(
                system_prompt="@prompts/documentation-validation.txt",
                tools=["read_file"]
            )
        assert "user_prompt" in str(exc_info.value).lower()

    def test_executor_config_tools_optional(self):
        """ExecutorConfig tools are optional (agent provides defaults)."""
        from dokumen.test_scaffold import ExecutorConfig

        config = ExecutorConfig(
            system_prompt="@prompts/documentation-validation.txt",
            user_prompt="Read the file."
        )
        assert config.tools is None

    def test_executor_config_tools_must_be_list(self):
        """ExecutorConfig tools must be a list."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import ExecutorConfig

        with pytest.raises(ValidationError):
            ExecutorConfig(
                system_prompt="@prompts/documentation-validation.txt",
                user_prompt="Read the file.",
                tools="read_file"  # Not a list
            )


class TestJudgeConfigModel:
    """Tests for the JudgeConfig Pydantic model."""

    def test_judge_config_exists(self):
        """JudgeConfig class should exist."""
        from dokumen.test_scaffold import JudgeConfig
        assert JudgeConfig is not None

    def test_judge_config_minimal(self):
        """JudgeConfig with only required fields."""
        from dokumen.test_scaffold import JudgeConfig

        config = JudgeConfig(name="accuracy")
        assert config.name == "accuracy"
        assert config.system_prompt is None
        assert config.tools == []
        assert config.include_executor_output is True

    def test_judge_config_full(self):
        """JudgeConfig with all fields."""
        from dokumen.test_scaffold import JudgeConfig

        config = JudgeConfig(
            name="accuracy-judge",
            system_prompt="Evaluate the response.",
            tools=["read_file"],
            include_executor_output=False
        )
        assert config.name == "accuracy-judge"
        assert config.system_prompt == "Evaluate the response."
        assert config.tools == ["read_file"]
        assert config.include_executor_output is False

    def test_judge_config_requires_name(self):
        """JudgeConfig requires name field."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import JudgeConfig

        with pytest.raises(ValidationError) as exc_info:
            JudgeConfig(system_prompt="Evaluate.")
        assert "name" in str(exc_info.value).lower()

    def test_judge_config_tools_default_empty(self):
        """JudgeConfig tools defaults to empty list."""
        from dokumen.test_scaffold import JudgeConfig

        config = JudgeConfig(name="test")
        assert config.tools == []


class TestDockerMountModel:
    """Tests for the DockerMount Pydantic model."""

    def test_docker_mount_exists(self):
        """DockerMount class should exist."""
        from dokumen.test_scaffold import DockerMount
        assert DockerMount is not None

    def test_docker_mount_valid(self):
        """Valid DockerMount should parse correctly."""
        from dokumen.test_scaffold import DockerMount

        mount = DockerMount(
            source="./data",
            target="/data",
            readonly=True
        )
        assert mount.source == "./data"
        assert mount.target == "/data"
        assert mount.readonly is True

    def test_docker_mount_readonly_default(self):
        """DockerMount readonly defaults to False."""
        from dokumen.test_scaffold import DockerMount

        mount = DockerMount(source="./data", target="/data")
        assert mount.readonly is False


class TestSandboxConfigModel:
    """Tests for the SandboxConfig Pydantic model."""

    def test_sandbox_config_exists(self):
        """SandboxConfig class should exist."""
        from dokumen.test_scaffold import SandboxConfig
        assert SandboxConfig is not None

    def test_sandbox_config_valid_docker(self):
        """Valid docker SandboxConfig should parse correctly."""
        from dokumen.test_scaffold import SandboxConfig

        config = SandboxConfig(
            type="docker",
            docker_image="python:3.11-slim",
            docker_network="none"
        )
        assert config.type == "docker"
        assert config.docker_image == "python:3.11-slim"
        assert config.docker_network == "none"

    def test_sandbox_config_type_required(self):
        """SandboxConfig requires type field."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import SandboxConfig

        with pytest.raises(ValidationError) as exc_info:
            SandboxConfig(docker_image="python:3.11-slim")
        assert "type" in str(exc_info.value).lower()

    def test_sandbox_config_type_validation(self):
        """SandboxConfig type must be valid option."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import SandboxConfig

        with pytest.raises(ValidationError):
            SandboxConfig(type="invalid_sandbox_type")

    def test_sandbox_config_valid_types(self):
        """All valid sandbox types should be accepted."""
        from dokumen.test_scaffold import SandboxConfig

        valid_types = ["none", "whitelist", "subprocess", "docker", "virtual_fs"]
        for sandbox_type in valid_types:
            config = SandboxConfig(type=sandbox_type)
            assert config.type == sandbox_type

    def test_sandbox_config_defaults(self):
        """SandboxConfig should have sensible defaults."""
        from dokumen.test_scaffold import SandboxConfig

        config = SandboxConfig(type="docker")
        assert config.docker_image == "python:3.11-slim"
        assert config.docker_network == "none"
        assert config.docker_mount_readonly is False
        assert config.docker_workdir == "/workspace"
        assert config.timeout == 60
        assert config.max_memory_mb == 512

    def test_sandbox_config_docker_network_validation(self):
        """docker_network must be 'none' or 'bridge'."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import SandboxConfig

        with pytest.raises(ValidationError):
            SandboxConfig(type="docker", docker_network="invalid")

    def test_sandbox_config_docker_mounts(self):
        """SandboxConfig should accept docker mounts."""
        from dokumen.test_scaffold import SandboxConfig, DockerMount

        config = SandboxConfig(
            type="docker",
            docker_mounts=[
                DockerMount(source="./data", target="/data", readonly=True)
            ]
        )
        assert len(config.docker_mounts) == 1
        assert config.docker_mounts[0].source == "./data"


class TestTestScaffoldModel:
    """Tests for the TestScaffold Pydantic model."""

    def test_test_scaffold_exists(self):
        """TestScaffold class should exist."""
        from dokumen.test_scaffold import TestScaffold
        assert TestScaffold is not None

    def test_test_scaffold_minimal(self):
        """TestScaffold with only required fields."""
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig, FileRef

        scaffold = TestScaffold(
            name="my-test",
            files=[FileRef(path="docs/test.md")],
            executor=ExecutorConfig(
                system_prompt="@prompts/documentation-validation.txt",
                user_prompt="Validate the document.",
                tools=["read_file"]
            ),
            judges=[JudgeConfig(name="accuracy", system_prompt="Evaluate.")]
        )
        assert scaffold.name == "my-test"
        assert scaffold.reason is None
        assert len(scaffold.files) == 1
        assert scaffold.timeout == 60.0
        assert scaffold.retries == 0
        assert scaffold.sandbox is None

    def test_test_scaffold_full(self):
        """TestScaffold with all fields."""
        from dokumen.test_scaffold import (
            TestScaffold, ExecutorConfig, JudgeConfig,
            FileRef, SandboxConfig
        )

        scaffold = TestScaffold(
            name="full-test",
            reason="Test all the things",
            files=[FileRef(path="docs/api.md")],
            executor=ExecutorConfig(
                system_prompt="@prompts/documentation-validation.txt",
                user_prompt="Read the file.",
                tools=["read_file", "glob"]
            ),
            judges=[
                JudgeConfig(name="accuracy", system_prompt="Evaluate."),
                JudgeConfig(name="completeness", system_prompt="Check all.")
            ],
            timeout=90.0,
            retries=2,
            sandbox=SandboxConfig(type="docker")
        )
        assert scaffold.name == "full-test"
        assert scaffold.reason == "Test all the things"
        assert len(scaffold.files) == 1
        assert scaffold.timeout == 90.0
        assert scaffold.retries == 2

    def test_test_scaffold_requires_name(self):
        """TestScaffold requires name field."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig

        with pytest.raises(ValidationError) as exc_info:
            TestScaffold(
                executor=ExecutorConfig(
                    system_prompt="@prompts/documentation-validation.txt",
                    user_prompt="Read the file.",
                    tools=["read_file"]
                ),
                judges=[JudgeConfig(name="accuracy")]
            )
        assert "name" in str(exc_info.value).lower()

    def test_test_scaffold_requires_executor(self):
        """TestScaffold requires executor field."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import TestScaffold, JudgeConfig

        with pytest.raises(ValidationError) as exc_info:
            TestScaffold(
                name="my-test",
                judges=[JudgeConfig(name="accuracy")]
            )
        assert "executor" in str(exc_info.value).lower()

    def test_test_scaffold_requires_judges(self):
        """TestScaffold requires judges field."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig

        with pytest.raises(ValidationError) as exc_info:
            TestScaffold(
                name="my-test",
                executor=ExecutorConfig(
                    system_prompt="@prompts/documentation-validation.txt",
                    user_prompt="Read the file.",
                    tools=["read_file"]
                )
            )
        assert "judges" in str(exc_info.value).lower()

    def test_test_scaffold_judges_non_empty(self):
        """TestScaffold judges must be non-empty list."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig

        with pytest.raises(ValidationError) as exc_info:
            TestScaffold(
                name="my-test",
                executor=ExecutorConfig(
                    system_prompt="@prompts/documentation-validation.txt",
                    user_prompt="Read the file.",
                    tools=["read_file"]
                ),
                judges=[]
            )
        # Should mention that list must have at least 1 item
        assert "judges" in str(exc_info.value).lower() or "min" in str(exc_info.value).lower()

    def test_test_scaffold_name_kebab_case(self):
        """TestScaffold name must be kebab-case."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig

        with pytest.raises(ValidationError) as exc_info:
            TestScaffold(
                name="Not_Valid_Name",
                executor=ExecutorConfig(
                    system_prompt="@prompts/documentation-validation.txt",
                    user_prompt="Read the file.",
                    tools=["read_file"]
                ),
                judges=[JudgeConfig(name="accuracy")]
            )
        assert "kebab" in str(exc_info.value).lower() or "name" in str(exc_info.value).lower()

    def test_test_scaffold_name_valid_kebab_cases(self):
        """Valid kebab-case names should be accepted."""
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig, FileRef

        valid_names = ["my-test", "test", "a-b-c", "test123", "my-test-123"]
        for name in valid_names:
            scaffold = TestScaffold(
                name=name,
                files=[FileRef(path="docs/test.md")],
                executor=ExecutorConfig(
                    system_prompt="@prompts/documentation-validation.txt",
                    user_prompt="Validate the document.",
                    tools=["read_file"]
                ),
                judges=[JudgeConfig(name="accuracy")]
            )
            assert scaffold.name == name

    def test_test_scaffold_timeout_range(self):
        """TestScaffold timeout must be between 1 and 600."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig

        # Too low
        with pytest.raises(ValidationError):
            TestScaffold(
                name="my-test",
                executor=ExecutorConfig(
                    system_prompt="@prompts/documentation-validation.txt",
                    user_prompt="Read the file.",
                    tools=["read_file"]
                ),
                judges=[JudgeConfig(name="accuracy")],
                timeout=0
            )

        # Too high
        with pytest.raises(ValidationError):
            TestScaffold(
                name="my-test",
                executor=ExecutorConfig(
                    system_prompt="@prompts/documentation-validation.txt",
                    user_prompt="Read the file.",
                    tools=["read_file"]
                ),
                judges=[JudgeConfig(name="accuracy")],
                timeout=601
            )

    def test_test_scaffold_retries_range(self):
        """TestScaffold retries must be between 0 and 5."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig

        # Negative
        with pytest.raises(ValidationError):
            TestScaffold(
                name="my-test",
                executor=ExecutorConfig(
                    system_prompt="@prompts/documentation-validation.txt",
                    user_prompt="Read the file.",
                    tools=["read_file"]
                ),
                judges=[JudgeConfig(name="accuracy")],
                retries=-1
            )

        # Too high
        with pytest.raises(ValidationError):
            TestScaffold(
                name="my-test",
                executor=ExecutorConfig(
                    system_prompt="@prompts/documentation-validation.txt",
                    user_prompt="Read the file.",
                    tools=["read_file"]
                ),
                judges=[JudgeConfig(name="accuracy")],
                retries=6
            )

    def test_test_scaffold_sandbox_string(self):
        """TestScaffold sandbox can be a string reference."""
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig, FileRef

        scaffold = TestScaffold(
            name="my-test",
            files=[FileRef(path="docs/test.md")],
            executor=ExecutorConfig(
                system_prompt="@prompts/documentation-validation.txt",
                user_prompt="Validate the document.",
                tools=["read_file"]
            ),
            judges=[JudgeConfig(name="accuracy")],
            sandbox="default-sandbox"
        )
        assert scaffold.sandbox == "default-sandbox"

    def test_test_scaffold_sandbox_object(self):
        """TestScaffold sandbox can be a SandboxConfig object."""
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig, SandboxConfig, FileRef

        scaffold = TestScaffold(
            name="my-test",
            files=[FileRef(path="docs/test.md")],
            executor=ExecutorConfig(
                system_prompt="@prompts/documentation-validation.txt",
                user_prompt="Validate the document.",
                tools=["read_file"]
            ),
            judges=[JudgeConfig(name="accuracy")],
            sandbox=SandboxConfig(type="docker")
        )
        assert isinstance(scaffold.sandbox, SandboxConfig)
        assert scaffold.sandbox.type == "docker"


class TestLoadScaffold:
    """Tests for loading scaffolds from YAML files."""

    def test_load_scaffold_exists(self):
        """load_scaffold function should exist."""
        from dokumen.test_scaffold import load_scaffold
        assert callable(load_scaffold)

    def test_load_scaffold_valid_minimal(self, valid_minimal_scaffold_path: Path):
        """Loading valid minimal scaffold should return TestScaffold."""
        from dokumen.test_scaffold import load_scaffold, TestScaffold

        scaffold = load_scaffold(str(valid_minimal_scaffold_path))
        assert isinstance(scaffold, TestScaffold)
        assert scaffold.name == "my-test"
        assert len(scaffold.judges) == 1

    def test_load_scaffold_valid_complete(self, valid_complete_scaffold_path: Path):
        """Loading valid complete scaffold should return TestScaffold with all fields."""
        from dokumen.test_scaffold import load_scaffold

        scaffold = load_scaffold(str(valid_complete_scaffold_path))
        assert scaffold.name == "verify-api-authentication-methods"
        assert scaffold.reason is not None
        assert len(scaffold.files) == 2
        assert len(scaffold.judges) == 2
        assert scaffold.timeout == 90
        assert scaffold.retries == 1
        assert scaffold.sandbox is not None

    def test_load_scaffold_missing_file(self, tmp_path: Path):
        """Loading non-existent file should raise ScaffoldError."""
        from dokumen.test_scaffold import load_scaffold, ScaffoldError

        missing_path = tmp_path / "nonexistent.test.yaml"
        with pytest.raises(ScaffoldError) as exc_info:
            load_scaffold(str(missing_path))
        assert "not found" in str(exc_info.value).lower()

    def test_load_scaffold_invalid_yaml(self, invalid_malformed_scaffold_path: Path):
        """Loading malformed YAML should raise ScaffoldError."""
        from dokumen.test_scaffold import load_scaffold, ScaffoldError

        with pytest.raises(ScaffoldError) as exc_info:
            load_scaffold(str(invalid_malformed_scaffold_path))
        assert "yaml" in str(exc_info.value).lower() or "parse" in str(exc_info.value).lower()

    def test_load_scaffold_missing_name(self, invalid_missing_name_path: Path):
        """Loading scaffold without name should raise ScaffoldError."""
        from dokumen.test_scaffold import load_scaffold, ScaffoldError

        with pytest.raises(ScaffoldError) as exc_info:
            load_scaffold(str(invalid_missing_name_path))
        assert "name" in str(exc_info.value).lower()

    def test_load_scaffold_missing_executor(self, invalid_missing_executor_path: Path):
        """Loading scaffold without executor should raise ScaffoldError."""
        from dokumen.test_scaffold import load_scaffold, ScaffoldError

        with pytest.raises(ScaffoldError) as exc_info:
            load_scaffold(str(invalid_missing_executor_path))
        assert "executor" in str(exc_info.value).lower()


class TestValidateScaffoldWithWarnings:
    """Tests for scaffold validation that returns warnings."""

    def test_validate_scaffold_file_exists(self):
        """validate_scaffold_file function should exist."""
        from dokumen.test_scaffold import validate_scaffold_file
        assert callable(validate_scaffold_file)

    def test_validate_scaffold_file_valid(self, tmp_path: Path):
        """Validating valid scaffold should return (True, [])."""
        from dokumen.test_scaffold import validate_scaffold_file
        import os

        # Create a valid scaffold with files that exist
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "test.md").write_text("# Test\n\nTest content.")

        scaffold_content = """
name: valid-test
files:
  - path: docs/test.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Validate the document."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate if the executor correctly validated the document."
"""
        scaffold_path = tmp_path / "valid.test.yaml"
        scaffold_path.write_text(scaffold_content)

        # Change to temp dir so file paths resolve correctly
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            valid, errors, warnings = validate_scaffold_file(str(scaffold_path))
            assert valid is True, f"Expected valid=True, got errors: {errors}"
            assert errors == []
        finally:
            os.chdir(original_cwd)

    def test_validate_scaffold_file_error_unknown_tool(self, warning_unknown_tool_path: Path):
        """Unknown tool should generate an error (not just warning)."""
        from dokumen.test_scaffold import validate_scaffold_file

        valid, errors, warnings = validate_scaffold_file(str(warning_unknown_tool_path))
        assert valid is False  # Unknown tools are now errors
        assert any("tool" in e.lower() for e in errors)

    def test_validate_scaffold_file_error_no_files(self, warning_no_files_path: Path):
        """Missing files section should generate error (files is required)."""
        from dokumen.test_scaffold import validate_scaffold_file

        valid, errors, warnings = validate_scaffold_file(str(warning_no_files_path))
        assert valid is False
        assert any("files" in e.lower() for e in errors)

    def test_validate_scaffold_file_warning_missing_judge_prompt(self, tmp_path: Path):
        """Missing judge system_prompt should generate warning."""
        from dokumen.test_scaffold import validate_scaffold_file
        import os

        # Create scaffold with missing judge system_prompt
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "refund.md").write_text("# Refund Policy\n\nRefund content.")

        scaffold_content = """
name: missing-judge-prompt-test
files:
  - path: docs/refund.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Validate the refund policy document."
  tools:
    - read_file
judges:
  - name: accuracy
    # Missing system_prompt - should generate warning
"""
        scaffold_path = tmp_path / "test.yaml"
        scaffold_path.write_text(scaffold_content)

        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            valid, errors, warnings = validate_scaffold_file(str(scaffold_path))
            assert valid is True, f"Expected valid=True, got errors: {errors}"
            assert any("system_prompt" in w.lower() or "judge" in w.lower() for w in warnings)
        finally:
            os.chdir(original_cwd)

    def test_validate_scaffold_file_invalid_name(self, invalid_bad_name_path: Path):
        """Invalid name format should return (False, [error])."""
        from dokumen.test_scaffold import validate_scaffold_file

        valid, errors, warnings = validate_scaffold_file(str(invalid_bad_name_path))
        assert valid is False
        assert any("name" in e.lower() or "kebab" in e.lower() for e in errors)


class TestValidateScaffoldFileErrors:
    """Tests for error cases in validate_scaffold_file."""

    def test_validate_scaffold_file_not_found(self, tmp_path):
        """Non-existent file returns error."""
        from dokumen.test_scaffold import validate_scaffold_file

        nonexistent_path = tmp_path / "nonexistent.test.yaml"
        valid, errors, warnings = validate_scaffold_file(str(nonexistent_path))

        assert valid is False
        assert any("not found" in e.lower() for e in errors)

    def test_validate_scaffold_file_yaml_error(self, tmp_path):
        """Invalid YAML syntax returns error."""
        from dokumen.test_scaffold import validate_scaffold_file

        invalid_yaml = tmp_path / "invalid.test.yaml"
        invalid_yaml.write_text("name: test\n  invalid: [unclosed")

        valid, errors, warnings = validate_scaffold_file(str(invalid_yaml))

        assert valid is False
        assert any("yaml" in e.lower() for e in errors)

    def test_validate_scaffold_file_empty(self, tmp_path):
        """Empty scaffold file returns error."""
        from dokumen.test_scaffold import validate_scaffold_file

        empty_file = tmp_path / "empty.test.yaml"
        empty_file.write_text("")

        valid, errors, warnings = validate_scaffold_file(str(empty_file))

        assert valid is False
        assert any("empty" in e.lower() for e in errors)


class TestLoadScaffoldErrors:
    """Tests for error cases in load_scaffold."""

    def test_load_empty_scaffold_raises(self, tmp_path):
        """Loading empty scaffold file raises ScaffoldError."""
        from dokumen.test_scaffold import load_scaffold, ScaffoldError

        empty_file = tmp_path / "empty.test.yaml"
        empty_file.write_text("")

        with pytest.raises(ScaffoldError, match="Empty"):
            load_scaffold(str(empty_file))


class TestScaffoldError:
    """Tests for the ScaffoldError exception."""

    def test_scaffold_error_exists(self):
        """ScaffoldError exception class should exist."""
        from dokumen.test_scaffold import ScaffoldError
        assert issubclass(ScaffoldError, Exception)

    def test_scaffold_error_message(self):
        """ScaffoldError should preserve error message."""
        from dokumen.test_scaffold import ScaffoldError

        error = ScaffoldError("Test error message")
        assert "Test error message" in str(error)


class TestExecutorPromptValidation:
    """Tests for executor system_prompt validation (pre-defined prompts only)."""

    def test_valid_documentation_validation_prompt(self):
        """Valid @prompts/documentation-validation.txt reference."""
        from dokumen.test_scaffold import ExecutorConfig

        config = ExecutorConfig(
            system_prompt="@prompts/documentation-validation.txt",
            user_prompt="Read the file.",
            tools=["read_file"]
        )
        assert config.system_prompt == "@prompts/documentation-validation.txt"

    def test_valid_general_prompt(self):
        """Valid @prompts/general.txt reference."""
        from dokumen.test_scaffold import ExecutorConfig

        config = ExecutorConfig(
            system_prompt="@prompts/general.txt",
            user_prompt="Do something.",
            tools=["read_file"]
        )
        assert config.system_prompt == "@prompts/general.txt"

    def test_inline_prompt_accepted(self):
        """Inline prompt text is accepted (used by agent definitions)."""
        from dokumen.test_scaffold import ExecutorConfig

        config = ExecutorConfig(
            system_prompt="You are a helpful assistant.",
            user_prompt="Read the file.",
            tools=["read_file"]
        )
        assert config.system_prompt == "You are a helpful assistant."

    def test_unknown_prompt_rejected(self):
        """Unknown prompt name should be rejected."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import ExecutorConfig

        with pytest.raises(ValidationError) as exc_info:
            ExecutorConfig(
                system_prompt="@prompts/unknown-type.txt",
                user_prompt="Read the file.",
                tools=["read_file"]
            )
        assert "unknown" in str(exc_info.value).lower() or "valid options" in str(exc_info.value).lower()

    def test_missing_system_prompt_allowed(self):
        """Missing system_prompt is allowed (agent provides default)."""
        from dokumen.test_scaffold import ExecutorConfig

        config = ExecutorConfig(
            user_prompt="Read the file.",
            tools=["read_file"]
        )
        assert config.system_prompt is None

    def test_empty_system_prompt_rejected(self):
        """Empty system_prompt should be rejected."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import ExecutorConfig

        with pytest.raises(ValidationError) as exc_info:
            ExecutorConfig(
                system_prompt="",
                user_prompt="Read the file.",
                tools=["read_file"]
            )
        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_system_prompt_rejected(self):
        """Whitespace-only system_prompt should be rejected."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import ExecutorConfig

        with pytest.raises(ValidationError) as exc_info:
            ExecutorConfig(
                system_prompt="   ",
                user_prompt="Read the file.",
                tools=["read_file"]
            )
        assert "empty" in str(exc_info.value).lower()

    def test_at_prefix_treated_as_inline_prompt(self):
        """@prompts/ syntax is no longer special — treated as inline string."""
        from dokumen.test_scaffold import ExecutorConfig

        config = ExecutorConfig(
            system_prompt="@prompts/some-old-style.txt",
            user_prompt="Read the file.",
            tools=["read_file"]
        )
        assert config.system_prompt == "@prompts/some-old-style.txt"


class TestFilesFieldRequired:
    """Tests for the required files field validation.

    The files field is REQUIRED because retrieval of the correct documentation
    is a direct success criteria of ALL tests. The explore phase must discover
    the documents specified in files.
    """

    def test_scaffold_requires_files_field(self):
        """TestScaffold should require at least one file in files field."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig

        with pytest.raises(ValidationError) as exc_info:
            TestScaffold(
                name="my-test",
                executor=ExecutorConfig(
                    system_prompt="@prompts/documentation-validation.txt",
                    user_prompt="Explain the margin requirements.",
                    tools=["read_file"]
                ),
                judges=[JudgeConfig(name="accuracy", system_prompt="Evaluate.")]
                # Missing files field - should raise
            )
        assert "files" in str(exc_info.value).lower()

    def test_scaffold_rejects_empty_files_list(self):
        """TestScaffold should reject empty files list."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig

        with pytest.raises(ValidationError) as exc_info:
            TestScaffold(
                name="my-test",
                executor=ExecutorConfig(
                    system_prompt="@prompts/documentation-validation.txt",
                    user_prompt="Explain the margin requirements.",
                    tools=["read_file"]
                ),
                judges=[JudgeConfig(name="accuracy", system_prompt="Evaluate.")],
                files=[]  # Empty list - should raise
            )
        assert "files" in str(exc_info.value).lower()

    def test_scaffold_accepts_valid_files(self):
        """TestScaffold should accept valid files list."""
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig, FileRef

        scaffold = TestScaffold(
            name="my-test",
            executor=ExecutorConfig(
                system_prompt="@prompts/documentation-validation.txt",
                user_prompt="Explain the margin requirements.",
                tools=["read_file"]
            ),
            judges=[JudgeConfig(name="accuracy", system_prompt="Evaluate.")],
            files=[FileRef(path="docs/policies/margin.md")]
        )
        assert len(scaffold.files) == 1
        assert scaffold.files[0].path == "docs/policies/margin.md"


class TestPromptPathValidation:
    """Tests for validating that file paths don't appear in executor prompts.

    The AI should DISCOVER which documents are relevant via explore, not be
    told explicitly via hardcoded paths in prompts.
    """

    def test_user_prompt_rejects_hardcoded_file_path(self):
        """User prompt should NOT contain file paths from the files list."""
        from dokumen.test_scaffold import validate_scaffold_file
        import tempfile
        import os

        # Create a scaffold with path in user_prompt
        scaffold_content = """
name: bad-test
files:
  - path: docs/policies/margin.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Read docs/policies/margin.md and explain the margin requirements."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.test.yaml', delete=False
        ) as f:
            f.write(scaffold_content)
            f.flush()
            temp_path = f.name

        try:
            valid, errors, warnings = validate_scaffold_file(temp_path)
            # Should have an error about path in prompt
            assert not valid or len(errors) > 0 or any("path" in w.lower() for w in warnings)
        finally:
            os.unlink(temp_path)

    def test_user_prompt_without_path_is_valid(self):
        """User prompt without hardcoded paths should be valid."""
        from dokumen.test_scaffold import validate_scaffold_file
        import tempfile
        import os

        # Create a scaffold without path in user_prompt
        scaffold_content = """
name: good-test
files:
  - path: docs/policies/margin.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Explain the margin requirements."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.test.yaml', delete=False
        ) as f:
            f.write(scaffold_content)
            f.flush()
            temp_path = f.name

        try:
            valid, errors, warnings = validate_scaffold_file(temp_path)
            # Should not have path-related errors
            path_errors = [e for e in errors if "path" in e.lower() and "prompt" in e.lower()]
            assert len(path_errors) == 0, f"Unexpected path errors: {path_errors}"
        finally:
            os.unlink(temp_path)

    def test_validate_detects_multiple_paths_in_prompt(self):
        """Should detect when multiple file paths from files list appear in prompt."""
        from dokumen.test_scaffold import validate_scaffold_file
        import tempfile
        import os

        scaffold_content = """
name: multi-path-test
files:
  - path: docs/api/auth.md
  - path: docs/api/security.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Read docs/api/auth.md and docs/api/security.md and explain auth."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.test.yaml', delete=False
        ) as f:
            f.write(scaffold_content)
            f.flush()
            temp_path = f.name

        try:
            valid, errors, warnings = validate_scaffold_file(temp_path)
            # Should detect both paths
            all_messages = errors + warnings
            assert any("path" in m.lower() for m in all_messages)
        finally:
            os.unlink(temp_path)

    def test_system_prompt_path_not_checked(self):
        """System prompt uses @prompts/ references, not file paths - should not trigger validation."""
        from dokumen.test_scaffold import validate_scaffold_file
        import tempfile
        import os

        scaffold_content = """
name: system-prompt-test
files:
  - path: docs/policies/margin.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Explain the margin requirements."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Check if docs/policies/margin.md was used correctly."
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.test.yaml', delete=False
        ) as f:
            f.write(scaffold_content)
            f.flush()
            temp_path = f.name

        try:
            valid, errors, warnings = validate_scaffold_file(temp_path)
            # Judge system_prompt with path reference should NOT trigger validation error
            # Only executor user_prompt is checked
            executor_path_errors = [
                e for e in errors
                if "executor" in e.lower() and "user_prompt" in e.lower() and "path" in e.lower()
            ]
            assert len(executor_path_errors) == 0
        finally:
            os.unlink(temp_path)


# =============================================================================
# Browser Scaffold Model Tests
# =============================================================================


class TestBrowserScaffoldConfig:
    """Tests for BrowserScaffoldConfig model and browser type support."""

    def test_scaffold_accepts_type_browser(self):
        """TestScaffold should accept type: browser."""
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig, FileRef

        scaffold = TestScaffold(
            name="login-flow-test",
            type="browser",
            files=[FileRef(path="docs/credentials/pat.txt")],
            executor=ExecutorConfig(
                system_prompt="@prompts/browser-testing.txt",
                user_prompt="Navigate to https://app.dokumen.app and verify login.",
                tools=["browser_navigate", "browser_click", "browser_type", "read_file"]
            ),
            judges=[JudgeConfig(name="ui-check", system_prompt="Verify login page loaded.")]
        )
        assert scaffold.type == "browser"

    def test_scaffold_accepts_browser_config_with_viewport(self):
        """TestScaffold should accept browser config with viewport."""
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig, FileRef, BrowserScaffoldConfig

        scaffold = TestScaffold(
            name="browser-viewport-test",
            type="browser",
            files=[FileRef(path="docs/test.md")],
            browser=BrowserScaffoldConfig(
                headless=False,
                viewport="1920x1080",
                save_video="1920x1080",
            ),
            executor=ExecutorConfig(
                system_prompt="@prompts/browser-testing.txt",
                user_prompt="Test the page.",
                tools=["browser_navigate"]
            ),
            judges=[JudgeConfig(name="check", system_prompt="Evaluate.")]
        )
        assert scaffold.browser is not None
        assert scaffold.browser.viewport == "1920x1080"
        assert scaffold.browser.headless is False

    def test_scaffold_accepts_browser_config_with_viewport_size_alias(self):
        """TestScaffold should accept browser config with viewport_size alias."""
        from dokumen.test_scaffold import BrowserScaffoldConfig

        config = BrowserScaffoldConfig(viewport_size="1280x720")
        assert config.viewport == "1280x720"

    def test_scaffold_rejects_browser_without_type(self):
        """TestScaffold with browser config but no type: browser should fail."""
        from pydantic import ValidationError
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig, FileRef, BrowserScaffoldConfig

        with pytest.raises(ValidationError, match="browser.*requires.*type"):
            TestScaffold(
                name="bad-browser-test",
                files=[FileRef(path="docs/test.md")],
                browser=BrowserScaffoldConfig(headless=False),
                executor=ExecutorConfig(
                    system_prompt="@prompts/browser-testing.txt",
                    user_prompt="Test the page.",
                    tools=["browser_navigate"]
                ),
                judges=[JudgeConfig(name="check", system_prompt="Evaluate.")]
            )

    def test_scaffold_type_defaults_none(self):
        """TestScaffold type should default to None for standard tests."""
        from dokumen.test_scaffold import TestScaffold, ExecutorConfig, JudgeConfig, FileRef

        scaffold = TestScaffold(
            name="standard-test",
            files=[FileRef(path="docs/test.md")],
            executor=ExecutorConfig(
                system_prompt="@prompts/documentation-validation.txt",
                user_prompt="Validate the doc.",
                tools=["read_file"]
            ),
            judges=[JudgeConfig(name="check", system_prompt="Evaluate.")]
        )
        assert scaffold.type is None

    def test_browser_scaffold_allows_file_paths_in_prompt(self):
        """Browser type scaffolds should allow file paths in user_prompt (for credentials)."""
        from dokumen.test_scaffold import validate_scaffold_file
        import tempfile
        import os

        scaffold_content = """
name: browser-login-test
type: browser
files:
  - path: docs/credentials/pat.txt
browser:
  headless: false
executor:
  system_prompt: "@prompts/browser-testing.txt"
  user_prompt: "Read docs/credentials/pat.txt for the PAT, then navigate to the login page."
  tools:
    - browser_navigate
    - browser_click
    - read_file
judges:
  - name: login-check
    system_prompt: "Verify the login was successful."
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.test.yaml', delete=False
        ) as f:
            f.write(scaffold_content)
            f.flush()
            temp_path = f.name

        try:
            valid, errors, warnings = validate_scaffold_file(temp_path)
            # Browser tests should NOT get errors about hardcoded paths
            path_errors = [e for e in errors if "hardcoded path" in e.lower()]
            assert len(path_errors) == 0, f"Browser test should allow file paths: {path_errors}"
        finally:
            os.unlink(temp_path)

    def test_browser_config_viewport_dict(self):
        """BrowserScaffoldConfig should accept viewport as dict."""
        from dokumen.test_scaffold import BrowserScaffoldConfig

        config = BrowserScaffoldConfig(viewport={"width": 1920, "height": 1080})
        assert config.viewport == {"width": 1920, "height": 1080}

    def test_browser_config_viewport_list(self):
        """BrowserScaffoldConfig should accept viewport as list."""
        from dokumen.test_scaffold import BrowserScaffoldConfig

        config = BrowserScaffoldConfig(viewport=[1920, 1080])
        assert config.viewport == [1920, 1080]


class TestScaffoldValidationBrowserType:
    """Tests for scaffold.py validate_scaffold() browser type support."""

    def test_validate_scaffold_rejects_browser_without_type(self):
        """validate_scaffold should reject browser config without type: browser."""
        from dokumen.scaffold import validate_scaffold

        data = {
            "name": "bad-browser-test",
            "files": [{"path": "docs/test.md"}],
            "browser": {"headless": False},
            "executor": {
                "system_prompt": "@prompts/browser-testing.txt",
                "user_prompt": "Test.",
                "tools": ["browser_navigate"],
            },
            "judges": [{"name": "check", "system_prompt": "Evaluate."}],
        }
        result = validate_scaffold(data)
        assert not result.valid
        assert any("browser" in e.lower() for e in result.errors)

    def test_validate_scaffold_accepts_browser_with_type(self):
        """validate_scaffold should accept browser config with type: browser."""
        from dokumen.scaffold import validate_scaffold

        data = {
            "name": "good-browser-test",
            "type": "browser",
            "files": [{"path": "docs/test.md"}],
            "browser": {"headless": False},
            "executor": {
                "system_prompt": "@prompts/browser-testing.txt",
                "user_prompt": "Test.",
                "tools": ["browser_navigate"],
            },
            "judges": [{"name": "check", "system_prompt": "Evaluate."}],
        }
        result = validate_scaffold(data)
        type_errors = [e for e in result.errors if "type" in e.lower() and "browser" in e.lower()]
        assert len(type_errors) == 0, f"Unexpected type errors: {type_errors}"

    def test_validate_scaffold_rejects_invalid_type(self):
        """validate_scaffold should reject invalid type values."""
        from dokumen.scaffold import validate_scaffold

        data = {
            "name": "bad-type-test",
            "type": "invalid",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "system_prompt": "@prompts/documentation-validation.txt",
                "user_prompt": "Test.",
                "tools": ["read_file"],
            },
            "judges": [{"name": "check", "system_prompt": "Evaluate."}],
        }
        result = validate_scaffold(data)
        assert not result.valid
        assert any("invalid" in e.lower() or "type" in e.lower() for e in result.errors)
