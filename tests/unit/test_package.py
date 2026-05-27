"""Package smoke tests to catch dependency and import issues early."""

import pytest


class TestCLIEntryPoint:
    """Verify CLI entry point is properly configured."""

    def test_cli_entry_point_exists(self):
        """CLI entry point should be importable and callable."""
        from dokumen.cli import cli

        assert cli is not None
        assert callable(cli)

    def test_cli_is_click_group(self):
        """CLI should be a Click group with commands."""
        import click
        from dokumen.cli import cli

        assert isinstance(cli, click.core.Group)


class TestVersion:
    """Verify version is accessible."""

    def test_version_accessible(self):
        """Package version should be accessible."""
        import dokumen

        assert hasattr(dokumen, "__version__")
        # Version is dynamically set, just verify it's a valid semver
        parts = dokumen.__version__.split(".")
        assert len(parts) == 3

    def test_version_is_string(self):
        """Version should be a string in semver format."""
        import dokumen

        version = dokumen.__version__
        assert isinstance(version, str)
        # Basic semver format check (X.Y.Z)
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)


class TestCoreImports:
    """Verify core modules are importable (catches missing dependencies)."""

    def test_config_imports(self):
        """Config module should import without errors."""
        from dokumen.config import DokumenConfig, ProviderConfig

        assert DokumenConfig is not None
        assert ProviderConfig is not None

    def test_scaffold_imports(self):
        """Test scaffold module should import without errors."""
        from dokumen.scaffold import (
            ValidationResult,
            discover_scaffolds,
            load_scaffold_yaml,
            validate_scaffold,
        )

        assert ValidationResult is not None
        assert discover_scaffolds is not None
        assert load_scaffold_yaml is not None
        assert validate_scaffold is not None

    def test_agent_imports(self):
        """Agent types module should import without errors."""
        from dokumen.agent_object import (
            AgentType,
            ExecutorOutput,
            JudgeResult,
            Provider,
        )

        assert AgentType is not None
        assert ExecutorOutput is not None
        assert JudgeResult is not None
        assert Provider is not None

    def test_output_imports(self):
        """Output schemas module should import without errors."""
        from dokumen.output_schemas import (
            ResultsJsonOutput,
            CoverageJsonOutput,
            DebugTraceOutput,
        )

        assert ResultsJsonOutput is not None
        assert CoverageJsonOutput is not None
        assert DebugTraceOutput is not None

    def test_tools_imports(self):
        """Tools module should import without errors."""
        from dokumen.tools_object import (
            ToolsObject,
            ToolDefinition,
            ToolResult,
            BUILTIN_TOOLS,
        )

        assert ToolsObject is not None
        assert ToolDefinition is not None
        assert ToolResult is not None
        assert BUILTIN_TOOLS is not None

    def test_loader_imports(self):
        """Loader module should import without errors."""
        from dokumen.loader import (
            load_scaffold,
            load_all_scaffolds,
            load_test_from_yaml,
        )

        assert load_scaffold is not None
        assert load_all_scaffolds is not None
        assert load_test_from_yaml is not None

    def test_test_suite_imports(self):
        """Test suite module should import without errors."""
        from dokumen.test_suite import TestSuite

        assert TestSuite is not None

    def test_test_object_imports(self):
        """Test object module should import without errors."""
        from dokumen.test_object import TestObject

        assert TestObject is not None

    def test_provider_imports(self):
        """Provider modules should import without errors."""
        from dokumen.providers import AnthropicProvider

        assert AnthropicProvider is not None
