"""Tests for dokumen.sdk.hooks — validation hooks for the Agent SDK path."""

import pytest
from unittest.mock import patch, MagicMock

from dokumen.sdk.hooks import (
    _deny,
    is_path_allowed,
    is_command_allowed,
    build_validation_hooks,
)


# -- Helper to build PreToolUseHookInput dicts --

def _pre_input(tool_name: str, tool_input: dict) -> dict:
    return {
        "session_id": "test",
        "transcript_path": "/tmp/test",
        "cwd": "/tmp",
        "agent_id": "test",
        "agent_type": "main",
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_use_id": "tu_123",
    }


def _post_input(tool_name: str, tool_input: dict, tool_response: str = "ok") -> dict:
    return {
        "session_id": "test",
        "transcript_path": "/tmp/test",
        "cwd": "/tmp",
        "agent_id": "test",
        "agent_type": "main",
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_response": tool_response,
        "tool_use_id": "tu_123",
    }


# -- _deny helper --

def test_deny_helper():
    result = _deny("bad path")
    assert result == {"decision": "deny", "reason": "bad path"}


# -- is_path_allowed --

@patch("os.getcwd", return_value="/project/root")
def test_is_path_allowed_within_project(mock_cwd):
    assert is_path_allowed("/project/root/docs/readme.md") is True


@patch("os.getcwd", return_value="/project/root")
def test_is_path_allowed_outside_project(mock_cwd):
    assert is_path_allowed("/etc/passwd") is False


@patch("os.getcwd", return_value="/project/root")
def test_is_path_allowed_traversal(mock_cwd):
    assert is_path_allowed("/project/root/docs/../../etc/passwd") is False


@patch("os.getcwd", return_value="/project/root")
def test_is_path_allowed_empty_path(mock_cwd):
    assert is_path_allowed("") is False


# -- is_command_allowed --

def test_is_command_allowed_default():
    assert is_command_allowed("ls -la") is True


# -- build_validation_hooks structure --

def test_build_validation_hooks_structure():
    hooks = build_validation_hooks()
    assert "PreToolUse" in hooks
    assert len(hooks["PreToolUse"]) == 1
    matcher = hooks["PreToolUse"][0]
    assert matcher.matcher == "Read|Write|Edit|Bash"


def test_build_validation_hooks_no_callback():
    hooks = build_validation_hooks()
    assert "PostToolUse" not in hooks


# -- PreToolUse hook behavior --

@pytest.mark.asyncio
@patch("dokumen.sdk.hooks.is_path_allowed", return_value=True)
async def test_pre_tool_use_allows_valid_read(mock_allowed):
    hooks = build_validation_hooks()
    pre_hook = hooks["PreToolUse"][0].hooks[0]

    input_data = _pre_input("Read", {"file_path": "/project/root/docs/readme.md"})
    result = await pre_hook(input_data, "tu_123", None)

    assert result == {}
    mock_allowed.assert_called_once_with("/project/root/docs/readme.md", None)


@pytest.mark.asyncio
@patch("dokumen.sdk.hooks.is_path_allowed", return_value=False)
async def test_pre_tool_use_denies_invalid_path(mock_allowed):
    hooks = build_validation_hooks()
    pre_hook = hooks["PreToolUse"][0].hooks[0]

    input_data = _pre_input("Read", {"file_path": "/etc/passwd"})
    result = await pre_hook(input_data, "tu_123", None)

    assert result["decision"] == "deny"
    assert "/etc/passwd" in result["reason"]


@pytest.mark.asyncio
async def test_pre_tool_use_bash_timeout_injection():
    """Bash gets timeout from tools_config when not set or exceeds limit."""
    tools_config = MagicMock()
    tools_config.config.run_shell_command.timeout = 30.0

    hooks = build_validation_hooks(tools_config=tools_config)
    pre_hook = hooks["PreToolUse"][0].hooks[0]

    tool_input = {"command": "echo hello"}
    input_data = _pre_input("Bash", tool_input)
    result = await pre_hook(input_data, "tu_123", None)

    assert result == {}
    assert tool_input["timeout"] == 30000


@pytest.mark.asyncio
@patch("dokumen.sdk.hooks.is_command_allowed", return_value=False)
async def test_pre_tool_use_denies_disallowed_command(mock_cmd):
    """Bash command denied when is_command_allowed returns False."""
    hooks = build_validation_hooks()
    pre_hook = hooks["PreToolUse"][0].hooks[0]

    input_data = _pre_input("Bash", {"command": "rm -rf /"})
    result = await pre_hook(input_data, "tu_123", None)

    assert result["decision"] == "deny"
    assert "rm -rf /" in result["reason"]
    mock_cmd.assert_called_once_with("rm -rf /", None)


@pytest.mark.asyncio
async def test_pre_tool_use_bash_timeout_attribute_error():
    """Bash timeout gracefully handles tools_config missing expected attributes."""
    # tools_config exists but lacks config.run_shell_command.timeout
    tools_config = MagicMock()
    del tools_config.config  # Force AttributeError on access

    hooks = build_validation_hooks(tools_config=tools_config)
    pre_hook = hooks["PreToolUse"][0].hooks[0]

    tool_input = {"command": "echo hello"}
    input_data = _pre_input("Bash", tool_input)
    result = await pre_hook(input_data, "tu_123", None)

    # Should not crash, should return empty (allowed)
    assert result == {}
    assert "timeout" not in tool_input


@pytest.mark.asyncio
async def test_pre_tool_use_bash_timeout_no_config():
    """Bash without tools_config does not modify timeout."""
    hooks = build_validation_hooks(tools_config=None)
    pre_hook = hooks["PreToolUse"][0].hooks[0]

    tool_input = {"command": "echo hello"}
    input_data = _pre_input("Bash", tool_input)
    result = await pre_hook(input_data, "tu_123", None)

    assert result == {}
    assert "timeout" not in tool_input


@pytest.mark.asyncio
async def test_pre_tool_use_bash_timeout_not_overridden_when_lower():
    """Bash timeout is NOT overridden when existing timeout is within limit."""
    tools_config = MagicMock()
    tools_config.config.run_shell_command.timeout = 60.0

    hooks = build_validation_hooks(tools_config=tools_config)
    pre_hook = hooks["PreToolUse"][0].hooks[0]

    tool_input = {"command": "echo hello", "timeout": 5000}
    input_data = _pre_input("Bash", tool_input)
    result = await pre_hook(input_data, "tu_123", None)

    assert result == {}
    assert tool_input["timeout"] == 5000  # kept original, lower than 60000


# -- PostToolUse hook behavior --

@pytest.mark.asyncio
async def test_post_tool_use_callback():
    """on_tool_call is called with correct args."""
    calls = []

    def track(name, inp, resp):
        calls.append((name, inp, resp))

    hooks = build_validation_hooks(on_tool_call=track)
    assert "PostToolUse" in hooks

    post_hook = hooks["PostToolUse"][0].hooks[0]
    input_data = _post_input("Read", {"file_path": "/tmp/f.txt"}, "file contents")
    result = await post_hook(input_data, "tu_123", None)

    assert result == {}
    assert len(calls) == 1
    assert calls[0] == ("Read", {"file_path": "/tmp/f.txt"}, "file contents")
