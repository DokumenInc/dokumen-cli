"""Shared constants for scaffold and tool validation."""

KNOWN_MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-0-20250514",
}

CLI_RESOLVABLE_TOOLS = {
    "read_file",
    "write_file",
    "glob",
    "list_directory",
    "read_many_files",
    "run_shell_command",
    "search_file_content",
    "web_fetch",
    "web_search",
    "task_create",
    "task_update",
    "task_list",
    "task_output",
}

SDK_TOOL_ALIASES = {
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "Bash",
    "WebFetch",
    "WebSearch",
}

BROWSER_TOOLS = {
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_snapshot",
    "browser_screenshot",
    "browser_take_screenshot",
    "browser_wait",
    "browser_close",
    "browser_evaluate",
}

AGENT_TOOLS = {"explore"}

VALID_EXECUTOR_TOOLS = CLI_RESOLVABLE_TOOLS | SDK_TOOL_ALIASES | BROWSER_TOOLS | AGENT_TOOLS
