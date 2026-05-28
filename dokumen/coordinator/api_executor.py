"""
direct API executor — runs worker tasks via provider.complete() instead of SDK.

bypasses the bundled claude CLI entirely. uses the same tool handlers
from tools_object.py but drives the agent loop ourselves:
  1. send messages + tools to provider.complete()
  2. if response has tool_use, execute tool, append result, loop
  3. when response is text-only (end_turn), return the final text

this avoids the exit code 1 crash from the bundled CLI message reader.
"""

import asyncio
import logging
import os
import glob as glob_module
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_TURNS = 50


async def _handle_tool_call(tool_name: str, tool_input: Dict[str, Any], base_dir: str) -> str:
    """execute a tool call and return the result as text.

    supports the core read-only tools that workers need:
    read_file, list_directory, glob, search_file_content.
    """
    try:
        if tool_name == "read_file":
            path = tool_input.get("file_path") or tool_input.get("path", "")
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            if not os.path.exists(path):
                return f"error: file not found: {path}"
            with open(path, "r", errors="replace") as f:
                content = f.read()
            # truncate large files
            if len(content) > 100_000:
                content = content[:100_000] + f"\n\n... truncated ({len(content)} chars total)"
            return content

        elif tool_name == "list_directory":
            path = tool_input.get("path", ".")
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            if not os.path.isdir(path):
                return f"error: not a directory: {path}"
            entries = sorted(os.listdir(path))
            return "\n".join(entries) if entries else "(empty directory)"

        elif tool_name == "glob":
            pattern = tool_input.get("pattern", "*")
            path = tool_input.get("path", base_dir)
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            full_pattern = os.path.join(path, pattern)
            matches = sorted(glob_module.glob(full_pattern, recursive=True))
            # return relative paths
            rel = [os.path.relpath(m, base_dir) for m in matches]
            if len(rel) > 200:
                rel = rel[:200] + [f"... and {len(rel) - 200} more"]
            return "\n".join(rel) if rel else "no matches"

        elif tool_name == "search_file_content":

            pattern = tool_input.get("pattern", "")
            search_path = tool_input.get("path", ".")
            if not os.path.isabs(search_path):
                search_path = os.path.join(base_dir, search_path)
            case_flag = "-i" if tool_input.get("case_insensitive") else ""
            cmd = f"grep -rn {case_flag} -- {_shell_quote(pattern)} {_shell_quote(search_path)}"
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=base_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            output = stdout.decode("utf-8", errors="replace")
            if proc.returncode == 1:
                return "no matches found"
            if proc.returncode > 1:
                return f"error: {stderr.decode('utf-8', errors='replace')}"
            # truncate
            if len(output) > 50_000:
                output = output[:50_000] + "\n... truncated"
            return output

        elif tool_name == "write_file":
            path = tool_input.get("file_path") or tool_input.get("path", "")
            content = tool_input.get("content", "")
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"wrote {len(content)} chars to {path}"

        elif tool_name == "spawn_worker":
            # handled separately in the agent loop — should not reach here
            return "error: spawn_worker must be handled by the agent loop"

        else:
            return f"error: unknown tool '{tool_name}'"

    except Exception as e:
        return f"error executing {tool_name}: {e}"


def _shell_quote(s: str) -> str:
    """simple shell quoting."""
    import shlex

    return shlex.quote(s)


def _tools_to_anthropic_format(tool_names: List[str]) -> List[Dict[str, Any]]:
    """convert tool names to anthropic API tool definitions."""
    tool_schemas = {
        "read_file": {
            "name": "read_file",
            "description": "read a file and return its contents",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "path to the file to read"},
                },
                "required": ["file_path"],
            },
        },
        "list_directory": {
            "name": "list_directory",
            "description": "list files and directories in a path",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "directory path (default: current dir)",
                    },
                },
                "required": [],
            },
        },
        "glob": {
            "name": "glob",
            "description": "find files matching a glob pattern (e.g. **/*.py)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob pattern"},
                    "path": {"type": "string", "description": "base directory"},
                },
                "required": ["pattern"],
            },
        },
        "search_file_content": {
            "name": "search_file_content",
            "description": "search file contents with a regex pattern (grep -rn)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "regex pattern to search for"},
                    "path": {"type": "string", "description": "file or directory to search"},
                    "case_insensitive": {"type": "boolean", "description": "ignore case"},
                },
                "required": ["pattern"],
            },
        },
        "write_file": {
            "name": "write_file",
            "description": "write content to a file (creates directories as needed)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "path to write to"},
                    "content": {"type": "string", "description": "file content"},
                },
                "required": ["file_path", "content"],
            },
        },
        "spawn_worker": {
            "name": "spawn_worker",
            "description": (
                "spawn a sub-agent to handle a scoped task autonomously. "
                "the worker gets its own context and tool access (read_file, glob, list_directory, search_file_content, write_file). "
                "use this to delegate independent pieces of work — e.g. 'implement the tokenizer module' or "
                "'write tests for the grammar module'. the worker runs to completion and returns its output. "
                "you can spawn multiple workers for independent tasks."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "clear description of what the worker should accomplish",
                    },
                    "context": {
                        "type": "string",
                        "description": "relevant context the worker needs (blueprint excerpts, file paths, etc.)",
                    },
                },
                "required": ["goal"],
            },
        },
    }

    result = []
    for name in tool_names:
        if name in tool_schemas:
            result.append(tool_schemas[name])
        else:
            logger.warning("api_executor: unknown tool requested", extra={"tool_name": name})
    return result


async def run_api_executor(
    provider: Any,
    system_prompt: str,
    user_prompt: str,
    tool_names: Optional[List[str]] = None,
    max_turns: int = MAX_TURNS,
    timeout: float = 300.0,
    base_dir: str = ".",
    max_tokens: int = 64000,
    _depth: int = 0,
) -> Dict[str, Any]:
    """run an agent loop using direct provider.complete() calls.

    args:
        _depth: internal recursion depth for spawn_worker (max 2 levels)

    returns:
        dict with keys: success, output, error, input_tokens, output_tokens
    """
    if tool_names is None:
        tool_names = ["read_file", "list_directory", "glob", "search_file_content"]

    # always include spawn_worker unless we're already a sub-agent (depth limit)
    effective_tools = list(tool_names)
    if "spawn_worker" not in effective_tools and _depth < 2:
        effective_tools.append("spawn_worker")

    tools = _tools_to_anthropic_format(effective_tools)
    messages = [{"role": "user", "content": user_prompt}]
    total_input = 0
    total_output = 0

    logger.info(
        "api_executor starting",
        extra={"tools": tool_names, "max_turns": max_turns, "base_dir": base_dir},
    )

    for turn in range(max_turns):
        try:
            result = await asyncio.wait_for(
                provider.complete(
                    messages=messages,
                    tools=tools,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("api_executor: provider call timed out", extra={"turn": turn})
            return {
                "success": False,
                "output": "",
                "error": f"timed out on turn {turn}",
                "input_tokens": total_input,
                "output_tokens": total_output,
            }
        except Exception as e:
            logger.error(
                "api_executor: provider call failed", extra={"turn": turn, "error": str(e)}
            )
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "input_tokens": total_input,
                "output_tokens": total_output,
            }

        # track tokens
        usage = result.get("usage", {})
        total_input += usage.get("input_tokens", 0)
        total_output += usage.get("output_tokens", 0)

        content = result.get("content", "")
        tool_calls = result.get("tool_use", [])

        if not tool_calls:
            # no tool calls — this is the final response
            logger.info(
                "api_executor: completed",
                extra={"turns": turn + 1, "output_length": len(content)},
            )
            return {
                "success": True,
                "output": content,
                "error": None,
                "input_tokens": total_input,
                "output_tokens": total_output,
            }

        # build assistant message with text + tool_use blocks
        assistant_content = []
        if content:
            assistant_content.append({"type": "text", "text": content})
        for tc in tool_calls:
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                }
            )
        messages.append({"role": "assistant", "content": assistant_content})

        # execute tools and build tool_result message
        tool_results = []
        tool_counts: Dict[str, int] = {}
        for tc in tool_calls:
            if tc["name"] == "spawn_worker":
                # recursive sub-agent
                goal = tc["input"].get("goal", "")
                context = tc["input"].get("context", "")
                worker_prompt = f"{context}\n\ngoal: {goal}" if context else goal
                worker_system = (
                    "you are a worker agent. complete your assigned task using the provided tools. "
                    "ALWAYS use tools — never narrate what you would do. "
                    "if your task involves writing files, use write_file for every file."
                )
                print(f"  ⚡ spawning worker: {goal[:100]}", flush=True)
                logger.info("spawn_worker", extra={"goal": goal[:200], "depth": _depth + 1})

                worker_result = await run_api_executor(
                    provider=provider,
                    system_prompt=worker_system,
                    user_prompt=worker_prompt,
                    tool_names=[t for t in tool_names if t != "spawn_worker"],
                    max_turns=max(20, max_turns // 3),
                    timeout=timeout,
                    base_dir=base_dir,
                    max_tokens=max_tokens,
                    _depth=_depth + 1,
                )
                total_input += worker_result.get("input_tokens", 0)
                total_output += worker_result.get("output_tokens", 0)

                if worker_result["success"]:
                    tool_output = (
                        f"worker completed successfully:\n{worker_result.get('output', '')}"
                    )
                else:
                    tool_output = f"worker failed: {worker_result.get('error', 'unknown error')}"
                print(
                    f"  ✓ worker done: {'ok' if worker_result['success'] else 'failed'}", flush=True
                )
            else:
                tool_output = await _handle_tool_call(tc["name"], tc["input"], base_dir)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": tool_output,
                }
            )
            tool_counts[tc["name"]] = tool_counts.get(tc["name"], 0) + 1

        messages.append({"role": "user", "content": tool_results})

        # compact progress line
        parts = [f"{name} ×{count}" if count > 1 else name for name, count in tool_counts.items()]
        text_preview = ""
        if content:
            first_line = content.strip().split("\n")[0][:80]
            text_preview = f"  {first_line}"
        print(f"  ↳ turn {turn + 1}: {', '.join(parts)}{text_preview}", flush=True)

        logger.debug(
            "api_executor: turn complete",
            extra={"turn": turn, "tool_calls": len(tool_calls)},
        )

    # exhausted turns
    logger.warning("api_executor: max turns reached", extra={"max_turns": max_turns})
    return {
        "success": False,
        "output": "",
        "error": f"max turns ({max_turns}) reached",
        "input_tokens": total_input,
        "output_tokens": total_output,
    }
