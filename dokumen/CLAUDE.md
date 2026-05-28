# Package Agent Guide

This package contains the core Dokumen runtime. Prefer the root README and docs
for public-facing explanations; use this file for package-level orientation.

## Execution Model

Dokumen turns YAML scaffolds into executable agent SOP tests. Each test has an
executor that performs a grounded business task and judges that evaluate the
result. The test pipeline keeps setup, exploration, execution, judging, memory
extraction, and artifact collection as separate stages.

## Working Rules

- Keep scaffold parsing, tool resolution, and execution concerns separated.
- Add comments for non-obvious intent, compatibility behavior, and security
  decisions. Do not add comments that merely restate simple code.
- Keep result schemas stable unless the output contract is intentionally being
  changed.
- Treat debug traces and cache output as sensitive.

## Useful Commands

```bash
dokumen validate
dokumen run --dry-run
pytest tests/unit -q
pytest tests/unit/test_config.py -q
```

## Extension Map

- Commands: add a Click command and register it with the main CLI group.
- Configuration: update the Pydantic config model and validation tests.
- Runtime behavior: add or modify a pipeline stage.
- Tools: update tool definitions, SDK/MCP mapping, validation hooks, and tests.
- Output: update schemas, formatters, and command tests together.
