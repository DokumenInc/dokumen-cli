# Agent Guide

This repository contains the Dokumen CLI, an AI-powered documentation testing
framework. Use the public README and docs first; this file is a short operating
guide for coding agents working inside the repo.

## Priorities

- Preserve the executor-judge separation.
- Keep user-facing behavior covered by tests.
- Treat credentials, debug traces, and `.dokumen-cache/` output as sensitive.
- Prefer existing pipeline stages, tool-resolution paths, and SDK wrappers over
  new abstractions.

## Common Commands

```bash
dokumen --help
dokumen validate
dokumen run --dry-run
pytest tests/contracts -q
```

## Where Changes Usually Belong

- CLI behavior: command modules and command tests.
- Scaffold parsing or configuration: loader, parser, config models, validation
  tests.
- Agent execution: SDK wrappers, pipeline stages, and result types.
- Tool access: tool definitions, SDK/MCP mapping, validation hooks, and security
  tests.
- Output shape: output schemas, formatters, and command tests.

## Commenting Standard

Add comments where they explain non-obvious intent, security constraints,
compatibility behavior, or failure handling. Avoid comments that restate simple
code. Long-lived behavior should be documented in README or `docs/`.
