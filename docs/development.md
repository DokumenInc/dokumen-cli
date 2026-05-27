# Development

This guide is for engineers changing Dokumen itself. The main product path is a
CLI command that runs a Claude Code-style skill attempt and then asks LLM judges
whether the attempt met the scaffold's success criteria.

## Setup

Use Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Some optional integration paths require external credentials. Keep local smoke
checks offline and deterministic unless you are intentionally testing
provider-backed commands.

## Useful Commands

```bash
dokumen --help
dokumen validate
dokumen run --dry-run
pytest tests/contracts -q
python -m compileall -q dokumen dokumen_schema
```

When editing behavior that touches agents or tools, run the smallest relevant
contract test first, then broaden to command-level smoke checks. The repository
no longer carries the old ad hoc unit-test scripts; add focused contract tests
when a cleanup needs a durable guardrail.

## Code Organization

- CLI commands are Click commands grouped under the main `dokumen` command.
- Configuration is parsed with Pydantic and should reject unsafe values early.
- The loader is the boundary between YAML scaffolds and runtime objects.
- The pipeline owns test execution order and cleanup behavior.
- Stages should do one job and communicate through the shared pipeline context.
- SDK wrappers keep external agent behavior behind stable Dokumen result types.
- Tools should validate inputs, limit side effects, and report useful errors.
- Coverage and status commands are experimental; do not let coverage concerns
  complicate the core executor-plus-judge path.
- Coordinator mode should remain disabled by default. The standard behavior is
  one executor prompted to use a skill, followed by LLM judge evaluation.

## Adding A Command

1. Add a command module.
2. Register it with the main CLI group.
3. Keep command parsing separate from runtime logic where practical.
4. Add tests for flags, exit codes, and output formats.
5. Document user-facing behavior in the README or a focused docs page.

## Adding A Tool

1. Define the tool schema and handler.
2. Add it to tool resolution.
3. Decide whether it maps to an SDK built-in, Dokumen MCP tool, or external MCP
   server.
4. Add validation hooks if it can read, write, shell out, or access the network.
5. Test successful execution, bad inputs, and blocked behavior.

## Adding A Pipeline Stage

Add a stage only when the behavior is a distinct step in the test lifecycle.
Stages should:

- read and write the shared context explicitly,
- mark the context failed instead of raising for expected failure modes,
- leave cleanup to pipeline cleanup callbacks,
- log enough state to debug CI failures.

## Documentation Standard

Public docs should answer what the reader can do next. Prefer examples and
explicit commands over implementation history. Keep internal notes out of the
repository root, and never commit credentials or local run output.
