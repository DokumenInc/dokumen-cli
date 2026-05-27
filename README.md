# Dokumen CLI

Dokumen is an AI-powered documentation testing framework. It checks whether
documentation can actually support the tasks it claims to support.

This README is for engineers evaluating, presenting, or running the CLI. After
reading it, you should be able to install Dokumen, write a test scaffold, run the
test suite, and understand the main execution model.

## What It Does

Dokumen uses an executor-judge workflow:

1. An executor agent reads documentation and performs a task.
2. One or more judge agents evaluate the executor output against explicit
   assertions.
3. Dokumen writes machine-readable results, coverage data, debug traces, and any
   generated artifacts.

This makes documentation testable in CI. Instead of asking whether docs look
complete, Dokumen asks whether an agent can use them successfully.

## Core Capabilities

- Run documentation tests from YAML scaffolds.
- Validate scaffolds and project configuration before CI execution.
- Track which documentation files are covered by tests.
- Run exploration before execution so agents can discover relevant files.
- Support browser-oriented tests through Playwright MCP tools.
- Generate summaries for text, image, and PDF documentation with
  `dokumen summarize`.
- Generate new test scaffolds from a natural-language goal.
- Ask documentation-grounded questions against an existing test corpus.
- Analyze codebases with `mimick` and produce architecture blueprints.
- Emit JSON, JUnit, TAP, and text output for CI and dashboards.

## Installation

Dokumen requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Set provider credentials before running agent-backed commands:

```bash
export ANTHROPIC_API_KEY=...
```

## Quick Start

Create `dokumen.yaml`:

```yaml
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
coverage:
  include:
    - docs/**/*.md
execution:
  timeout: 600
```

Create a test scaffold under `tests/`:

```yaml
name: api-authentication-docs
reason: Verify that the authentication docs explain supported auth methods.

files:
  - path: docs/authentication.md

executor:
  user_prompt: |
    Read the authentication docs and explain the supported authentication
    methods, required headers, and setup steps.
  tools:
    - read_file
    - glob

judges:
  - name: groundedness
    system_prompt: |
      Pass only if the answer is fully grounded in the referenced docs and
      clearly answers the user's question.
      Return JSON: {"verdict": "PASS" or "FAIL", "reason": "..."}
```

Validate and run:

```bash
dokumen validate
dokumen run
```

CI-friendly output:

```bash
dokumen run --output json
dokumen run --output junit
```

## Command Overview

| Command | Purpose |
| --- | --- |
| `dokumen run` | Execute documentation tests. |
| `dokumen validate` | Validate configuration and test scaffolds. |
| `dokumen list` | List tests, files, or tools. |
| `dokumen coverage` | Show documentation coverage. |
| `dokumen status` | Emit a compact CI status summary. |
| `dokumen explore` | Discover files relevant to a topic. |
| `dokumen ask` | Answer questions using documentation and test context. |
| `dokumen create` | Generate a scaffold from a natural-language goal. |
| `dokumen summarize` | Build summary indexes for large documentation sets. |
| `dokumen mimick` | Analyze a codebase and produce an architecture blueprint. |
| `dokumen config` | View or edit project configuration. |

## Test Lifecycle

Each test runs through a stage pipeline:

1. Prepare output directories and setup resources when requested.
2. Explore the workspace and inject discovered context.
3. Run the SDK-backed executor.
4. Compact context when enabled.
5. Run judges concurrently.
6. Extract memory and collect output artifacts.
7. Write results and coverage files.

The important design choice is that execution and evaluation are separate. The
executor proves the docs are usable. The judges prove the result satisfies the
test's criteria.

The default test path is intentionally narrow: scaffolds are loaded locally,
executor and judge agents run through the Claude Agent SDK, browser actions go
through SDK-managed Playwright MCP, and Dokumen-specific helpers are exposed as a
small in-process MCP server only when a scaffold asks for them.

## Outputs

Dokumen writes run artifacts under `.dokumen-cache/`:

- `results.json` for dashboards and API ingestion.
- `junit.xml` for CI test reports.
- `coverage.json` for documentation coverage.
- `debug-traces/` when `--debug` is enabled.
- `output/` for files produced by executors or judges.

The cache directory is local run output and should not be committed.

## Documentation

- [Architecture](docs/architecture.md)
- [Development](docs/development.md)
- [Authentication and credentials](docs/authentication.md)
- [Security](SECURITY.md)

## Development

Run the repository smoke checks while iterating:

```bash
python -m compileall -q dokumen dokumen_schema
dokumen --help
dokumen validate --config-only
pytest tests/contracts -q
```

Run formatting and linting when those tools are installed:

```bash
ruff check dokumen tests
black dokumen tests
```

The committed test suite is intentionally small. `tests/contracts` protects the
public tool surface and SDK mapping behavior without carrying the old ad hoc
unit-test scripts or credential-backed integration fixtures.

## Security Notes

Dokumen tests may run agents with file, shell, web, or browser tools. Keep tool
allowlists narrow, store credentials in environment variables or CI secrets, and
do not commit `.dokumen-cache/` output.
