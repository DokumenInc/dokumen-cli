# Dokumen CLI

```text
    ____        __
   / __ \____  / /____  ______ ___  ___  ____
  / / / / __ \/ //_/ / / / __ `__ \/ _ \/ __ \
 / /_/ / /_/ / ,< / /_/ / / / / / /  __/ / / /
/_____/\____/_/|_|\__,_/_/ /_/ /_/\___/_/ /_/

       Business SOP Agent Test CLI
```

Dokumen is a CLI for testing whether agents can follow business SOPs. You
define a task, the procedure the agent should follow, the tools the agent may
use, and the success criteria. Dokumen runs the agent attempt and then asks LLM
judges whether the result followed the SOP.

The CLI is standalone: it loads local config, local scaffolds, and local SOPs,
then calls the configured agent/runtime provider directly. It does not require a
hosted Dokumen service.

This README is for engineers evaluating, presenting, or running the CLI. After
reading it, you should be able to install Dokumen, write an SOP test scaffold
with success criteria, run it from the command line, and understand the main
executor-judge model.

## What It Does

Dokumen turns SOP adherence into a repeatable command:

1. A scaffold describes the business case, source files, tools, SOP references,
   and judge criteria.
2. An executor agent attempts the task with the allowed tools.
3. One or more LLM judges evaluate the final output and tool log against the
   success criteria.
4. Dokumen writes machine-readable results, CI output, debug traces, and any
   generated artifacts.

This makes agent behavior testable in CI. Instead of asking whether a workflow
or prompt looks complete, Dokumen asks whether an agent can execute the
procedure and whether an independent judge agrees that it met the stated bar.

## Core Capabilities

- Run agent SOP tests from YAML scaffolds.
- Define pass/fail success criteria as judge prompts.
- Validate scaffolds and project configuration before CI execution.
- Inject reusable SOPs or instructions into executor and judge prompts.
- Optionally run exploration before execution so agents can discover relevant
  files.
- Support browser-oriented tests through Playwright MCP tools.
- Emit JSON, JUnit, TAP, and text output for CI and dashboards.

Useful supporting commands:

- Generate summaries for text, image, and PDF source material with
  `dokumen summarize`.
- Author new test scaffolds with the packaged authoring skill named
  `dokumen-test-author` when working inside Claude Code.
- Track file-level source coverage with `dokumen coverage` and `dokumen status`
  as experimental commands.

## Installation

Dokumen requires Python 3.11 or newer.

Install the CLI as a user-level command:

```bash
uv tool install git+https://github.com/DokumenInc/dokumen-cli.git
dokumen --help
```

From a local checkout:

```bash
uv tool install --force .
```

If `dokumen` is not found after installation, add uv's tool directory to your
shell:

```bash
uv tool update-shell
```

For an editable development checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
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
execution:
  timeout: 600
explore:
  enabled: false
compaction:
  enabled: false
coordinator:
  enabled: false
```

Create a reusable SOP under `sops/refund-escalation-sop.md`:

```markdown
# Refund Escalation SOP

When reviewing a refund request:

- Identify the customer's plan, request amount, and stated reason.
- Confirm whether the request is inside the 30-day refund window.
- Escalate to Finance when the amount is over $500 or the customer is on an
  enterprise plan.
- Include the recommended next action and the reason for that action.
```

Create a test scaffold under `tests/`:

```yaml
name: refund-escalation
reason: Verify that the executor follows the refund escalation SOP.

files:
  - path: docs/customer-ticket.md

executor:
  sops:
    - refund-escalation-sop
  tools:
    - read_file
  user_prompt: |
    Follow the refund-escalation-sop while reviewing the referenced customer
    ticket. Report the customer's plan, request amount, refund-window status,
    escalation requirement, and recommended next action.

judges:
  - name: sop-success-criteria
    include_executor_output: true
    system_prompt: |
      Pass only if the executor output proves it followed the refund escalation
      SOP and clearly reports the plan, amount, refund-window status,
      escalation requirement, and recommended next action.
      Return JSON: {"verdict": "PASS" or "FAIL", "reason": "..."}
```

The judge prompt is the success criteria. Keep it specific enough that a fresh
LLM judge can decide pass or fail from the executor output and tool log.

## SOP Test Pattern

The normal Dokumen test shape is a single executor prompted to follow a named
SOP, followed by an LLM judge that checks whether the procedure was followed
correctly. Coordinator mode is off by default and is not part of this path.

Place SOPs in `sops/`:

```markdown
# Account Cancellation SOP

When processing a cancellation request, identify the customer's account tier,
contract term, cancellation date, required approvals, and retention-risk notes.
```

Reference and prompt the SOP from a scaffold:

```yaml
name: account-cancellation

files:
  - path: docs/cancellation-ticket.md

executor:
  sops:
    - account-cancellation-sop
  tools:
    - read_file
  user_prompt: |
    Follow the account-cancellation-sop while reviewing the referenced ticket.
    Report the required approvals, risks, and next action.

judges:
  - name: sop-success-criteria
    include_executor_output: true
    system_prompt: |
      Pass only if the executor output follows the SOP and clearly reports the
      required approvals, risks, and next action.
      Return JSON: {"verdict": "PASS" or "FAIL", "reason": "..."}
```

The legacy `executor.skills` field still works for existing scaffolds. New
business-process tests should prefer `executor.sops` so the intent is clear.

A complete copyable example lives in
[`examples/business-sop`](examples/business-sop/README.md).

## Authoring Tests

This repo packages an authoring helper named `dokumen-test-author` for creating
or revising Dokumen tests in Claude Code. It replaces the removed
scaffold-generation command and keeps authoring aligned with the preferred SOP
test pattern: define or reuse an SOP, prompt the executor to follow it, and
write a judge that evaluates explicit success criteria.

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

Core commands:

| Command | Purpose |
| --- | --- |
| `dokumen run` | Execute agent SOP tests. |
| `dokumen validate` | Validate configuration and test scaffolds. |
| `dokumen list` | List tests, files, or tools. |

Supporting commands:

| Command | Purpose |
| --- | --- |
| `dokumen help` | Show general help or help for a command. |
| `dokumen explore` | Discover files relevant to a topic. |
| `dokumen summarize` | Build summary indexes for large source sets. |
| `dokumen config` | View or edit project configuration. |

Experimental commands:

| Command | Purpose |
| --- | --- |
| `dokumen coverage` | Show file-level source coverage. |
| `dokumen status` | Emit a compact coverage status summary. |

## Test Lifecycle

Each test runs through a stage pipeline:

1. Prepare output directories and setup resources when requested.
2. Optionally explore the workspace and inject discovered context.
3. Run the SDK-backed executor.
4. Compact context when enabled.
5. Run judges concurrently.
6. Extract memory and collect output artifacts.
7. Write results and optional experimental coverage files.

The important design choice is that execution and evaluation are separate. The
executor attempts the business task. The judges prove the result satisfies the
test's criteria.

The default test path is intentionally narrow: scaffolds are loaded locally,
executor and judge agents run through the Claude Agent SDK, browser actions go
through SDK-managed Playwright MCP, and Dokumen-specific helpers are exposed as a
small in-process MCP server only when a scaffold asks for them.

## Outputs

Dokumen writes run artifacts under `.dokumen-cache/`:

- `results.json` for dashboards and automation.
- `junit.xml` for CI test reports.
- `coverage.json` for experimental source coverage.
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
uv run --extra dev python -m compileall -q dokumen dokumen_schema tests/contracts
uv run --extra dev dokumen --help
uv run --extra dev dokumen validate --config-only
uv run --extra dev pytest tests/contracts -q
```

Run formatting and linting when those tools are installed:

```bash
uv run --extra dev ruff check dokumen dokumen_schema tests
uv run --extra dev black --check dokumen dokumen_schema tests
```

The committed test suite is intentionally small. `tests/contracts` protects the
public tool surface and SDK mapping behavior without carrying the old ad hoc
unit-test scripts or credential-backed integration fixtures.

## Security Notes

Dokumen tests may run agents with file, shell, web, or browser tools. Keep tool
allowlists narrow, store credentials in environment variables or CI secrets, and
do not commit `.dokumen-cache/` output.
