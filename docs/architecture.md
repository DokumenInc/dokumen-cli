# Architecture

This document is for engineers who need to understand how Dokumen executes a
skill test or safely extend the runtime. After reading it, you should be able to
trace a scaffold from YAML into agent execution and result output.

## System Model

Dokumen is organized around four concepts:

- A project configuration defines provider/model settings, coverage, tools,
  memory, optional coordinator behavior, and execution defaults.
- A test scaffold describes an agent skill, the files it covers, the tools
  the executor can use, and the judges that evaluate the result.
- A loader resolves scaffolds into executable test objects by applying
  configuration, skills, agents, tools, and model overrides.
- A pipeline runs each test through independent stages and emits a structured
  result.

The executor and judge roles stay separate by design. Executors perform work
using project knowledge and tools. Judges evaluate whether the work meets the
assertion.

## Runtime Flow

1. The CLI loads project configuration and discovers test scaffolds.
2. The loader validates each scaffold and resolves provider, agent, skill, and
   tool settings.
3. A test suite filters tests by command-line flags and manages cache state.
4. Each test runs through the pipeline.
5. Results are aggregated into console output and cache artifacts.

## Pipeline Stages

The test pipeline uses small stages with a shared context object:

- Browser setup prepares output directories for browser artifacts. The Claude
  Agent SDK starts Playwright MCP when browser tools are allowed.
- Setup runs pre-test commands or background processes.
- Explore discovers relevant files and injects retrieval context.
- Executor runs the main SDK-backed agent. Optional coordinator mode can
  decompose larger work across multiple workers when explicitly enabled.
- Compaction reduces long context before judging.
- Judge runs all assertions and records pass, fail, or infrastructure errors.
- Memory extracts reusable facts when enabled.
- Artifact collection gathers output files, reports, screenshots, and videos.

Stages fail fast by marking the shared context as failed. Cleanup callbacks still
run so setup processes and temporary resources are closed.

## Agents

Dokumen uses the Claude Agent SDK for the primary executor and judge path. The
base agent builds SDK options from:

- system and user prompts,
- allowed SDK tools,
- Dokumen MCP tools,
- external MCP servers such as Playwright,
- validation hooks,
- optional SDK-native subagents.

Wrappers adapt SDK results back into Dokumen's stable result types so older code
and tests can keep using the same interfaces.

## Tool Resolution

Scaffold tool names are resolved in layers:

- SDK built-ins, such as reading files, writing files, running shell commands,
  searching, fetching web pages, and web search.
- Browser tools, exposed through Playwright MCP.
- Dokumen-specific tools, exposed through the in-process MCP server.
- Agent delegation, mapped to SDK-native subagents when available.

Project configuration can define default, allowed, and blocked tools. The loader
also tracks tool provenance so CI logs show whether a tool came from a scaffold,
global defaults, auto-injection, or an agent definition.

Core scaffold execution does not expose the removed code graph tools or the old
PDF tree-section reader. PDF and image source material is supported through the
summary index workflow, while agent execution relies on SDK-native file and
browser handling.

## Current Boundaries

The presentation-ready path is local and SDK-backed:

- `dokumen validate`, `dokumen list`, `dokumen coverage`, and `dokumen status`
  operate on local config, scaffolds, and cache artifacts.
- `dokumen run` uses the Claude Agent SDK for executor and judge agents.
- Browser tools are passed to the SDK-managed Playwright MCP server.
- Dokumen-specific helpers, such as `read_many_files`, `explore`, and `ask`, are
  exposed as Dokumen MCP tools only when explicitly requested.

Some integration code remains intentionally optional:

- `DokuRouter` and direct provider classes still support non-test commands such
  as `ask`, `create`, and `summarize`.
- Backend-oriented stdin modes and workspace resolution are adapters for the
  hosted product, not required for local scaffold execution.
- Coordinator, task tracking, and memory are advanced paths and are disabled by
  default.

## Coordinator Mode

Coordinator mode is used for larger goals. It decomposes a goal into tasks,
resolves dependencies, runs independent workers in parallel, stores shared
memory, and synthesizes the worker outputs.

The coordinator supports three synthesis strategies:

- `merge`: concatenate and reconcile findings.
- `vote`: prefer findings with majority support.
- `chain`: feed results through workers sequentially.

If decomposition fails, the coordinator falls back to a simpler execution plan
instead of blocking the entire run. Keep this path optional; the default
presentation path is the single SDK executor plus judges.

## Output Contract

The output layer writes a consistent set of artifacts:

- run results for dashboards and API consumers,
- JUnit for CI systems,
- coverage summaries,
- explore traces,
- optional debug traces,
- executor and judge output files.

The output contract is intentionally separate from console formatting so the CLI
can be human-friendly without breaking integrations.

## Extension Points

Use the existing extension points before adding new framework concepts:

- Add commands through the CLI command group.
- Add prompt behavior through prompts, skills, or agent definitions.
- Add execution behavior through pipeline stages.
- Add tools through the tool resolver and SDK/MCP mapping.
- Add provider support through the provider abstraction or router.

Keep new features observable. A failed CI run should show which scaffold, model,
tool set, stage, and judge caused the failure.
