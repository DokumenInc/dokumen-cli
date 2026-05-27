"""
mimick command — architecture analysis via agent.

points our executor agent (with the mimick skill prompt) at a target
codebase and generates an architecture blueprint. the agent does the
heavy lifting — exploring files, detecting patterns, mapping modules.
"""
import logging
import os
import sys
from typing import Optional

import click

from ..helpers import load_config, run_async

logger = logging.getLogger(__name__)


def _load_mimick_config() -> dict:
    """load coordinator config from dokumen.yaml if available."""
    try:
        config_path = os.path.join(os.getcwd(), "dokumen.yaml")
        if os.path.exists(config_path):
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


async def _run_mimick(
    source_path: str,
    name: str,
    output: Optional[str],
    timeout: float,
    config: Optional[dict] = None,
) -> str:
    """run the mimick skill agent against a codebase.

    uses coordinator mode if enabled in config, otherwise single agent.
    """
    from dokumen.test_builder import get_configured_provider

    # get provider
    provider = get_configured_provider()
    if provider is None:
        raise click.ClickException(
            "no provider configured — set DOKUMEN_PROVIDER and DOKUMEN_API_KEY, "
            "or add a provider section to dokumen.yaml"
        )

    model = getattr(provider, "model", None)

    # read mimick config from dokumen.yaml
    yaml_config = _load_mimick_config()
    mimick_config = yaml_config.get("mimick", {})
    mimick_max_turns = mimick_config.get("max_turns", 50)
    mimick_timeout = mimick_config.get("timeout", timeout)
    mimick_model = mimick_config.get("model", "") or model
    build_max_turns = mimick_config.get("build_max_turns", 80)

    # read the mimick prompt
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "prompts", "executors", "mimick.txt"
    )
    try:
        with open(prompt_path) as f:
            system_prompt = f.read()
    except FileNotFoundError:
        system_prompt = (
            "you are an architecture analysis agent. explore the codebase thoroughly, "
            "identify modules, patterns, and dependencies, then produce a YAML blueprint."
        )

    tool_names = ["read_file", "list_directory", "glob", "search_file_content"]

    user_prompt = (
        f"analyze the codebase at: {source_path}\n"
        f"project name: {name}\n\n"
        "explore the entire codebase, map all modules, detect architectural patterns, "
        "and produce a comprehensive YAML blueprint. be thorough."
    )

    # check if coordinator mode is enabled
    yaml_config = _load_mimick_config()
    coord_config = yaml_config.get("coordinator", {})
    use_coordinator = coord_config.get("enabled", False)

    if use_coordinator:
        try:
            from dokumen.coordinator.coordinator import CoordinatorAgent
            from dokumen.coordinator.types import WorkerTask

            max_workers = coord_config.get("max_workers", 5)
            strategy = coord_config.get("synthesis_strategy", "merge")
            worker_timeout = coord_config.get("worker_timeout", 300.0)
            decompose_timeout = coord_config.get("decompose_timeout", 60.0)
            decompose_model = coord_config.get("decompose_model", "") or None
            executor_mode = coord_config.get("executor_mode", "api")
            worker_model = coord_config.get("worker_model", "") or None

            click.echo(f"coordinator mode: {max_workers} workers, {strategy} synthesis, executor={executor_mode}")

            coordinator = CoordinatorAgent(
                provider=provider,
                max_workers=max_workers,
                synthesis_strategy=strategy,
                default_timeout=worker_timeout,
                decompose_timeout=decompose_timeout,
                decompose_model=decompose_model,
                executor_mode=executor_mode,
                base_dir=source_path,
                worker_model=worker_model,
            )

            result = await coordinator.run(
                goal=user_prompt,
            )

            # coordinator returns a dict with synthesis key
            if isinstance(result, dict):
                return result.get("synthesis", str(result))
            return str(result)

        except ImportError:
            click.echo("coordinator not available, falling back to single agent")
        except Exception as e:
            click.echo(f"coordinator failed ({e}), falling back to single agent")

    # single agent mode (default) — direct API, no bundled CLI
    from dokumen.coordinator.api_executor import run_api_executor

    result = await run_api_executor(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_names=tool_names,
        max_turns=mimick_max_turns,
        timeout=mimick_timeout,
        base_dir=source_path,
    )

    if not result["success"]:
        raise click.ClickException(f"mimick failed: {result.get('error') or 'unknown error'}")

    return result.get("output", "")


BUILD_SYSTEM_PROMPT = """you are a code generation agent. you receive an architecture blueprint
(YAML) produced by the mimick analysis phase. your job is to generate a complete, runnable implementation.

## what to generate

1. **implementation files**: for each module in the blueprint, produce python files with classes/functions
   matching the described interfaces and responsibilities. use protocol-based abstractions where the
   blueprint indicates swappable backends.

2. **test files**: for each module, produce `tests/test_{module}.py` with comprehensive coverage:
   - happy path tests for every public function/method
   - error path tests (bad inputs, missing deps, timeouts)
   - edge cases and boundary values
   - guard clause tests — if a line prevents a bug, test it
   - use simple assert-based tests (no pytest fixtures needed)
   - each test file should be runnable standalone: `python tests/test_{module}.py`

3. **Dockerfile**: generate a `Dockerfile` that:
   - uses an appropriate python base image (e.g. python:3.11-slim)
   - installs all dependencies from requirements.txt
   - copies the source code
   - sets up the working directory
   - has a default CMD that runs all tests
   - includes a healthcheck or entrypoint for running the app

4. **requirements.txt**: list all python dependencies with pinned versions

5. **docker-compose.yml** (if the project needs services like redis, postgres, etc.)

6. **run_tests.sh**: a simple script that runs all test files and reports results:
   ```bash
   #!/bin/bash
   set -e
   failed=0
   for f in tests/test_*.py; do
     echo "running $f..."
     python "$f" || failed=1
   done
   exit $failed
   ```

## rules

- CRITICAL: you MUST use the write_file tool to create every file. do NOT output code as text — call write_file for each file. if you don't call write_file, the file won't exist.
- every file must be written with write_file(file_path="{output_dir}/path/to/file.py", content="...")
- make tests thorough — aim for 90%+ coverage of the public API
- use lowercase comments
- don't over-engineer but don't leave stubs — every function should have a real implementation
- the docker container should be self-contained: `docker build -t project . && docker run project` runs all tests"""


async def _run_build(
    blueprint: str,
    name: str,
    output_dir: str,
    timeout: float,
    build_max_turns: int = 80,
) -> str:
    """second pass: take a blueprint and generate implementation + tests."""
    from dokumen.test_builder import get_configured_provider

    provider = get_configured_provider()
    if provider is None:
        raise click.ClickException("no provider configured")

    tool_names = ["read_file", "list_directory", "glob", "search_file_content", "write_file"]

    user_prompt = (
        f"project name: {name}\n"
        f"output directory: {output_dir}\n\n"
        f"## architecture blueprint\n\n{blueprint}\n\n"
        "generate the implementation files and test files based on this blueprint.\n\n"
        "IMPORTANT: first, use read_file and glob to find any existing tests in the source "
        "codebase (look for test_*.py, *_test.py, tests/ directories). copy and adapt those "
        "tests for the new implementation — they capture the original author's intent and edge "
        "cases. then add additional tests to fill coverage gaps.\n\n"
        "IMPORTANT: you MUST call write_file for every file you generate. "
        "do not output code as text — use write_file(file_path, content) for each file. "
        "files that aren't written with write_file will not exist on disk."
    )

    # check coordinator mode
    yaml_config = _load_mimick_config()
    coord_config = yaml_config.get("coordinator", {})
    use_coordinator = coord_config.get("enabled", False)

    if use_coordinator:
        try:
            from dokumen.coordinator.coordinator import CoordinatorAgent

            max_workers = coord_config.get("max_workers", 5)
            strategy = coord_config.get("synthesis_strategy", "merge")
            worker_timeout = coord_config.get("worker_timeout", 300.0)
            decompose_timeout = coord_config.get("decompose_timeout", 60.0)
            decompose_model = coord_config.get("decompose_model", "") or None
            executor_mode = coord_config.get("executor_mode", "api")
            worker_model = coord_config.get("worker_model", "") or None

            click.echo(f"coordinator mode: {max_workers} workers, {strategy} synthesis, executor={executor_mode}")

            coordinator = CoordinatorAgent(
                provider=provider,
                max_workers=max_workers,
                synthesis_strategy=strategy,
                default_timeout=worker_timeout,
                decompose_timeout=decompose_timeout,
                decompose_model=decompose_model,
                executor_mode=executor_mode,
                base_dir=output_dir,
                worker_model=worker_model,
            )

            result = await coordinator.run(goal=user_prompt)

            if isinstance(result, dict):
                return result.get("synthesis", str(result))
            return str(result)

        except ImportError:
            click.echo("coordinator not available, falling back to single agent")
        except Exception as e:
            click.echo(f"coordinator failed ({e}), falling back to single agent")

    # single agent fallback — direct API
    from dokumen.coordinator.api_executor import run_api_executor

    result = await run_api_executor(
        provider=provider,
        system_prompt=BUILD_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_names=tool_names,
        max_turns=build_max_turns,
        timeout=timeout,
        base_dir=output_dir,
    )

    if not result["success"]:
        raise click.ClickException(f"build failed: {result.get('error') or 'unknown error'}")

    return result.get("output", "")


@click.command("mimick")
@click.argument("source_path", type=click.Path(exists=True))
@click.option("--name", "-n", default=None, help="project name (defaults to directory name)")
@click.option("--output", "-o", default=None, type=click.Path(), help="output file path (default: stdout)")
@click.option("--timeout", "-t", default=3600.0, type=float, help="timeout in seconds (default: 3600)")
@click.option("--build", "-b", default=None, type=click.Path(), help="generate implementation + tests into this directory")
@click.option("--blueprint", "-bp", default=None, type=click.Path(exists=True), help="skip analysis, use existing blueprint file")
@click.pass_context
def mimick(ctx, source_path: str, name: Optional[str], output: Optional[str], timeout: float, build: Optional[str], blueprint: Optional[str]):
    """analyze a codebase and generate an architecture blueprint.

    uses an AI agent to explore the codebase, identify modules,
    detect architectural patterns, and produce a blueprint.

    with --build, a second agent pass generates implementation
    files and tests based on the blueprint.

    with --blueprint, skip analysis and use an existing blueprint file.

    examples:

        dokumen mimick ./my-project

        dokumen mimick /path/to/repo --name my-app --output blueprint.yaml

        dokumen mimick . --build ./new-project

        dokumen mimick . --blueprint tree-diffusion-blueprint.yaml --build ./output

        dokumen mimick . --timeout 600
    """
    source = os.path.abspath(source_path)
    project_name = name or os.path.basename(source)

    logger.info("mimick starting", extra={"source": source, "project_name": project_name})

    if blueprint:
        # skip analysis, load existing blueprint
        with open(blueprint) as f:
            blueprint_text = f.read()
        click.echo(f"using existing blueprint: {blueprint}")
    else:
        click.echo(f"analyzing {source}...")

        config = ctx.obj.get("config_path") if ctx.obj else None

        try:
            blueprint_text = run_async(_run_mimick(source, project_name, output, timeout, config))
        except Exception as e:
            click.echo(f"error: {e}", err=True)
            sys.exit(1)

        # auto-save blueprint
        blueprint_path = output or f"{project_name}-blueprint.yaml"
        with open(blueprint_path, "w") as f:
            f.write(blueprint_text)
        click.echo(f"blueprint saved to {blueprint_path}")

        if not output:
            click.echo(blueprint_text)

    # build phase
    if build:
        if not blueprint_text or blueprint_text.strip() in ("", "all workers failed — no results to synthesize"):
            click.echo("analysis produced no usable blueprint — skipping build", err=True)
            sys.exit(1)

        build_dir = os.path.abspath(build)
        os.makedirs(build_dir, exist_ok=True)
        click.echo(f"\nbuilding implementation into {build_dir}...")

        try:
            yaml_config = _load_mimick_config()
            mimick_cfg = yaml_config.get("mimick", {})
            build_result = run_async(_run_build(
                blueprint_text, project_name, build_dir, timeout,
                build_max_turns=mimick_cfg.get("build_max_turns", 80),
            ))
            click.echo(build_result)
            click.echo(f"\nbuild complete — files written to {build_dir}")
        except Exception as e:
            click.echo(f"build error: {e}", err=True)
            sys.exit(1)

    logger.info("mimick complete", extra={"project_name": project_name})
