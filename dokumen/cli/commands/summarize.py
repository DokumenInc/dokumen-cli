"""
Summarize command for dokumen CLI.

Generates DOKUMEN_SUMMARIES_INDEX.md with AI-powered summaries of documentation files.
The explore agent reads this index first to quickly identify relevant files.
"""

import base64
import logging
import time
from pathlib import Path
from typing import Optional

import click

from ..helpers import load_config, discover_doc_files, run_async
from dokumen.config import DEFAULT_FAST_MODEL
from dokumen.loader import get_configured_provider
from dokumen.summary_index import (
    IMAGE_TYPES,
    INDEX_FILENAME,
    SummaryIndex,
    compute_content_hash,
    compute_staleness,
    generate_summary_index,
    is_image_file,
    is_pdf_file,
    parse_summary_index,
    render_summary_index,
)

logger = logging.getLogger(__name__)


async def _run_summarize(
    config: dict,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run the summarize operation.

    Args:
        config: Loaded dokumen.yaml config.
        force: If True, regenerate all summaries (ignore existing index).
        dry_run: If True, report what would change without writing.

    Returns:
        Dict with stats: files_processed, files_skipped, files_removed.
    """
    start_time = time.time()

    logger.info(f"[SUMMARIZE] Starting: force={force}, dry_run={dry_run}")

    # Discover doc files
    doc_file_paths = discover_doc_files(config)
    logger.info(f"[SUMMARIZE] Discovered {len(doc_file_paths)} doc files")

    if not doc_file_paths:
        logger.info("[SUMMARIZE] No doc files found")
        return {
            "files_processed": 0,
            "files_skipped": 0,
            "files_removed": 0,
            "total_files": 0,
            "duration": time.time() - start_time,
        }

    # Read file contents, separating text, image, and PDF files
    doc_files = {}
    image_files = {}
    pdf_files = {}
    for path in doc_file_paths:
        if is_pdf_file(path):
            try:
                raw_bytes = Path(path).read_bytes()
                b64 = base64.b64encode(raw_bytes).decode("ascii")
                pdf_files[path] = b64
            except Exception as e:
                logger.warning(f"[SUMMARIZE] Failed to read PDF {path}: {e}")
        elif is_image_file(path):
            try:
                raw_bytes = Path(path).read_bytes()
                b64 = base64.b64encode(raw_bytes).decode("ascii")
                ext = Path(path).suffix.lower()
                media_type = IMAGE_TYPES.get(ext, "application/octet-stream")
                image_files[path] = (b64, media_type)
            except Exception as e:
                logger.warning(f"[SUMMARIZE] Failed to read image {path}: {e}")
        else:
            try:
                content = Path(path).read_text(encoding="utf-8")
                doc_files[path] = content
            except Exception as e:
                logger.warning(f"[SUMMARIZE] Failed to read {path}: {e}")

    logger.info(f"[SUMMARIZE] Read {len(doc_files)} text files, {len(image_files)} image files, {len(pdf_files)} PDF files")

    # Load existing index (unless --force)
    existing_index = None
    index_path = Path(INDEX_FILENAME)
    if not force and index_path.exists():
        try:
            existing_content = index_path.read_text(encoding="utf-8")
            existing_index = parse_summary_index(existing_content)
            logger.info(f"[SUMMARIZE] Loaded existing index with {len(existing_index.entries)} entries")
        except Exception as e:
            logger.warning(f"[SUMMARIZE] Failed to parse existing index: {e}")

    # Compute what would change (for dry-run reporting)
    all_file_count = len(doc_files) + len(image_files) + len(pdf_files)
    if existing_index and not force:
        current_hashes = {p: compute_content_hash(c) for p, c in doc_files.items()}
        for p, (b64, _mt) in image_files.items():
            current_hashes[p] = compute_content_hash(b64)
        for p, b64 in pdf_files.items():
            current_hashes[p] = compute_content_hash(b64)
        new_files, changed_files, removed_files = compute_staleness(
            existing_index, current_hashes
        )
        files_to_process = len(new_files) + len(changed_files)
        files_to_skip = all_file_count - files_to_process
    else:
        new_files = list(doc_files.keys()) + list(image_files.keys()) + list(pdf_files.keys())
        changed_files = []
        removed_files = list(existing_index.entries.keys()) if existing_index else []
        files_to_process = all_file_count
        files_to_skip = 0

    if dry_run:
        duration = time.time() - start_time
        logger.info(f"[SUMMARIZE] Dry run: would_process={files_to_process}, would_skip={files_to_skip}, would_remove={len(removed_files)}")
        return {
            "files_processed": files_to_process,
            "files_skipped": files_to_skip,
            "files_removed": len(removed_files),
            "total_files": all_file_count,
            "new_files": new_files,
            "changed_files": changed_files,
            "removed_files": removed_files,
            "duration": duration,
            "dry_run": True,
        }

    # Get provider (use Haiku for summaries)
    explore_config = config.get("explore", {})
    model = explore_config.get("model", DEFAULT_FAST_MODEL)
    provider = get_configured_provider(model_override=model)

    if provider is None:
        logger.error("[SUMMARIZE] No LLM provider configured")
        raise click.ClickException(
            "No LLM provider configured. Set ANTHROPIC_API_KEY or configure provider in dokumen.yaml."
        )

    # Generate summaries
    def on_progress(event, data):
        if event == "generating":
            file_path = data.get("file_path", "")
            index = data.get("index", 0)
            total = data.get("total", 0)
            click.echo(f"  [{index}/{total}] {file_path}")

    result_index = await generate_summary_index(
        provider=provider,
        doc_files=doc_files,
        existing_index=existing_index if not force else None,
        on_progress=on_progress,
        image_files=image_files if image_files else None,
        pdf_files=pdf_files if pdf_files else None,
    )

    # Write index file
    rendered = render_summary_index(result_index)
    index_path.write_text(rendered, encoding="utf-8")

    duration = time.time() - start_time
    logger.info(f"[SUMMARIZE] Complete: {len(result_index.entries)} entries in {int(duration * 1000)}ms")

    return {
        "files_processed": files_to_process,
        "files_skipped": files_to_skip,
        "files_removed": len(removed_files),
        "total_files": all_file_count,
        "entries_written": len(result_index.entries),
        "duration": duration,
    }


@click.command()
@click.option("--force", is_flag=True, help="Regenerate all summaries (ignore cache)")
@click.option("--dry-run", is_flag=True, help="Show what would change without writing")
@click.pass_context
def summarize(ctx, force, dry_run):
    """Generate documentation summaries index.

    Creates DOKUMEN_SUMMARIES_INDEX.md with AI-generated summaries of all
    documentation files. The explore agent reads this index to quickly
    identify relevant files.

    Examples:

        dokumen summarize

        dokumen summarize --force

        dokumen summarize --dry-run
    """
    logger.info(f"[SUMMARIZE_CMD] Command invoked: force={force}, dry_run={dry_run}")

    # Load config
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    config = load_config(config_path)

    try:
        stats = run_async(_run_summarize(config, force=force, dry_run=dry_run))
    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"[SUMMARIZE_CMD] Failed: {e}", exc_info=True)
        raise click.ClickException(str(e))

    # Display results
    if dry_run:
        click.echo(click.style("Dry run — would not write any files", fg="yellow", bold=True))
        click.echo(f"  Files to process: {stats['files_processed']}")
        click.echo(f"  Files to skip:    {stats['files_skipped']}")
        click.echo(f"  Files to remove:  {stats['files_removed']}")
        if stats.get("new_files"):
            click.echo(click.style("\n  New files:", bold=True))
            for f in stats["new_files"]:
                click.echo(f"    + {f}")
        if stats.get("changed_files"):
            click.echo(click.style("\n  Changed files:", bold=True))
            for f in stats["changed_files"]:
                click.echo(f"    ~ {f}")
        if stats.get("removed_files"):
            click.echo(click.style("\n  Removed files:", bold=True))
            for f in stats["removed_files"]:
                click.echo(f"    - {f}")
    elif stats["total_files"] == 0:
        click.echo(click.style("No documentation files found.", fg="yellow"))
    else:
        click.echo(click.style("Summary index generated", fg="green", bold=True))
        click.echo(f"  Processed:  {stats['files_processed']} files")
        click.echo(f"  Skipped:    {stats['files_skipped']} (unchanged)")
        click.echo(f"  Removed:    {stats['files_removed']}")
        click.echo(f"  Total:      {stats.get('entries_written', 0)} entries in {INDEX_FILENAME}")
        click.echo(f"  Duration:   {stats['duration']:.1f}s")
