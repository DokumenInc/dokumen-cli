"""
Summary index module for the Documentation Unit Test Framework.

Generates, parses, and manages DOKUMEN_SUMMARIES_INDEX.md files that
contain pre-computed summaries of documentation files. The explore agent
reads this index first to quickly identify relevant files.

NOTE: The summary prompt and hash function are copied from
backend/api/summaries/service.py. Keep them in sync to prevent drift.
"""

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from .logging_config import get_logger

logger = get_logger(__name__)

INDEX_FILENAME = "DOKUMEN_SUMMARIES_INDEX.md"

# Supported image types for multimodal summarization
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Supported PDF type
PDF_EXTENSION = ".pdf"

# Summary prompt - kept in sync with backend/api/summaries/service.py:20-33
SUMMARY_SYSTEM_PROMPT = """You are a documentation analyst. Given a file path and its content, produce a structured summary listing every important topic on the page.

Output format:
1. First line: A single sentence stating the document's purpose, referencing the file path.
2. Followed by a blank line, then bullet points (one per line, prefixed with "- ").
3. Each bullet describes one key topic, fact, concept, or instruction from the document.
4. Order bullets by importance (most critical first).
5. Use 5-15 bullets depending on document length and density.

Rules:
- Be factual and objective — do not editorialize
- Each bullet must be a single concise line (no sub-bullets)
- Do not include code blocks, markdown headers, or rich formatting
- Respond with ONLY the structured summary, nothing else"""

# Image summary prompt - kept in sync with backend/api/summaries/service.py
IMAGE_SUMMARY_SYSTEM_PROMPT = """You are a documentation analyst. Given a file path and an image, produce a structured summary describing the visual content.

Output format:
1. First line: A single sentence stating what the image depicts, referencing the file path.
2. Followed by a blank line, then bullet points (one per line, prefixed with "- ").
3. Each bullet describes one key visual element, diagram component, data point, or piece of information visible in the image.
4. Order bullets by importance (most prominent first).
5. Use 3-10 bullets depending on image complexity.

Rules:
- Be factual and objective — describe what you see
- Each bullet must be a single concise line (no sub-bullets)
- If the image contains text, transcribe key text content
- If the image is a diagram, describe the structure and relationships
- If the image is a chart/graph, describe the data and trends
- Respond with ONLY the structured summary, nothing else"""

# PDF summary prompt - sends PDF as document for multimodal analysis
PDF_SUMMARY_SYSTEM_PROMPT = """You are a documentation analyst. Given a file path and a PDF document, produce a structured summary listing every important topic on each page.

Output format:
1. First line: A single sentence stating the document's purpose, referencing the file path.
2. Followed by a blank line, then bullet points (one per line, prefixed with "- ").
3. Each bullet describes one key topic, fact, concept, diagram, or visual element from the document.
4. Order bullets by importance (most critical first).
5. Use 5-20 bullets depending on document length and density.

Rules:
- Be factual and objective — do not editorialize
- Each bullet must be a single concise line (no sub-bullets)
- If the document contains diagrams, charts, or figures, describe their content and what they illustrate
- If the document contains tables, summarize the key data points
- Capture visual elements that would be lost in a text-only extraction
- Respond with ONLY the structured summary, nothing else"""


def is_pdf_file(path: str) -> bool:
    """Check if a file path refers to a PDF file.

    Args:
        path: File path to check.

    Returns:
        True if the file extension is .pdf (case-insensitive).
    """
    if not path:
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext == PDF_EXTENSION


def is_image_file(path: str) -> bool:
    """Check if a file path refers to a supported image type.

    Args:
        path: File path to check.

    Returns:
        True if the file extension matches a supported image type.
    """
    if not path:
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in IMAGE_TYPES


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for staleness detection.

    Reuses exact format from backend/api/summaries/service.py:36-46.

    Args:
        content: File content to hash.

    Returns:
        Hash string in format "sha256:<hex>".
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def is_binary_content(content: str) -> bool:
    """Check if content appears to be binary.

    Args:
        content: File content to check.

    Returns:
        True if content contains null bytes (binary indicator).
    """
    return "\x00" in content


@dataclass
class FileSummaryEntry:
    """A single file's summary in the index."""

    file_path: str
    content_hash: str
    summary_text: str


@dataclass
class SummaryIndex:
    """The full summary index containing all file summaries."""

    entries: Dict[str, FileSummaryEntry]
    generated_at: str
    version: str


def render_summary_index(index: SummaryIndex) -> str:
    """Render a SummaryIndex to markdown string.

    Args:
        index: The summary index to render.

    Returns:
        Markdown string suitable for writing to DOKUMEN_SUMMARIES_INDEX.md.
    """
    logger.info(
        "summary_index.render",
        entry_count=len(index.entries),
        generated_at=index.generated_at,
    )

    lines = [
        "<!-- DOKUMEN SUMMARIES INDEX -->",
        f"<!-- Generated by: dokumen summarize -->",
        f"<!-- Generated at: {index.generated_at} -->",
        f"<!-- File count: {len(index.entries)} -->",
        "",
    ]

    sorted_paths = sorted(index.entries.keys())
    for i, path in enumerate(sorted_paths):
        entry = index.entries[path]
        if i > 0:
            lines.append("---")
            lines.append("")

        lines.append(f"## {entry.file_path}")
        lines.append(f"<!-- hash: {entry.content_hash} -->")
        lines.append("")
        lines.append(entry.summary_text)
        lines.append("")

    return "\n".join(lines)


def parse_summary_index(content: str) -> SummaryIndex:
    """Parse a DOKUMEN_SUMMARIES_INDEX.md file into a SummaryIndex.

    Args:
        content: Raw markdown content of the index file.

    Returns:
        Parsed SummaryIndex. Returns empty index for empty/malformed content.
    """
    logger.info("summary_index.parse", content_length=len(content))

    if not content or not content.strip():
        return SummaryIndex(entries={}, generated_at="", version="1.0")

    # Extract generated_at from header
    generated_at = ""
    at_match = re.search(r"<!-- Generated at: (.+?) -->", content)
    if at_match:
        generated_at = at_match.group(1)

    # Split content into sections by ## headings
    entries: Dict[str, FileSummaryEntry] = {}

    # Split by sections (## heading or --- divider)
    sections = re.split(r"\n---\n", content)

    # Pattern: ## file_path followed by <!-- hash: ... --> and summary text
    section_pattern = re.compile(
        r"## (.+?)\n"            # ## file_path
        r"<!-- hash: (.+?) -->"  # <!-- hash: sha256:xxx -->
        r"\n\n"                  # blank line
        r"(.+)",                 # summary text (everything remaining)
        re.DOTALL,
    )

    for section in sections:
        match = section_pattern.search(section)
        if match:
            file_path = match.group(1).strip()
            content_hash = match.group(2).strip()
            summary_text = match.group(3).strip()

            entries[file_path] = FileSummaryEntry(
                file_path=file_path,
                content_hash=content_hash,
                summary_text=summary_text,
            )

    logger.info("summary_index.parsed", entry_count=len(entries))
    return SummaryIndex(entries=entries, generated_at=generated_at, version="1.0")


def compute_staleness(
    index: SummaryIndex, current_files: Dict[str, str]
) -> Tuple[List[str], List[str], List[str]]:
    """Compute which files are new, changed, or removed.

    Args:
        index: Existing summary index.
        current_files: Dict mapping file_path to content_hash for current files.

    Returns:
        Tuple of (new_files, changed_files, removed_files) as lists of paths.
    """
    logger.info(
        "summary_index.staleness",
        index_count=len(index.entries),
        current_count=len(current_files),
    )

    new_files: List[str] = []
    changed_files: List[str] = []
    removed_files: List[str] = []

    # Check current files against index
    for path, current_hash in current_files.items():
        if path not in index.entries:
            new_files.append(path)
        elif index.entries[path].content_hash != current_hash:
            changed_files.append(path)

    # Check for removed files
    for path in index.entries:
        if path not in current_files:
            removed_files.append(path)

    logger.info(
        "summary_index.staleness.result",
        new=len(new_files),
        changed=len(changed_files),
        removed=len(removed_files),
    )
    return new_files, changed_files, removed_files


async def generate_file_summary(
    provider,
    file_path: str,
    content: str,
) -> Optional[str]:
    """Generate a summary for a single file using the LLM provider.

    Args:
        provider: LLM provider instance.
        file_path: Path to the file being summarized.
        content: File content to summarize.

    Returns:
        Summary text string, or None if generation failed.
    """
    logger.info(
        "summary_index.generate_file",
        file_path=file_path,
        content_length=len(content),
    )

    try:
        user_message = f"File: {file_path}\n\n{content}"
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        response = await provider.complete(messages)

        summary_text = response.get("content", "")
        if not summary_text:
            logger.warning(
                "summary_index.generate_file.empty_response",
                file_path=file_path,
            )
            return None

        logger.info(
            "summary_index.generate_file.complete",
            file_path=file_path,
            summary_length=len(summary_text),
        )
        return summary_text

    except Exception as e:
        logger.error(
            "summary_index.generate_file.error",
            file_path=file_path,
            error=str(e),
        )
        return None


async def generate_image_summary(
    provider,
    file_path: str,
    base64_data: str,
    media_type: str,
) -> Optional[str]:
    """Generate a summary for an image file using multimodal LLM.

    Args:
        provider: LLM provider instance.
        file_path: Path to the image file.
        base64_data: Base64-encoded image data.
        media_type: MIME type of the image (e.g. "image/png").

    Returns:
        Summary text string, or None if generation failed.
    """
    logger.info(
        "summary_index.generate_image",
        file_path=file_path,
        media_type=media_type,
        base64_length=len(base64_data),
    )

    try:
        messages = [
            {"role": "system", "content": IMAGE_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"File: {file_path}"},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                }},
            ]},
        ]

        response = await provider.complete(messages)

        summary_text = response.get("content", "")
        if not summary_text:
            logger.warning(
                "summary_index.generate_image.empty_response",
                file_path=file_path,
            )
            return None

        logger.info(
            "summary_index.generate_image.complete",
            file_path=file_path,
            summary_length=len(summary_text),
        )
        return summary_text

    except Exception as e:
        logger.error(
            "summary_index.generate_image.error",
            file_path=file_path,
            error=str(e),
        )
        return None


async def generate_pdf_summary(
    provider,
    file_path: str,
    base64_data: str,
) -> Optional[str]:
    """Generate a summary for a PDF file using multimodal document block.

    Sends the PDF as a document content block so the LLM can analyze
    visual content (diagrams, charts, tables) in addition to text.

    Args:
        provider: LLM provider instance.
        file_path: Path to the PDF file.
        base64_data: Base64-encoded PDF data.

    Returns:
        Summary text string, or None if generation failed.
    """
    logger.info(
        "summary_index.generate_pdf",
        file_path=file_path,
        base64_length=len(base64_data),
    )

    try:
        messages = [
            {"role": "system", "content": PDF_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"File: {file_path}"},
                {"type": "document", "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64_data,
                }},
            ]},
        ]

        response = await provider.complete(messages)

        summary_text = response.get("content", "")
        if not summary_text:
            logger.warning(
                "summary_index.generate_pdf.empty_response",
                file_path=file_path,
            )
            return None

        logger.info(
            "summary_index.generate_pdf.complete",
            file_path=file_path,
            summary_length=len(summary_text),
        )
        return summary_text

    except Exception as e:
        logger.error(
            "summary_index.generate_pdf.error",
            file_path=file_path,
            error=str(e),
        )
        return None


async def generate_summary_index(
    provider,
    doc_files: Dict[str, str],
    existing_index: Optional[SummaryIndex] = None,
    on_progress: Optional[Callable] = None,
    image_files: Optional[Dict[str, Tuple[str, str]]] = None,
    pdf_files: Optional[Dict[str, str]] = None,
) -> SummaryIndex:
    """Generate a full summary index, with incremental updates.

    Args:
        provider: LLM provider instance.
        doc_files: Dict mapping file_path to text file content.
        existing_index: Optional existing index for incremental updates.
        on_progress: Optional callback(event, data) for progress reporting.
        image_files: Optional dict mapping file_path to (base64_data, media_type).
        pdf_files: Optional dict mapping file_path to base64_data.

    Returns:
        New SummaryIndex with all summaries.
    """
    if image_files is None:
        image_files = {}
    if pdf_files is None:
        pdf_files = {}

    total_file_count = len(doc_files) + len(image_files) + len(pdf_files)

    logger.info(
        "summary_index.generate",
        file_count=total_file_count,
        text_files=len(doc_files),
        image_files=len(image_files),
        pdf_files=len(pdf_files),
        has_existing=existing_index is not None,
    )

    # Compute hashes for all current files
    current_hashes: Dict[str, str] = {}
    for path, content in doc_files.items():
        current_hashes[path] = compute_content_hash(content)
    for path, (base64_data, _media_type) in image_files.items():
        current_hashes[path] = compute_content_hash(base64_data)
    for path, base64_data in pdf_files.items():
        current_hashes[path] = compute_content_hash(base64_data)

    # Determine what needs (re)generation
    if existing_index:
        new_files, changed_files, removed_files = compute_staleness(
            existing_index, current_hashes
        )
        files_to_generate = new_files + changed_files
    else:
        files_to_generate = list(current_hashes.keys())
        removed_files = []

    logger.info(
        "summary_index.generate.plan",
        to_generate=len(files_to_generate),
        to_skip=total_file_count - len(files_to_generate),
        to_remove=len(removed_files) if existing_index else 0,
    )

    # Start with existing entries (minus removed files)
    entries: Dict[str, FileSummaryEntry] = {}
    if existing_index:
        for path, entry in existing_index.entries.items():
            if path not in removed_files and path in current_hashes:
                # Keep unchanged entries
                if path not in files_to_generate:
                    entries[path] = entry

    # Generate summaries for new/changed files
    for i, path in enumerate(sorted(files_to_generate)):
        if on_progress:
            on_progress("generating", {
                "file_path": path,
                "index": i + 1,
                "total": len(files_to_generate),
            })

        # Route to PDF, image, or text summarization
        if path in pdf_files:
            pdf_b64 = pdf_files[path]
            summary = await generate_pdf_summary(provider, path, pdf_b64)
        elif path in image_files:
            base64_data, media_type = image_files[path]
            summary = await generate_image_summary(
                provider, path, base64_data, media_type
            )
        elif path in doc_files:
            content = doc_files[path]
            # Skip binary files
            if is_binary_content(content):
                logger.info(
                    "summary_index.generate.skip_binary",
                    file_path=path,
                )
                continue
            summary = await generate_file_summary(provider, path, content)
        else:
            continue

        if summary:
            entries[path] = FileSummaryEntry(
                file_path=path,
                content_hash=current_hashes[path],
                summary_text=summary,
            )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(
        "summary_index.generate.complete",
        total_entries=len(entries),
        generated=len(files_to_generate),
    )

    return SummaryIndex(
        entries=entries,
        generated_at=generated_at,
        version="1.0",
    )
