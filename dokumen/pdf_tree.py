"""Helpers for reading Dokumen PDF tree index JSON files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PdfTreeNode:
    """A section node from a PDF tree index."""

    node_id: str
    title: str
    summary: str = ""
    text: str = ""
    level: int = 0
    page_index: Optional[int] = None
    children: list["PdfTreeNode"] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PdfTreeNode":
        """Build a node from either current or legacy tree index shapes."""
        raw_children = data.get("children") or data.get("nodes") or []
        return cls(
            node_id=str(data.get("node_id") or data.get("id") or ""),
            title=str(data.get("title") or "Untitled section"),
            summary=str(data.get("summary") or ""),
            text=str(data.get("text") or ""),
            level=int(data.get("level") or 0),
            page_index=data.get("page_index"),
            children=[cls.from_dict(child) for child in raw_children if isinstance(child, dict)],
        )


@dataclass
class PdfDocumentTree:
    """A parsed PDF tree index."""

    file_path: str
    title: str
    description: str = ""
    nodes: list[PdfTreeNode] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PdfDocumentTree":
        """Build a document tree from `_tree_index.json` data."""
        nodes = [
            PdfTreeNode.from_dict(node) for node in data.get("nodes", []) if isinstance(node, dict)
        ]
        file_path = str(data.get("file_path") or "")
        return cls(
            file_path=file_path,
            title=str(data.get("title") or file_path or "PDF document"),
            description=str(data.get("description") or ""),
            nodes=nodes,
        )

    def get_outline(self, max_depth: int = 3) -> str:
        """Render the tree as a compact markdown outline."""
        lines: list[str] = [f"# {self.title}"]
        if self.description:
            lines.extend(["", self.description])

        for node in self.nodes:
            _append_outline_node(lines, node, depth=0, max_depth=max_depth)

        return "\n".join(lines)


def parse_pdf_tree(data: dict[str, Any]) -> PdfDocumentTree:
    """Parse tree index JSON data into local dataclasses."""
    return PdfDocumentTree.from_dict(data)


def _append_outline_node(
    lines: list[str],
    node: PdfTreeNode,
    depth: int,
    max_depth: int,
) -> None:
    """Append one node and its visible children to the outline."""
    if depth >= max_depth:
        return

    indent = "  " * depth
    page = f" (p.{node.page_index + 1})" if node.page_index is not None else ""
    summary = f" - {node.summary}" if node.summary else ""
    lines.append(f"{indent}- [{node.node_id}] {node.title}{page}{summary}")

    for child in node.children:
        _append_outline_node(lines, child, depth + 1, max_depth)
