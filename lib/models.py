"""Data models for paper2md."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paper:
    """Represents a research paper with extracted metadata and content."""
    pdf_path: Path
    title: str
    text: str
    summary_md: str | None = None
