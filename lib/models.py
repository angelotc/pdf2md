"""Data models for paper2md."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Figure:
    """A detected figure region in a PDF.

    rect is (x0, y0, x1, y1) in PDF points, page is 0-based.
    label is e.g. "Figure 3"; caption may be None if no match found.
    """
    page: int
    rect: tuple[float, float, float, float]
    label: str | None = None
    caption: str | None = None
    png_path: Path | None = None
    description: str | None = None


@dataclass(frozen=True)
class Paper:
    """Represents a research paper with extracted metadata and content."""
    pdf_path: Path
    title: str
    text: str
    figures: tuple[Figure, ...] = ()
    summary_md: str | None = None
