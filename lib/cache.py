"""Caching: hash-based incremental processing for papers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from lib.models import Paper, Figure


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = _REPO_ROOT / ".paper2md"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "cache.json"

SCHEMA_VERSION = 3


def compute_pdf_hash(pdf_path: Path) -> str:
    """Compute SHA-256 hash of PDF file content."""
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class PaperCache:
    """
    Cache for extracted PDF text + figure metadata keyed by PDF content hash.

    Schema (v3):
    {
        "version": 3,
        "papers": {
            "filename.pdf": {
                "hash": "sha256...",
                "title": "Paper Title",
                "text": "Full extracted text...",
                "figures": [
                    {"page": 0, "rect": [x0,y0,x1,y1], "label": "Figure 1:",
                     "caption": "...", "png": "path/to/crop.png"}
                ]
            }
        }
    }

    Note: We cache extraction output (text + figure crops/metadata), not
    summaries or vision descriptions, because those depend on LLM/prompts
    and would go stale when either changes.
    """

    def __init__(self, cache_path: Path | str = DEFAULT_CACHE_FILE):
        self.cache_path = Path(cache_path)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load cache from disk or initialize empty."""
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == SCHEMA_VERSION:
                    return data
                # Invalidate older cache versions
            except (json.JSONDecodeError, OSError):
                pass
        return {"version": SCHEMA_VERSION, "papers": {}}

    def save(self) -> None:
        """Persist cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def get_cached(self, pdf_path: Path) -> Paper | None:
        """
        Return cached Paper if PDF hasn't changed, else None.

        Checks if:
        1. Entry exists for this filename
        2. Stored hash matches current file hash
        3. Text exists
        4. Every referenced figure crop still exists on disk
        """
        entry = self._data["papers"].get(pdf_path.name)
        if not entry:
            return None

        current_hash = compute_pdf_hash(pdf_path)
        if entry.get("hash") != current_hash:
            return None

        text = entry.get("text")
        if text is None:  # Allow empty string (some PDFs have no extractable text)
            return None

        figures: list[Figure] = []
        for f in entry.get("figures", []):
            png = f.get("png")
            png_path = Path(png) if png else None
            if png_path is not None and not png_path.exists():
                return None  # Crops vanished (e.g. cleaned dir): re-extract
            figures.append(Figure(
                page=f.get("page", 0),
                rect=tuple(f.get("rect", (0, 0, 0, 0))),
                label=f.get("label"),
                caption=f.get("caption"),
                png_path=png_path,
            ))

        return Paper(
            pdf_path=pdf_path,
            title=entry.get("title", pdf_path.stem),
            text=text,
            figures=tuple(figures),
            summary_md=None  # Always regenerate summaries
        )

    def store(self, paper: Paper, pdf_hash: str | None = None) -> None:
        """
        Store extracted paper text + figure metadata in cache.

        Args:
            paper: Paper with text and figures extracted
            pdf_hash: Pre-computed hash (to avoid re-hashing)
        """
        if pdf_hash is None:
            pdf_hash = compute_pdf_hash(paper.pdf_path)

        self._data["papers"][paper.pdf_path.name] = {
            "hash": pdf_hash,
            "title": paper.title,
            "text": paper.text,
            "figures": [
                {
                    "page": fig.page,
                    "rect": list(fig.rect),
                    "label": fig.label,
                    "caption": fig.caption,
                    "png": str(fig.png_path) if fig.png_path else None,
                }
                for fig in paper.figures
            ],
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._data["papers"] = {}
