"""Caching: hash-based incremental processing for papers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lib.models import Paper


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = _REPO_ROOT / ".paper2md"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "cache.json"


def compute_pdf_hash(pdf_path: Path) -> str:
    """Compute SHA-256 hash of PDF file content."""
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class PaperCache:
    """
    Cache for extracted PDF text keyed by PDF content hash.
    
    Schema:
    {
        "version": 2,
        "papers": {
            "filename.pdf": {
                "hash": "sha256...",
                "title": "Paper Title",
                "text": "Full extracted text..."
            }
        }
    }
    
    Note: We cache text (not summaries) because:
    - Text extraction is deterministic
    - Summaries depend on LLM/prompts (would be stale when switching modes)
    - Text extraction requires pdfminer which may not always be available
    """

    def __init__(self, cache_path: Path | str = DEFAULT_CACHE_FILE):
        self.cache_path = Path(cache_path)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load cache from disk or initialize empty."""
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == 2:
                    return data
                # Invalidate old cache versions (v1 stored summaries, v2 stores text)
            except (json.JSONDecodeError, OSError):
                pass
        return {"version": 2, "papers": {}}

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

        return Paper(
            pdf_path=pdf_path,
            title=entry.get("title", pdf_path.stem),
            text=text,
            summary_md=None  # Always regenerate summaries
        )

    def store(self, paper: Paper, pdf_hash: str | None = None) -> None:
        """
        Store extracted paper text in cache.
        
        Args:
            paper: Paper with text extracted
            pdf_hash: Pre-computed hash (to avoid re-hashing)
        """
        if pdf_hash is None:
            pdf_hash = compute_pdf_hash(paper.pdf_path)

        self._data["papers"][paper.pdf_path.name] = {
            "hash": pdf_hash,
            "title": paper.title,
            "text": paper.text
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._data["papers"] = {}
