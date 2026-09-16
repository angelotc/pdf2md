"""Pure analysis functions for extracting structured content from paper text."""

from __future__ import annotations

import re

from lib.text_clean import _clean_text, _strip_boilerplate_lines


def find_doi(text: str) -> str | None:
    """Extract DOI from text using regex."""
    m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, re.I)
    if not m:
        return None
    doi = m.group(0)
    # Normalize common trailing punctuation
    doi = doi.rstrip(").,;")
    return doi


def extract_abstract(text: str) -> str | None:
    """Extract abstract section from paper text."""
    # Try common patterns: "Abstract" heading then text until next heading
    m = re.search(
        r"(?is)\babstract\b\s*[:\n]\s*(.{200,4500}?)(?:\n\s*\b(?:1\s+(?:introduction|intro)|keywords|ccs concepts|index terms|background|related work)\b)",
        text,
    )
    if m:
        return _clean_text(_strip_boilerplate_lines(m.group(1)))

    # Fallback 1: look for Introduction as the end marker
    m2 = re.search(r"(?is)\babstract\b\s*[:\n]\s*(.{200,3000}?)(?:\n\s*1\.?\s+Introduction)", text)
    if m2:
        return _clean_text(_strip_boilerplate_lines(m2.group(1)))

    # Fallback 2: first ~1.5k chars after the word "Abstract"
    m3 = re.search(r"(?is)\babstract\b\s*[:\n]\s*(.{200,1500})", text)
    if m3:
        return _clean_text(_strip_boilerplate_lines(m3.group(1)))
    return None


def chunk_text_for_llm(text: str, max_chars: int = 12000) -> list[str]:
    """
    Split text into paragraph-aligned chunks for LLM processing.

    Preserves paragraph boundaries to avoid breaking mid-thought.
    A single paragraph longer than max_chars is hard-split at word
    boundaries so the chunk budget is always respected.
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    for p in text.split("\n\n"):
        p = p.strip()
        if not p:
            continue
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            pieces.append(p[:cut])
            p = p[cut:].lstrip()
        if p:
            pieces.append(p)

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for p in pieces:
        add_len = len(p) + 2
        if cur and (cur_len + add_len) > max_chars:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = add_len
        else:
            cur.append(p)
            cur_len += add_len

    if cur:
        chunks.append("\n\n".join(cur))
    return chunks
