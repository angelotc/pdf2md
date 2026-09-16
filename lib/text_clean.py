"""Pure text cleaning and normalization functions."""

from __future__ import annotations

import re
import unicodedata


# Boilerplate patterns to strip from PDFs
_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*permission to make digital or hard copies\b", re.I),
    re.compile(r"^\s*request permissions from\b", re.I),
    re.compile(r"^\s*copyrights for components of this work\b", re.I),
    re.compile(r"^\s*abstracting with credit is permitted\b", re.I),
    re.compile(r"^\s*to copy otherwise\b", re.I),
    re.compile(r"^\s*republish, to post on servers\b", re.I),
    re.compile(r"^\s*©\s*\d{4}\b", re.I),
    re.compile(r"^\s*acm isbn\b", re.I),
    re.compile(r"^\s*this work is licensed under\b", re.I),
    re.compile(r"^\s*please use nonacm option\b", re.I),
    re.compile(r"^\s*recsys\s*['']?\s*\d{2}\b", re.I),
    re.compile(r"^\s*conference acronym\b", re.I),
)

_WHITESPACE_RE = re.compile(r"\s+")
_BOM_PREFIX_RE = re.compile(r"^\s*(?:\ufeff|þÿ|ÿþ|\ufffe|\ufeff)\s*")


def _clean_text(s: str) -> str:
    """Normalize whitespace without destroying paragraphs too aggressively."""
    s = s.replace("\x00", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _dehyphenate_wrapped_words(s: str) -> str:
    """Fix common PDF line-wrapping artifacts like: personal-\\npersonalization."""
    return re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", s)


def _strip_boilerplate_lines(s: str) -> str:
    """Remove common academic paper boilerplate (copyright notices, etc.)."""
    lines = s.splitlines()
    kept: list[str] = []
    for ln in lines:
        raw = ln.strip()
        if not raw:
            kept.append("")
            continue
        if any(p.search(raw) for p in _BOILERPLATE_PATTERNS):
            continue
        # Drop short author-footnote-only lines
        if re.match(r"^[\*\u2217\u2020\u2021]+\s*\w+", raw) and len(raw) < 90:
            continue
        kept.append(ln)
    return _clean_text("\n".join(kept))


def clean_pdf_text(raw_text: str) -> str:
    """
    Clean raw PDF text: dehyphenate, strip boilerplate, normalize whitespace.

    Pure function: deterministic, no side effects.
    """
    txt = _dehyphenate_wrapped_words(raw_text)
    txt = _strip_boilerplate_lines(txt)
    # Fold real ligature codepoints (ﬁ ﬂ ﬀ ...) into plain letters. Blind
    # string replacement of "f i"/"f l"/"f f" would also corrupt legitimate
    # text like "of features" or "of if", so NFKC only.
    txt = unicodedata.normalize("NFKC", txt)
    return txt


def normalize_for_sentences(text: str) -> str:
    """
    Merge hard-wrapped lines inside paragraphs, preserving paragraph breaks.

    Useful for extracting sentence-level content from PDFs.
    """
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    merged: list[str] = []
    for p in paras:
        p2 = re.sub(r"(?<!\n)\n(?!\n)", " ", p)
        p2 = re.sub(r"\s{2,}", " ", p2).strip()
        merged.append(p2)
    return "\n\n".join(merged)
