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

# Filename sanitization patterns (from rename_papers_by_title.py)
_INVALID_WIN_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
_WHITESPACE_RE = re.compile(r"\s+")
_PRIVATE_USE_RE = re.compile(r"[\uE000-\uF8FF]")
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


def to_safe_filename(title: str, max_len: int = 160) -> str:
    """
    Convert paper title to Windows-safe filename.

    Handles:
    - Unicode normalization (NFKC)
    - Invalid Windows characters
    - BOM artifacts from PDF extraction
    - Private-use Unicode characters
    - Length limits
    - Spaced-out titles (e.g., "C o a r s e" -> "Coarse")
    """
    # Unicode normalize to clean up ligatures / presentation forms
    t = unicodedata.normalize("NFKC", title)

    # Strip common BOM-ish garbage
    t = _BOM_PREFIX_RE.sub("", t)

    # Known PDF extraction artifact: "\uE039" should be "ft"
    t = t.replace("\uE039", "ft")

    # Remove any remaining private-use glyphs
    t = _PRIVATE_USE_RE.sub("", t)

    # Normalize whitespace and strip
    t = _WHITESPACE_RE.sub(" ", t).strip()

    # Heuristic: some PDFs yield titles like "C o a r s e - t o - f i n e ..."
    # If there are many single-letter tokens, de-space runs of letters.
    tokens = t.split(" ")
    if tokens:
        def is_single_letter_token(tok: str) -> bool:
            letters = [ch for ch in tok if ch.isalpha()]
            return len(letters) == 1 and len(tok) <= 3

        single_letter = sum(1 for tok in tokens if is_single_letter_token(tok))
        if single_letter / max(len(tokens), 1) > 0.35:
            rebuilt: list[str] = []
            buf: list[str] = []
            for tok in tokens:
                letters = [ch for ch in tok if ch.isalpha()]
                if len(letters) == 1 and len(tok) <= 3:
                    buf.append(letters[0])
                else:
                    if buf:
                        rebuilt.append("".join(buf))
                        buf = []
                    rebuilt.append(tok)
            if buf:
                rebuilt.append("".join(buf))
            t = " ".join(rebuilt)
            t = _WHITESPACE_RE.sub(" ", t).strip()

    # Replace invalid Windows characters with spaces, then re-collapse
    t = _INVALID_WIN_CHARS_RE.sub(" ", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()

    # Avoid trailing dot/space (illegal on Windows)
    t = t.rstrip(" .")

    # Prevent empty filenames
    if not t:
        t = "untitled"

    # Conservative length limit (path length safety)
    if len(t) > max_len:
        t = t[:max_len].rstrip(" .")
        if not t:
            t = "untitled"

    return t + ".pdf"
