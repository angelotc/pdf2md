"""Markdown rendering: per-paper docs, index, and the derived combined doc.

Per-paper files (output/<stem>.md) are the canonical output. INDEX.md and the
combined PAPERS_SUMMARY.md are derived from those files on disk, so they can
be rebuilt at any time without re-running extraction or the LLM.
"""

from __future__ import annotations

import re
from pathlib import Path

from tqdm import tqdm

from lib.models import Paper, Figure
from lib.content_analysis import find_doi, annotate_text_with_figures
from lib.vision import NO_VISION_PLACEHOLDER

INDEX_FILENAME = "INDEX.md"
COMBINED_FILENAME = "PAPERS_SUMMARY.md"

_HEADING_RE = re.compile(r"^#{1,5}\s")
_H1_RE = re.compile(r"^#\s+(.+?)\s*$")


def github_slug(title: str) -> str:
    """Mimic GitHub's heading anchor slugger so index links resolve on github.com."""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    return re.sub(r"\s", "-", slug)


def _figure_link(paper: Paper, fig: Figure) -> str | None:
    """Relative markdown link target for a figure crop, if it exists."""
    if not fig.png_path or not fig.png_path.exists():
        return None
    return f"figures/{paper.pdf_path.stem}/{fig.png_path.name}"


def _paper_lines(paper: Paper, heading_level: int) -> list[str]:
    """Lines for one paper's section, shared by the combined and per-paper docs."""
    h = "#" * heading_level
    lines: list[str] = [f"{h} {paper.title}", ""]
    lines.append(f"- **Source PDF**: `{paper.pdf_path.as_posix()}`")

    doi = find_doi(paper.text)
    if doi:
        lines.append(f"- **DOI**: `https://doi.org/{doi}`")
    lines.append("")

    if paper.figures:
        lines.append(f"{'#' * (heading_level + 1)} Figures")
        lines.append("")
        for fig in paper.figures:
            rel = _figure_link(paper, fig)
            if rel:
                alt = (fig.caption or fig.label or "figure").replace("[", "(").replace("]", ")")
                lines.append(f"![{alt}]({rel})")
                lines.append("")
        lines.append("---")
        lines.append("")

    if paper.summary_md:
        lines.append(paper.summary_md.strip())
    lines.append("")

    return lines


def build_paper_markdown(paper: Paper) -> str:
    """Build a standalone markdown document for a single paper."""
    return "\n".join(_paper_lines(paper, heading_level=1)).rstrip() + "\n"


def _raw_figure_block(fig: Figure, img_rel: str | None) -> str:
    """Markdown block for one figure inside the raw doc: header, crop, description.

    The vision description is blockquoted to distinguish model output from
    the paper's own text.
    """
    label = (fig.label or "Figure").rstrip(": .")
    header = f"**{label}** (page {fig.page + 1})"
    if fig.caption and fig.caption != fig.label:
        header += f" — {fig.caption}"

    lines = [header]
    if img_rel:
        lines += ["", f"![{label}]({img_rel})"]
    if fig.description and fig.description != NO_VISION_PLACEHOLDER:
        lines += [""] + [f"> {ln}" if ln.strip() else ">" for ln in fig.description.splitlines()]
    return "\n".join(lines)


def build_paper_raw_markdown(paper: Paper) -> str:
    """Standalone raw doc: the paper's full extracted text with figures inline.

    Each figure's crop (linked) and vision description are inserted right
    after its caption in the text; figures whose captions aren't found are
    appended at the end. This is the exact material the summarizer saw,
    minus the plain-text annotation markup.
    """
    stem = paper.pdf_path.stem

    def block(fig: Figure) -> str:
        img_rel = None
        if fig.png_path and fig.png_path.exists():
            img_rel = f"../figures/{stem}/{fig.png_path.name}"
        return _raw_figure_block(fig, img_rel)

    if paper.figures:
        body = annotate_text_with_figures(paper.text or "", paper.figures, block_fn=block)
    else:
        body = paper.text or ""

    header = (
        f"# {paper.title}\n\n"
        f"> Raw extract of `{paper.pdf_path.as_posix()}` — full paper text with "
        "figure crops and their vision-model descriptions (quoted). Layout is "
        "not faithful to the original PDF.\n\n---\n\n"
    )
    return header + body.strip() + "\n"


def bump_headings(md: str, levels: int = 1) -> str:
    """Raise every ATX heading by `levels`; fenced code blocks are left alone."""
    had_trailing_newline = md.endswith("\n")
    out: list[str] = []
    in_fence = False
    for line in md.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and _HEADING_RE.match(line):
            line = "#" * levels + line
        out.append(line)
    bumped = "\n".join(out)
    return bumped + "\n" if had_trailing_newline else bumped


def extract_h1(md: str) -> str | None:
    """Text of the first H1 heading in a markdown document."""
    for line in md.splitlines():
        m = _H1_RE.match(line)
        if m:
            return m.group(1)
    return None


def first_tldr_line(md: str) -> str | None:
    """First TL;DR bullet of a per-paper doc, flattened to plain text."""
    lines = md.splitlines()
    for i, line in enumerate(lines):
        low = line.strip().lower().replace(";", "")
        if low.startswith("### tldr"):
            for candidate in lines[i + 1:]:
                s = candidate.strip()
                if not s:
                    continue
                if s.startswith(("* ", "- ")):
                    s = s[2:].replace("**", "").strip()
                    if len(s) > 140:
                        cut = s.rfind(" ", 0, 137)
                        s = (s[:cut] if cut > 0 else s[:137]) + "…"
                    return s
                break  # first non-empty line after the heading is not a bullet
            return None
    return None


def collect_paper_docs(
    out_dir: Path,
    exclude: frozenset[str] = frozenset({INDEX_FILENAME, COMBINED_FILENAME}),
) -> list[tuple[Path, str]]:
    """Read every per-paper markdown in out_dir, sorted by filename.

    The output dir is tool-managed: any other .md file there is treated as a
    per-paper doc. Unreadable files are skipped with a warning.
    """
    docs: list[tuple[Path, str]] = []
    for path in sorted(out_dir.glob("*.md")):
        if path.name in exclude:
            continue
        try:
            docs.append((path, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError) as e:
            tqdm.write(f"[WARN] Could not read {path}: {e}")
    return docs


def build_index_markdown(docs: list[tuple[Path, str]]) -> str:
    """Build INDEX.md: one line per paper linking to its per-paper file."""
    lines = ["# Papers Index", ""]
    for path, content in docs:
        title = extract_h1(content) or path.stem.replace("_", " ")
        entry = f"- [{title}]({path.name})"
        if (path.parent / "raw" / path.name).exists():
            entry += f" ([raw](raw/{path.name}))"
        tldr = first_tldr_line(content)
        if tldr:
            entry += f" — {tldr}"
        lines.append(entry)
    return "\n".join(lines).rstrip() + "\n"


def build_combined_markdown(docs: list[tuple[Path, str]]) -> str:
    """Build the combined PAPERS_SUMMARY.md from per-paper docs.

    Bodies are the per-paper files with headings bumped one level, under an
    index of anchor links. Pure derivation: no LLM, no extraction.
    """
    lines = ["# Papers Summary", "", "## Index", ""]
    for path, content in docs:
        title = extract_h1(content) or path.stem.replace("_", " ")
        lines.append(f"- [{title}](#{github_slug(title)})")
    lines += ["", "---", ""]
    bodies = [bump_headings(content).rstrip() for _, content in docs]
    return "\n".join(lines) + "\n\n" + "\n\n".join(bodies) + "\n"


def write_if_changed(path: Path, content: str) -> bool:
    """Write only when content differs from what's on disk.

    Returns True if the file was written. Keeps git diffs and mtimes clean
    when derived or unchanged outputs would otherwise be rewritten verbatim.
    """
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == content:
                return False
        except (OSError, UnicodeDecodeError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True
