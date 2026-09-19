#!/usr/bin/env python3
"""
Summarize PDFs in /papers into a single markdown file for repo context.

Behavior:
- Extracts text + title from each PDF via sophisticated metadata + text extraction.
- Generates structured summaries using OpenAI-compatible LLM (required).
- Caches extracted text in .paper2md/ to skip re-extraction of unchanged PDFs.

Usage (PowerShell):
  python summarize_papers.py [--papers-dir papers] [--out output/PAPERS_SUMMARY.md]
  python summarize_papers.py --no-cache  # Force re-summarize all papers

Required env vars:
  OPENAI_API_KEY          -> API key for LLM provider

Optional env vars:
  OPENAI_MODEL            -> default: google/gemini-3.1-flash-lite
  OPENAI_BASE_URL         -> API base URL (default: https://openrouter.ai/api/v1)
  OPENAI_VISION_MODEL     -> model for figure descriptions (default: OPENAI_MODEL)
"""

from __future__ import annotations

import argparse
import os
import re
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from lib.models import Paper, Figure
from lib.pdf_extract import extract_paper_from_pdf
from lib.figure_extract import extract_figures
from lib.vision import describe_figures
from lib.summarization import summarize_paper
from lib.content_analysis import find_doi, annotate_text_with_figures
from lib.cache import PaperCache, compute_pdf_hash

# Load environment variables from root .env if it exists
load_dotenv(Path(__file__).parent / ".env")


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v not in {"", "0", "false", "no", "off"}


def _format_exc(e: Exception) -> str:
    msg = str(e).strip()
    if msg:
        return f"{type(e).__name__}: {msg}"
    return type(e).__name__


def _report_error(stage: str, pdf: Path, e: Exception) -> None:
    tqdm.write(f"[ERROR] {stage} failed for {pdf.name}: {_format_exc(e)}")
    if _truthy_env("PAPER2MD_DEBUG_TRACE"):
        tqdm.write(traceback.format_exc())


def load_papers(
    papers_dir: Path,
    max_pages: int | None = None,
    cache: PaperCache | None = None,
    extract_figs: bool = True,
    max_figures: int = 12,
) -> tuple[list[Paper], int]:
    """
    Extract title + text (+ figures) from all PDFs in directory.
    Uses cache to avoid re-extracting unchanged PDFs.

    Returns: (list of Paper objects with text extracted, extraction failure count)
    """
    pdfs = sorted(papers_dir.glob("*.pdf"))
    papers: list[Paper] = []
    failures = 0
    cached_count = 0
    new_extractions = 0

    for pdf in tqdm(pdfs, desc="Extracting PDFs"):
        # Check cache first
        if cache:
            cached_paper = cache.get_cached(pdf)
            if cached_paper:
                if not extract_figs and cached_paper.figures:
                    cached_paper = replace(cached_paper, figures=())
                papers.append(cached_paper)
                cached_count += 1
                continue

        try:
            paper = extract_paper_from_pdf(pdf, max_pages=max_pages)
            if extract_figs:
                figures = extract_figures(pdf, max_pages=max_pages, max_figures=max_figures)
                paper = replace(paper, figures=tuple(figures))

            # Store extracted text + figure metadata in cache
            if cache:
                pdf_hash = compute_pdf_hash(pdf)
                cache.store(paper, pdf_hash)
                new_extractions += 1

        except Exception as e:
            failures += 1
            _report_error("extract", pdf, e)
            continue

        if len(paper.text) < 500:
            tqdm.write(
                f"[WARN] Very little text extracted for {pdf.name} "
                f"(chars={len(paper.text)}). It may be scanned or protected."
            )
        papers.append(paper)

    # Save cache if we extracted new papers
    if cache and new_extractions:
        cache.save()
        tqdm.write(f"[INFO] Cached text for {new_extractions} newly extracted papers")
    
    if cached_count:
        tqdm.write(f"[INFO] Using cached text for {cached_count} unchanged papers")
    if failures:
        tqdm.write(f"[WARN] Extraction failures: {failures}/{len(pdfs)} PDFs")

    return papers, failures


def generate_summaries(papers: list[Paper]) -> tuple[list[Paper], int]:
    """Describe figures via vision LLM, weave into text, then summarize.
    Returns (papers, failure count)."""
    summarized: list[Paper] = []
    failures = 0

    for paper in tqdm(papers, desc="Summarizing"):
        # Skip papers with nothing to summarize
        if not paper.text and not paper.figures:
            summarized.append(replace(paper, summary_md="_No extractable text in this PDF._"))
            continue

        try:
            if paper.figures:
                paper = describe_figures(paper)
                annotated = annotate_text_with_figures(paper.text or "", paper.figures)
                paper = replace(paper, text=annotated)
            result = summarize_paper(paper)
            summarized.append(result)
        except Exception as e:
            failures += 1
            _report_error("summarize", paper.pdf_path, e)
            summarized.append(replace(
                paper,
                summary_md=f"> **Summary generation failed**: {_format_exc(e)}",
            ))

    if failures:
        tqdm.write(f"[WARN] Summarization failures: {failures}/{len(papers)} PDFs")

    return summarized, failures


def _github_slug(title: str) -> str:
    """Mimic GitHub's heading anchor slugger so index links resolve on github.com."""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    return re.sub(r"\s", "-", slug)


def build_markdown(papers: list[Paper]) -> str:
    """Build final markdown document from papers."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []

    lines.append("# Papers Summary")
    lines.append("")
    lines.append(f"_Generated: {now}_")
    lines.append("")

    # Index
    lines.append("## Index")
    lines.append("")
    for p in papers:
        anchor = _github_slug(p.title)
        lines.append(f"- [{p.title}](#{anchor})")
    lines.append("")

    # Summaries
    lines.append("---")
    lines.append("")
    for p in papers:
        lines.append(f"## {p.title}")
        lines.append("")
        lines.append(f"- **Source PDF**: `{p.pdf_path.as_posix()}`")

        doi = find_doi(p.text)
        if doi:
            lines.append(f"- **DOI**: `https://doi.org/{doi}`")
        lines.append("")

        if p.figures:
            lines.append("### Figures")
            lines.append("")
            for fig in p.figures:
                rel = _figure_link(p, fig)
                if rel:
                    alt = (fig.caption or fig.label or "figure").replace("[", "(").replace("]", ")")
                    lines.append(f"![{alt}]({rel})")
                    lines.append("")
            lines.append("---")
            lines.append("")

        if p.summary_md:
            lines.append(p.summary_md.strip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _figure_link(paper: Paper, fig: Figure) -> str | None:
    """Relative markdown link target for a figure crop, if it exists."""
    if not fig.png_path or not fig.png_path.exists():
        return None
    return f"figures/{paper.pdf_path.stem}/{fig.png_path.name}"


def export_figure_crops(papers: list[Paper], out_path: Path) -> int:
    """Copy figure crops next to the output markdown (output/figures/<stem>/).

    Returns the number of crops copied.
    """
    import shutil

    copied = 0
    for paper in papers:
        for fig in paper.figures:
            if not fig.png_path or not fig.png_path.exists():
                continue
            dest_dir = out_path.parent / "figures" / paper.pdf_path.stem
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fig.png_path, dest_dir / fig.png_path.name)
            copied += 1
    return copied


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", default="papers", help="Directory containing PDFs (default: papers)")
    ap.add_argument(
        "--out",
        default="output/PAPERS_SUMMARY.md",
        help="Output markdown path (default: output/PAPERS_SUMMARY.md)",
    )
    ap.add_argument("--max-pages", type=int, default=0, help="Limit pages per PDF (0 = all pages)")
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching, re-extract text from all PDFs"
    )
    ap.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before running"
    )
    ap.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure extraction and vision interpretation"
    )
    ap.add_argument(
        "--max-figures",
        type=int,
        default=12,
        help="Max figures to process per paper (default: 12)"
    )
    args = ap.parse_args()

    papers_dir = Path(args.papers_dir)
    out_path = Path(args.out)
    max_pages = None if args.max_pages == 0 else args.max_pages

    if not papers_dir.exists():
        raise SystemExit(f"papers dir not found: {papers_dir}")

    # Initialize cache (unless disabled)
    cache = None if args.no_cache else PaperCache()
    if cache and args.clear_cache:
        cache.clear()
        cache.save()
        print("[INFO] Cache cleared")

    # Pipeline: load → summarize → write
    papers, extract_failures = load_papers(
        papers_dir,
        max_pages=max_pages,
        cache=cache,
        extract_figs=not args.no_figures,
        max_figures=args.max_figures,
    )
    papers, summarize_failures = generate_summaries(papers)

    md = build_markdown(papers)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    n_crops = export_figure_crops(papers, out_path)
    print(f"Wrote: {out_path}")
    if n_crops:
        print(f"Exported {n_crops} figure crops to: {out_path.parent / 'figures'}")

    if extract_failures or summarize_failures:
        print(
            f"[WARN] Completed with failures: extraction={extract_failures}, "
            f"summarization={summarize_failures}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
