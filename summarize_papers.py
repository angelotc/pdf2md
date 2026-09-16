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
  OPENAI_MODEL            -> default: gpt-5-mini-2025-08-07
  OPENAI_BASE_URL         -> API base URL (e.g. OpenRouter, Gemini)
"""

from __future__ import annotations

import argparse
import os
import re
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from lib.models import Paper
from lib.pdf_extract import extract_paper_from_pdf
from lib.summarization import summarize_paper
from lib.content_analysis import find_doi
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
    cache: PaperCache | None = None
) -> list[Paper]:
    """
    Extract title + text from all PDFs in directory.
    Uses cache to avoid re-extracting unchanged PDFs.
    
    Returns: list of Paper objects with text extracted
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
                papers.append(cached_paper)
                cached_count += 1
                continue

        try:
            paper = extract_paper_from_pdf(pdf, max_pages=max_pages)
            
            # Store extracted text in cache
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

    return papers


def generate_summaries(papers: list[Paper]) -> list[Paper]:
    """Generate summaries for all papers using LLM."""
    summarized: list[Paper] = []
    failures = 0

    for paper in tqdm(papers, desc="Summarizing"):
        # Skip papers with no text (nothing to summarize)
        if not paper.text:
            summarized.append(paper)
            continue

        try:
            result = summarize_paper(paper)
            summarized.append(result)
        except Exception as e:
            failures += 1
            _report_error("summarize", paper.pdf_path, e)
            summarized.append(paper)

    if failures:
        tqdm.write(f"[WARN] Summarization failures: {failures}/{len(papers)} PDFs")

    return summarized


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

        if p.summary_md:
            lines.append(p.summary_md.strip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


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
    papers = load_papers(papers_dir, max_pages=max_pages, cache=cache)
    papers = generate_summaries(papers)

    md = build_markdown(papers)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
