#!/usr/bin/env python3
"""
Summarize PDFs in /papers into a single markdown file for repo context.

Behavior:
- Extracts text + title from each PDF via sophisticated metadata + text extraction.
- Generates structured summaries using OpenAI-compatible LLM (required).
- Caches extracted text in .paper2md/ to skip re-extraction of unchanged PDFs.
- Writes one standalone markdown per paper (output/<stem>.md) plus INDEX.md;
  --combined additionally derives output/PAPERS_SUMMARY.md from those files.
  Every paper also gets a raw doc (output/raw/<stem>.md): the full extracted
  text with figure crops and their vision descriptions inline. Files whose
  content is unchanged are not rewritten.

Usage:
  python summarize_papers.py                       # per-paper files + INDEX.md in output/
  python summarize_papers.py --combined            # also derive output/PAPERS_SUMMARY.md
  python summarize_papers.py --paper wikiskills    # single paper (stem substring ok)
  python summarize_papers.py --no-cache            # force re-extract all PDFs

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
import traceback
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from lib.models import Paper
from lib.pdf_extract import extract_paper_from_pdf
from lib.figure_extract import extract_figures
from lib.vision import describe_figures
from lib.summarization import summarize_paper
from lib.content_analysis import annotate_text_with_figures
from lib.render import (
    INDEX_FILENAME,
    COMBINED_FILENAME,
    build_paper_markdown,
    build_paper_raw_markdown,
    build_index_markdown,
    build_combined_markdown,
    collect_paper_docs,
    write_if_changed,
)
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
    only: Path | None = None,
) -> tuple[list[Paper], int]:
    """
    Extract title + text (+ figures) from all PDFs in directory.
    Uses cache to avoid re-extracting unchanged PDFs.
    When `only` is given, process just that PDF.

    Returns: (list of Paper objects with text extracted, extraction failure count)
    """
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if only:
        pdfs = [pdf for pdf in pdfs if pdf == only]
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


def generate_summaries(
    papers: list[Paper], max_chunks: int | None = None
) -> tuple[list[Paper], int]:
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
            # The annotated text (descriptions woven in) is the summarizer's
            # input only; the returned paper keeps the original text so the
            # raw doc can render figures with markdown blocks instead.
            llm_input = paper
            if paper.figures:
                paper = describe_figures(paper)
                annotated = annotate_text_with_figures(paper.text or "", paper.figures)
                llm_input = replace(paper, text=annotated)
            summarized_result = summarize_paper(llm_input, max_chunks=max_chunks)
            summarized.append(replace(paper, summary_md=summarized_result.summary_md))
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


def export_figure_crops(papers: list[Paper], out_dir: Path) -> int:
    """Copy figure crops into out_dir/figures/<pdf-stem>/.

    Returns the number of crops copied.
    """
    import shutil

    copied = 0
    for paper in papers:
        for fig in paper.figures:
            if not fig.png_path or not fig.png_path.exists():
                continue
            dest_dir = out_dir / "figures" / paper.pdf_path.stem
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fig.png_path, dest_dir / fig.png_path.name)
            copied += 1
    return copied


def _select_paper(papers_dir: Path, name: str) -> Path:
    """Resolve --paper NAME to exactly one PDF in papers_dir.

    Accepts the filename ("wikiskills.pdf"), the stem ("wikiskills"), or a
    unique substring of the stem, all case-insensitive.
    """
    key = name.casefold()
    pdfs = sorted(papers_dir.glob("*.pdf"))
    matches = [p for p in pdfs if key in {p.name.casefold(), p.stem.casefold()}]
    if not matches:
        matches = [p for p in pdfs if key in p.stem.casefold()]
    if len(matches) == 1:
        return matches[0]
    available = "\n  ".join(p.name for p in pdfs) or "(none)"
    if not matches:
        raise SystemExit(f"--paper {name!r} matched no PDF in {papers_dir}.\nAvailable:\n  {available}")
    listed = "\n  ".join(p.name for p in matches)
    raise SystemExit(f"--paper {name!r} matched multiple PDFs:\n  {listed}")


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", default="papers", help="Directory containing PDFs (default: papers)")
    ap.add_argument(
        "--out-dir",
        default="output",
        help="Directory for all outputs: per-paper markdown, INDEX.md, figure "
        "crops, and PAPERS_SUMMARY.md with --combined (default: output)",
    )
    ap.add_argument(
        "--paper",
        metavar="NAME",
        help="Process a single PDF from the papers dir, matched by filename, "
        "stem, or unique stem substring (e.g. 'wikiskills.pdf' or 'wikiskills')",
    )
    ap.add_argument(
        "--combined",
        action="store_true",
        help="Also write the combined PAPERS_SUMMARY.md, derived from the "
        "per-paper files on disk (no re-extraction or LLM calls)",
    )
    ap.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Max text chunks to summarize per paper (overrides prompts.json's max_chunks)",
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
    out_dir = Path(args.out_dir)
    max_pages = None if args.max_pages == 0 else args.max_pages

    if not papers_dir.exists():
        raise SystemExit(f"papers dir not found: {papers_dir}")

    only: Path | None = _select_paper(papers_dir, args.paper) if args.paper else None

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
        only=only,
    )
    papers, summarize_failures = generate_summaries(papers, max_chunks=args.max_chunks)

    if only and not papers:
        print(f"[ERROR] No text could be extracted for {only.name}; nothing to write.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    n_crops = export_figure_crops(papers, out_dir)

    # Per-paper files are canonical; write only those whose content changed.
    written = 0
    total = 0
    for p in papers:
        targets = [(out_dir / f"{p.pdf_path.stem}.md", build_paper_markdown(p))]
        if p.text or p.figures:
            raw_path = out_dir / "raw" / f"{p.pdf_path.stem}.md"
            targets.append((raw_path, build_paper_raw_markdown(p)))
        for path, content in targets:
            total += 1
            if write_if_changed(path, content):
                print(f"Wrote: {path}")
                written += 1
    unchanged = total - written
    if unchanged:
        print(f"[INFO] {unchanged} per-paper file(s) unchanged, not rewritten")

    # INDEX and the combined doc are derived from the per-paper files on disk,
    # so they reflect every paper summarized so far — not just this run's.
    docs = collect_paper_docs(out_dir)
    if write_if_changed(out_dir / INDEX_FILENAME, build_index_markdown(docs)):
        print(f"Wrote: {out_dir / INDEX_FILENAME}")
    if args.combined:
        if write_if_changed(out_dir / COMBINED_FILENAME, build_combined_markdown(docs)):
            print(f"Wrote: {out_dir / COMBINED_FILENAME}")

    if n_crops:
        print(f"Exported {n_crops} figure crops to: {out_dir / 'figures'}")

    if extract_failures or summarize_failures:
        print(
            f"[WARN] Completed with failures: extraction={extract_failures}, "
            f"summarization={summarize_failures}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
