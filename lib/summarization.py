"""Summarization: LLM-based (OpenAI required)."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Final, Any

from tqdm import tqdm

from lib.models import Paper
from lib.content_analysis import chunk_text_for_llm


DEFAULT_MAX_CHARS_PER_CHUNK: Final[int] = 12000
DEFAULT_MAX_CHUNKS: Final[int] = 8

DEFAULT_CHUNK_PROMPT: Final[str] = (
    "You are summarizing an ML/RecSys research paper for an engineering codebase context.\n"
    "Write a concise, high-signal summary of THIS CHUNK.\n"
    "Focus on: problem, key methods, key claims/results, assumptions/limitations.\n"
    "Return 6-10 bullet points.\n\n"
    "Paper title: {title}\n"
    "Chunk {idx}/{total}:\n\n{chunk}"
)

DEFAULT_REDUCE_PROMPT: Final[str] = (
    "Combine the chunk summaries into a final paper summary for engineers.\n"
    "Output markdown with EXACTLY these sections:\n"
    "### TL;DR (3 bullets)\n"
    "### Problem\n"
    "### Approach\n"
    "### Results (include numbers)\n"
    "### Practical Takeaways\n"
    "### Limitations / Open Questions\n"
    "Rules:\n"
    "- Prefer concrete metrics/numbers if stated.\n"
    "- If something isn't supported, write 'Unclear from text'.\n\n"
    "Paper title: {title}\n\n"
    "Chunk summaries:\n\n{summaries}"
)


_CONFIG_CACHE: dict[str, Any] | None = None
_CLIENT: Any = None


def _get_client() -> Any:
    """Create the shared OpenAI client once per run."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it in .env or export OPENAI_API_KEY=sk-..."
        )

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        )

    base_url = os.environ.get("OPENAI_BASE_URL")
    _CLIENT = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return _CLIENT


def _load_config() -> dict[str, Any]:
    """Load configuration from prompts.json if it exists (cached)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = Path(__file__).resolve().parent.parent / "prompts.json"
    if not path.exists():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE
    try:
        _CONFIG_CACHE = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _CONFIG_CACHE = {}
    
    return _CONFIG_CACHE


def _summarize_chunk(client, model: str, paper: Paper, chunk: str, idx: int, total: int) -> str:
    """Summarize a single chunk of paper text using OpenAI."""
    config = _load_config()
    prompt_template = config.get("chunk_prompt", DEFAULT_CHUNK_PROMPT)
    prompt = (
        prompt_template
        .replace("{title}", paper.title)
        .replace("{idx}", str(idx))
        .replace("{total}", str(total))
        .replace("{chunk}", chunk)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()


def _reduce_chunk_summaries(client, model: str, paper: Paper, chunk_summaries: list[str]) -> str:
    """Combine chunk summaries into final paper summary using OpenAI."""
    config = _load_config()
    reduce_prompt_template = config.get("reduce_prompt", DEFAULT_REDUCE_PROMPT)
    reduce_prompt = (
        reduce_prompt_template
        .replace("{title}", paper.title)
        .replace("{summaries}", "\n\n".join(chunk_summaries))
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": reduce_prompt}]
    )
    return resp.choices[0].message.content.strip()


def summarize_paper(paper: Paper, max_chunks: int | None = None) -> Paper:
    """
    Generate summary using OpenAI LLM (required).

    Strategy: Map-reduce over chunks
    1. Split text into chunks
    2. Summarize each chunk
    3. Reduce chunk summaries into final summary

    Requires: OPENAI_API_KEY environment variable

    Args:
        paper: Paper with text to summarize.
        max_chunks: Overrides prompts.json's max_chunks when given.

    Returns: New Paper object with summary_md populated.
    Raises: RuntimeError if OPENAI_API_KEY not set
    """
    client = _get_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-5-mini-2025-08-07")

    config = _load_config()

    # Priority: Config > Default
    max_chars = config.get("chunk_max_chars", DEFAULT_MAX_CHARS_PER_CHUNK)
    config_max_chunks = config.get("max_chunks", DEFAULT_MAX_CHUNKS)

    effective_max_chunks = max_chunks if max_chunks is not None else config_max_chunks

    chunks = chunk_text_for_llm(paper.text, max_chars=max_chars)
    if len(chunks) > effective_max_chunks:
        skipped_chars = sum(len(c) for c in chunks[effective_max_chunks:])
        tqdm.write(
            f"[WARN] {paper.pdf_path.name}: text produced {len(chunks)} chunks, "
            f"summarizing only the first {effective_max_chunks} "
            f"(~{skipped_chars} chars skipped)"
        )
    chunks = chunks[:effective_max_chunks]
    chunk_summaries: list[str] = []

    # Use tqdm for chunk summarization if there are multiple
    chunk_iter = enumerate(chunks, start=1)
    if len(chunks) > 1:
        chunk_iter = tqdm(
            chunk_iter,
            total=len(chunks),
            desc=f"  Summarizing {paper.pdf_path.name}",
            leave=False
        )

    for idx, ch in chunk_iter:
        summary = _summarize_chunk(client, model, paper, ch, idx, len(chunks))
        chunk_summaries.append(summary)

    final_summary = _reduce_chunk_summaries(client, model, paper, chunk_summaries)
    return dataclasses.replace(paper, summary_md=final_summary)
