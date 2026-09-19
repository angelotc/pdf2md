# pdf2md

Automatically generate structured markdown summaries of academic PDFs for use as context in engineering codebases.

## Features

- **Sophisticated title extraction**: PDF metadata (/Title, XMP) → first-page text heuristics → filename fallback
- **Figure extraction + vision interpretation**: figure regions (raster + vector) detected via PyMuPDF geometry, cropped to PNG, caption-matched, and described by a vision-capable LLM; descriptions are woven into the text sent to the summarizer and crops are linked in the output
- **LLM-based summarization**: OpenAI-compatible API with map-reduce strategy for high-quality summaries
- **Multiple LLM providers**: Supports OpenAI, OpenRouter, Gemini, or any OpenAI-compatible endpoint
- **Structured output**: TL;DR, Problem, Approach, Results, Practical Takeaways, Limitations
- **Incremental processing**: Caches extracted text + figure metadata, skips re-extraction for unchanged PDFs
- **Customizable prompts**: Configure via `prompts.json`

## Installation

```bash
# Create & activate a virtual environment (recommended)
uv venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Required: Set up environment variables
cp .env.local .env
# Edit .env and add your OPENAI_API_KEY (required)
```

## Configuration

### Environment variables

- `OPENAI_API_KEY` - **Required** for LLM summarization (with the default OpenRouter base URL, use your OpenRouter key)
- `OPENAI_MODEL` - Model to use (default: `google/gemini-3.1-flash-lite`)
- `OPENAI_BASE_URL` - API base URL (default: `https://openrouter.ai/api/v1`; set `https://api.openai.com/v1` for OpenAI)
- `OPENAI_VISION_MODEL` - Model for figure descriptions (default: same as `OPENAI_MODEL`); must accept image input

### Default provider

The defaults target [OpenRouter](https://openrouter.ai) with
[Gemini 3.1 Flash Lite](https://openrouter.ai/google/gemini-3.1-flash-lite) —
multimodal (text + image input), ~1M-token context, $0.25/M input tokens, so the
same model handles both chunk summarization and figure descriptions:

```env
OPENAI_API_KEY=sk-or-...          # your OpenRouter API key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=google/gemini-3.1-flash-lite
```

### prompts.json

Customize summarization prompts and chunking via `prompts.json`:

```json
{
  "chunk_prompt": "...",
  "reduce_prompt": "...",
  "figure_prompt": "...",
  "chunk_max_chars": 12000,
  "max_chunks": 8
}
```

**Available Keys:**
- `chunk_prompt`: Template for summarizing individual chunks. Placeholders: `{title}`, `{idx}`, `{total}`, `{chunk}`.
- `reduce_prompt`: Template for the final combination step. Placeholders: `{title}`, `{summaries}`.
- `figure_prompt`: Template for vision descriptions of figure crops. Placeholders: `{title}`, `{label}`, `{caption}`.
- `chunk_max_chars`: Maximum characters per text chunk (default: `12000`).
- `max_chunks`: Maximum number of chunks to process per paper (default: `8`).

## Usage

```bash
# Basic usage (requires OPENAI_API_KEY in .env or environment)
python summarize_papers.py

# Or set API key inline
OPENAI_API_KEY=sk-... python summarize_papers.py

# Custom options
python summarize_papers.py --papers-dir papers --out output/PAPERS_SUMMARY.md --max-pages 10

# Skip figure extraction / vision interpretation (text-only pipeline)
python summarize_papers.py --no-figures

# Limit figures per paper
python summarize_papers.py --max-figures 6

# Force re-summarize all papers (ignore cache)
python summarize_papers.py --no-cache

# Clear cache and re-run
python summarize_papers.py --clear-cache
```

The script exits non-zero if any PDF fails to extract or summarize; failed papers are marked inline in the output markdown.

### Command-line options

- `--papers-dir DIR` - Directory containing PDFs (default: `papers`)
- `--out FILE` - Output markdown path (default: `output/PAPERS_SUMMARY.md`)
- `--max-pages N` - Limit pages per PDF, 0 = all pages (default: 0)
- `--no-figures` - Skip figure extraction and vision interpretation
- `--max-figures N` - Max figures to process per paper (default: 12)
- `--no-cache` - Disable caching, re-extract text from all PDFs
- `--clear-cache` - Clear cache before running

## Process Flow

```mermaid
graph TD
    A[📁 PDF Directory] --> B[Extract PDFs]
    B --> C[For each PDF]

    subgraph "1. Extraction"
        C --> D{Title Strategy}
        D -->|1. Override| E[Manual Dict]
        D -->|2. Metadata| F[PDF /Title or XMP]
        D -->|3. Heuristic| G[First Page Text]
        D -->|4. Fallback| H[Filename]
        
        E --> I[Extract Full Text<br/>pdfminer.six]
        F --> I
        G --> I
        H --> I
        I --> J[Clean Text]
        J --> F2[Extract Figures<br/>PyMuPDF regions + captions + PNG crops]
    end

    J --> K[Paper Object]
    F2 --> K

    subgraph "2. Summarization"
        K --> L[Load Config<br/>prompts.json]
        L --> V[Describe Figures<br/>Vision LLM]
        V --> W[Weave Descriptions<br/>into Text]
        W --> M[Chunk Text]
        M --> N[Map: Summarize Chunks<br/>OpenAI API]
        N --> O[Reduce: Combine<br/>OpenAI API]
    end

    O --> P[Final Summary]
    P --> Q{More PDFs?}
    Q -->|Yes| C
    Q -->|No| R[Build Markdown<br/>+ figure crops]
    R --> S[Write Output]
```

**Key stages:**
1. **PDF → Paper** - Title extraction cascade + text extraction (pdfminer.six) + figure region detection (PyMuPDF: raster image bboxes + clustered vector drawings, caption-matched, cropped to PNG)
2. **Paper → Summarized Paper** - Vision descriptions of figure crops → descriptions woven into text → map-reduce LLM summarization (chunk → summarize → combine)
3. **Papers → Markdown** - Build structured output with index, summaries, and linked figure crops

## Architecture

The codebase follows a **deep modules** design pattern with strict separation of concerns:

- `lib/pdf_extract.py` - Deep module hiding all PDF parsing complexity
- `lib/figure_extract.py` - Deep module for figure region detection, caption matching, PNG crops
- `lib/vision.py` - Vision LLM descriptions of figure crops (graceful fallback)
- `lib/text_clean.py` - Pure text transformation functions
- `lib/content_analysis.py` - Pure analysis functions (DOI, abstract, figure-text annotation) + LLM chunking
- `lib/summarization.py` - LLM-based summarization (OpenAI)
- `lib/cache.py` - Caches extracted text + figure metadata (not summaries/descriptions) for incremental processing
- `lib/models.py` - Immutable dataclasses (Paper, Figure)
- `summarize_papers.py` - Thin orchestration layer

## Adding PDFs with Missing Metadata

If a PDF's title metadata is corrupt or missing, add an override to `lib/pdf_extract.py`:

```python
TITLE_OVERRIDES: dict[str, str] = {
    "filename.pdf": "Actual Paper Title",
}
```

## Output Format

Generated markdown includes:

- Index of all papers with anchor links
- Per-paper summaries with:
  - Figure crops (linked from `output/figures/<pdf-stem>/`)
  - TL;DR (3 bullets)
  - Problem statement
  - Approach/methodology
  - Results with metrics
  - Practical takeaways
  - Limitations and open questions
  - DOI link (if available)

## Testing

```bash
python -m unittest discover -s tests
```

## Dependencies

- `pdfminer.six` - PDF text extraction (preferred for two-column layouts)
- `pymupdf` - Figure region detection (image bboxes, vector drawings) and PNG crops
- `python-dotenv` - Environment variable loading
- `tqdm` - Progress bars
- `openai` - **Required** for LLM-based summarization and vision figure descriptions
