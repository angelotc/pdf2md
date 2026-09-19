"""Vision interpretation: describe figure crops with a multimodal LLM.

Each figure crop is sent as a base64 PNG alongside a text prompt; the model
returns bullet points describing what the figure shows. Failures degrade to
caption-only descriptions so the pipeline never blocks on vision errors.
"""

from __future__ import annotations

import base64
import dataclasses
import os
from typing import Final

from tqdm import tqdm

from lib.models import Paper, Figure
from lib.summarization import DEFAULT_MODEL


DEFAULT_FIGURE_PROMPT: Final[str] = (
    "You are analyzing one figure cropped from an ML/RecSys research paper.\n"
    "Describe what the figure ACTUALLY SHOWS in 3-6 bullet points.\n"
    "Include concrete values: axis ranges, key numbers, trends, comparisons,\n"
    "named models/systems and how they compare. Read legends and axis labels.\n"
    "If the crop is not a chart/diagram (e.g. plain text or empty), say so briefly.\n\n"
    "Paper title: {title}\n"
    "Figure: {label}\n"
    "Caption: {caption}\n"
    "Respond with bullets only, no preamble."
)

NO_VISION_PLACEHOLDER: Final[str] = "[No vision interpretation available — caption only]"


def _vision_model() -> str:
    """Model for figure descriptions; falls back to the text model setting."""
    return os.environ.get("OPENAI_VISION_MODEL") or os.environ.get(
        "OPENAI_MODEL", DEFAULT_MODEL
    )


def describe_figures(
    paper: Paper,
    client=None,
    load_config=None,
) -> Paper:
    """Populate Figure.description for each figure via a vision LLM call.

    Degrades gracefully: any API/model error logs one warning for the paper
    and leaves descriptions as the caption-only placeholder, so text-only
    models or missing images never abort summarization.

    Args:
        paper: Paper with figures extracted (png_path set).
        client: OpenAI-compatible client (default: shared one from summarization).
        load_config: config loader (default: prompts.json loader).

    Returns: New Paper with described figures.
    """
    from lib.summarization import _get_client, _load_config

    client = client or _get_client()
    load_config = load_config or _load_config
    config = load_config()
    prompt_template = config.get("figure_prompt", DEFAULT_FIGURE_PROMPT)
    model = _vision_model()

    described: list[Figure] = []
    warned = False
    for fig in tqdm(paper.figures, desc=f"  Figures {paper.pdf_path.name}", leave=False):
        prompt = (
            prompt_template
            .replace("{title}", paper.title)
            .replace("{label}", fig.label or "unlabeled figure")
            .replace("{caption}", fig.caption or "(none found)")
        )

        description: str | None = None
        if fig.png_path and fig.png_path.exists():
            try:
                png_b64 = base64.b64encode(fig.png_path.read_bytes()).decode("ascii")
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                            },
                        ],
                    }],
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    description = content
            except Exception as e:
                if not warned:
                    tqdm.write(
                        f"[WARN] Vision description failed for {paper.pdf_path.name} "
                        f"({type(e).__name__}: {e}); falling back to caption-only"
                    )
                    warned = True

        described.append(dataclasses.replace(fig, description=description or NO_VISION_PLACEHOLDER))

    return dataclasses.replace(paper, figures=tuple(described))
