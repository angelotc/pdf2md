"""Figure extraction: detect figure regions in PDFs, crop them to PNG.

Deep module hiding all region-detection complexity:
- Raster figures via placed-image bounding boxes.
- Vector figures (matplotlib/gnuplot plots) via clustered drawing paths.
- Captions matched from nearby text blocks ("Figure N", "Fig. N", "Table N").
- Crops rendered to .paper2md/figures/<pdf-stem>/fig_p{page}_{i}.png.

Geometry is in PDF points; page is 0-based.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from lib.models import Figure


_REPO_ROOT = Path(__file__).resolve().parent.parent

# Region filters, as fractions of page area
MIN_RASTER_AREA = 0.01   # icons/logos below this are skipped
MAX_REGION_AREA = 0.85   # near-full-page images are likely scans/backgrounds
MIN_FIGURE_AREA = 0.04   # merged regions below this are skipped
CLUSTER_GAP = 15.0       # pts; rects closer than this merge into one region
RECT_PAD = 6.0           # pts of padding around the final crop rect

_CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table)\s*\S+", re.IGNORECASE)


def _area(rect: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = rect
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """True if rects overlap or are separated by less than CLUSTER_GAP on both axes."""
    return not (a[2] + CLUSTER_GAP < b[0] or b[2] + CLUSTER_GAP < a[0]) and not (
        a[3] + CLUSTER_GAP < b[1] or b[3] + CLUSTER_GAP < a[1]
    )


def _union(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def cluster_rects(rects: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Union-find merge of rects that intersect or nearly touch.

    Repeatedly merges any two rects closer than CLUSTER_GAP until stable.
    """
    clusters = list(rects)
    merged = True
    while merged:
        merged = False
        out: list[tuple[float, float, float, float]] = []
        for rect in clusters:
            for i, existing in enumerate(out):
                if _intersect(existing, rect):
                    out[i] = _union(existing, rect)
                    merged = True
                    break
            else:
                out.append(rect)
        clusters = out
    return clusters


def _raster_regions(page: pymupdf.Page, page_area: float) -> list[tuple[float, float, float, float]]:
    """Bounding boxes of placed raster images, filtered to figure-sized ones."""
    rects = []
    for info in page.get_image_info():
        bbox = info.get("bbox")
        if not bbox:
            continue
        rect = tuple(float(v) for v in bbox)
        frac = _area(rect) / page_area
        if MIN_RASTER_AREA <= frac <= MAX_REGION_AREA:
            rects.append(rect)
    return rects


def _vector_regions(page: pymupdf.Page, page_area: float) -> list[tuple[float, float, float, float]]:
    """Clustered bounding boxes of vector drawings, excluding rules and page furniture."""
    raw: list[tuple[float, float, float, float]] = []
    page_rect = page.rect
    for drawing in page.get_drawings():
        rect = tuple(float(v) for v in drawing["rect"])
        # Skip hairline horizontal/vertical rules (section separators, table lines)
        if rect[3] - rect[1] < 2.0 or rect[2] - rect[0] < 2.0:
            continue
        # Skip rects covering (nearly) the whole page: borders/backgrounds
        if _area(rect) / page_area > MAX_REGION_AREA:
            continue
        # Skip rects pinned to the page edge on all sides (page frames)
        if (
            rect[0] <= page_rect.x0 + 1 and rect[1] <= page_rect.y0 + 1
            and rect[2] >= page_rect.x1 - 1 and rect[3] >= page_rect.y1 - 1
        ):
            continue
        raw.append(rect)
    return cluster_rects([r for r in raw if _area(r) / page_area >= MIN_RASTER_AREA])


def _match_caption(
    page: pymupdf.Page,
    region: tuple[float, float, float, float],
) -> tuple[str | None, str | None]:
    """Find the caption block nearest to the region (preferring directly below).

    A candidate must start with "Figure N" / "Fig. N" / "Table N", sit within
    40pts of the region (overlap counts as distance 0 — caption baselines often
    fall inside the drawing cluster's bbox), and overlap horizontally.
    """
    best: tuple[float, str, str] | None = None  # (score, label, caption)
    rx0, ry0, rx1, ry1 = region
    for block in page.get_text("blocks"):
        bx0, by0, bx1, by1, text = block[0], block[1], block[2], block[3], block[4]
        stripped = " ".join(text.split())
        m = _CAPTION_RE.match(stripped)
        if not m:
            continue
        gap = max(0.0, ry0 - by1, by0 - ry1)  # 0 when overlapping
        if gap > 40:
            continue
        # Horizontal overlap
        if bx1 < rx0 - 20 or bx0 > rx1 + 20:
            continue
        above = by1 <= ry0 + 20  # caption sits above the region
        score = gap + (100 if above else 0)
        if best is None or score < best[0]:
            best = (score, m.group(0).rstrip(), stripped)

    if best is None:
        return None, None
    return best[1], best[2]


def extract_figures(
    pdf_path: Path,
    max_pages: int | None = None,
    max_figures: int = 12,
    dpi: int = 150,
    out_dir: Path | None = None,
) -> list[Figure]:
    """Detect figure regions in a PDF, crop them to PNG, and match captions.

    Returns Figures ordered by (page, position), cropped PNGs written under
    out_dir (default .paper2md/figures/<pdf-stem>/). Regions exceeding
    max_figures are dropped, largest first.
    """
    figures_dir = out_dir or (_REPO_ROOT / ".paper2md" / "figures" / pdf_path.stem)

    results: list[tuple[float, int, Figure]] = []  # (area, page, figure) for truncation
    with pymupdf.open(pdf_path) as doc:
        n_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
        per_page_count: dict[int, int] = {}
        for page_num in range(n_pages):
            page = doc[page_num]
            page_area = abs(page.rect.width * page.rect.height)
            if page_area <= 0:
                continue

            # Match captions BEFORE the area filter: two subplots may each be
            # too small to keep alone but form one caption-sized figure together.
            clusters = cluster_rects(
                _raster_regions(page, page_area) + _vector_regions(page, page_area)
            )

            # Regions matched to the same caption are subpanels of one figure
            # (e.g. side-by-side plots 20pt apart) — merge their rects.
            by_caption: dict[str, tuple[str, tuple[float, float, float, float]]] = {}
            standalone: list[tuple[float, float, float, float]] = []
            for region in clusters:
                label, caption = _match_caption(page, region)
                if caption:
                    if caption in by_caption:
                        prev_label, prev_rect = by_caption[caption]
                        by_caption[caption] = (prev_label, _union(prev_rect, region))
                    else:
                        by_caption[caption] = (label, region)
                else:
                    standalone.append(region)

            page_regions: list[tuple[tuple[float, float, float, float], str | None, str | None]] = [
                (rect, label, caption) for caption, (label, rect) in by_caption.items()
            ] + [(rect, None, None) for rect in standalone]
            page_regions = [
                (rect, label, caption) for rect, label, caption in page_regions
                if _area(rect) / page_area >= MIN_FIGURE_AREA
            ]
            page_regions.sort(key=lambda t: (t[0][1], t[0][0]))

            for region, label, caption in page_regions:
                # Expand cluster rects a little to catch axis labels at the edge
                rect = (
                    max(0.0, region[0] - RECT_PAD),
                    max(0.0, region[1] - RECT_PAD),
                    region[2] + RECT_PAD,
                    region[3] + RECT_PAD,
                )
                # Clip to page
                rect = (
                    max(rect[0], page.rect.x0),
                    max(rect[1], page.rect.y0),
                    min(rect[2], page.rect.x1),
                    min(rect[3], page.rect.y1),
                )

                idx = per_page_count.get(page_num, 0)
                per_page_count[page_num] = idx + 1
                figures_dir.mkdir(parents=True, exist_ok=True)
                png_path = figures_dir / f"fig_p{page_num}_{idx}.png"
                pix = page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(rect))
                pix.save(png_path)

                fig = Figure(
                    page=page_num,
                    rect=region,
                    label=label,
                    caption=caption,
                    png_path=png_path,
                )
                results.append((_area(region), page_num, fig))

    # Keep the largest max_figures regions, then restore document order
    results.sort(key=lambda t: -t[0])
    kept = sorted(results[:max_figures], key=lambda t: (t[1], t[2].rect[1], t[2].rect[0]))
    return [fig for _, _, fig in kept]
