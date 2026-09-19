"""Tests for figure extraction: clustering, caption matching, end-to-end on a synthetic PDF."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf

from lib.figure_extract import (
    cluster_rects,
    extract_figures,
    _match_caption,
)
from lib.models import Figure
from lib.content_analysis import annotate_text_with_figures


def make_synthetic_pdf(path: Path) -> None:
    """One page with: two nearby vector subplots (one figure), a caption below,
    an isolated vector figure with a caption above it, and body text."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)  # US Letter, like arXiv

    # Body text at top
    page.insert_text((72, 60), "A Synthetic Paper about Recommendation Models")

    # Figure 1: two subplots 20pt apart (beyond CLUSTER_GAP) sharing one caption
    for x0 in (72, 252):
        box = pymupdf.Rect(x0, 100, x0 + 140, 220)
        page.draw_rect(box, color=(0, 0, 0), width=1)
        page.draw_line(
            pymupdf.Point(x0 + 10, 210), pymupdf.Point(x0 + 130, 120),
            color=(0.8, 0.1, 0.1), width=1.5,
        )
        # tick-label-ish text inside the plot box
        page.insert_text((x0 + 8, 218), "0 1 2 3", fontsize=6)
    # Shared caption below, within 40pt
    page.insert_text((72, 232), "Figure 1: Synthetic training curves (two panels).", fontsize=9)

    # Figure 2: isolated box with caption ABOVE it
    page.insert_text((72, 330), "Figure 2: Isolated architecture diagram.", fontsize=9)
    box2 = pymupdf.Rect(72, 342, 320, 470)
    page.draw_rect(box2, color=(0, 0, 0.6), width=1.5)
    page.draw_circle(pymupdf.Point(200, 400), 40, color=(0, 0.6, 0), width=1.5)

    # Body text at bottom (should never be detected as a figure)
    page.insert_text((72, 520), "This paragraph is plain body text and contains no drawings.")

    doc.save(path)
    doc.close()


class TestClusterRects(unittest.TestCase):
    def test_disjoint_rects_stay_separate(self):
        rects = [(0, 0, 10, 10), (100, 100, 110, 110)]
        self.assertEqual(sorted(cluster_rects(rects)), sorted(rects))

    def test_intersecting_rects_merge(self):
        merged = cluster_rects([(0, 0, 10, 10), (5, 5, 15, 15)])
        self.assertEqual(merged, [(0, 0, 15, 15)])

    def test_nearby_rects_merge_within_gap(self):
        # 10pt apart < CLUSTER_GAP=15
        merged = cluster_rects([(0, 0, 10, 10), (20, 0, 30, 10)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0], (0, 0, 30, 10))

    def test_far_rects_do_not_merge(self):
        # 30pt apart > CLUSTER_GAP=15
        merged = cluster_rects([(0, 0, 10, 10), (40, 0, 50, 10)])
        self.assertEqual(len(merged), 2)

    def test_chain_merges_into_one(self):
        merged = cluster_rects([(0, 0, 10, 10), (15, 0, 25, 10), (30, 0, 40, 10)])
        self.assertEqual(merged, [(0, 0, 40, 10)])

    def test_empty(self):
        self.assertEqual(cluster_rects([]), [])


class TestExtractFiguresSynthetic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.tmp.name) / "synthetic.pdf"
        make_synthetic_pdf(self.pdf_path)
        self.out_dir = Path(self.tmp.name) / "figs"

    def tearDown(self):
        self.tmp.cleanup()

    def test_extracts_both_figures_with_captions(self):
        figs = extract_figures(self.pdf_path, out_dir=self.out_dir)
        labels = [f.label for f in figs]
        self.assertIn("Figure 1:", labels)
        self.assertIn("Figure 2:", labels)

        fig1 = next(f for f in figs if f.label == "Figure 1:")
        self.assertIn("Synthetic training curves", fig1.caption or "")
        # Subplots merged: rect spans both panels (x0 ~72 to x1 ~392)
        self.assertLess(fig1.rect[0], 100)
        self.assertGreater(fig1.rect[2], 380)

        fig2 = next(f for f in figs if f.label == "Figure 2:")
        self.assertIn("Isolated architecture", fig2.caption or "")

        # PNG crops written and non-trivial
        for f in figs:
            self.assertTrue(f.png_path and f.png_path.exists())
            self.assertGreater(f.png_path.stat().st_size, 500)

    def test_max_figures_truncates(self):
        figs = extract_figures(self.pdf_path, max_figures=1, out_dir=self.out_dir)
        self.assertEqual(len(figs), 1)

    def test_no_false_positive_on_text_only_page(self):
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 100), "Only text here. Figure 1: a caption with no figure nearby.")
        p = Path(self.tmp.name) / "textonly.pdf"
        doc.save(p)
        doc.close()
        figs = extract_figures(p, out_dir=self.out_dir)
        self.assertEqual(figs, [])


class TestMatchCaption(unittest.TestCase):
    def test_caption_overlapping_region_bottom_matches(self):
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 100), "Figure 3: Overlapping caption case.", fontsize=9)
            region = (70, 50, 300, 98)  # bottom edge overlaps the caption block
            label, caption = _match_caption(page, region)
        self.assertEqual(label, "Figure 3:")
        self.assertIn("Overlapping", caption)

    def test_far_caption_rejected(self):
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 500), "Figure 4: Too far away.", fontsize=9)
            region = (70, 50, 300, 200)  # 300pt away
            label, caption = _match_caption(page, region)
        self.assertIsNone(label)
        self.assertIsNone(caption)


class TestAnnotateTextWithFigures(unittest.TestCase):
    def test_inserts_after_caption(self):
        text = "Intro line.\nFigure 1: Training curves.\nConclusion line."
        figs = [Figure(page=0, rect=(0, 0, 1, 1), label="Figure 1:",
                       caption="Figure 1: Training curves.",
                       description="- AUC rises 0.84 -> 0.85")]
        out = annotate_text_with_figures(text, figs)
        self.assertIn("[Figure 1 (page 1)] Figure 1: Training curves.", out)
        self.assertIn("- AUC rises 0.84 -> 0.85", out)
        # Description block sits between caption line and conclusion
        cap_idx = out.find("Figure 1: Training curves.\n")
        desc_idx = out.find("- AUC rises")
        concl_idx = out.find("Conclusion line.")
        self.assertLess(cap_idx, desc_idx)
        self.assertLess(desc_idx, concl_idx)

    def test_unlocatable_figure_appended(self):
        text = "Body text without the caption."
        figs = [Figure(page=2, rect=(0, 0, 1, 1), label="Figure 9:",
                       caption=None, description="- mystery plot")]
        out = annotate_text_with_figures(text, figs)
        self.assertTrue(out.startswith("Body text without the caption."))
        self.assertIn("[Figure 9 (page 3)]\n- mystery plot", out)

    def test_no_figures_noop(self):
        text = "unchanged"
        self.assertEqual(annotate_text_with_figures(text, []), "unchanged")


if __name__ == "__main__":
    unittest.main()
