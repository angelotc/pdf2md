"""Tests for markdown rendering: per-paper docs, derived index and combined doc."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.models import Paper, Figure
from lib.render import (
    INDEX_FILENAME,
    COMBINED_FILENAME,
    github_slug,
    build_paper_markdown,
    bump_headings,
    extract_h1,
    first_tldr_line,
    collect_paper_docs,
    build_index_markdown,
    build_combined_markdown,
    write_if_changed,
)


class TestGithubSlug(unittest.TestCase):
    def test_plain_title(self):
        self.assertEqual(github_slug("Unified Embedding"), "unified-embedding")

    def test_keeps_existing_hyphens(self):
        self.assertEqual(
            github_slug("Prompt-to-Slate Diffusion Models"),
            "prompt-to-slate-diffusion-models",
        )

    def test_punctuation_is_stripped_without_inserting_hyphens(self):
        # GitHub slugs "Don't" as "dont", not "don-t".
        self.assertEqual(
            github_slug("You Don't Bring Me Flowers: Mitigating Unwanted Recommendations"),
            "you-dont-bring-me-flowers-mitigating-unwanted-recommendations",
        )

    def test_acronyms_survive_lowercasing(self):
        self.assertEqual(
            github_slug("Exploring Scaling Laws of CTR Model"),
            "exploring-scaling-laws-of-ctr-model",
        )


class TestBuildPaperMarkdown(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.png = Path(self.tmp.name) / "fig_p0_0.png"
        self.png.write_bytes(b"fake-png")
        self.addCleanup(self.tmp.cleanup)

    def _paper(self, with_figure=True, with_doi=True) -> Paper:
        text = "body text"
        if with_doi:
            text = "See https://doi.org/10.1145/3600006.3613165 for details. " + text
        figures = ()
        if with_figure:
            figures = (Figure(
                page=0, rect=(0, 0, 1, 1), label="Figure 1", caption="A test caption",
                png_path=self.png,
            ),)
        return Paper(
            pdf_path=Path("papers/wikiskills.pdf"),
            title="WikiSkill: A Test",
            text=text,
            figures=figures,
            summary_md="### TL;DR\n* bullet one",
        )

    def test_standalone_doc_structure(self):
        md = build_paper_markdown(self._paper())
        self.assertTrue(md.startswith("# WikiSkill: A Test\n"))
        self.assertIn("- **Source PDF**: `papers/wikiskills.pdf`", md)
        self.assertIn("- **DOI**: `https://doi.org/10.1145/3600006.3613165`", md)
        self.assertIn("\n## Figures\n", md)
        self.assertIn("![A test caption](figures/wikiskills/fig_p0_0.png)", md)
        self.assertIn("### TL;DR", md)

    def test_missing_doi_and_figures_omitted(self):
        md = build_paper_markdown(self._paper(with_figure=False, with_doi=False))
        self.assertNotIn("**DOI**", md)
        self.assertNotIn("## Figures", md)

    def test_figure_with_missing_png_is_skipped(self):
        paper = self._paper()
        gone = Paper(
            pdf_path=paper.pdf_path, title=paper.title, text=paper.text,
            figures=(Figure(page=0, rect=(0, 0, 1, 1), label="Fig 9",
                            png_path=Path("/nonexistent/fig.png")),),
        )
        md = build_paper_markdown(gone)
        self.assertNotIn("![", md)


class TestBumpHeadings(unittest.TestCase):
    def test_bumps_each_level(self):
        self.assertEqual(bump_headings("# T\n\ntext\n\n### S"), "## T\n\ntext\n\n#### S")

    def test_leaves_fenced_code_alone(self):
        md = "# T\n\n```python\n# not a heading\n```\n"
        self.assertEqual(bump_headings(md), "## T\n\n```python\n# not a heading\n```\n")

    def test_h6_not_bumped_past_limit(self):
        self.assertEqual(bump_headings("###### six"), "###### six")

    def test_plain_text_untouched(self):
        self.assertEqual(bump_headings("just text\n#no-space-not-heading"), "just text\n#no-space-not-heading")


class TestDocParsing(unittest.TestCase):
    def test_extract_h1(self):
        self.assertEqual(extract_h1("# My Title\n\n## Sub"), "My Title")
        self.assertIsNone(extract_h1("## only h2"))
        self.assertIsNone(extract_h1("plain text"))

    def test_first_tldr_line(self):
        md = "# T\n\n### TL;DR (3 bullets)\n* **Label:** the first insight\n* second\n"
        self.assertEqual(first_tldr_line(md), "Label: the first insight")

    def test_first_tldr_line_truncates_long_bullets(self):
        md = "### TL;DR\n* " + "word " * 50 + "\n"
        line = first_tldr_line(md)
        self.assertLessEqual(len(line), 141)
        self.assertTrue(line.endswith("…"))

    def test_first_tldr_line_absent(self):
        self.assertIsNone(first_tldr_line("# T\n\n### Problem\nbody"))


class TestDerivedDocs(unittest.TestCase):
    DOCS = [
        (Path("alpha.md"), "# Alpha Paper\n\n### TL;DR\n* **A:** first alpha point\n"),
        (Path("beta.md"), "# Beta Beats\n\n### TL;DR\n* **B:** first beta point\n"),
    ]

    def test_combined_structure(self):
        md = build_combined_markdown(self.DOCS)
        self.assertTrue(md.startswith("# Papers Summary\n"))
        self.assertIn("- [Alpha Paper](#alpha-paper)", md)
        self.assertIn("- [Beta Beats](#beta-beats)", md)
        # Bodies appear with headings bumped under the index
        self.assertIn("\n## Alpha Paper\n", md)
        self.assertIn("\n#### TL;DR\n", md)
        # Exactly one H1
        self.assertEqual(md.count("\n# "), 0)
        self.assertTrue(md.startswith("# Papers Summary"))

    def test_index_structure(self):
        md = build_index_markdown(self.DOCS)
        self.assertEqual(
            md,
            "# Papers Index\n\n"
            "- [Alpha Paper](alpha.md) — A: first alpha point\n"
            "- [Beta Beats](beta.md) — B: first beta point\n",
        )

    def test_index_falls_back_to_stem_without_h1(self):
        md = build_index_markdown([(Path("notes.md"), "no heading here")])
        self.assertIn("- [notes](notes.md)", md)
        self.assertNotIn("—", md)


class TestCollectAndWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_collect_excludes_index_and_combined(self):
        for name in ("a.md", "b.md", INDEX_FILENAME, COMBINED_FILENAME):
            (self.dir / name).write_text(f"# {name}\n", encoding="utf-8")
        docs = collect_paper_docs(self.dir)
        self.assertEqual([p.name for p, _ in docs], ["a.md", "b.md"])

    def test_collect_skips_unreadable(self):
        (self.dir / "good.md").write_text("# good\n", encoding="utf-8")
        (self.dir / "bad.md").write_bytes(b"\xff\xfe\x00invalid")
        docs = collect_paper_docs(self.dir)
        self.assertEqual([p.name for p, _ in docs], ["good.md"])

    def test_write_if_changed(self):
        path = self.dir / "out.md"
        self.assertTrue(write_if_changed(path, "v1"))          # new file
        mtime = path.stat().st_mtime_ns
        self.assertFalse(write_if_changed(path, "v1"))         # identical: no write
        self.assertEqual(path.stat().st_mtime_ns, mtime)
        self.assertTrue(write_if_changed(path, "v2"))          # changed: write
        self.assertEqual(path.read_text(encoding="utf-8"), "v2")


if __name__ == "__main__":
    unittest.main()
