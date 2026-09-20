"""Tests for lib.content_analysis: find_doi, extract_abstract, chunk_text_for_llm."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.content_analysis import chunk_text_for_llm, extract_abstract, find_doi


class TestFindDoi(unittest.TestCase):
    def test_extracts_doi_from_url(self):
        self.assertEqual(
            find_doi("See https://doi.org/10.1145/3601335.2966887 for details."),
            "10.1145/3601335.2966887",
        )

    def test_strips_trailing_punctuation(self):
        self.assertEqual(find_doi("doi: 10.1145/123456."), "10.1145/123456")

    def test_no_doi(self):
        self.assertIsNone(find_doi("No identifier in this text."))


class TestExtractAbstract(unittest.TestCase):
    def _paper(self, body: str) -> str:
        return (
            "Author One, Author Two\n\n"
            "Abstract\n" + body + "\n\n1 Introduction\n\nIntro text goes here."
        )

    def test_extracts_between_abstract_and_introduction(self):
        abstract_body = "We study " + "recommendation systems. " * 20
        result = extract_abstract(self._paper(abstract_body))
        self.assertIsNotNone(result)
        self.assertIn("We study", result)
        self.assertNotIn("Introduction", result)

    def test_returns_none_without_abstract(self):
        self.assertIsNone(extract_abstract("Just body text with no heading."))


class TestChunkTextForLlm(unittest.TestCase):
    def test_short_text_is_single_chunk(self):
        self.assertEqual(chunk_text_for_llm("hello world"), ["hello world"])

    def test_splits_on_paragraph_boundaries(self):
        paras = [f"para {i} " + "x" * 50 for i in range(10)]
        text = "\n\n".join(paras)
        chunks = chunk_text_for_llm(text, max_chars=200)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)
        # No text lost
        self.assertEqual(
            sum(len(c) for c in chunks),
            len(text) - (len(chunks) - 1) * 2,  # joins replace the "\n\n"
        )

    def test_splits_oversized_single_paragraph(self):
        big = " ".join(["word"] * 6000)  # 30k chars, no paragraph break
        chunks = chunk_text_for_llm(big, max_chars=1000)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 1000)
        self.assertEqual("".join(c.replace(" ", "") for c in chunks), "word" * 6000)


class TestAnnotateBlockFn(unittest.TestCase):
    def test_custom_block_renderer_is_used(self):
        from lib.content_analysis import annotate_text_with_figures
        from lib.models import Figure

        fig = Figure(page=0, rect=(0, 0, 1, 1), label="Figure 1",
                     caption="The caption.", png_path=None, description="* desc")
        text = "Before.\nFigure 1: The caption.\nAfter."
        out = annotate_text_with_figures(
            text, [fig], block_fn=lambda f: f"[[IMG {f.label}]]"
        )
        self.assertIn("[[IMG Figure 1]]", out)
        self.assertNotIn("[Figure 1 (page 1)]", out)  # default block not used


if __name__ == "__main__":
    unittest.main()
