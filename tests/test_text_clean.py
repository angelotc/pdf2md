"""Tests for lib.text_clean.clean_pdf_text."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.text_clean import clean_pdf_text, normalize_for_sentences


class TestCleanPdfText(unittest.TestCase):
    def test_preserves_normal_english(self):
        # Regression: blind "f i"/"f l"/"f f" replacement used to mangle these.
        self.assertEqual(clean_pdf_text("a set of features"), "a set of features")
        self.assertIn("of if", clean_pdf_text("what of it, of if"))
        self.assertIn("self learning", clean_pdf_text("we study self learning"))
        self.assertIn("off our", clean_pdf_text("we took off our coats"))

    def test_folds_ligature_codepoints(self):
        self.assertEqual(clean_pdf_text("e\ufb03ciency"), "efficiency")  # ﬃ -> ffi
        self.assertEqual(clean_pdf_text("o\ufb02ine"), "ofline")  # ﬂ -> fl

    def test_dehyphenates_wrapped_words(self):
        self.assertEqual(clean_pdf_text("per-\nsonalization"), "personalization")

    def test_strips_boilerplate_lines(self):
        text = "Title line\n\u00a9 2024 Association for Computing Machinery\nACM ISBN 123-4-5678\n\nBody text here."
        cleaned = clean_pdf_text(text)
        self.assertNotIn("\u00a9 2024", cleaned)
        self.assertNotIn("ACM ISBN", cleaned)
        self.assertIn("Body text here.", cleaned)
        self.assertIn("Title line", cleaned)

    def test_collapses_excess_blank_lines(self):
        self.assertNotIn("\n\n\n", clean_pdf_text("a\n\n\n\nb"))


class TestNormalizeForSentences(unittest.TestCase):
    def test_merges_wrapped_lines_within_paragraphs(self):
        self.assertEqual(
            normalize_for_sentences("First line\nwrapped here.\n\nNew paragraph."),
            "First line wrapped here.\n\nNew paragraph.",
        )


if __name__ == "__main__":
    unittest.main()
