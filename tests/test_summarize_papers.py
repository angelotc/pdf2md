"""Tests for the index anchor slugger in summarize_papers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from summarize_papers import _github_slug


class TestGithubSlug(unittest.TestCase):
    def test_plain_title(self):
        self.assertEqual(_github_slug("Unified Embedding"), "unified-embedding")

    def test_keeps_existing_hyphens(self):
        self.assertEqual(
            _github_slug("Prompt-to-Slate Diffusion Models"),
            "prompt-to-slate-diffusion-models",
        )

    def test_punctuation_is_stripped_without_inserting_hyphens(self):
        # GitHub slugs "Don't" as "dont", not "don-t".
        self.assertEqual(
            _github_slug("You Don't Bring Me Flowers: Mitigating Unwanted Recommendations"),
            "you-dont-bring-me-flowers-mitigating-unwanted-recommendations",
        )

    def test_acronyms_survive_lowercasing(self):
        self.assertEqual(
            _github_slug("Exploring Scaling Laws of CTR Model"),
            "exploring-scaling-laws-of-ctr-model",
        )


if __name__ == "__main__":
    unittest.main()
