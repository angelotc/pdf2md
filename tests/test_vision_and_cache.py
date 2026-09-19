"""Tests for vision description (with fake client) and cache v3 round-trip."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lib.models import Figure, Paper
from lib.vision import describe_figures, NO_VISION_PLACEHOLDER
from lib.cache import PaperCache


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]


class _FakeClient:
    def __init__(self, content: str | Exception):
        self._content = content
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, model, messages):
        self.calls.append({"model": model, "messages": messages})
        if isinstance(self._content, Exception):
            raise self._content
        return _FakeResponse(self._content)


def _make_paper(tmp: Path, with_png: bool = True) -> Paper:
    png = tmp / "fig_p0_0.png"
    if with_png:
        png.write_bytes(b"\x89PNG fake bytes")
    fig = Figure(page=0, rect=(0, 0, 100, 100), label="Figure 1:",
                 caption="Figure 1: Test caption.", png_path=png if with_png else None)
    return Paper(pdf_path=tmp / "paper.pdf", title="Test Paper", text="body",
                 figures=(fig,))


class TestDescribeFigures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _config(self):
        return {}

    def test_describes_figure_with_image_part(self):
        paper = _make_paper(self.root)
        client = _FakeClient("- Chart shows AUC 0.85 vs baseline 0.83")
        out = describe_figures(paper, client=client, load_config=self._config)

        self.assertEqual(out.figures[0].description, "- Chart shows AUC 0.85 vs baseline 0.83")
        # Multimodal message: text part + image_url part with base64 PNG
        msg = client.calls[0]["messages"][0]
        types = [p["type"] for p in msg["content"]]
        self.assertEqual(types, ["text", "image_url"])
        self.assertIn("data:image/png;base64,", msg["content"][1]["image_url"]["url"])

    def test_api_failure_falls_back_to_placeholder(self):
        paper = _make_paper(self.root)
        client = _FakeClient(RuntimeError("model does not support images"))
        out = describe_figures(paper, client=client, load_config=self._config)
        self.assertEqual(out.figures[0].description, NO_VISION_PLACEHOLDER)

    def test_missing_png_falls_back_to_placeholder(self):
        paper = _make_paper(self.root, with_png=False)
        client = _FakeClient("- should not be called")
        out = describe_figures(paper, client=client, load_config=self._config)
        self.assertEqual(out.figures[0].description, NO_VISION_PLACEHOLDER)
        self.assertEqual(client.calls, [])


class TestCacheV3(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache_path = self.root / "cache.json"
        self.pdf = self.root / "paper.pdf"
        self.pdf.write_bytes(b"%PDF-fake-bytes")

    def test_round_trip_with_figures(self):
        png = self.root / "fig_p0_0.png"
        png.write_bytes(b"png")
        paper = Paper(
            pdf_path=self.pdf, title="T", text="text",
            figures=(Figure(page=2, rect=(1.0, 2.0, 3.0, 4.0), label="Figure 5:",
                            caption="Figure 5: Cap.", png_path=png),),
        )
        cache = PaperCache(self.cache_path)
        cache.store(paper)
        cache.save()

        cache2 = PaperCache(self.cache_path)
        got = cache2.get_cached(self.pdf)
        self.assertIsNotNone(got)
        self.assertEqual(got.text, "text")
        self.assertEqual(len(got.figures), 1)
        fig = got.figures[0]
        self.assertEqual(fig.page, 2)
        self.assertEqual(fig.rect, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(fig.label, "Figure 5:")
        self.assertEqual(fig.caption, "Figure 5: Cap.")
        self.assertEqual(fig.png_path, png)
        self.assertIsNone(fig.description)  # descriptions never cached

    def test_missing_crop_invalidates_entry(self):
        png = self.root / "fig_p0_0.png"
        png.write_bytes(b"png")
        paper = Paper(
            pdf_path=self.pdf, title="T", text="text",
            figures=(Figure(page=0, rect=(0, 0, 1, 1), png_path=png),),
        )
        cache = PaperCache(self.cache_path)
        cache.store(paper)
        cache.save()

        png.unlink()
        cache2 = PaperCache(self.cache_path)
        self.assertIsNone(cache2.get_cached(self.pdf))

    def test_v2_cache_invalidated(self):
        self.cache_path.write_text(
            '{"version": 2, "papers": {"paper.pdf": {"hash": "x", "title": "T", "text": "t"}}}',
            encoding="utf-8",
        )
        cache = PaperCache(self.cache_path)
        self.assertIsNone(cache.get_cached(self.pdf))


if __name__ == "__main__":
    unittest.main()
