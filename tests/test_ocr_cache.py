"""OCR memoization: identical pixels hit the cache, changed pixels miss."""
import numpy as np

from poke_hunter import ocr


def test_cache_hits_identical_and_misses_changed(monkeypatch):
    calls = []

    def fake(img, psm, whitelist, upscale, threshold):
        calls.append(1)
        return ocr.TextResult("X", "X", 90.0)

    monkeypatch.setattr(ocr, "_read_text_uncached", fake)
    ocr.clear_ocr_cache()
    a = np.zeros((10, 20, 3), np.uint8)
    b = a.copy()
    b[0, 0, 0] = 255
    ocr.read_text(a)
    ocr.read_text(a.copy())
    assert len(calls) == 1
    ocr.read_text(b)
    assert len(calls) == 2
    ocr.read_text(a, psm=ocr.PSM_BLOCK)
    assert len(calls) == 3


def test_failed_results_not_cached(monkeypatch):
    calls = []

    def fake(img, psm, whitelist, upscale, threshold):
        calls.append(1)
        return ocr.TextResult("", "", -1.0)

    monkeypatch.setattr(ocr, "_read_text_uncached", fake)
    ocr.clear_ocr_cache()
    a = np.zeros((5, 5), np.uint8)
    ocr.read_text(a)
    ocr.read_text(a)
    assert len(calls) == 2
