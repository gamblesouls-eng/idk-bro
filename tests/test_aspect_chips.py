"""Aspect chip reader tests: spec parsing, slot gates, calibration wiring.

Ground truth comes from the 2026-10-22 calibration night:
  * reskin slot: bright italic letters on the streaky panel; streaks alone
    must not trip (shape-filter drops full-width thin lines).
  * gradient slot: a dark chip box (v ~121) renders only while a chip is
    shown; chipless backgrounds (v ~202) must not trip. The dark box is
    the ONLY presence gate: it is part of the chip graphic itself and
    renders in every biome, while the slot otherwise shows the biome
    background (saturated grass tripped the old sat-letters fallback).
"""
import json
import os

import cv2
import numpy as np
import pytest

from poke_hunter.config import (AspectChipSpec, Calibration, ConfigError,
                                _parse_aspect_chips)
from poke_hunter.detectors import detect_color_chip, detect_reskin_letters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hsv_img(hsv):
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _panel_base(w=380, h=42):
    """Panel-gray crop: v=143, low saturation."""
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 25
    hsv[..., 2] = 143
    return hsv


def _draw_letters(rng, hsv):
    """Bright letter-like blobs: small, disjoint, not full-width."""
    for _ in range(60):
        x = int(rng.integers(4, hsv.shape[1] - 10))
        y = int(rng.integers(4, hsv.shape[0] - 26))
        w = int(rng.integers(5, 26))
        h = int(rng.integers(12, 30))
        hsv[y:y + h, x:x + w, 2] = 190
        hsv[y:y + h, x:x + w, 1] = 20
    return hsv


def _draw_streaks(hsv):
    """Panel streak texture: thin full-width bright lines."""
    for y in (6, 18, 30):
        hsv[y:y + 2, :, 2] = 165
        hsv[y:y + 2, :, 1] = 30
    return hsv


@pytest.fixture
def spec():
    return AspectChipSpec(reskin_zone=(0, 0, 380, 42),
                          gradient_zone=(0, 0, 380, 44))


def test_color_chip_prep_draws_chip_letters_black_on_white():
    """Production chip-OCR mode (_prep color='chip')."""
    from poke_hunter.ocr import _prep
    bgr = np.full((44, 380, 3), 121, dtype=np.uint8)     # box v=121
    bgr[30:36, 180:220] = (255, 255, 60)                 # letter pixels, v~198
    bgr[16:20, 100:130] = (40, 40, 40)                   # dark letter pixels
    canvas = _prep(bgr, 1, color="chip")
    # letters -> black glyphs on a white background
    assert canvas.shape[:2] == (44, 380)
    dark = (canvas == 0)
    assert dark[31:34, 185:215].sum() >= 10   # bright letter captured
    assert dark[16:20, 105:125].sum() >= 5    # dark letter captured too
    assert not dark[5, 0:].any()              # frame bar area stays white (margin cut)
    # box interior -> white
    assert not dark[38, 200:260].any()


def test_read_text_caches_identical_crops_by_pixels():
    """Memoization: identical pixels skip the Tesseract subprocess."""
    imgs = []
    canv = np.full((44, 300), 255, np.uint8)
    cv2.putText(canv, "repeatable", (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 3, cv2.LINE_AA)
    from poke_hunter import ocr
    ocr.clear_ocr_cache()
    img = cv2.cvtColor(canv, cv2.COLOR_GRAY2BGR)
    first = ocr.read_text(img, psm=7)
    second = ocr.read_text(img.copy(), psm=7)
    assert first.value == second.value
    # cache stores >= 1 entry and returns the same object
    assert ocr._OCR_CACHE
    third = ocr.read_text(img.copy(), psm=7)
    assert third is second
    # failed reads are not cached: empty engine result stays uncached
    blank = np.full((40, 200, 3), 128, np.uint8)
    _ = ocr.read_text(blank, psm=7)
def test_reskin_letters_trip_on_blobs(spec):
    rng = np.random.default_rng(7)
    hsv = _draw_letters(rng, _panel_base())
    crop = _hsv_img(hsv)
    present, stats = detect_reskin_letters(crop, spec)
    assert present
    assert stats["letter_pixels"] >= spec.reskin_min_pixels
    # canvas: letters turned black on white
    canvas = stats["canvas"]
    assert canvas.dtype == np.uint8 and canvas.max() == 255
    assert (canvas == 0).sum() >= spec.reskin_min_pixels


def test_reskin_letters_reject_streaks_only(spec):
    hsv = _draw_streaks(_panel_base())
    crop = _hsv_img(hsv)
    present, stats = detect_reskin_letters(crop, spec)
    assert not present
    assert stats["letter_pixels"] == 0


def test_color_chip_trips_on_dark_box(spec):
    hsv = np.zeros((44, 380, 3), dtype=np.uint8)
    hsv[..., 1] = 25
    hsv[..., 2] = 202                      # battle background
    hsv[6:38, 20:360, 2] = 121             # the dark chip box
    hsv[6:38, 20:360, 0] = 5
    hsv[6:38, 20:360, 1] = 35
    present, stats = detect_color_chip(_hsv_img(hsv), spec)
    assert present
    assert stats["dark_pixels"] >= spec.dark_box_min_pixels


def test_color_chip_rejects_chipless_background(spec):
    hsv = np.zeros((44, 380, 3), dtype=np.uint8)
    hsv[..., 1] = 25
    hsv[..., 2] = 202
    hsv[:, :, 0] = 130                     # lavender pastel
    present, stats = detect_color_chip(_hsv_img(hsv), spec)
    assert not present
    assert stats["dark_pixels"] < spec.dark_box_min_pixels


def test_color_chip_ignores_saturated_letters_without_box(spec):
    """Biome invariant: saturated lettering alone must not trip.

    The 2026-10-23 grass-biome false MATCH came from the old
    sat-letters fallback: the green grass below the panel tripped the
    sat gate with no chip present, labeling `gradient-present` and
    matching a reskin-only wild. Only the dark box gates presence.
    """
    hsv = np.zeros((44, 380, 3), dtype=np.uint8)
    hsv[..., 1] = 25
    hsv[..., 2] = 202
    hsv[30:36, 100:280, 1] = 160           # saturated lettering, no box
    hsv[30:36, 100:280, 2] = 120
    hsv[30:36, 100:280, 0] = 5
    present, stats = detect_color_chip(_hsv_img(hsv), spec)
    assert not present
    assert stats["dark_pixels"] < spec.dark_box_min_pixels


def test_parse_aspect_chips_full_roundtrip():
    raw = {
        "reskin_zone": [480, 354, 380, 42],
        "gradient_zone": [420, 398, 380, 44],
        "reskin_min_pixels": 180, "grad_min_pixels": 430,
        "dark_box_min_pixels": 6000, "wait_ms": 2500, "poll_ms": 300,
    }
    spec = _parse_aspect_chips(raw)
    assert spec.reskin_zone == (480, 354, 380, 42)
    assert spec.gradient_zone == (420, 398, 380, 44)
    assert spec.dark_box_min_pixels == 6000
    assert spec.wait_ms == 2500


def test_parse_aspect_chips_missing_or_bad():
    assert _parse_aspect_chips(None) is None
    assert _parse_aspect_chips({}) is None
    with pytest.raises(ConfigError):
        _parse_aspect_chips({"reskin_zone": [1, 2]})
    with pytest.raises(ConfigError):
        _parse_aspect_chips({"reskin_zone": [1, 2, 3, 4]})


def test_calibration_json_has_aspect_chips():
    path = os.path.join(ROOT, "config", "calibration.json")
    calib = load_calibration_safe(path)
    assert calib.aspect_chips is not None
    assert calib.aspect_chips.reskin_zone == (480, 354, 380, 42)
    assert calib.aspect_chips.gradient_zone == (420, 398, 380, 44)
    assert calib.aspect_chips.dark_box_min_pixels == 6000


def load_calibration_safe(path):
    from poke_hunter.config import load_calibration
    return load_calibration(path)
