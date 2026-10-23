"""OCR wrapper tests against Tesseract with synthetic text images."""
import numpy as np
import pytest
import cv2

pytesseract = pytest.importorskip("pytesseract")

from poke_hunter.ocr import PSM_BLOCK, PSM_SINGLE_LINE, read_stable, read_text, verify_tesseract
from poke_hunter.util import close_match, normalize_text


def text_image(text, size=(360, 60), fontscale=1.6, color=(0, 0, 0), bg=(255, 255, 255)):
    img = np.full((size[1], size[0], 3), bg, dtype=np.uint8)
    cv2.putText(img, text, (18, 44), cv2.FONT_HERSHEY_SIMPLEX,
                fontscale, color, 2, cv2.LINE_AA)
    return img


class StaticGrabber:
    def __init__(self, img):
        self.img = img

    def capture_region(self, region, fresh=False):
        return self.img.copy()


def test_tesseract_available():
    assert "Tesseract" in verify_tesseract()


def test_species_ocr():
    res = read_text(text_image("Quaxly"), psm=PSM_SINGLE_LINE)
    assert res.value in ("quaxly", "quaxty") and len(res.value) >= 5


def test_reskin_ocr_case_normalization():
    res = read_text(text_image("Inferno"), psm=PSM_SINGLE_LINE)
    assert res.value == "inferno"


def test_empty_region_is_empty_and_stable():
    grab = StaticGrabber(text_image("", color=(255, 255, 255)))
    value, stable = read_stable(grab, (0, 0, 100, 40), attempts=2, retry_delay_s=0.0)
    assert value == "" and stable is True


def test_unstable_reads_marked_unknown():
    frames = [text_image("Cinder"), text_image("Glacier"), text_image("Cinder"),
              text_image("Glacier")]
    grab = StaticGrabber(frames[0])
    seq = []
    real_capture = grab.capture_region
    idx = {"i": 0}

    def cycling(region, fresh=False):
        img = frames[idx["i"] % len(frames)]
        idx["i"] += 1
        return img

    grab.capture_region = cycling
    value, stable = read_stable(grab, (0, 0, 360, 60), attempts=3, retry_delay_s=0.0)
    assert stable is False


def test_whitelist_numbers():
    res = read_text(text_image("Lv. 150"), psm=PSM_SINGLE_LINE,
                    whitelist="0123456789.")
    assert "150" in res.raw or "150" in res.value


def test_close_match_boundaries():
    assert close_match("  Quaxly ", "quaxly")
    assert close_match("quaxlyy", "quaxly")   # 1-char OCR artifact
    assert not close_match("mudkip", "quaxly")


def test_normalize_text():
    assert normalize_text("  Inferno\n ") == "inferno"
    assert normalize_text(None) == ""
