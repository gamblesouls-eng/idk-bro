"""Small shared utilities (kept dependency-light)."""
from __future__ import annotations

import re
from typing import Tuple

import cv2
import numpy as np

Region = Tuple[int, int, int, int]


def clamp_crop(frame: np.ndarray, region: Region) -> np.ndarray:
    return crop_region(frame, region)


def crop_region(frame: np.ndarray, region: Region) -> np.ndarray:
    """Crop a window-relative region out of a frame, clamped to bounds."""
    x, y, w, h = region
    fh, fw = frame.shape[:2]
    x = max(0, min(int(x), fw - 1))
    y = max(0, min(int(y), fh - 1))
    w = max(1, min(int(w), fw - x))
    h = max(1, min(int(h), fh - y))
    return frame[y : y + h, x : x + w]


def downscale_if_huge(img: np.ndarray, max_side: int = 1200) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    if side > max_side:
        scale = max_side / side
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize harmless OCR formatting differences (case, whitespace)."""
    if text is None:
        return ""
    t = text.replace("\u2022", " ").replace("\u2019", "'").replace("\n", " ")
    t = _WS_RE.sub(" ", t).strip().lower()
    return t


def fuzzy_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def close_match(observed: str, expected: str, fuzzy_threshold: float = 0.88) -> bool:
    """Exact match first; fuzzy only above a conservative threshold.

    Prevents substituting a similar-but-different species name.
    """
    a, b = normalize_text(observed), normalize_text(expected)
    if not a:
        return False
    if a == b:
        return True
    return fuzzy_ratio(a, b) >= fuzzy_threshold
