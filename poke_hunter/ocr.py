"""Tesseract OCR wrapper with per-region settings, confidence and retries."""
from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytesseract

from .logging_setup import get_logger
from .util import normalize_text

log = get_logger("ocr")

# Page segmentation modes used by our regions
PSM_SINGLE_LINE = 7
PSM_SINGLE_WORD = 8
PSM_SPARSE_TEXT = 11
PSM_BLOCK = 6


class OCRInitError(Exception):
    pass


def verify_tesseract() -> str:
    """Check Tesseract is reachable through PATH; return its version string."""
    exe = shutil.which("tesseract")
    if not exe:
        raise OCRInitError(
            "Tesseract executable not found on PATH. Install it or add its "
            "install directory to PATH."
        )
    try:
        version = pytesseract.get_tesseract_version()
    except Exception as exc:  # noqa: BLE001
        raise OCRInitError(f"Tesseract found ({exe}) but could not be executed: {exc}")
    return f"Tesseract {version} at {exe}"


@dataclass
class TextResult:
    value: str            # normalized text ("" if nothing readable)
    raw: str              # raw OCR text (diagnostics)
    confidence: float     # mean tesseract confidence, -1 when unavailable


def _prep(img: np.ndarray, upscale: int = 3,
          color: Optional[str] = None) -> np.ndarray:
    """Preprocess a small UI crop for Tesseract.

    luma (default): BGR2GRAY + upscale, INTER_LANCZOS4 for integer scales —
    cubic reached its ringing limit on glyph edges (squiggle artifacts).
    color "max"/"min": the chip invariants. Luma (0.299R 0.587G 0.114B)
    collapses saturated lettering onto the dark chip box ('Frawn' cyan ends
    up ~19 luma levels off the box) — squiggle bait; the min channel keeps
    darker-than-box lettering. color "chip" is the production mode:
    adaptive difference from the crop's median V captures bright AND dark
    lettering in one pass, margins remove the box frame bars.
    """
    if color == "chip":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2].astype(np.int16)
        box_v = np.median(v)   # the box dominates the crop's histogram
        mask = (np.abs(v - box_v) >= 26).astype(np.uint8) * 255
        # kill the box frame bars: they border the zone edges, letters are
        # central - unsigned black lines around glyphs ruin PSM 7 (the min/
        # max modes for comparison read 0/92 real chips without this)
        mask[:8, :] = 0
        mask[-6:, :] = 0
        mask[:, :10] = 0
        mask[:, -10:] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)))
        canvas = np.where(mask > 0, 0, 255).astype(np.uint8)
        if upscale != 1:
            canvas = cv2.resize(canvas, None, fx=upscale, fy=upscale,
                                interpolation=cv2.INTER_NEAREST)
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
        return canvas
    if color in ("max", "min"):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2]
        if color == "max":
            mask = (v >= 165).astype(np.uint8) * 255   # bright lettering
        else:
            mask = (v <= 96).astype(np.uint8) * 255    # dark lettering
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)))
        canvas = np.where(mask > 0, 0, 255).astype(np.uint8)
        if upscale != 1:
            canvas = cv2.resize(canvas, None, fx=upscale, fy=upscale,
                                interpolation=cv2.INTER_NEAREST)
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
        return canvas
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if upscale > 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_LANCZOS4)
    return img


# Memoized OCR: identical crops (stability re-reads, repeated panel OCR)
# skip the Tesseract subprocess entirely; any pixel change misses.
_OCR_CACHE: "OrderedDict[tuple, TextResult]" = OrderedDict()
_OCR_CACHE_MAX = 256
_OCR_CACHE_LOCK = Lock()


def clear_ocr_cache() -> None:
    with _OCR_CACHE_LOCK:
        _OCR_CACHE.clear()


def read_text(img: np.ndarray, psm: int = PSM_SINGLE_LINE,
              whitelist: Optional[str] = None, upscale: int = 3,
              threshold: Optional[int] = None,
              color: Optional[str] = None) -> TextResult:
    """OCR one small region (memoized on exact pixels + settings).

    `color` (None | "max" | "min") selects the preprocessing in `_prep`;
    chip readers pass a channel mode because luma merges colored lettering
    into the dark box. Failed/empty-engine results are never cached."""
    key = (img.shape,
           hashlib.blake2b(np.ascontiguousarray(img).tobytes(),
                           digest_size=16).digest(),
           psm, whitelist, upscale, threshold, color)
    with _OCR_CACHE_LOCK:
        hit = _OCR_CACHE.get(key)
        if hit is not None:
            _OCR_CACHE.move_to_end(key)
            return hit

    cfgs: List[Tuple[str, List[str]]] = []
    flags = ["--psm", str(psm)]
    extra = []
    if whitelist:
        extra = ["-c", f"tessedit_char_whitelist={whitelist}"]
    cfgs.append((flags + extra, []))

    # Second fallback attempt without any whitelist (whitelists can hurt when
    # the game font renders odd glyphs).
    if whitelist:
        cfgs.append((list(flags), []))

    last = TextResult("", "", -1.0)
    for extra_args, _ in cfgs:
        args = extra_args if extra_args else []
        try:
            data = pytesseract.image_to_data(
                _prep(img, upscale, color), config=" ".join(args),
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractError:  # bad psm/whitelist combo -> retry plain
            if len(cfgs) > 1:
                try:
                    args = ["--psm", str(psm)]
                    data = pytesseract.image_to_data(
                        _prep(img, upscale, color), config=" ".join(args),
                        output_type=pytesseract.Output.DICT,
                    )
                except Exception:  # noqa: BLE001
                    continue
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            log.warning("OCR call failed: %s", exc)
            return last

        words, confs = [], []
        for token, conf in zip(data["text"], data["conf"]):
            t = (token or "").strip()
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1.0
            if t and c >= 0:
                words.append(t)
                confs.append(c)
        if words:
            # Tesseract hallucinates short tokens on blank/ambiguous regions;
            # suppress single low-confidence tokens entirely.
            mean_conf = sum(confs) / len(confs)
            confident = [w for w, c in zip(words, confs) if c >= 40]
            if len(confident) == 1 and mean_conf < 60:
                confident = []
            if not confident and mean_conf < 40:
                continue  # try the next OCR configuration instead of failing fast
            last = TextResult(
                value=normalize_text(" ".join(confident or words)),
                raw=" ".join(words),
                confidence=mean_conf,
            )
            with _OCR_CACHE_LOCK:
                _OCR_CACHE[key] = last
                while len(_OCR_CACHE) > _OCR_CACHE_MAX:
                    _OCR_CACHE.popitem(last=False)
            return last
    return last


def read_stable(grab, region, psm: int = PSM_SINGLE_LINE,
                whitelist: Optional[str] = None, attempts: int = 3,
                retry_delay_s: float = 0.45, expect_change: bool = False) -> Tuple[str, bool]:
    """Read a region repeatedly until OCR output is consistent.

    Returns (value, stable). value=="" with stable=True means the region was
    consistently blank. stable=False means unstable / unreliable (unknown).
    """
    seen: List[str] = []
    for attempt in range(max(1, attempts)):
        img = grab.capture_region(region, fresh=True)
        res = read_text(img, psm=psm, whitelist=whitelist)
        seen.append(res.value)
        if len(seen) >= 2 and seen[-1] == seen[-2] and len(set(seen)) == 1:
            return seen[0], True  # unanimous so far
        if expect_change and seen[-1] != "":
            break
        time.sleep(retry_delay_s)
    from collections import Counter
    counter = Counter(seen)
    value, count = counter.most_common(1)[0]
    stable = count == len(seen)
    return value, stable


def dump_candidates(img: np.ndarray, region_name: str) -> List[str]:
    """Debug helper: what does plain unfiltered OCR think about this region?"""
    return [
        pytesseract.image_to_string(_prep(img), config=f"--psm {p}").strip()
        for p in (PSM_SINGLE_LINE, PSM_BLOCK, PSM_SPARSE_TEXT)
    ]
