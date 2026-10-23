"""Tesseract OCR wrapper with per-region settings, confidence and retries."""
from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections import OrderedDict
from threading import Lock
from dataclasses import dataclass
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


def _prep(img: np.ndarray, upscale: int = 3) -> np.ndarray:
    """Upscale + grayscale: small UI labels OCR much better enlarged."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if upscale > 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return img


_OCR_CACHE: "OrderedDict[tuple, TextResult]" = OrderedDict()
_OCR_CACHE_MAX = 256
_OCR_CACHE_LOCK = Lock()


def clear_ocr_cache() -> None:
    with _OCR_CACHE_LOCK:
        _OCR_CACHE.clear()


def read_text(img: np.ndarray, psm: int = PSM_SINGLE_LINE,
              whitelist: Optional[str] = None, upscale: int = 3,
              threshold: Optional[int] = None) -> TextResult:
    """OCR one small region (memoized on the exact pixels + settings).

    Identical crops (idle panels, repeated stability re-reads) skip the
    Tesseract subprocess entirely; any pixel change misses the cache."""
    key = (img.shape, hashlib.blake2b(np.ascontiguousarray(img).tobytes(),
                                      digest_size=16).digest(),
           psm, whitelist, upscale, threshold)
    with _OCR_CACHE_LOCK:
        hit = _OCR_CACHE.get(key)
        if hit is not None:
            _OCR_CACHE.move_to_end(key)
            return hit
    res = _read_text_uncached(img, psm, whitelist, upscale, threshold)
    if res.confidence >= 0:  # never cache failed/empty-engine results
        with _OCR_CACHE_LOCK:
            _OCR_CACHE[key] = res
            while len(_OCR_CACHE) > _OCR_CACHE_MAX:
                _OCR_CACHE.popitem(last=False)
    return res


def _read_text_uncached(img: np.ndarray, psm: int, whitelist: Optional[str],
                        upscale: int, threshold: Optional[int]) -> TextResult:
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
                _prep(img, upscale), config=" ".join(args),
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractError:  # bad psm/whitelist combo -> retry plain
            if len(cfgs) > 1:
                try:
                    args = ["--psm", str(psm)]
                    data = pytesseract.image_to_data(
                        _prep(img, upscale), config=" ".join(args),
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
