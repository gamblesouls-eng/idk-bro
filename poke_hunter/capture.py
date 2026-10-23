"""Screen capture via mss, window-relative coordinates."""
from __future__ import annotations

import hashlib
import time
from typing import Optional, Tuple

import cv2
import mss
import numpy as np

from .logging_setup import get_logger
from .util import crop_region
from .window import GameWindow, WindowNotFound

log = get_logger("capture")

Region = Tuple[int, int, int, int]  # x, y, w, h relative to window client area


class CaptureError(Exception):
    pass


PW_RENDERFULLCONTENT = 2
_frame_contexts: dict = {}


def grab_window_content(window) -> "np.ndarray | None":
    """Capture one window's contents via Win32 PrintWindow.

    Advantages over screen-region capture:
      * works while the window is partially/fully covered,
      * cannot capture a different window's pixels,
      * multi-instance safe by construction (bound to the exact hwnd).
    Returns a BGR numpy array or None if the call failed.
    """
    try:
        import ctypes
        import win32ui
    except ImportError:
        return None
    hwnd = window.hwnd
    # client size in pixels
    rect = window.client_size()
    width, height = int(rect[0]), int(rect[1])
    if width <= 0 or height <= 0:
        return None
    # device context is cached per window for speed; recreate if size changed
    key = (hwnd, width, height)
    ctx = _frame_contexts.get(key)
    if ctx is None:
        try:
            import win32gui
            hwnd_dc = win32gui.GetWindowDC(hwnd)
        except Exception:
            return None
        mfc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc, width, height)
        save_dc.SelectObject(bmp)
        _frame_contexts[key] = (hwnd_dc, mfc, save_dc, bmp, width, height)
        ctx = _frame_contexts[key]
        # keep only one context per hwnd: drop stale sizes
        stale = [k for k in _frame_contexts if k[0] == hwnd and k != key]
        for k in stale:
            _frame_contexts.pop(k, None)
    _, _, save_dc, bmp, w, h = ctx
    result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
    if not result:
        return None
    data = bmp.GetBitmapBits(True)
    img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))
    return np.ascontiguousarray(img[:, :, :3])  # keep top-down orientation


class Grabber:
    """Captures one game window; regions are window-relative.

    Primary path is PrintWindow (works while the window is occluded); the
    whole-screen mss rectangle capture is the fallback (foreground only)."""

    def __init__(self, window: GameWindow):
        self.window = window
        self.last_full: Optional[np.ndarray] = None
        self.last_full_time: float = 0.0
        self.last_signature: Optional[str] = None
        # parallel aspect readers share this Grabber; PrintWindow/mss capture
        # and the last-frame cache must stay serialized
        from threading import Lock
        self._capture_lock = Lock()

    def frame(self, fresh: bool = False, min_age_ms: int = 0) -> np.ndarray:
        """Capture the whole game client area as a BGR numpy array.

        fresh=True forces a new hardware capture even if last frame is recent.
        Tries window-content capture (PrintWindow - safe while occluded),
        falls back to screen-region capture.
        """
        age_ms = (time.monotonic() - self.last_full_time) * 1000.0
        if (not fresh) and self.last_full is not None and age_ms < min_age_ms:
            return self.last_full
        with self._capture_lock:
            try:
                x, y, w, h = self.window.box()
            except WindowNotFound as exc:
                raise CaptureError(str(exc)) from exc
            if w <= 0 or h <= 0:
                raise CaptureError("game window client area is empty")
            img = grab_window_content(self.window)
            if img is not None and img.shape[0] == h and img.shape[1] == w:
                self.last_full = img
                self.last_full_time = time.monotonic()
                return img
            monitor = {"left": x, "top": y, "width": w, "height": h}
            with mss.mss() as sct:  # fresh context each time: avoids stale region caches
                try:
                    raw = sct.grab(monitor)
                except Exception as exc:  # noqa: BLE001
                    raise CaptureError(f"mss capture failed: {exc}") from exc
            img = np.asarray(raw, dtype=np.uint8)[:, :, :3]  # BGRA -> BGR
            img = np.ascontiguousarray(img)
            self.last_full = img
            self.last_full_time = time.monotonic()
            return img

    def crop(self, frame: np.ndarray, region: Region) -> np.ndarray:
        from .util import crop_region

        return crop_region(frame, region)

    def capture_region(self, region: Region, fresh: bool = False) -> np.ndarray:
        return self.crop(self.frame(fresh=fresh), region)

    def frame_signature(self, frame: Optional[np.ndarray] = None,
                        region: Optional[Region] = None) -> str:
        """Cheap perceptual signature for skip-identical-frame logic.

        With `region`, only that crop is hashed: regional stability lets the
        bot continue once the UI it actually reads has settled, even while
        background battle animations keep rendering."""
        frame = frame if frame is not None else self.last_full
        if frame is None:
            return ""
        if region is not None:
            from .util import clamp_crop, crop_region
            frame = crop_region(frame, region)
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        return hashlib.md5(small.tobytes()).hexdigest()

    @staticmethod
    def diff_changed(a: np.ndarray, b: np.ndarray, threshold: float = 8.0) -> bool:
        """True when two full frames clearly differ (mean abs diff > threshold)."""
        if a.shape != b.shape:
            return True
        d = cv2.absdiff(
            cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        )
        return float(d.mean()) > threshold


def save_debug_image(path: str, frame: np.ndarray, region: Optional[Region] = None) -> str:
    """Persist a diagnostic image (used when recognition repeatedly fails)."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if region is not None and all(0 <= v for v in region):
        img = crop_region(frame, region) if frame is not None else frame
    else:
        img = frame
    from .util import downscale_if_huge
    cv2.imwrite(path, downscale_if_huge(img))
    return path
