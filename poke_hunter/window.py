"""Game window identification (Roblox) via Win32."""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import ctypes.wintypes as wintypes  # alias used for readability

from .config import InstanceConfig
from .logging_setup import get_logger

log = get_logger("window")

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wt.HWND, wt.LPARAM)


class WindowNotFound(Exception):
    pass


class GameWindow:
    """A bounded reference to one game client window (client-area coordinates)."""

    def __init__(self, hwnd: int, title: str, name: str):
        self.hwnd = hwnd
        self.title = title
        self.name = name
        self._validate()

    def _validate(self) -> None:
        if not _user32.IsWindow(self.hwnd):
            raise WindowNotFound(f"window {self.hwnd:#x} is no longer valid")
        if _user32.IsIconic(self.hwnd):
            # restore minimized windows so client rect is meaningful
            _user32.ShowWindow(self.hwnd, 9)  # SW_RESTORE

    # -- geometry ----------------------------------------------------------
    def client_origin(self) -> tuple:
        rect = wt.RECT()
        if not _user32.GetClientRect(self.hwnd, ctypes.byref(rect)):
            raise WindowNotFound("GetClientRect failed")
        pt = wt.POINT(0, 0)
        _user32.ClientToScreen(self.hwnd, ctypes.byref(pt))
        return pt.x, pt.y

    def client_size(self) -> tuple:
        rect = wt.RECT()
        if not _user32.GetClientRect(self.hwnd, ctypes.byref(rect)):
            raise WindowNotFound("GetClientRect failed")
        return rect.right, rect.bottom  # width, height

    def box(self) -> tuple:
        """(left, top, width, height) of the client area in screen coordinates."""
        x, y = self.client_origin()
        w, h = self.client_size()
        return x, y, w, h

    def is_foreground(self) -> bool:
        return _user32.GetForegroundWindow() == self.hwnd

    def focus(self) -> None:
        try:
            _user32.SetForegroundWindow(self.hwnd)
        except Exception as exc:  # noqa: BLE001
            log.warning("SetForegroundWindow failed for %s: %s", self.name, exc)
        if _user32.IsIconic(self.hwnd):
            _user32.ShowWindow(self.hwnd, 9)

    def describe(self) -> str:
        x, y, w, h = self.box()
        return f"hwnd={self.hwnd:#x} title={self.title!r} box=({x},{y},{w},{h})"


def find_windows(patterns, instance_index: int = 0, name: str = "instance") -> GameWindow:
    """Find game windows whose title contains any of the given patterns.

    Returns the i-th match where i = instance_index, so multiple instances can
    be pinned by index. Raises WindowNotFound with a helpful message.
    """
    patterns = [p.lower() for p in patterns]
    matches: list = []

    @WNDENUMPROC
    def _cb(hwnd, _lparam):
        if not _user32.IsWindowVisible(hwnd):
            return True
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.lower()
        if any(p in title for p in patterns):
            matches.append((hwnd, buf.value))
        return True

    _user32.EnumWindows(_cb, 0)
    if not matches:
        raise WindowNotFound(
            f"No window matching {patterns}. Is the game running and visible "
            f"(not only in the taskbar)?"
        )
    if instance_index >= len(matches):
        raise WindowNotFound(
            f"Requested window_index {instance_index}, but only {len(matches)} "
            f"matching window(s) found: {[m[1] for m in matches]}"
        )
    hwnd, title = matches[instance_index]
    gw = GameWindow(hwnd, title, name)
    log.info("Identified window: %s", gw.describe())
    return gw
