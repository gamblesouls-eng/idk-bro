"""Window-targeted mouse and keyboard input.

All coordinates are window-relative; conversions are applied per call and the
window is verified + focused beforehand, so input cannot reach another client.
Uses pydirectinput (SendInput with scan codes) — verified to work with the
Roblox client, where pyautogui's synthetic clicks are silently swallowed.
"""
from __future__ import annotations

import random
import time
from typing import Tuple

import pydirectinput

from .logging_setup import get_logger
from .window import GameWindow

log = get_logger("input")

pydirectinput.PAUSE = 0.02


class InputError(Exception):
    pass


def _abs_point(window: GameWindow, wx: int, wy: int) -> Tuple[int, int]:
    x0, y0 = window.client_origin()
    return x0 + int(wx), y0 + int(wy)


def focus(window: GameWindow) -> None:
    if not window.is_foreground():
        window.focus()
        time.sleep(0.15)


def click(window: GameWindow, x: int, y: int, verify_enabled: bool = True) -> None:
    """Single left click at a window-relative coordinate."""
    try:
        window._validate()
    except Exception as exc:  # noqa: BLE001
        raise InputError(f"window invalid before click: {exc}")
    focus(window)
    ax, ay = _abs_point(window, x, y)
    jitter = 2
    ax += random.randint(-jitter, jitter)
    ay += random.randint(-jitter, jitter)
    pydirectinput.moveTo(ax, ay)
    time.sleep(0.05)
    pydirectinput.click(ax, ay)
    log.debug("click @ window(%s,%s) -> screen(%s,%s)", x, y, ax, ay)


def click_region_center(window: GameWindow, region: Tuple[int, int, int, int]) -> None:
    x, y, w, h = region
    click(window, x + w // 2, y + h // 2)


# Window-relative inert parking spot: black letterbox band right of the Mute
# button, above the game canvas. No buttons live there; a stray press while
# parked cannot hit Run/Fight/Switch (user-reported ready-fire hazard: the
# cursor used to rest on the Run button between actions).
PARK_POINT = (1850, 120)


def park(window: GameWindow) -> None:
    """Move (never click) the cursor to the inert park point."""
    try:
        window._validate()
        ax, ay = _abs_point(window, *PARK_POINT)
        pydirectinput.moveTo(ax, ay)
    except Exception as exc:  # noqa: BLE001
        log.debug("park skipped: %s", exc)


def type_text(window: GameWindow, text: str, interval: float = 0.045) -> None:
    """Type via DirectInput scan codes (works with Roblox text fields)."""
    focus(window)
    pydirectinput.typewrite(text, interval=interval)
    log.debug("typed %r (%d chars)", text, len(text))


def press_key(window: GameWindow, key: str, times: int = 1) -> None:
    focus(window)
    for _ in range(times):
        pydirectinput.press(key)
        time.sleep(0.05)


def select_all_and_delete(window: GameWindow, sweep: bool = True) -> None:
    """Ctrl+A + DEL, then a backspace sweep, inside a focused field: removes
    ALL existing text. The sweep is the safety net — a Roblox field that
    ignores Ctrl+A or swallows the single DEL is still guaranteed emptied
    (backspace nukes the whole content char by char). ``sweep=False`` skips
    that slow fallback; callers that verify emptiness enable it on retry."""
    focus(window)
    pydirectinput.keyDown("ctrl")
    time.sleep(0.05)
    pydirectinput.press("a")
    time.sleep(0.06)
    pydirectinput.keyUp("ctrl")
    time.sleep(0.06)
    pydirectinput.press("delete")
    time.sleep(0.15)
    if not sweep:
        return
    for _ in range(34):
        pydirectinput.press("backspace")
        time.sleep(0.02)
    time.sleep(0.25)


def pause(seconds: float) -> None:
    time.sleep(seconds)
