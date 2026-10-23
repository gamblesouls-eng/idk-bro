"""Verified bag navigation test with click-verify-retry and clipboard paste.
Targets the real UI measured on calib_shots screenshots.
"""
import time
import cv2
import win32clipboard
import pydirectinput

from poke_hunter.capture import Grabber
from poke_hunter.inputio import click
from poke_hunter.ocr import read_text
from poke_hunter.util import crop_region
from poke_hunter.window import find_windows

pydirectinput.PAUSE = 0.05

# measured from real screenshots (window-relative)
BAG_BUTTON = (984, 876, 347, 89)
BACKPACK_TITLE = (388, 178, 340, 80)   # 'Backpack' mint band incl. padding
TABS = {
    "recovers": (420, 310, 130, 60),
    "pokeballs": (420, 375, 130, 60),
}
SEARCH_FIELD = (1075, 289, 445, 54)
CLOSE_BUTTON = (1270, 183, 250, 75)

WINDOW_PATTERNS = ["Roblox"]


def ocr(img, region, psm=7):
    crop = crop_region(img, region)
    return read_text(crop, psm=psm)


def click_and_verify(window, grab, region, expect_region, expect_text,
                     tries=4, settle=1.0, name=""):
    """Click center of region; verify expected text appeared in expect_region."""
    for attempt in range(1, tries + 1):
        click(window, region[0] + region[2] // 2, region[1] + region[3] // 2)
        time.sleep(settle)
        frame = grab.frame(fresh=True)
        res = ocr(frame, expect_region)
        hit = expect_text in res.raw.lower()
        print(f"{name} click try {attempt}: got {res.raw!r} -> {'OK' if hit else 'no'}")
        if hit:
            return frame, True
    return grab.frame(fresh=True), False


def main():
    window = find_windows(WINDOW_PATTERNS, 0, "calib")
    grab = Grabber(window)

    # 1. open Bag, verified by 'Backpack' title
    frame, ok = click_and_verify(window, grab, BAG_BUTTON, BACKPACK_TITLE,
                                 "backpack", name="bag")
    if not ok:
        cv2.imwrite("calib_shots/debug_bag_fail.png", frame)
        return 3

    # 2. switch to Pokeballs tab, verified by 'Pokeball' header
    header = (560, 280, 260, 70)
    frame, ok = click_and_verify(window, grab, TABS["pokeballs"], header,
                                 "pokeball", name="pokeballs-tab")
    if not ok:
        cv2.imwrite("calib_shots/debug_tab_fail.png", frame)
        return 3

    # 3. select search field and paste with verification of field contents
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardText("Repeat Ball", win32clipboard.CF_UNICODETEXT)
    win32clipboard.CloseClipboard()

    field_ok = None
    for attempt in range(1, 4):
        click(window, 1200, 315)
        time.sleep(0.4)
        pydirectinput.keyDown("ctrl")
        time.sleep(0.12)
        pydirectinput.press("v")
        time.sleep(0.12)
        pydirectinput.keyUp("ctrl")
        time.sleep(0.9)
        frame = grab.frame(fresh=True)
        res = ocr(frame, SEARCH_FIELD)
        print(f"paste try {attempt}: field={res.raw!r} conf={res.confidence:.0f}")
        if "repeat" in res.raw.lower() or "ball" in res.raw.lower():
            field_ok = True
            break
    cv2.imwrite("calib_shots/verified_catch_flow.png", frame)
    print("final:", "field verified" if field_ok else "field NOT verified")
    return 0 if field_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
