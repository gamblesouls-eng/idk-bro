"""Test mouse input mechanisms against the live game with verification.

Mechanisms:
    A) PostMessage WM_LBUTTONDOWN/UP directly to the Roblox window
       (no foreground needed, no synthetic cursor events)
    B) hardware-style SendInput via pydirectinput.mouse
Each click is verified by an expected OCR change before it counts.
"""
import time
import cv2
import win32api
import win32con

from poke_hunter.capture import Grabber
from poke_hunter.ocr import read_text
from poke_hunter.util import crop_region
from poke_hunter.window import find_windows

BAG_BUTTON = (984, 876, 347, 89)
BACKPACK_TITLE = (388, 170, 350, 90)


def doclick_postmessage(hwnd, x, y):
    lparam = (int(y) << 16) | int(x)
    win32api.SendMessage(hwnd, win32con.WM_MOUSEACTIVATE, win32con.NULL, 0)
    win32api.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
    time.sleep(0.05)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
    time.sleep(0.05)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lparam)


def try_postmessage(window, grab):
    for i in range(3):
        doclick_postmessage(window.hwnd, 1157, 920)
        time.sleep(1.1)
        frame = grab.frame(fresh=True)
        res = read_text(crop_region(frame, BACKPACK_TITLE), psm=7)
        print(f"post message try {i + 1}: {res.raw!r}")
        if "backpack" in res.raw.lower():
            return True, frame
    return False, grab.frame(fresh=True)


def try_sendinput(window, grab):
    import pydirectinput
    pydirectinput.PAUSE = 0.1
    for i in range(2):
        pydirectinput.click(1157, 920)
        time.sleep(1.1)
        frame = grab.frame(fresh=True)
        res = read_text(crop_region(frame, BACKPACK_TITLE), psm=7)
        print(f"sendinput try {i + 1}: {res.raw!r}")
        if "backpack" in res.raw.lower():
            return True, frame
    return False, grab.frame(fresh=True)


if __name__ == "__main__":
    w = find_windows(["Roblox"], 0, "input")
    grab = Grabber(w)
    print("== A: PostMessage ==")
    ok, frame = try_postmessage(w, grab)
    cv2.imwrite("calib_shots/pm_result.png", frame)
    if not ok:
        print("== B: SendInput ==")
        ok2, frame2 = try_sendinput(w, grab)
        cv2.imwrite("calib_shots/si_result.png", frame2)
    print("postmessage:", "OK" if ok else "failed")
