"""Test whether keyboard input reaches the bag search field, letter by letter."""
import time
import cv2

from poke_hunter.capture import Grabber
from poke_hunter.inputio import click, type_text
from poke_hunter.ocr import read_text
from poke_hunter.util import crop_region
from poke_hunter.window import find_windows

SEARCH_FIELD = (1075, 289, 445, 54)

w = find_windows(["Roblox"], 0, "calib")
grab = Grabber(w)


def field_text():
    frame = grab.frame(fresh=True)
    crop = crop_region(frame, SEARCH_FIELD)
    return read_text(crop, psm=7, whitelist=None), frame


res, frame = field_text()
print("before click:", res.value, res.raw)

click(w, 1200, 315)
time.sleep(0.5)
res, _ = field_text()
print("after click :", res.value, res.raw)

type_text(w, "Ab", interval=0.15)
time.sleep(0.8)
res, frame = field_text()
print("after typing:", res.value, res.raw, "conf", res.confidence)
cv2.imwrite("calib_shots/search_test_ab.png", frame)
