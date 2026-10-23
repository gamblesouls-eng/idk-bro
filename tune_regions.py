"""Tune region crops by scoring OCR confidence over candidate rectangles."""
import cv2
import numpy as np

from poke_hunter.ocr import read_text
from poke_hunter.util import crop_region

img = cv2.imread("calib_shots/idle_02.png")


def try_region(name, x, y, w, h):
    crop = crop_region(img, (x, y, w, h))
    res = read_text(crop, psm=7)
    res6 = read_text(crop, psm=6)
    best = max((res, res6), key=lambda r: r.confidence)
    print(f"{name:8s} ({x},{y},{w},{h}) -> {best.value!r} raw={best.raw!r} conf={best.confidence:.0f}")
    return best


print("== species candidates ==")
for y in (250, 258, 266):
    for x in (370, 380, 395):
        try_region("spec", x, y, 165, 58)
        try_region("spec", x, y, 190, 66)

print("== reskin candidates ==")
for x in (510, 525, 540):
    for w in (300, 320, 340):
        try_region("reskin", x, 358, w, 32)
