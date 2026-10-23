"""Quick OCR verification of provisional calibration regions on a screenshot."""
import sys
import cv2

from poke_hunter.ocr import read_text
from poke_hunter.util import crop_region

regions = {
    "mute": (1471, 103, 209, 59),
    "fight": (288, 876, 347, 89),
    "switch": (636, 876, 347, 88),
    "bag": (984, 876, 347, 89),
    "run": (1332, 876, 347, 89),
    "species": (380, 256, 165, 60),
    "level_wild": (640, 256, 185, 60),
    "reskin": (500, 356, 350, 36),
    "box_own": (1123, 578, 487, 163),
    "dynamax": (636, 965, 573, 89),
}

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "calib_shots/idle_02.png"
    img = cv2.imread(path)
    for name, r in regions.items():
        crop = crop_region(img, r)
        res = read_text(crop, psm=6)
        print(f"{name:10s} {r} -> {res.value!r} (conf {res.confidence:.0f}; raw {res.raw!r})")
