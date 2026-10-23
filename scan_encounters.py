"""Scan saved encounter screenshots for species + candidate star indicators.

The shiny/tint star regions are not yet calibrated. As a safety net, this scans
the whole wild-info box for isolated saturated-blue clusters (shiny star) and
small solid-white clusters over dark background (tint star). Anything that
looks like a small isolated highlight is reported as a candidate.

Evidence PNGs may be downscaled on save (max side 1200): regions are rescaled
per image by (frame_width/1920, frame_height/1080).
"""
import glob
import os
import sys

import cv2

from poke_hunter.ocr import read_text
from poke_hunter.util import crop_region

SPECIES = (378, 246, 320, 66)
LEVEL = (640, 256, 185, 60)
BOX = (358, 240, 500, 170)          # whole wild info box (+padding)
CANDIDATE_AREA = (300, 200, 760, 280)


def scaler(shape):
    fw, fh = shape[1], shape[0]
    sx, sy = fw / 1920.0, fh / 1080.0
    if abs(sx - 1.0) < 0.02 and abs(sy - 1.0) < 0.02:
        return lambda r: r
    def s(r):
        x, y, w, h = r
        return (int(x * sx), int(y * sy), max(1, int(w * sx)), max(1, int(h * sy)))
    return s


def upsc(crop, factor=3):
    return cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def clusters(mask, min_px=12):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    hits = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_px:
            hits.append((int(x), int(y), int(w), int(h), int(area)))
    return hits


def scan(path):
    img = cv2.imread(path)
    if img is None:
        return None
    s = scaler(img.shape)
    sp = read_text(upsc(crop_region(img, s(SPECIES))), psm=7).raw
    lv = read_text(upsc(crop_region(img, s(LEVEL))), psm=7).raw
    box = crop_region(img, s(BOX))
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (90, 120, 150), (135, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 230), (179, 45, 255))
    blue[80:118, :] = 0          # HP-bar band excluded
    white[80:118, 150:290] = 0   # HP-bar band excluded
    return {"species": sp, "level": lv,
            "blue": clusters(blue), "white": clusters(white)}


def main():
    files = sorted(glob.glob("logs/instance1_encounter_*_full.png"))
    if len(sys.argv) > 1:
        # basename-based cutoff (avoids win32 path-separator mismatch)
        cut = os.path.basename(sys.argv[1])
        files = [f for f in files if os.path.basename(f) >= cut]
    for p in files:
        r = scan(p)
        if not r:
            continue
        flag = ""
        if r["blue"]:
            flag += f"  BLUE {r['blue']}"
        if r["white"]:
            flag += f"  WHITE {r['white']}"
        print(f"{os.path.basename(p)[22:36]}  species={r['species']!r:<26} "
              f"lvl={r['level']!r:<10}{flag}")
        if flag:
            out = p.replace("_full.png", "_starcandidate.png")
            full = cv2.imread(p)
            crop = crop_region(full, scaler(full.shape)(CANDIDATE_AREA))
            cv2.imwrite(out, crop)
            print("   candidate crop:", out)


if __name__ == "__main__":
    main()
