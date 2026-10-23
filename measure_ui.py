"""Measure UI element bounding boxes from a calibration screenshot by
detecting solid-color blocks (buttons, HP bars, bands). Prints windows-relative
regions ready for calibration.json.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np


def load(path):
    img = cv2.imread(path)
    return img


def find_solid_bbox(img, color_bgr, tol=18, y_range=None, x_range=None, min_area=400):
    """Bounding box of the dominant connected region matching a BGR color."""
    mask = (np.abs(img.astype(int) - np.array(color_bgr)).max(axis=2) <= tol)
    if y_range is not None:
        m = np.zeros_like(mask)
        m[y_range[0]:y_range[1], x_range[0]:x_range[1]] = mask[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        mask = m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if best is None or area > best[4]:
            best = (int(x), int(y), int(w), int(h), int(area))
    return best


def scan_row_caption(img, y, desc=""):
    row = img[y]
    # print distinct color runs along a scanline
    runs = []
    start = 0
    for x in range(1, row.shape[0]):
        if np.abs(row[x].astype(int) - row[x - 1].astype(int)).max() > 24:
            if x - start > 12:
                runs.append((start, x, tuple(int(v) for v in row[(start + x) // 2])))
            start = x
    print(f"scanline y={y} {desc}: " + " | ".join(f"[{a}-{b}) {c}" for a, b, c in runs))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "calib_shots/idle_02.png"
    img = load(path)
    print("size:", img.shape[1], "x", img.shape[0])

    # Mute button: strong red block in the top-right area
    print("mute:", find_solid_bbox(img, (40, 40, 210), tol=40, y_range=(80, 200), x_range=(1300, 1900)))
    # Fight button (pale yellow-green): bottom-left quadrant
    print("fight:", find_solid_bbox(img, (210, 240, 220), tol=30, y_range=(800, 1000), x_range=(280, 700), min_area=5000))
    # Switch (sage gray-green)
    print("switch:", find_solid_bbox(img, (150, 180, 155), tol=25, y_range=(800, 1000), x_range=(600, 1000), min_area=5000))
    # Bag (slate blue-green)
    print("bag:", find_solid_bbox(img, (140, 130, 100), tol=25, y_range=(800, 1000), x_range=(950, 1400), min_area=5000))
    # Run (dark teal)
    print("run:", find_solid_bbox(img, (110, 90, 60), tol=25, y_range=(800, 1000), x_range=(1300, 1900), min_area=5000))
    # Dynamax (magenta)
    print("dynamax:", find_solid_bbox(img, (140, 30, 190), tol=60, y_range=(900, 1080), x_range=(600, 1400), min_area=5000))

    # wild HP bar: green rectangle on the LEFT side
    print("hp_wild:", find_solid_bbox(img, (40, 180, 60), tol=60, y_range=(280, 420), x_range=(450, 900), min_area=1500))
    # player HP bar: green rect on the RIGHT side
    print("hp_own:", find_solid_bbox(img, (40, 180, 60), tol=60, y_range=(560, 830), x_range=(1250, 1700), min_area=1500))

    # info boxes: light gray rounded boxes; detect gray blocks
    gray = (134, 134, 134)
    print("box_wild:", find_solid_bbox(img, gray, tol=14, y_range=(200, 480), x_range=(300, 950), min_area=20000))
    print("box_own:", find_solid_bbox(img, gray, tol=14, y_range=(500, 800), x_range=(1050, 1750), min_area=20000))

    # dialog band: dark slate blue (approx (140, 100, 70)? measured visually)
    # sample some candidate pixels
    for (x, y) in [(400, 800), (900, 810), (600, 790)]:
        print("dialog band sample", (x, y), tuple(int(v) for v in img[y, x]))
