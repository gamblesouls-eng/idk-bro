"""OCR preprocessing A/B on real gradient-chip frames.

Question (user review 2026-10-23): does the luma pipeline squiggle colored
chip lettering that per-channel preprocessing would keep separated, and was
INTER_CUBIC the wrong interpolation?

Variants (replicating _prep modes):
  luma_cubic   - OLD baseline: BGR2GRAY + INTER_CUBIC 3x
  luma_lanczos - new default: BGR2GRAY + INTER_LANCZOS4 3x
  max_canvas   - V>=165 letters -> black on white (bright lettering class)
  min_canvas   - V<=96 letters -> black on white (darker-than-box class)

Scoring: a chip crop counts as READ when the joined raw token(s) look like
a chip name (^[A-Za-z][A-Za-z -]*$, len >= 4). Also reports hits of the
labels we ground-truthed visually from calib crops.
"""
import glob
import re
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))                       # project root

from poke_hunter.ocr import _prep, read_text

GRAD = (420, 398, 380, 44)          # calibration gradient chip zone (1920x1080)
CHIP_DARK_THRESHOLD = 6000          # detect_color_chip dark-box presence gate

VALID = re.compile(r"^[A-Za-z][A-Za-z -]*$")


def luma_cubic(img, upscale=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, None, fx=upscale, fy=upscale,
                      interpolation=cv2.INTER_CUBIC)


def load_chip_crops():
    """(label_key, crop) for frames that trip the dark-box gate."""
    out = []
    for path in sorted(glob.glob(r"logs\instance1_encounter_2026102*_full.png")):
        img = cv2.imread(path)
        big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
        z = hsv[GRAD[1]:GRAD[1] + GRAD[3], GRAD[0]:GRAD[0] + GRAD[2]]
        dark = int(((z[..., 1] <= 60) & (z[..., 2] >= 95)
                    & (z[..., 2] <= 145)).sum())
        if dark < CHIP_DARK_THRESHOLD:
            continue                    # chipless frame
        x, y, w, h = GRAD
        out.append((path[-11:-5], big[y:y + h, x:x + w].copy()))
    return out


def read_variant(crop, variant):
    imgs = {"luma_cubic": luma_cubic(crop, 3),
            "luma_lanczos": _prep(crop, 3),
            "max_canvas": _prep(crop, 3, color="max"),
            "chip_canvas": _prep(crop, 3, color="chip")}
    res = read_text(imgs[variant], psm=7, color=None)
    return res.raw.strip()


def main():
    chips = load_chip_crops()
    if not chips:
        print("no chip frames found (run from the project root)")
        return
    print("chip frames:", len(chips))
    known = ["Frawn", "Tyrian", "Skittles", "WITCHFIRE", "HARVEST", "Rufescent",
             "Virtual", "Radiance", "Nebula", "Reaper"]
    per_variant = {v: [0, 0, 0] for v in ("luma_cubic", "luma_lanczos",
                                          "max_canvas", "chip_canvas")}
    for _k, crop in chips:
        for variant in per_variant:
            raw = read_variant(crop, variant)
            ok = bool(VALID.match(raw)) and len(raw) >= 4
            per_variant[variant][0] += 1 if ok else 0
            per_variant[variant][1] += len(raw) if ok else 0
            for name in known:
                if name.lower() in raw.lower():
                    per_variant[variant][2] += 1
    for variant, (reads, chars, knowns) in per_variant.items():
        print(f"{variant:12s} readable {reads}/{len(chips)}"
              f"  mean-chars {chars / max(1, reads):5.1f}  known-hits {knowns}")


if __name__ == "__main__":
    main()
