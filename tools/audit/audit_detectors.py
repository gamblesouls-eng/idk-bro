"""Offline detector audit: replay gradient/reskin detection on saved frames.

No screen grabbing, no mouse, no window access -- only files already on disk.
"""
import os
import sys
import glob as globmod
import csv

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poke_hunter.ocr import read_text
from poke_hunter.detectors import detect_gradient

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
CROPS_DIR = os.path.join(LOGS, "audit_crops")

# calibration.json (1920x1080 coordinate space)
RESKIN_LABEL = [510, 356, 340, 36]
GRADIENT_LABEL = [400, 392, 400, 56]
BAND = {"sat_min": 120, "val_min": 80, "min_fraction": 0.03}


class _Spec:
    sat_min = BAND["sat_min"]
    val_min = BAND["val_min"]
    min_fraction = BAND["min_fraction"]


def crop_1920(frame_small, region):
    """Upscale a 1200x675 frame to 1920x1080 exactly like the recalib logic."""
    big = cv2.resize(frame_small, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    x, y, w, h = region
    return big[y:y + h, x:x + w]


def main():
    os.makedirs(CROPS_DIR, exist_ok=True)
    pattern = os.path.join(LOGS, "instance1_encounter_20261022_22*_full.png")
    frames = sorted(globmod.glob(pattern))
    print(f"frames found: {len(frames)}")

    rows = []
    classes = {"reskin_present": 0, "reskin_empty": 0, "reskin_none": 0,
               "grad_present": 0, "grad_empty": 0, "grad_none": 0,
               "both": 0}
    for path in frames:
        img = cv2.imread(path)
        if img is None:
            rows.append({"frame": os.path.basename(path), "error": "unreadable"})
            continue
        big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)

        # --- gradient: exact detector ---
        gx, gy, gw, gh = GRADIENT_LABEL
        gcrop = big[gy:gy + gh, gx:gx + gw]
        present, stats = detect_gradient(gcrop.copy(), _Spec())
        sat_frac = stats["sat_fraction"]
        if present:
            g_out = "PRESENT"
            classes["grad_present"] += 1
        elif sat_frac <= 0.0:
            g_out = "ABSENT"
            classes["grad_empty"] += 1
        else:
            g_out = "absent(weak)"
            classes["grad_empty"] += 1

        # --- reskin: exact OCR path ---
        rx, ry, rw_, rh = RESKIN_LABEL
        rcrop = big[ry:ry + rh, rx:rx + rw_]
        res = read_text(rcrop.copy())
        r_out = res.raw.strip()
        if not res.value:
            r_tag = "EMPTY/UNSTABLE"
            classes["reskin_empty"] += 1
        else:
            r_tag = "PRESENT"
            classes["reskin_present"] += 1

        both = bool(res.value) and bool(present)
        if both:
            classes["both"] += 1
        rows.append({
            "frame": os.path.basename(path),
            "reskin_raw": r_out,
            "reskin_norm": res.value,
            "reskin_tag": r_tag,
            "gradient": g_out,
            "sat_fraction": sat_frac,
            "both": "YES" if both else "-",
        })

        # save the highest-information crops: gradient band always for
        # borderline/exciting frames, plus every 'both' frame and top-20 frractions
        sat = sat_frac
        thumbs = sat >= 0.005 or both
        if thumbs:
            tag = os.path.basename(path).replace("instance1_encounter_", "").replace("_full.png", "")
            gr = cv2.resize(gcrop, None, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            rr = cv2.resize(rcrop, None, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(CROPS_DIR, f"{tag}_g_{sat:.4f}_{g_out}.png"), gr)
            if not both:
                cv2.imwrite(os.path.join(CROPS_DIR, f"{tag}_r_{res.value or 'empty'}.png"), rr)

    out_csv = os.path.join(LOGS, "audit_table_20261022.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("classes:", classes)
    print("csv:", out_csv)
    # distribution buckets
    fracs = [r["sat_fraction"] for r in rows if "sat_fraction" in r]
    if fracs:
        buckets = [0, 0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 1.0]
        hist = []
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            n = sum(1 for f in fracs if lo <= f < hi)
            hist.append(f"{lo:.3f}-{hi:.3f}: {n}")
        print("\n".join(hist))


if __name__ == "__main__":
    main()
