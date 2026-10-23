"""Prototype the two final gates:
  gradient slot: dark chip-box pixel count (chips ~12k, chipless ~400)
  reskin slot:   shape-filtered letter blobs (drop streak lines) + canvas OCR
Run over known-truth frames and pick thresholds.
"""
import os
import cv2
import numpy as np
import sys

sys.path.insert(0, r'C:\Users\Seb\Projects\poke_hunter')
from poke_hunter.ocr import read_text
from poke_hunter.util import normalize_text

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'

CASES = [
    # tag, expect_reskin, expect_gradient
    ('223252', True, False),    # MAGMA
    ('204334', True, False),    # FOREST GUARDIAN
    ('223056', False, True),    # GLITCHED Rufescent
    ('222159', False, True),    # Frawn (dim)
    ('202538', False, True),    # Nebula (dim)
    ('200439', False, True),    # Reaper (desaturated)
    ('223252x', False, False),  # placeholder
    ('200400', False, False),
    ('201534', False, False),
    ('201836', None, False),    # hallucinated 'doodle' -> must be rejected
    ('203721', None, False),    # 'spectre'
    ('205221', None, False),    # 'goblin'
    ('204257', None, False),    # garbage
    ('202208', False, False),
    ('203054', False, False),
]


def load(tag):
    img = cv2.imread(os.path.join(LOGS, f'instance1_encounter_20261022_{tag}_full.png'))
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(big, cv2.COLOR_BGR2HSV), big


def grad_stats(hsv):
    z = hsv[398:442, 420:800]
    s = z[..., 1].astype('int16')
    v = z[..., 2].astype('int16')
    dark = int(((s <= 60) & (v >= 95) & (v <= 145)).sum())
    sat = int(((s >= 100) & (v >= 90)).sum())
    return dark, sat


def reskin_letters(hsv, big):
    z = hsv[354:396, 480:860]
    s = z[..., 1].astype('int16')
    v = z[..., 2].astype('int16')
    mask = ((s <= 80) & (v >= 158)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    total = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 5 <= w <= 80 and 9 <= h <= 34 and area >= 25:
            keep[lab == i] = 255
            total += int(area)
    # canvas: letters dark on white
    canvas = np.full(keep.shape, 255, np.uint8)
    canvas[keep > 0] = 0
    canvas_big = cv2.resize(canvas, (keep.shape[1] * 3, keep.shape[0] * 3),
                            interpolation=cv2.INTER_CUBIC)
    res = read_text(canvas_big)
    txt = normalize_text(res.raw).strip()
    return total, txt


if __name__ == '__main__':
    for tag, e_r, e_g in CASES:
        if tag.endswith('x'):
            continue
        hsv, big = load(tag)
        dark, sat = grad_stats(hsv)
        gcount, gtxt = reskin_letters(hsv, big)
        print(f'{tag}  grad(dark={dark:5d} sat={sat:4d})   '
              f'reskin(blobs={gcount:5d} canvas_ocr={gtxt!r})')
