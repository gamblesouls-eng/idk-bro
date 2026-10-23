"""Measure where aspect chips actually render, across every saved frame.

Finds saturated "lettering" pixels in the zone under the species panel
(y 385..480, x 360..960 in 1920-space) and clusters them into chip boxes.
"""
import os
import glob as globmod
import cv2
import numpy as np

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
TO19 = (1920, 1080)

# zone under the panel, in 1920-space
ZX0, ZY0, ZX1, ZY1 = 340, 385, 980, 485

rows = []
two_chip = 0
no_chip = 0
for path in sorted(globmod.glob(os.path.join(LOGS, 'instance1_encounter_20261022_22*_full.png'))):
    img = cv2.imread(path)
    if img is None:
        continue
    big = cv2.resize(img, TO19, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
    zone = hsv[ZY0:ZY1, ZX0:ZX1]
    s = zone[..., 1].astype(np.int16)
    v = zone[..., 2].astype(np.int16)
    mask = ((s >= 100) & (v >= 90)).astype(np.uint8)
    # y distribution of saturated pixels
    ys, xs = np.nonzero(mask)
    if ys.size < 40:
        no_chip += 1
        rows.append({'frame': os.path.basename(path)[22:33], 'chips': 0})
        continue
    # histogram rows: which y band holds the letters
    yhist = np.bincount(ys, minlength=ZY1 - ZY0)
    # x clustering: split xs into groups with gaps > 45 px
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    groups = []
    start = 0
    for i in range(1, xs_sorted.size):
        if xs_sorted[i] - xs_sorted[i - 1] > 45:
            groups.append((start, i))
            start = i
    groups.append((start, xs_sorted.size))
    chips = []
    for g0, g1 in groups:
        gx = xs_sorted[g0:g1]
        gy = ys_sorted[g0:g1]
        if gx.size < 30:
            continue
        chips.append((int(gx.min()), int(gy.min()), int(gx.max()), int(gy.max()), int(gx.size)))
    chips = [c for c in chips if (c[2] - c[0]) >= 30]
    if not chips:
        no_chip += 1
        rows.append({'frame': os.path.basename(path)[20:35], 'chips': 0})
        continue
    if len(chips) >= 2:
        two_chip += 1
    rows.append({'frame': os.path.basename(path)[20:35],
                 'chips': len(chips),
                 'boxes': chips})

print('frames scanned:', len(rows), '| no-chip frames:', no_chip, '| >=2 chip frames:', two_chip)
ymid = {}
multi = [r for r in rows if r.get('chips', 0) >= 2]
print('multi-chip examples:')
for r in multi[:10]:
    print('  ', r['frame'], r['boxes'])
ys_all = [c[1] + ZY0 for r in rows for c in r.get('boxes', [])]
ye_all = [c[3] + ZY0 for r in rows for c in r.get('boxes', [])]
if ys_all:
    print('chip text y-extent in 1920-space: min %d .. p05 %d .. median %d .. p95 %d .. max %d'
          % (min(ys_all), int(np.percentile(ys_all, 5)), int(np.median(ys_all)),
             int(np.percentile(ys_all, 95)), max(ys_all)))
import json
with open(os.path.join(LOGS, 'chip_measure_20261022.json'), 'w') as fh:
    json.dump(rows, fh, indent=1)
print('saved chip_measure_20261022.json')
