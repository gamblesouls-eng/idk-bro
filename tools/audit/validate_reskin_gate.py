"""Offline validation of the reskin-slot dark-letter gate over all frames.

Sweeps (v_min, v_max, s_max, min_pixels) candidates and reports per-frame
gate statistic extremes, so thresholds keep a clean margin between
chipless noise and lettered frames.
"""
import os
import glob as globmod
import cv2
import numpy as np

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
ZONE = (480, 354, 380, 42)   # x, y, w, h in 1920-space

rows = []
for path in sorted(globmod.glob(os.path.join(LOGS, 'instance1_encounter_20261022_2*_full.png'))):
    img = cv2.imread(path)
    if img is None:
        continue
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
    x, y, w, h = ZONE
    zone = hsv[y:y + h, x:x + w]
    s = zone[..., 1].astype(np.int16)
    v = zone[..., 2].astype(np.int16)
    counts = {}
    for vmin in (130, 140, 150):
        for vmax in (188, 192, 196):
            for smax in (55, 60, 70):
                m = ((s <= smax) & (v >= vmin) & (v <= vmax))
                counts[(vmin, vmax, smax)] = int(m.sum())
    rows.append((os.path.basename(path)[20:35], counts))

# summarize: for one chosen config show distribution of counts
cfg = (140, 192, 60)
vals = sorted(r[1][cfg] for r in rows)
print(f'config v{cfg}: min..max = {vals[0]}..{vals[-1]}  p50 {int(np.percentile(vals, 50))}'
      f'  p90 {int(np.percentile(vals, 90))}  p99 {int(np.percentile(vals, 99))}')
top = sorted(rows, key=lambda r: -r[1][cfg])[:25]
print('top frames (should be the lettered ones):')
for tag, c in top:
    print('  ', tag, c[cfg])
