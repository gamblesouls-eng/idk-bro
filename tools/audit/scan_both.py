"""Scan all frames for BOTH badge types:
- white reskin letters in y 350..400 (v>=200, s<=60)
- saturated colored chip letters in y 400..440
"""
import os
import glob as globmod
import cv2
import numpy as np

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'

hits = []
both = []
for path in sorted(globmod.glob(os.path.join(LOGS, 'instance1_encounter_20261022_22*_full.png'))):
    img = cv2.imread(path)
    if img is None:
        continue
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
    h_, s, v = hsv[..., 0], hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)

    # white reskin letters
    wm = ((s <= 70) & (v >= 215))
    wzone = wm[350:400, 400:950]
    wcount = int(wzone.sum())

    # colored chip letters
    cm = ((s >= 100) & (v >= 90))
    czone = cm[400:440, 400:950]
    ccount = int(czone.sum())

    tag = os.path.basename(path)[20:35]
    if wcount >= 150:
        hits.append((tag, 'W', wcount, ccount))
    if ccount >= 400:
        hits.append((tag, 'C', wcount, ccount))
    if wcount >= 150 and ccount >= 400:
        both.append(tag)

for t, kind, w, c in hits:
    print(kind, t, 'white_px:', w, 'color_px:', c)
print('frames with BOTH white reskin text AND colored chip:', both)
