"""Re-validate the corrected reskin gate over all frames."""
import os
import glob as globmod
import cv2
import numpy as np

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
RZONE = (480, 354, 380, 42)
GZONE = (420, 398, 380, 44)

chip_frames = {'220427','220754','221055','221523','221557','221627','222159',
               '222255','222806','223056','223242','223320','223455'}
white_chip = {'223252'}  # MAGMA

res = []
for path in sorted(globmod.glob(os.path.join(LOGS, 'instance1_encounter_20261022_2*_full.png'))):
    img = cv2.imread(path)
    if img is None:
        continue
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
    x, y, w, h = RZONE
    zone = hsv[y:y+h, x:x+w]
    s = zone[..., 1].astype(np.int16); v = zone[..., 2].astype(np.int16)
    rcount = int(((s <= 60) & (v >= 150) & (v <= 200)).sum())
    x, y, w, h = GZONE
    zone = hsv[y:y+h, x:x+w]
    s = zone[..., 1].astype(np.int16); v = zone[..., 2].astype(np.int16)
    gcount = int(((s >= 100) & (v >= 90)).sum())
    tag = os.path.basename(path)[20:35]
    short = tag[9:15]
    rc, gc = rcount, gcount
    truth = ('R' if short in chip_frames or short in white_chip else '-')
    if short in chip_frames:
        truth = 'C'
    if short in white_chip:
        truth = 'W'
    res.append((tag, rc, gc, truth))

import numpy as np
r = np.array([x[1] for x in res]); g = np.array([x[2] for x in res])
print('reskin-slot count: p50 %d p90 %d p99 %d max %d' % tuple(np.percentile(r, [50, 90, 99, 100])))
print('color-slot count:  p50 %d p90 %d p99 %d max %d' % tuple(np.percentile(g, [50, 90, 99, 100])))
print()
r_th, g_th = 480, 400
errs = 0
for tag, rc, gc, truth in sorted(res, key=lambda x: -max(x[1], x[2]))[:30]:
    verdict = ''
    ok = True
    if 'W' in truth:
        ok = rc >= r_th
    if 'C' in truth:
        ok = gc >= g_th
    if truth == '-' and (rc >= r_th or gc >= g_th):
        ok = False
    errs += not ok
    print(f'{tag}  reskin {rc:6d} color {gc:6d}  truth {truth:2s}  '
          f'{"OK" if ok else "MISS/FP"}')
print()
print('gate errors over all frames:', errs, '/', len(res))
