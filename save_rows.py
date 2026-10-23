import os
import cv2
import numpy as np

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
OUT = os.path.join(LOGS, 'audit_crops2')
os.makedirs(OUT, exist_ok=True)

RESKIN_LABEL = [510, 356, 340, 36]
GRADIENT_LABEL = [400, 392, 400, 56]

frames = ['220427', '220754', '221055', '221523', '221557', '221627',
          '222255', '222806', '223056', '223242', '223320', '223455']

for t in frames:
    path = os.path.join(LOGS, f'instance1_encounter_20261022_{t}_full.png')
    img = cv2.imread(path)
    if img is None:
        print('unreadable', t)
        continue
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    rx, ry, rw, rh = RESKIN_LABEL
    rcrop = big[ry - 8:ry + rh + 8, rx - 8:rx + rw + 8]   # slight pad
    gx, gy, gw, gh = GRADIENT_LABEL
    gcrop = big[gy - 8:gy + gh + 8, gx - 8:gx + gw + 8]
    r3 = cv2.resize(rcrop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    g3 = cv2.resize(gcrop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    w = max(r3.shape[1], g3.shape[1])
    white = np.full((12, w, 3), 255, np.uint8)
    r3 = cv2.copyMakeBorder(r3, 0, 0, 0, w - r3.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    g3 = cv2.copyMakeBorder(g3, 0, 0, 0, w - g3.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    stack = np.vstack([r3, white, g3])
    cv2.imwrite(os.path.join(OUT, f'{t}_rows.png'), stack)
print('done ->', OUT)
