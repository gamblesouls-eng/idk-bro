import os
import cv2
import numpy as np

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
OUT = os.path.join(LOGS, 'audit_crops2')

# a 'failed=[gradient]' (reskin-present live) frame and neighbors
for t in ['223017', '223032', '222255', '220754', '221627', '222159']:
    path = os.path.join(LOGS, f'instance1_encounter_20261022_{t}_full.png')
    img = cv2.imread(path)
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    zone = big[330:470, 380:950]
    z2 = cv2.resize(zone, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT, f'{t}_z330.png'), z2)
print('ok')
