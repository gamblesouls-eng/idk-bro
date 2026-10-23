import os
import cv2

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
OUT = os.path.join(LOGS, 'audit_crops2')

for t in ['204515', '210427', '200740']:
    path = os.path.join(LOGS, f'instance1_encounter_20261022_{t}_full.png')
    img = cv2.imread(path)
    if img is None:
        print('missing', t)
        continue
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    zone = big[340:470, 380:980]
    z2 = cv2.resize(zone, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT, f'{t}_z340.png'), z2)
print('ok')
