import os
import cv2

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
OUT = os.path.join(LOGS, 'audit_crops2')

for t in ['200400', '200449', '201534']:
    path = os.path.join(LOGS, f'instance1_encounter_20261022_{t}_full.png')
    img = cv2.imread(path)
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    zone = big[300:480, 360:960]
    z2 = cv2.resize(zone, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT, f'{t}_z300.png'), z2)
print('ok')
