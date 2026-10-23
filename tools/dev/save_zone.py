import os
import cv2

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
OUT = os.path.join(LOGS, 'audit_crops2')

for t in ['220427', '223056']:
    path = os.path.join(LOGS, f'instance1_encounter_20261022_{t}_full.png')
    img = cv2.imread(path)
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    # aspect region zone: x 380..920, y 300..500
    zone = big[300:520, 360:940]
    z2 = cv2.resize(zone, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT, f'{t}_zone.png'), z2)
print('ok')
