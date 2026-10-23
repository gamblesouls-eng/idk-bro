import os
import cv2

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
OUT = os.path.join(LOGS, 'audit_crops2')

# frames behind encounters that live-read reskin (failed=['gradient']):
# 0011 -> 220719, 0034 -> 221509, 0075 -> 223252, 0079 -> 223341,
# 0086 -> 223521, 0089 -> 223555, 0096 -> 223722
for t in ['220719', '221509', '223252', '223341', '223521', '223555', '223722']:
    path = os.path.join(LOGS, f'instance1_encounter_20261022_{t}_full.png')
    img = cv2.imread(path)
    if img is None:
        print('missing', t)
        continue
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    zone = big[330:470, 380:950]
    z2 = cv2.resize(zone, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT, f'{t}_z330.png'), z2)
print('ok')
