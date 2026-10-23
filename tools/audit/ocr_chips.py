import os
import json
import cv2
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Seb\Projects\poke_hunter')
from poke_hunter.ocr import read_text

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
data = json.load(open(os.path.join(LOGS, 'chip_measure_20261022.json')))
chipped = [r for r in data if r.get('chips')]
print('chip frames:', len(chipped))
for r in chipped:
    path = os.path.join(LOGS, 'instance1_encounter_' + r['frame'] + '_full.png')
    img = cv2.imread(path)
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    texts = []
    for box in r['boxes']:
        x0, y0, x1, y1 = box[:4]
        X0, Y0, X1, Y1 = x0 + 340 - 10, y0 + 385 - 8, x1 + 340 + 10, y1 + 385 + 8
        crop = big[Y0:Y1, X0:X1]
        res = read_text(crop.copy())
        texts.append(res.raw.strip() or '<unreadable>')
        zoom = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(LOGS, 'audit_crops2', 'chip_' + r['frame'] + '.png'), zoom)
    x0, y0, x1, y1 = r['boxes'][0][:4]
    x0 += 340; x1 += 340; y0 += 385; y1 += 385
    sat = None
    print(f"{r['frame']}  chipbox x[{x0}..{x1}] y[{y0}..{y1}]  text: {texts}")
