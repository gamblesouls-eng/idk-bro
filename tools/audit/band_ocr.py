"""OCR the gradient-slot letters for every frame with 400 <= sat < 4000,
to establish which are real chips and where the noise ceiling sits."""
import os
import glob as globmod
import sys
import cv2

sys.path.insert(0, r'C:\Users\Seb\Projects\poke_hunter')
from poke_hunter.ocr import read_text
from poke_hunter.util import normalize_text

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
known = {'220427','220754','221055','221523','221557','221627','222159',
         '222255','222806','223056','223242','223320','223455',
         '204515','203816','210427','200740'}

for pat in ('2*', '1*'):
    for p in sorted(globmod.glob(os.path.join(LOGS, 'instance1_encounter_20261022_' + pat + '_full.png'))):
        tag = os.path.basename(p)[20:35][9:15]
        img = cv2.imread(p)
        big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
        z = hsv[398:442, 420:800]
        s = z[..., 1].astype('int16')
        v = z[..., 2].astype('int16')
        sat = int(((s >= 100) & (v >= 90)).sum())
        if not (400 <= sat < 4000):
            continue
        crop = big[405:438, 440:800]
        res = read_text(crop.copy())
        txt = normalize_text(res.raw).strip() or '<none>'
        mark = 'C' if tag in known else '?'
        print(f'{tag} sat={sat:5d} {mark} text={txt!r}')
