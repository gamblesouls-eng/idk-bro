import os
import cv2
import numpy as np

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'
img = cv2.imread(os.path.join(LOGS, 'instance1_encounter_20261022_223252_full.png'))
big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)

# MAGMA text area from zone inspection: 2.4x crop of [330..470, 380..950]
# letters approx y 365..395, x 650..830
crop = hsv[360:400, 630:850]
s = crop[..., 1].astype(np.int16)
v = crop[..., 2].astype(np.int16)
h = crop[..., 0]
print('area 220x40')
for vs, ve in [(150, 190), (190, 210), (210, 230), (230, 256)]:
    for ss, se in [(0, 40), (40, 70), (70, 100), (100, 160)]:
        n = int(((v >= vs) & (v < ve) & (s >= ss) & (s < se)).sum())
        if n > 200:
            print(f'  v[{vs}:{ve}] s[{ss}:{se}] -> {n}')
# background sample (right of panel text): x 860..940 y 360..400
bg = hsv[360:400, 860:940]
print('panel bg v median:', int(np.median(bg[..., 2])), 's median:', int(np.median(bg[..., 1])))
# gray empty-band region below (no chip): y 440..470 x 500..800
bg2 = hsv[440:470, 500:800]
print('below-band v median:', int(np.median(bg2[..., 2])), 's median:', int(np.median(bg2[..., 1])))

# same stats on a chipless frame for contrast
img2 = cv2.imread(os.path.join(LOGS, 'instance1_encounter_20261022_223017_full.png'))
b2 = cv2.resize(img2, (1920, 1080), interpolation=cv2.INTER_CUBIC)
h2 = cv2.cvtColor(b2, cv2.COLOR_BGR2HSV)
c2 = h2[360:400, 630:850]
s2, v2 = c2[..., 1].astype(np.int16), c2[..., 2].astype(np.int16)
n2 = [int(((v2 >= vs) & (v2 < ve) & (s2 >= ss) & (s2 < se)).sum())
      for vs, ve in [(150, 190), (190, 210), (210, 230), (230, 256)]
      for ss, se in [(0, 40), (40, 70), (70, 100)]]
print('223017 same-area counts (should be ~0):', n2)
