"""Offline end-to-end replay of Ui.read_aspect_chips over saved frames.

Uses a fake Grabber that serves each saved debug frame (resized to
1920x1080, mirroring the live native grab geometry). Verifies:
  * chip frames -> gradient value/marker
  * reskin frames (white slot) -> reskin word
  * chipless frames -> ('', '')
"""
import os
import sys
import glob as globmod
import time
import cv2

sys.path.insert(0, r'C:\Users\Seb\Projects\poke_hunter')
from poke_hunter.config import load_calibration
from poke_hunter.actions import Ui
from poke_hunter.capture import Grabber

LOGS = r'C:\Users\Seb\Projects\poke_hunter\logs'

KNOWN_CHIP = {   # colered-slot frames (gradient present)
    '220427': 'sherbert', '220754': '?', '221055': 'elysium',
    '221523': 'verdalis', '221557': 'verdalis', '221627': '?',
    '222159': '?', '222255': 'verdalis', '222806': 'glitched reseste',
    '223056': '?', '223242': 'horizon', '223320': 'ignited',
    '223455': 'azurine', '204515': '?', '203816': 'radiance',
    '210427': 'roseate', '200740': 'skittles', '200439': 'reaper',
    '202538': 'nebula', '205745': '?',
}
WHITE_RESKIN = {  # white-slot frames (reskin word)
    '223252': 'magma', '204334': 'forest guardian', '200740': 'luminous',
    '204515': 'crayon', '210427': 'doodle', '201836': 'doodle',
    '203721': 'spectre', '205221': 'goblin', '205745': 'doodle',
}


class FakeGrabber:
    """Serves a fixed frame from a saved debug image."""

    def __init__(self, img):
        self._img = img

    def frame(self, fresh: bool = False, min_age_ms: int = 0):
        return self._img

    def frame_signature(self, frame=None, threshold: float = 1.0):  # unused here
        return "fake"


class NullWindow:
    pass


calib = load_calibration(os.path.join(r'C:\Users\Seb\Projects\poke_hunter',
                                      'config', 'calibration.json'))
assert calib.aspect_chips is not None, "aspect_chips missing from calibration"

ui = Ui(NullWindow(), None, calib, None)

# Build the test set: all chip frames + white-reskin frames + chipless sample
targets = sorted(set(KNOWN_CHIP) | set(WHITE_RESKIN))
chipless = []
for p in sorted(globmod.glob(os.path.join(LOGS, 'instance1_encounter_20261022_2*_full.png'))):
    tag = os.path.basename(p)[20:35][9:15]
    if tag not in KNOWN_CHIP and tag not in WHITE_RESKIN:
        chipless.append(p)
chipless = chipless[::15][:12]   # a manageable sample
paths = ([os.path.join(LOGS, f'instance1_encounter_20261022_{t}_full.png') for t in targets]
         + chipless)

errors = 0
for p in paths:
    tag = os.path.basename(p)[20:35][9:15]
    img = cv2.imread(p)
    if img is None:
        continue
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    ui.grab = FakeGrabber(big)
    t0 = time.monotonic()
    reskin, gradient, stats = ui.read_aspect_chips()
    dt = (time.monotonic() - t0) * 1000
    truth_g = KNOWN_CHIP.get(tag)
    truth_r = WHITE_RESKIN.get(tag)
    if truth_g is not None and truth_r is not None:
        pass  # both-badges frame, both slots must read
    ok_g = (truth_g is None and gradient == '') or \
           (truth_g is not None and gradient != '')
    ok_r = (truth_r is None and reskin == '') or \
           (truth_r is not None and reskin != '')
    if not (ok_g and ok_r):
        errors += 1
    print(f"{tag}  reskin={reskin!r:24s} gradient={gradient!r:20s} "
          f"({dt:.0f} ms) {'...' if (ok_g and ok_r) else '  << ERROR'}")
print('errors:', errors)
