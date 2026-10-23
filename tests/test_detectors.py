"""Synthetic-image tests for star/star-stability and HP-bar detection."""
import numpy as np
import pytest

from poke_hunter.config import HPSpec, StarSpec
from poke_hunter.detectors import detect_star, detect_star_stable, hp_state

BLUE = ((95, 60, 120), (125, 255, 255))   # shiny blue star range (OpenCV H=95..125)
WHITE = ((0, 0, 200), (179, 40, 255))     # tint white star range

SHINY = StarSpec(hsv_min=BLUE[0], hsv_max=BLUE[1], min_pixels=8)
TINT = StarSpec(hsv_min=WHITE[0], hsv_max=WHITE[1], min_pixels=8)

HP = HPSpec(colors={
    "red": ((0, 100, 100), (8, 255, 255)),
    "yellow": ((18, 100, 100), (35, 255, 255)),
    "green": ((40, 100, 100), (80, 255, 255)),
}, min_fraction=0.30)


def make(bgr_color, size=(40, 24)):
    img = np.full((size[1], size[0], 3), bgr_color, dtype=np.uint8)
    return img


def add_dot(img, center, radius, bgr):
    import cv2
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    img[mask == 255] = bgr
    return img


class FakeGrabber:
    """Stand-in for Grabber.capture_region: returns queued frames."""
    def __init__(self, frames):
        self.frames = list(frames)
        self.i = 0

    def capture_region(self, region, fresh=False):
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return f


# ------------------------------------------------------------------ star
def test_star_present_blue():
    img = make((255, 255, 255))          # white background
    add_dot(img, (20, 12), 5, (255, 120, 40))  # BGR blue-ish
    verdict, stats = detect_star(img, SHINY)
    assert verdict is True
    assert stats["star_pixels"] >= 8


def test_star_absent_clean_background():
    img = make((30, 30, 30))  # dark panel background
    verdict, stats = detect_star(img, TINT)
    assert verdict is False and stats["bg_fraction"] >= 0.85


def test_star_absent_over_bright_white_region_is_unknown():
    # a white region is NOT a confident 'no star' verdict for the blue range
    img = make((255, 255, 255))
    verdict, _ = detect_star(img, SHINY)
    assert verdict is None


def test_star_unknown_when_obscured():
    # no white pixels, no blue pixels -> neither present nor clean background
    img = make((30, 90, 160))
    verdict, _ = detect_star(img, SHINY)
    assert verdict is None


def test_shiny_vs_tint_distinguished():
    shiny_img = make((30, 30, 30))                   # dark background
    add_dot(shiny_img, (20, 12), 5, (20, 90, 255))   # orange-red dot (BGR)
    tint_img = make((10, 90, 130))                    # dark bg
    add_dot(tint_img, (20, 12), 5, (255, 255, 255))   # white dot
    assert detect_star(shiny_img, SHINY)[0] is False      # blue star absent
    assert detect_star(tint_img, TINT)[0] is True         # white star present


def test_star_stable_two_reads_same():
    img = make((255, 255, 255))
    add_dot(img, (20, 12), 5, (255, 120, 40))
    grab = FakeGrabber([img, img])
    verdict, _ = detect_star_stable(grab, (0, 0, 40, 24), SHINY, attempts=3)
    assert verdict is True


def test_star_stable_flips_marked_unknown():
    img1 = make((255, 255, 255))
    img2 = make((30, 90, 160))
    grab = FakeGrabber([img1, img2, img1, img2])
    verdict, _ = detect_star_stable(grab, (0, 0, 40, 24), SHINY, attempts=4)
    assert verdict is None


# ------------------------------------------------------------------ HP bar
def test_hp_red():
    verdict, fr = hp_state(make((40, 40, 230)), HP)   # BGR red
    assert verdict == "red"


def test_hp_yellow():
    verdict, fr = hp_state(make((30, 230, 240)), HP)
    assert verdict == "yellow"


def test_hp_green():
    verdict, fr = hp_state(make((90, 200, 80)), HP)
    assert verdict == "green"


def test_hp_unknown_gray():
    verdict, fr = hp_state(make((128, 128, 128)), HP)
    assert verdict is None


def test_hp_tie_is_unknown():
    # left 25% of the bar split evenly between red and green -> no confident call
    img = np.zeros((24, 40, 3), np.uint8)
    lg = make((90, 200, 80), size=(10, 24))
    lr = make((40, 40, 230), size=(10, 24))
    for i in range(10):
        img[:, i] = lg[:, i] if i % 2 == 0 else lr[:, i]
    verdict, counts = hp_state(img, HP)
    assert verdict is None


# --- favorite-star badge (bag favorites routing) ----------------------------- #
from poke_hunter.config import FavoriteSpec
from poke_hunter.detectors import favorite_star

FAV = FavoriteSpec(zone=(8, 45, 55, 33), hsv_min=(20, 110, 150),
                   hsv_max=(48, 255, 255), min_fraction=0.10)


def make_row(with_star: bool):
    """520x78 row crop: dark background, icon top-left, star badge when set."""
    import cv2
    row = np.full((78, 520, 3), (30, 32, 20), dtype=np.uint8)   # ~dark navy
    # icon glyph (part inside the star zone on x)
    row[10:44, 12:52] = (60, 120, 200)                          # blue-ish icon
    if with_star:
        # yellow star at the icon bottom-left corner
        mask = np.zeros(row.shape[:2], np.uint8)
        cv2.circle(mask, (26, 58), 9, 255, -1)
        row[mask == 255] = (40, 200, 255)                       # BGR warm yellow
    return row


def test_favorite_star_present():
    assert favorite_star(make_row(True), FAV) is True


def test_favorite_star_absent_on_plain_row():
    row = make_row(False)
    assert favorite_star(row, FAV) is False


def test_favorite_star_ignores_cyan_chip():
    row = make_row(False)
    # cyan quantity chip (far right) must not influence the left zone check
    row[20:60, 400:500] = (240, 200, 50)                        # BGR cyan
    assert favorite_star(row, FAV) is False


def test_favorite_star_zone_respects_config():
    # moving the zone away from the star means detection fails
    shifted = FavoriteSpec(zone=(300, 45, 55, 33), hsv_min=FAV.hsv_min,
                           hsv_max=FAV.hsv_max, min_fraction=0.10)
    assert favorite_star(make_row(True), shifted) is False

# --- search-field placeholder-aware emptiness helper -------------------------- #
from poke_hunter.actions import query_field_empty


def test_query_field_placeholder_counts_as_empty():
    assert query_field_empty("search... )")
    assert query_field_empty("Search...")
    assert query_field_empty("")
    assert query_field_empty("search")


def test_query_field_real_filter_is_not_empty():
    assert not query_field_empty("re")
    assert not query_field_empty("repeat ball")
    assert not query_field_empty("full restore")
