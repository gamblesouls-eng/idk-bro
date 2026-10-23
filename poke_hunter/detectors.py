"""Visual detectors.

All detection is config-driven (HSV specs in calibration.json) so thresholds
and regions can be re-calibrated without touching encounter logic.

Coordinate system: window-relative regions, OpenCV HSV scale
(H: 0-179, S: 0-255, V: 0-255).
"""
from __future__ import annotations

import time
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .capture import Grabber, Region
from .config import HPSpec, StarSpec
from .logging_setup import get_logger
from .util import crop_region

log = get_logger("detectors")


class Tri(Enum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------- #
# Star (shiny / tint) detection
# --------------------------------------------------------------------------- #

def detect_star(crop: np.ndarray, spec: StarSpec) -> Tuple[Optional[bool], dict]:
    """Detect a colored indicator star inside a region of interest.

    Logic:
      1. Count pixels inside the configured star HSV range.
      2. If enough pixels -> present (True).
      3. Else if the region is dominated by the *background* HSV range
         (dark panel with no star) -> confidently absent (False).
      4. Otherwise obscured / unreadable -> None (unknown).

    Returns (verdict, stats) - stats aid calibration via `main.py verify`.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(spec.hsv_min, dtype=np.uint8),
                       np.array(spec.hsv_max, dtype=np.uint8))
    count = int(cv2.countNonZero(mask))
    bg = cv2.inRange(hsv, np.array(spec.bg_hsv_min, dtype=np.uint8),
                     np.array(spec.bg_hsv_max, dtype=np.uint8))
    total = crop.shape[0] * crop.shape[1]
    bg_frac = float(cv2.countNonZero(bg)) / max(1, total)
    stats = {"star_pixels": count, "bg_fraction": round(bg_frac, 3)}
    if count >= spec.min_pixels:
        return True, stats
    if bg_frac >= spec.bg_min_fraction:
        return False, stats
    return None, stats


def detect_shiny(grab: Grabber, region: Region, spec: StarSpec) -> Tuple[Optional[bool], dict]:
    return detect_star(grab.capture_region(region, fresh=False), spec)


def detect_tint(grab: Grabber, region: Region, spec: StarSpec) -> Tuple[Optional[bool], dict]:
    return detect_star(grab.capture_region(region, fresh=False), spec)


# --------------------------------------------------------------------------- #
# Gradient chip presence (below the wild box)
# --------------------------------------------------------------------------- #

def detect_gradient(crop: np.ndarray, spec) -> Tuple[bool, dict]:
    """Present when a sufficient fraction of the band is saturated lettering.

    Gradient chips render in different colorways (dark red on gray, bright
    blue on charcoal, ...), but the letters are always strongly saturated
    while the surrounding background is pastel/striped."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)
    sat = ((s >= spec.sat_min) & (v >= spec.val_min))
    fraction = float(sat.mean())
    return fraction >= spec.min_fraction, {
        "sat_fraction": round(fraction, 4),
        "threshold": spec.min_fraction,
    }


def detect_gradient_stable(grab: Grabber, region: Region, spec,
                           attempts: int = 3, name: str = "gradient") -> Tuple[Optional[bool], dict]:
    """Repeat detection until consecutive reads agree (like detect_star_stable)."""
    verdicts: List[Optional[bool]] = []
    stats = {}
    for _ in range(max(1, attempts)):
        v, stats = detect_gradient(grab.capture_region(region, fresh=True), spec)
        verdicts.append(v)
        if len(verdicts) >= 2 and verdicts[0] == verdicts[1]:
            return verdicts[0], stats
        time.sleep(0.3)
    if all(v is False for v in verdicts):
        return False, stats
    if all(v is True for v in verdicts):
        return True, stats
    log.info("%s band unstable over %d reads, marking UNKNOWN (stats=%s)",
             name, attempts, stats)
    return None, stats


# --------------------------------------------------------------------------- #
# Aspect chip gates (two-slot reader, see config.AspectChipSpec)
# --------------------------------------------------------------------------- #

def detect_reskin_letters(crop: np.ndarray, spec) -> Tuple[bool, dict]:
    """White reskin-slot gate: shape-filtered letter blobs + clean canvas.

    The slot renders reskin badges ('MAGMA', 'DOODLE', 'FOREST GUARDIAN')
    as bright italic letter directly on the streaky panel; panel streaks
    alone produce full-width thin lines which the blob-shape filter drops
    (chipless frames -> 0 blob pixels, chipped -> ~700..1500). Returns a
    canvas (letters black on white) for reliable OCR in stats['canvas']."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)
    mask = ((s <= spec.reskin_letter_s_max)
            & (v >= spec.reskin_letter_v_min)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    total = 0
    for i in range(1, n):
        x, y, w, h, area = (int(x) for x in stats[i])
        if 5 <= w <= 80 and 9 <= h <= 34 and area >= 25:
            keep[lab == i] = 255
            total += area
    canvas = np.full(keep.shape, 255, dtype=np.uint8)
    canvas[keep > 0] = 0
    present = total >= spec.reskin_min_pixels
    return present, {
        "letter_pixels": total,
        "threshold": spec.reskin_min_pixels,
        "canvas": canvas,
    }


def detect_color_chip(crop: np.ndarray, spec) -> Tuple[bool, dict]:
    """Gradient-slot gate: the dark chip box under the letters.

    The box (v ~121) is part of the chip graphic itself and renders in
    every biome, while the area below the panel otherwise shows the
    biome background -- which can be saturated grass, lava or waves.
    Therefore a saturated-letter count alone cannot prove a chip: this
    biome's background trips it (false gradient marker -> catching
    reskin-only wilds). The dark-box pixel count is the gate; the
    saturated-letter count is kept for diagnostics only."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)
    dark = int(cv2.countNonZero(((s <= spec.dark_box_s_max)
                                 & (v >= spec.dark_box_v_min)
                                 & (v <= spec.dark_box_v_max))
                                .astype(np.uint8)))
    sat = int(cv2.countNonZero(((s >= spec.grad_sat_min)
                                & (v >= spec.grad_val_min)).astype(np.uint8)))
    present = dark >= spec.dark_box_min_pixels
    return present, {
        "dark_pixels": dark,
        "color_pixels": sat,
        "threshold": spec.dark_box_min_pixels,
    }


def detect_star_stable(grab: Grabber, region: Region, spec: StarSpec,
                       attempts: int = 3, name: str = "star") -> Tuple[Optional[bool], dict]:
    """Repeat detection until the verdict is stable (never flips to True from noise)."""
    verdicts: List[Optional[bool]] = []
    stats = {}
    for _ in range(max(1, attempts)):
        v, stats = detect_star(grab.capture_region(region, fresh=True), spec)
        verdicts.append(v)
        if len(verdicts) >= 2 and verdicts[0] == verdicts[1]:
            return verdicts[0], stats
        time.sleep(0.3)
    if all(v is False for v in verdicts):
        return False, stats
    if any(v is True for v in verdicts) and all(v in (True, False) for v in verdicts):
        return True, stats
    log.info("%s indicator unstable over %d reads, marking UNKNOWN (stats=%s)",
             name, attempts, stats)
    return None, stats


# --------------------------------------------------------------------------- #
# HP bar state
# --------------------------------------------------------------------------- #

def favorite_star(crop: np.ndarray, spec) -> bool:
    """True when the yellow favorite-star badge is drawn inside `spec.zone`
    of a backpack item-row crop.

    The badge sits at the icon's bottom-left corner; favorited rows are the
    only rows allowed to be clicked by the favorites routing, and a yellow
    quantity chip never lands inside this zone (chips live on the far right).
    """
    zx, zy, zw, zh = spec.zone
    zone = crop[zy:zy + zh, zx:zx + zw]
    if zone.size == 0:
        return False
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(spec.hsv_min, np.uint8),
                       np.array(spec.hsv_max, np.uint8))
    denom = zone.shape[0] * zone.shape[1]
    if denom <= 0:
        return False
    return (cv2.countNonZero(mask) / denom) >= spec.min_fraction


def hp_state(crop: np.ndarray, spec: HPSpec, sample_fraction: float = 0.25,
             min_fill_pixels: int = 20) -> Tuple[Optional[str], dict]:
    """Classify an HP bar as red / yellow / green / unknown.

    The game bars draw the *fill* at their LEFT edge; the empty remainder is a
    dark track. At 1 HP the red fill is only a few pixels wide, so classification
    samples the leftmost `sample_fraction` of the bar and requires the winning
    color to exceed min_fill_pixels. Ambiguous -> None (caller retries).
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    w = crop.shape[1]
    left = hsv[:, : max(1, int(w * sample_fraction))]
    counts = {}
    for name, (lo, hi) in spec.colors.items():
        mask = cv2.inRange(left, np.array(lo, np.uint8), np.array(hi, np.uint8))
        counts[name] = int(cv2.countNonZero(mask))
    best = max(counts, key=counts.get)
    counts_sorted = sorted(counts.values(), reverse=True)
    tied = len(counts_sorted) > 1 and counts_sorted[0] == counts_sorted[1]
    if tied:
        return None, counts
    if counts[best] >= min_fill_pixels:
        return best, counts
    return None, counts


def detect_hp(grab: Grabber, region: Region, spec: HPSpec,
              attempts: int = 3, sample_fraction: float = 0.25,
              min_fill_pixels: int = 20) -> Tuple[Optional[str], dict]:
    for i in range(max(1, attempts)):
        state, counts = hp_state(grab.capture_region(region, fresh=(i > 0)), spec,
                                 sample_fraction=sample_fraction,
                                 min_fill_pixels=min_fill_pixels)
        if state is not None:
            return state, counts
        time.sleep(0.35)
    return None, counts


# --------------------------------------------------------------------------- #
# MUTE-button encounter state monitor
# --------------------------------------------------------------------------- #

class MuteMonitor:
    """Tracks the [MUTE] button to detect active / idle encounter state.

    Verified against the real game (calib_shots): the red Mute[?] button exists
    ONLY while a battle/encounter is active; the idle overworld shows other UI
    (HIDE PLAYERS etc.) in that area. Detection = fraction of strong-red pixels
    inside the configured region, debounced across distinct frames.
    """

    def __init__(self, grab: Grabber, region: Region,
                 red_hsv_min: tuple = (0, 140, 120), red_hsv_max: tuple = (10, 255, 255),
                 min_red_fraction: float = 0.45,
                 active_frames: int = 3, idle_frames: int = 4):
        import cv2
        self.cv2 = cv2
        self.grab = grab
        self.region = region
        self.min_red_fraction = min_red_fraction
        self.red_hsv_min = red_hsv_min
        self.red_hsv_max = red_hsv_max
        self.active_frames_needed = active_frames
        self.idle_frames_needed = idle_frames
        self.texture_sig: Optional[str] = None  # dedupe: only process NEW frames
        self.state = "unknown"
        self.active_count = 0
        self.idle_count = 0

    def red_fraction(self, crop: np.ndarray) -> float:
        hsv = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2HSV)
        mask = self.cv2.inRange(hsv, np.array(self.red_hsv_min, np.uint8),
                                np.array(self.red_hsv_max, np.uint8))
        total = crop.shape[0] * crop.shape[1]
        return float(self.cv2.countNonZero(mask)) / max(1, total)

    def classify(self, frame: np.ndarray) -> Tuple[str, float]:
        """Returns ('active'|'idle', red_fraction) for one frame."""
        crop = crop_region(frame, self.region)
        frac = self.red_fraction(crop)
        return ("active" if frac >= self.min_red_fraction else "idle"), frac

    def update(self, frame: np.ndarray, signature: Optional[str] = None) -> str:
        """Debounced state update. Only NEW frames advance counters, so one
        encounter is never counted as several."""
        if signature is not None and signature == self.texture_sig:
            return self.state
        self.texture_sig = signature

        verdict, frac = self.classify(frame)
        if verdict == "active":
            self.active_count += 1
            self.idle_count = 0
        else:
            self.idle_count += 1
            self.active_count = 0

        if self.active_count >= self.active_frames_needed:
            if self.state != "active":
                log.info("encounter state -> ACTIVE (red fraction %.2f)", frac)
            self.state = "active"
        elif self.idle_count >= self.idle_frames_needed:
            if self.state != "idle":
                log.info("encounter state -> idle (red fraction %.2f)", frac)
            self.state = "idle"
        return self.state

    def learn_baseline(self) -> None:
        """No fingerprint baseline needed for red-button presence detection.

        Kept for API compatibility (called at bot startup).
        """
        frame = self.grab.frame(fresh=True)
        verdict, frac = self.classify(frame)
        self.state = verdict
        log.info("MUTE monitor uses red-button presence detection "
                 "(initial fraction %.2f -> %s)", frac, verdict)

    def wait_until(self, target: str, timeout_s: float, poll_ms: int = 180) -> bool:
        """Wait until the debounced state equals target (or timeout)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.grab.frame(fresh=True)
            sig = self.grab.frame_signature(frame)
            state = self.update(frame, sig)
            if state == target:
                return True
            time.sleep(poll_ms / 1000.0)
        return False
