"""Verified high-level UI actions.

Every action captures a fresh frame and verifies the expected outcome on the
screen before success is reported. Nothing is assumed in the absence of
verification. Verification failures raise RecognizedFailure so the encounter
orchestrator can route to retries / recovery / safe stop.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .capture import Grabber, Region
from .config import AppConfig, Calibration
from .inputio import click, click_region_center, park, select_all_and_delete, type_text
from .logging_setup import get_logger
from .ocr import PSM_BLOCK, PSM_SINGLE_LINE, read_stable, read_text
from .detectors import favorite_star, detect_color_chip, detect_reskin_letters
from .util import crop_region, fuzzy_ratio, normalize_text
from .window import GameWindow

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log = get_logger("actions")


def query_field_empty(value: str) -> bool:
    """True when a search/image-result field holds no real filter text.

    An empty field renders the 'Search...' placeholder overlay, which OCR
    reads back (e.g. raw 'search... )'); that counts as empty."""
    norm = normalize_text(value or "")
    return norm == "" or norm.startswith("search")


class RecognizedFailure(Exception):
    """A UI action could not be verified (expected vs. observed mismatch)."""


@dataclass
class SlotHit:
    index: int
    text: str
    exact: bool


class Ui:
    """Project-wide shared context: one game window + calibration + config."""

    def __init__(self, window: GameWindow, grab: Grabber,
                 calib: Calibration, cfg: AppConfig, instance_name: str = ""):
        self.window = window
        self.grab = grab
        self.calib = calib
        self.cfg = cfg
        self.instance_name = instance_name

    # -- low level -----------------------------------------------------------
    def settle(self, ms: Optional[int] = None) -> None:
        time.sleep((ms or self.cfg.automation.post_click_settle_ms) / 1000.0)

    def click_region(self, name: str) -> None:
        click_region_center(self.window, self.calib.region(name))
        log.debug("clicked region '%s'", name)

    def region_text(self, name: str, psm: int = PSM_SINGLE_LINE,
                    whitelist: Optional[str] = None,
                    attempts: Optional[int] = None, fresh: bool = True,
                    upscale: int = 3) -> Tuple[str, bool]:
        region = self.calib.region(name)
        return read_stable(
            self.grab, region, psm=psm, whitelist=whitelist,
            attempts=attempts or self.cfg.automation.max_ocr_retries,
            retry_delay_s=0.4,
        )

    def read_aspect_chips(self, wait_ms: Optional[int] = None) -> Tuple[str, str, dict]:
        """Poll both aspect-chip slots until they render or the window closes.

        The wild card's aspect chips fade in seconds after the panel settles,
        so a single early read races the fade and reports absent (the bug that
        made ~140 encounters yield zero matches). Slot semantics measured on
        2026-10-22 frame evidence:

          * white/colorless slot (upper, on the panel, y ~354..396) = RESKIN
            badge: mid-gray lettering ('MAGMA', 'CRAYON', 'FOREST
            GUARDIAN'). Letter pixel-counts are unreliable (panel streak
            texture matches letter counts), so this slot is OCR-only with
            the existing no-letter noise guard.
          * colored slot (lower, dark chip box, y ~398..442) = GRADIENT
            badge: saturated colorway lettering ('GLITCHED Rufescent',
            'JACK-O-LANTERN', 'Radiance', 'Verdalis').

        ``wait_ms`` optionally overrides the calibration window (the
        encounter orchestrator adapts it from observed chip-trip times).

        Returns (reskin, gradient, stats) using aspect semantics:
        '' = readable-but-absent | a word = present | '<x>-present' =
        sure-but-unlabelled. Absence at the window close is sure."""
        spec = self.calib.aspect_chips
        if spec is None:
            raise RecognizedFailure("aspect_chips calibration missing")
        t0 = time.monotonic()
        deadline = t0 + (wait_ms if wait_ms is not None
                         else spec.wait_ms) / 1000.0
        poll = spec.poll_ms / 1000.0
        reskin, gradient = "", ""
        reskin_done = False
        reskin_ocr_fails = 0
        grad_seen = False          # gate tripped at least once
        grad_reads = 0             # OCR attempts on tripped crops
        stats: dict = {}
        while True:
            frame = self.grab.frame(fresh=True)
            if not reskin_done:
                rcrop = crop_region(frame, spec.reskin_zone)
                r_trips, rstats = detect_reskin_letters(rcrop, spec)
                stats["reskin_pixels"] = rstats["letter_pixels"]
                if r_trips:
                    res = read_text(rstats["canvas"])
                    norm = normalize_text(res.raw)
                    if any(c.isalpha() for c in norm):
                        reskin = norm
                        reskin_done = True
                        stats["reskin_trip_ms"] = int((time.monotonic() - t0) * 1000)
                    elif reskin_ocr_fails >= 1:
                        # letter-shaped pixels persist but OCR fails twice
                        reskin = "reskin-present"
                        reskin_done = True
                        stats["reskin_trip_ms"] = int((time.monotonic() - t0) * 1000)
                    else:
                        reskin_ocr_fails += 1
            if reskin_done and gradient:
                break
            gcrop = crop_region(frame, spec.gradient_zone)
            g_trips, _gstats = detect_color_chip(gcrop, spec)
            if g_trips:
                if not grad_seen:
                    grad_seen = True
                    stats["gradient_trip_ms"] = int((time.monotonic() - t0) * 1000)
                grad_reads += 1
                grads = read_text(gcrop.copy())
                gnorm = normalize_text(grads.raw)
                if any(c.isalpha() for c in gnorm):
                    gradient = gnorm
                    break
                if grad_reads >= 2:
                    # sure (dark chip box) but unlabelled twice: keep it
                    gradient = "gradient-present"
                    break
            if time.monotonic() >= deadline:
                break
            time.sleep(poll)
        stats["reskin_pixels"] = int(stats.get("reskin_pixels", 0))
        stats["gradient_reads"] = grad_reads
        return reskin, gradient, stats

    @staticmethod
    def _ocr_chip_label(crop: np.ndarray, fallback: str) -> str:
        """OCR a chip crop for its word; unlabelled-but-sure falls back."""
        res = read_text(crop.copy())
        norm = normalize_text(res.raw)
        if any(c.isalpha() for c in norm):
            return norm
        return fallback

    def debug_save(self, tag: str, frame: Optional[np.ndarray] = None,
                   region_name: Optional[str] = None) -> str:
        from .capture import save_debug_image
        frame = frame if frame is not None else self.grab.frame()
        region = self.calib.regions.get(region_name) if region_name else None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(ROOT_DIR, "logs",
                            f"{self.instance_name}_{tag}_{stamp}")
        path = save_debug_image(f"{base}_full.png", frame, region)
        log.info("saved debug image %s", path)
        return path

    # -- move-slot scanning ---------------------------------------------------
    def scan_grid(self, grid_name: str, whitelist: Optional[str] = None,
                  psm: int = PSM_SINGLE_LINE, fresh: bool = True,
                  binary_threshold: Optional[int] = None
                  ) -> List[Tuple[str, float]]:
        """OCR every slot of a configured grid. Returns normalized texts.

        If the grid defines a text_zone, only that sub-rectangle (relative to
        each slot) is OCR'd — used for list rows whose icon/star/quantity tag
        break Tesseract's line parser. Click/click-verify geometry is not
        affected: slot(i) always returns the full-slot region.

        binary_threshold: binarize each slot crop at this gray level before
        OCR. Needed for the move grid's SELECTED tile (white text on a
        light-gray hover background reads as garbage in normal mode;
        live-verified '2'/40 :' instead of 'False Swipe 32/40')."""
        grid = self.calib.grid(grid_name)
        out: List[Tuple[str, float]] = []
        for i in range(grid.count):
            img = self.grab.capture_region(grid.slot(i), fresh=fresh)
            if grid.text_zone is not None:
                img = crop_region(img, grid.text_zone)
            if binary_threshold is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, img = cv2.threshold(gray, binary_threshold, 255,
                                       cv2.THRESH_BINARY)
            res = read_text(img, psm=psm, whitelist=whitelist)
            out.append((res.value, res.confidence))
        return out

    def find_slot(self, grid_name: str, wanted: str, whitelist: Optional[str] = None,
                  fuzzy: float = 0.86, psm: Optional[int] = None,
                  binary_threshold: Optional[int] = None) -> Optional[SlotHit]:
        """Locate a labelled slot by OCRing each slot; exact match preferred.

        psm: pass PSM_BLOCK for tiles whose text spans multiple lines
        (e.g. move tiles carry a second PP/type line)."""
        wanted_norm = normalize_text(wanted)
        texts = self.scan_grid(grid_name, whitelist=whitelist,
                               psm=psm if psm is not None else PSM_SINGLE_LINE,
                               binary_threshold=binary_threshold)
        log.debug("%s scan: %s", grid_name, texts)
        # de-noise: slot text like "false swipe" may include trailing junk
        for i, (t, _) in enumerate(texts):
            if t == wanted_norm:
                return SlotHit(index=i, text=t, exact=True)
        for i, (t, _) in enumerate(texts):
            if t and (wanted_norm in t or fuzzy_ratio(t[: len(wanted_norm)], wanted_norm) >= fuzzy):
                return SlotHit(index=i, text=t, exact=False)
        return None

    def click_grid_slot(self, grid_name: str, index: int) -> None:
        grid = self.calib.grid(grid_name)
        click_region_center(self.window, grid.slot(index))

    # -- favorites routing (no search typing) --------------------------------
    def select_favorited_item(self, item: str, scans: int = 2,
                              fuzzy: float = 0.92) -> int:
        """Favorites route: click the row showing `item` WITH its verified
        yellow favorite-star badge. Never types into the search field.

        The user favorited Repeat Ball / Full Restore / Max Elixir so they
        sort to the top of their category lists and stay in the first rows.
        Raises RecognizedFailure when no verified favorited row appears
        (callers fall back to the legacy search route)."""
        wanted = normalize_text(item)
        spec = self.calib.favorite_star
        grid = self.calib.grid("item_rows")
        for scan in range(1, scans + 1):
            texts = self.scan_grid("item_rows")
            log.debug("item_rows scan: %s", texts)
            for i, (t, _) in enumerate(texts):
                if not t:
                    continue
                is_match = (t == wanted
                            or fuzzy_ratio(t[: len(wanted)], wanted) >= fuzzy)
                if not is_match:
                    continue
                img = self.grab.capture_region(grid.slot(i), fresh=True)
                star_ok = favorite_star(img, spec) if spec is not None else True
                if not star_ok:
                    # Only relax the pixel check for an EXACT OCR match inside
                    # the two top rows (the user's favorites live there; the
                    # favorite badge calibrates after first live evidence).
                    if t != wanted or i > 1:
                        log.info("row %d matches %r but favorite star "
                                 "unverified", i, item)
                        continue
                    log.warning("favorite star unverified on exact-match row "
                                "%d; accepting (top-of-list favorite)", i)
                    self.debug_save(f"favorite_star_unverified_row{i}")
                self.click_row_verifying_detail("item_rows", i)
                log.info("SELECT_FAVORITE %s -> row %d (text %r, star %s)",
                         item, i, t, "ok" if star_ok else "relaxed")
                return i
            self.debug_save(f"favorite_row_missing_scan{scan}")
            time.sleep(0.8)
        raise RecognizedFailure(
            f"{item} not found as a favorited row after {scans} scans")

    # -- item detail panel -----------------------------------------------------
    def _use_button_green_fraction(self) -> float:
        """Green-pixel fraction inside the use_button region. The green Use
        button fills ~65% of that region while the detail bar is open and
        ~0% when closed — a far more reliable gate than OCR-ing the italic
        'Use' glyph, which flaked live (conf -1 with the button plainly
        visible) and once caused a re-click into the wrong row."""
        img = self.grab.capture_region(self.calib.region("use_button"), fresh=True)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) &
                (hsv[..., 1] >= 90) & (hsv[..., 2] >= 90))
        return float(mask.mean())

    def item_detail_open(self) -> bool:
        """True while an item row was clicked and its bottom detail bar shows
        the green Use button (pixel-gated; see _use_button_green_fraction).
        The detail bar RESIZES the window (rows shift up ~44px), so this
        must gate any re-click on list coordinates."""
        return self._use_button_green_fraction() >= 0.20

    def click_row_verifying_detail(self, grid_name: str, index: int) -> None:
        """Click an item row and confirm the detail panel opened.

        A swallowed first click is retried ONCE — safe, because we only
        reach the retry while `item_detail_open()` is False, i.e. the list
        still uses its full-size geometry. Re-clicking with stale row
        coordinates while the detail is open once selected the WRONG item
        (user-reported: mouse landed on Beast Ball after the rows shifted
        up with the detail bar open), hence the strict gating. The cursor
        is parked right after the click so it can never rest on a shifted
        row between the click and the verification."""
        self.click_grid_slot(grid_name, index)
        self.park_mouse()
        if self.item_detail_open():
            return
        time.sleep(0.5)
        self.click_grid_slot(grid_name, index)
        self.park_mouse()
        if self.item_detail_open():
            return
        self.debug_save(f"detail_after_select_missing_r{index}")
        raise RecognizedFailure("item detail panel did not open after row click")

    def press_use_verified(self) -> None:
        """Press the open detail panel's Use with click -> verify -> retry.

        Physical first presses can be swallowed (pydirectinput quirk); a
        swallowed Use press once left the flow believing a ball was thrown.
        The green button must be present before pressing and gone after."""
        for attempt in range(1, self.cfg.automation.max_action_retries + 1):
            if not self.item_detail_open():
                # detail may take a beat to (re)render
                time.sleep(0.6)
                if not self.item_detail_open():
                    log.info("Use button not visible before press "
                             "(attempt %d)", attempt)
                    continue
            self.click_region("use_button")
            self.park_mouse()
            self.settle(900)
            if self.item_detail_open():
                log.info("Use press %d not registered (button still "
                         "shown); pressing again", attempt)
                continue
            log.info("USE_BUTTON done (panel closed after press %d)", attempt)
            return
        self.debug_save("use_press_failed")
        raise RecognizedFailure("detail panel Use press could not be verified")

    # -- verified actions ------------------------------------------------------
    def fight_prompt_visible(self) -> bool:
        """True while the main action-menu prompt ('What should <X> do?')
        occupies the dialog strip, i.e. the move grid is NOT open.

        The move grid and the prompt share the same screen band; OCR text is
        the only reliable discriminator between the two states."""
        value, stable = self.region_text("dialog_strip", attempts=2)
        if not stable:
            return False
        return "what should" in normalize_text(value)

    def open_fight_menu(self) -> bool:
        """Click Fight; verify the MOVE grid opened (move text present AND
        the 'What should X do?' prompt gone). Any-text-only checks would
        false-verify on the main action menu's own buttons."""
        self.click_region("fight_button")
        self.settle()
        for attempt in range(1, self.cfg.automation.max_action_retries + 1):
            # PSM_BLOCK: tiles are two-line (name + "pp type category");
            # PSM_SINGLE_LINE reads ZERO tokens on them (live-verified).
            texts = self.scan_grid("move_slots", psm=PSM_BLOCK)
            if any(t for t, _ in texts) and not self.fight_prompt_visible():
                log.debug("fight menu verified, moves: %s", texts)
                return True
            log.info("move grid not visible yet (attempt %d)", attempt)
            self.click_region("fight_button")
            self.settle(self.cfg.automation.ui_settle_ms)
        self.debug_save("fight_menu_unverified")
        raise RecognizedFailure("Fight menu did not open (no move labels readable)")

    def click_move(self, hit: SlotHit) -> None:
        self.click_grid_slot("move_slots", hit.index)

    def navigate_to_tab(self, tabs: List[str]) -> None:
        """Open a sequence of tabs, verifying each by re-OCR the tab label."""
        for tab in tabs:
            region = self.calib.region(tab)
            expected = normalize_text(tab.replace("_tab", "").replace("_", " "))
            self.click_region(tab)
            self.settle(self.cfg.automation.ui_settle_ms)
            value, stable = None, False
            for attempt in range(1, self.cfg.automation.max_ocr_retries + 1):
                value, stable = self.region_text(tab, whitelist="abcdefghijklmnopqrstuvwxyz")
                if stable and value:
                    break
                log.info("tab '%s' not readable yet (attempt %d)", tab, attempt)
                self.click_region(tab)
                self.settle()
            if not (stable and value and fuzzy_ratio(value, expected) >= 0.7):
                self.debug_save("tab_unverified", region_name=tab)
                raise RecognizedFailure(f"Could not verify tab '{tab}' is active "
                                        f"(read {value!r}, stable={stable})")
            log.debug("tab '%s' verified as active", tab)

    def clear_search_field(self) -> bool:
        """Select-all + delete in the search field; verify the field is empty.

        An empty field renders the 'Search...' PLACEHOLDER text — OCR reads
        that overlay back, so it counts as empty. Distinct state with its own
        verification (Section 10.5).
        """
        self.click_region("search_field")
        time.sleep(0.2)
        for attempt in range(1, self.cfg.automation.max_action_retries + 1):
            # fast path first; the verification below escalates to the
            # backspace sweep only if the field is not actually empty
            select_all_and_delete(self.window, sweep=attempt > 1)
            time.sleep(0.35)
            value, stable = self.region_text("search_field", attempts=2)
            if stable and query_field_empty(value):
                log.debug("search field verified empty")
                return True
            log.info("search field still shows %r (attempt %d)", value, attempt)
        self.debug_save("search_clear_failed", region_name="search_field")
        raise RecognizedFailure(f"search field could not be cleared (shows {value!r})")

    def search_filter_is_clear(self) -> bool:
        """True when the search field holds no filter text (empty or only
        the 'Search...' placeholder overlay)."""
        value, stable = self.region_text(
            "search_field", whitelist="abcdefghijklmnopqrstuvwxyz ")
        if not stable:
            return False
        return query_field_empty(value)

    def flee_click(self) -> None:
        self.click_region("run_button")

    def park_mouse(self) -> None:
        """Cursor to the inert park point: nothing may accidentally press
        the button the mouse happens to rest on."""
        park(self.window)
