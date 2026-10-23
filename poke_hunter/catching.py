"""Repeat Ball catching procedure (Section 10).

Sequence per attempt:

    OPENING_BAG -> OPENING_BACKPACK -> OPENING_POKEBALLS
    -> CLEAR_STALE_SEARCH if needed
    -> SELECTING favorites row (zero typing; star-verified)
       fallback: exact-name OCR row select
    -> THROWING (wait for animation)
    -> VERIFYING capture result

The keyboard is NEVER used in the bag (user decision): the legacy typed-search
fallback was removed after live runs showed first-keystroke swallowing ("R"
and "B" eaten, field garbling 'epeat all') and unrecoverable clear loops.
"""
from __future__ import annotations

import time
from typing import Optional

from .actions import RecognizedFailure, Ui
from .logging_setup import get_logger
from .util import normalize_text

log = get_logger("catching")


class CatchAborted(Exception):
    """Catching cannot continue safely (no balls / encounter ended)."""


class CatchProcedure:
    def __init__(self, bot):
        self.bot = bot
        self.ui: Ui = bot.ui

    # ------------------------------------------------------------------ #
    def _verify_encounter_active(self) -> bool:
        # MuteMonitor exposes its state as an attribute (not a method)
        return self.bot.mute.state == "active"

    def _verify_hp_red(self) -> Optional[str]:
        state, fractions = self.bot.read_hp_state()
        return state

    def _search_field_text(self) -> str:
        value, stable = self.ui.region_text("search_field",
                                            whitelist="abcdefghijklmnopqrstuvwxyz ")
        return value if stable and value else ("" if stable else value)

    def _find_and_click_item(self, item: str, attempts: int) -> Optional[int]:
        for scan in range(1, attempts + 1):
            hit = self.ui.find_slot("item_rows", item)
            if hit and hit.exact:
                self.ui.click_row_verifying_detail("item_rows", hit.index)
                return hit.index
            if hit and not hit.exact:
                log.warning("item OCR not exact (%r); retry scan", hit.text)
            self.ui.debug_save(f"item_missing_scan{scan}")
            time.sleep(0.6)
        return None

    def _press_use_verified(self, item: str, row: int) -> None:
        """Press Use on the already-open detail panel (row click is verified
        upstream by the favorites/search selection step)."""
        self.ui.press_use_verified()

    def execute(self) -> str:
        """Run up to config max_ball_attempts catch attempts.

        Returns: 'captured' | 'out_of_balls' | 'hp_not_red' | 'catch_unconfirmed' | 'gave_up'
        """
        item = self.bot.cfg.battle.catch_item
        attempts_cap = self.bot.cfg.automation.max_ball_attempts
        attempt = 0
        outcome = "gave_up"
        while attempt < attempts_cap:
            attempt += 1
            log.info("catch attempt %d/%d (%s)", attempt, attempts_cap, item)
            if not self._verify_encounter_active():
                log.info("encounter already ended before throw %d", attempt)
                return "captured"

            # 10.2/10.3 open Bag -> sidebar "Pokeballs" tab (verified by header)
            opened = False
            for open_try in range(1, self.bot.cfg.automation.max_action_retries + 1):
                self.ui.click_region("bag_button")
                self.ui.settle()
                value, stable = self.ui.region_text("backpack_title")
                if stable and value and "backpack" in value:
                    opened = True
                    break
            if not opened:
                self.ui.debug_save("bag_open_failed", region_name="backpack_title")
                raise RecognizedFailure("Backpack inventory window did not open")

            opened = False
            for open_try in range(1, self.bot.cfg.automation.max_action_retries + 1):
                self.ui.click_region("pokeballs_tab")
                self.ui.settle()
                value, _ = self.ui.region_text("category_header")
                if value and "pokeball" in value:
                    opened = True
                    break
            if not opened:
                self.ui.debug_save("pokeballs_header_missing")
                raise RecognizedFailure("Pokeballs category tab did not open")
            log.debug("bag/backpack/pokeballs open (title + header verified)")

            # favorites routing: no typing. Favorites sort to the top of the
            # category list; first make sure no stale search filter is
            # narrowing the list (the 'Search...' placeholder counts as
            # empty -> no clearing, no typing at all).
            if not self.ui.search_filter_is_clear():
                log.info("search field holds a stale filter; clearing")
                self.ui.clear_search_field()
            selected_index = None
            try:
                selected_index = self.ui.select_favorited_item(item)
            except RecognizedFailure as exc:
                log.warning("favorites route failed for %s (%s); trying "
                            "exact-name row select", item, exc)
                self.ui.debug_save("ball_favorites_fallback")
            if selected_index is None:
                # fallback without a keyboard: exact-name OCR row select
                selected_index = self._find_and_click_item(item, attempts=2)
            if selected_index is None:
                log.error("%s not found in Pokeballs list", item)
                self.bot.notifier.failure(
                    f"{item} not found in inventory (favorites + name scan); "
                    f"cannot continue catching",
                    key=f"noitem:{item}")
                return "out_of_balls"
            self.ui.settle()
            log.info("SELECTING_%s done (row %d)",
                     item.replace(" ", "_").upper(), selected_index)

            # press Use after selecting the verified Repeat Ball entry
            self._press_use_verified(item, selected_index)

            # wait for throw animation
            self.ui.settle(self.bot.cfg.automation.throw_settle_ms)
            result = self.bot.await_capture_result(
                timeout_s=self.bot.cfg.automation.catch_result_timeout_s)
            if result == "captured":
                return "captured"
            if result == "stale_check":
                return "catch_unconfirmed"

            if not self._verify_encounter_active():
                return "captured"  # ended right after the throw

            hp = self._verify_hp_red()
            # None means the bar was unreadable (ball animation / transition):
            # wilds never heal, so already-weakened targets stay weakened ->
            # treat as red and throw again instead of attacking.
            # In sleep mode (Spore) full HP is INTENDED — sleep is the
            # weakening, no hp gate, just throw again while balls remain.
            if (hp is not None
                    and self.bot.cfg.battle.weakening_style != "sleep"
                    and hp != self.bot.cfg.battle.target_hp_state):
                log.info("HP state after failed throw: %r -> back to weakening", hp)
                return "hp_not_red"
            outcome = "catch_unconfirmed"
            log.info("ball failed; target still active (hp=%r%s)", hp,
                     "; sleep mode keeps full HP" if
                     self.bot.cfg.battle.weakening_style == "sleep" else
                     ", weakened to red")
        return outcome
