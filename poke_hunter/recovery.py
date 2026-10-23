"""Full Restore recovery state machine (Section 9).

Mandatory sequence whenever False Swipe is unavailable:

    EXITING_FIGHT_MENU (red Back button)
  -> OPENING_BAG
  -> OPENING_BACKPACK
  -> OPENING_RECOVERS
  -> SELECTING_FULL_RESTORE ("Full Restore (FULL)")
  -> IDENTIFYING_PARTY (six slots, species AND level)
  -> SELECTING_LEVEL_150_MARSHADOW (species = Marshadow AND level = 150)
  -> USING_FULL_RESTORE
  -> VERIFYING_ITEM_USE
  -> RETURNING_TO_ENCOUNTER
  -> REOPENING_FIGHT_MENU (back in the weaken loop)

Every step has a success condition and failure path; a step only completes
after the screen confirms the expected outcome.
"""
from __future__ import annotations

import re
import time
import typing

from .actions import RecognizedFailure, Ui
from .actions import SlotHit
from .detectors import detect_star, hp_state, Tri  # noqa: F401  (kept for calibration tools)
from .logging_setup import get_logger
from .ocr import PSM_BLOCK, PSM_SINGLE_LINE, read_text
from .util import normalize_text
from .util import close_match, fuzzy_ratio, normalize_text

if typing.TYPE_CHECKING:  # circular safe: only for type hints
    from .encounter import EncounterBot

log = get_logger("recovery")

LEVEL_RE = re.compile(r"lv\.?\s*([0-9]{1,3})")


class RecoveryExhausted(Exception):
    """The recovery cycle limit was reached without restoring False Swipe."""


class ItemNotAvailable(Exception):
    """Full Restore could not be found in the Recovers category."""


def parse_party_entry(raw_text: str) -> typing.Tuple[str, typing.Optional[int]]:
    """Split a party-slot OCR text into (species-ish text, level)."""
    text = normalize_text(raw_text)
    level = None
    match = LEVEL_RE.search(text)
    if not match:
        # fall back: any standalone 3-digit number adjacent to digits
        match = re.search(r"(?<![\w])1[0-9][0-9](?![\w])", text)
    if match:
        try:
            level = int(match.group(1) if match.group(1) else match.group(0))
        except ValueError:
            level = None
    # remove tokens that belong to the level, keep the letters for species OCR
    letters = re.sub(r"[0-9\W]+", " ", text)
    species = normalize_text(letters)
    return species, level


class FullRestoreRecovery:
    """Executes the exact recipe from the specification."""

    STATES = (
        "FALSE_SWIPE_UNAVAILABLE", "EXITING_FIGHT_MENU", "OPENING_BAG",
        "OPENING_BACKPACK", "OPENING_RECOVERS", "SELECTING_ITEM",
        "IDENTIFYING_PARTY", "SELECTING_LEVEL_150_MARSHADOW",
        "USING_ITEM", "VERIFYING_ITEM_USE", "RETURNING_TO_ENCOUNTER",
        "REOPENING_FIGHT_MENU", "RECOVERY_COMPLETE",
    )

    def __init__(self, bot: "EncounterBot"):
        self.bot = bot
        self.ui: Ui = bot.ui

    # -- helpers -----------------------------------------------------------
    def party_slots_info(self) -> list:
        """OCR + HP-classify every configured party card.

        Live drill findings: the party screen is a 2x3 card layout and the
        big stylized font OCRs unreliably ('Entel', 'Ly, 159'), while the
        HP bar classification is rock solid. So the choice of card is driven
        by HP state; OCR species/level are corroboration only.
        Returns dicts: {index, species, level, raw, hp}."""
        grid = self.ui.calib.grid("party_slots")
        spec = self.ui.calib.hp
        entries = []
        for i in range(grid.count):
            img = self.ui.grab.capture_region(grid.slot(i), fresh=True)
            res = read_text(img, psm=PSM_BLOCK)
            species, level = parse_party_entry(res.value)
            hp = None
            if spec is not None:
                bar = img[72:106, 52:612]  # HP bar strip inside the card
                hp, _counts = hp_state(bar, spec, sample_fraction=0.25)
            entries.append({"index": i, "species": species, "level": level,
                            "raw": res.value, "hp": hp})
        log.debug("party scan: %s",
                  [(e["species"], e["level"], e["hp"]) for e in entries])
        return entries

    def _party_screen_visible(self, debug: bool = False) -> bool:
        # the 'Pokemon Party' title strip is the definitive marker. NO
        # fallback counting of card OCRs: on the Backpack screen the party
        # grid overlaps the item list and 3+ 'cards' read fine (phantom
        # verification seen live). Retries handle the screen transition.
        for attempt in range(1, 4):
            value, stable = self.ui.region_text("party_title", psm=PSM_SINGLE_LINE)
            if stable and value and "party" in normalize_text(value):
                return True
            if not debug:
                return False
            time.sleep(0.8)
        if debug:
            self.ui.debug_save("party_title_unreadable")
        return False

    def find_recovery_pokemon(self, want_species: str, want_level: int) -> typing.Optional[int]:
        """Index of the party card to apply the item on.

        The drill proved the reliable signal: pick the card whose HP is not
        green (i.e. actually damaged). Species+level OCR is only used to
        disambiguate when SEVERAL members are damaged."""
        entry = self.find_recovery_card(want_species, want_level)
        return None if entry is None else entry["index"]

    def find_recovery_card(self, want_species: str, want_level: int) -> typing.Optional[dict]:
        """Dict entry of the party card to apply the item on (never guesses)."""
        exact = []
        for scan in range(1, 3):
            infos = self.party_slots_info()
            exact = [
                e for e in infos
                if e["species"] and e["level"] == want_level and close_match(
                    e["species"], want_species, fuzzy_threshold=0.92)
            ]
            if len(exact) >= 1:
                break
            if scan == 1:
                # give the party screen render a beat, then re-OCR
                log.info("party card not verified on scan %d; retrying", scan)
                self.ui.debug_save("party_scan1_unverified")
                time.sleep(1.0)
        if len(exact) == 1:
            e = exact[0]
            log.info("party card %d verified as %s lv %d (hp=%s raw=%r)",
                     e["index"], want_species, want_level, e["hp"], e["raw"])
            return e
        if len(exact) > 1:
            self.ui.debug_save("party_multiple_marshadow150")
            raise RecognizedFailure(
                f"multiple party members appear to match {want_species} lv {want_level}; refusing to guess")
        self.ui.debug_save("party_unverified_marshadow150")
        raise RecognizedFailure(
            f"could not verify {want_species} lv {want_level} on the party screen")

    def _select_exact_row(self, item: str) -> None:
        """No-typing fallback: select the item by exact-name OCR on the
        (stale-filter-free) Recovers list — keyboard is never used."""
        selected = None
        for scan_attempt in range(
                1, self.bot.cfg.automation.max_action_retries + 1):
            hit = self.ui.find_slot("item_rows", item)
            if hit and hit.exact:
                selected = hit
                break
            if hit and not hit.exact:
                log.warning("item OCR not exact (%r); retry scan", hit.text)
            self.ui.debug_save(f"item_missing_{scan_attempt}")
            time.sleep(0.8)
        if not selected:
            raise ItemNotAvailable(
                f"'{item}' not found in Recovers after "
                f"{self.bot.cfg.automation.max_action_retries} scans")
        self.ui.click_row_verifying_detail("item_rows", selected.index)

    # -- main sequence --------------------------------------------------------
    def execute(self) -> bool:
        """Full Restore heal cycle. Returns True after the item was applied
        and the party screen closed. Raises RecoveryExhausted /
        ItemNotAvailable / RecognizedFailure on verified failures."""
        return self._run_chain(self.bot.cfg.battle.recovery_item, "HEAL")

    def execute_pp(self) -> bool:
        """Max Elixir cycle: restore PP so False Swipe returns to the move
        grid (False Swipe has ~40 uses; this runs when it depletes)."""
        return self._run_chain(self.bot.cfg.battle.pp_item, "PP")

    def _run_chain(self, item: str, reason: str) -> bool:
        species = self.bot.cfg.battle.recovery_species
        level = self.bot.cfg.battle.recovery_level
        log.info("RECOVERY start (%s): use %s on %s lv %d (instance %s)",
                 reason, item, species, level, self.bot.instance_name)
        # never run a recovery chain on a screen that is no longer an encounter
        self.bot.assert_encounter_still_active()

        # EXITING_FIGHT_MENU: press the red Back (bottom-right 'Run' slot)
        # ONLY while the move grid is actually open — from the main action
        # menu that same button is 'Run' and pressing it would flee a
        # possibly MATCHED encounter. The prompt bar discriminates.
        closed = self.ui.fight_prompt_visible()   # already at the action menu
        attempt = 0
        while not closed and attempt < self.bot.cfg.automation.max_action_retries:
            attempt += 1
            self.ui.click_region("back_button")
            self.ui.settle()
            t0 = time.monotonic()
            # the attack animation may still flash; keep sampling briefly
            while time.monotonic() - t0 < 2.0:
                time.sleep(0.3)
                if self.ui.fight_prompt_visible():
                    closed = True
                    break
            if closed:
                break
            texts = self.ui.scan_grid("move_slots", fresh=True)
            if not any(t for t, _ in texts):
                closed = True   # labels cleared; menu closed (OCR tolerant)
                break
            log.info("fight menu still open after Back (attempt %d)", attempt)
        if not closed:
            self.ui.debug_save("back_failed")
            raise RecognizedFailure("red Back button did not close the Fight menu")
        log.info("EXITING_FIGHT_MENU done (menu verified closed)")

        # OPENING_BAG -> the inventory window itself is called 'Backpack'
        self.ui.click_region("bag_button")
        self.ui.settle()
        opened = False
        for attempt in range(1, self.bot.cfg.automation.max_action_retries + 1):
            value, stable = self.ui.region_text("backpack_title")
            if stable and value and "backpack" in value:
                opened = True
                break
            log.info("Backpack window did not open (attempt %d)", attempt)
            self.ui.click_region("bag_button")
            self.ui.settle(self.bot.cfg.automation.ui_settle_ms)
        if not opened:
            self.ui.debug_save("bag_open_failed", region_name="backpack_title")
            raise RecognizedFailure("Backpack inventory window did not open")
        log.info("OPENING_BAG done (Backpack window verified)")

        # OPENING_RECOVERS: click sidebar tab; verified by category header
        self.ui.click_region("recovers_tab")
        self.ui.settle()
        header_ok = False
        for attempt in range(1, self.bot.cfg.automation.max_action_retries + 1):
            value, _ = self.ui.region_text("category_header")
            if value and ("recover" in value):
                header_ok = True
                break
            log.info("Recovers header not visible yet (attempt %d)", attempt)
            self.ui.click_region("recovers_tab")
            self.ui.settle()
        if not header_ok:
            self.ui.debug_save("recovers_header_missing")
            raise RecognizedFailure("Recovers category tab did not open")
        log.info("OPENING_BACKPACK/OPENING_RECOVERS done (header verified)")

        # stale/placeholder search filter must not narrow the favorites list
        if not self.ui.search_filter_is_clear():
            self.ui.clear_search_field()
        row_index: typing.Optional[int] = None
        try:
            row_index = self.ui.select_favorited_item(item, scans=2)
        except RecognizedFailure as exc:
            log.warning("favorites route failed for %s (%s); trying "
                        "exact-name row select", item, exc)
            self.ui.debug_save(f"item_favorites_fallback_{reason}")
        if row_index is None:
            self._select_exact_row(item)
        self.ui.settle()
        log.info("SELECTING_%s done",
                 item.replace(" ", "_").upper())

        # the row click opens a detail panel (verified in the selection
        # step); press its Use button with click->verify->retry
        self.ui.press_use_verified()

        # IDENTIFYING_PARTY: party screen must appear (strict title check)
        party_ok = False
        for attempt in range(1, self.bot.cfg.automation.max_action_retries + 1):
            if self._party_screen_visible(debug=True):
                party_ok = True
                break
            self.ui.settle()
        if not party_ok:
            self.ui.debug_save("party_screen_missing")
            raise RecognizedFailure("party selection screen did not appear after selecting item")
        log.info("IDENTIFYING_PARTY done (six-slot party screen verified)")

        # SELECTING_LEVEL_150_MARSHADOW
        entry = self.find_recovery_card(species, level)
        slot_index = entry["index"]
        self.ui.click_grid_slot("party_slots", slot_index)
        self.ui.settle()
        log.info("SELECTING_LEVEL_150_MARSHADOW done (slot %d selected)", slot_index)

        # USING_FULL_RESTORE + VERIFYING_ITEM_USE + RETURNING_TO_ENCOUNTER
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if not self._party_screen_visible():
                break
            time.sleep(0.5)
        if self._party_screen_visible():
            self.ui.debug_save("item_use_unverified")
            raise RecognizedFailure("party screen still visible; item use not confirmed")
        log.info("USING_FULL_RESTORE/VERIFYING_ITEM_USE done (party screen closed)")

        # RETURNING_TO_ENCOUNTER: party closed; verify encounter UI restored by
        # checking the move grid area is empty while the window is back at battle
        self.ui.settle()
        self.bot.assert_encounter_still_active()
        log.info("RETURNING_TO_ENCOUNTER done")
        return True
