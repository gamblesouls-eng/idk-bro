"""Encounter orchestrator (Sections 5, 6, 7, 8, 10, 12).

Single-instance loop:

    WAITING_FOR_ENCOUNTER - MUTE monitor with debounce
    -> ENCOUNTER_DETECTED -> WAITING_FOR_UI_STABILITY
    -> species / shiny / reskin / gradient / tint identification
    -> MATCHING_TARGET
          MATCH    -> notify -> weaken (False Swipe) -> Repeat Ball catch
          NO_MATCH -> automatic RUN (flee)
          UNKNOWN  -> safe fallback (flee) after retries   (never catches on unknown)
    -> VERIFYING end -> RESETTING_ENCOUNTER -> loop

State transitions are logged; every action result is verified on-screen.
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Tuple

from .actions import RecognizedFailure, Ui
from .catching import CatchAborted, CatchProcedure
from .config import AppConfig, Calibration
from .detectors import (MuteMonitor, Tri, detect_gradient_stable, detect_hp,
                        detect_star_stable)
from .logging_setup import get_logger
from .matching import Detected, MatchResult, evaluate, evaluate_all
from .notify import Notifier
from .ocr import PSM_BLOCK, read_text
from .recovery import FullRestoreRecovery, RecoveryExhausted, ItemNotAvailable
from .saver import GameSaver, SaveUnverified
from .util import crop_region, normalize_text
from .window import GameWindow

log = get_logger("encounter")

CAUGHT_WORDS = ("caught", "captured", "caught it", "was caught")


class SafeStop(Exception):
    """Automation stopped for this instance because a safe continuation is unknown."""


class EncounterEnded(Exception):
    """Encounter ended while a battle action was being processed."""


class EncounterBot:
    def __init__(self, window: GameWindow, calib: Calibration,
                 cfg: AppConfig, notifier: Notifier, instance_name: str = ""):
        self.window = window
        self.calib = calib
        self.cfg = cfg
        self.notifier = notifier
        self.instance_name = instance_name
        self.ui = Ui(window, window_grabber(window), calib, cfg, instance_name)
        self.grab = self.ui.grab
        mute_region = calib.region("mute_button")
        self.mute = MuteMonitor(
            self.grab, mute_region,
            active_frames=cfg.automation.encounter_stability_frames,
            idle_frames=cfg.automation.encounter_end_frames,
        )
        self.recovery = FullRestoreRecovery(self)
        self.catching = CatchProcedure(self)
        self.saver = GameSaver(self.ui)
        self.recovery_cycles_used = 0
        self.heal_cycles_used = 0
        self.pp_cycles_used = 0
        self.state = "WAITING_FOR_ENCOUNTER"
        self.encounter_counter = 0
        self.current_encounter_id = ""
        self._current_detected_text: str = ""
        self._matched_target_ids: list[str] = []
        # default True: identify() overwrites it every encounter; a target
        # flow that never reached identify falls back to inline reads
        self._species_needed = True
        # aspect-chip reader started at encounter-confirm time (flee speed:
        # the chip window overlaps the UI-stability wait instead of adding
        # to it, and its length adapts to observed chip-render latency)
        self._chip_thread: Optional[threading.Thread] = None
        self._chip_result: Optional[Tuple[str, str, dict]] = None
        self._chip_error: Optional[BaseException] = None
        self._chip_started_at: float = 0.0
        self._chip_trip_max_ms: int = 0   # learned across this session
        self._chip_skip_window: bool = False

    def set_state(self, state: str) -> None:
        self.state = state
        log.info("state -> %s (instance=%s encounter=%s)",
                 state, self.instance_name, self.current_encounter_id or "-")

    def next_encounter_id(self) -> str:
        self.encounter_counter += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{self.instance_name or 'instance'}-{stamp}-{self.encounter_counter:04d}"

    # ------------------------------------------------------------------ #
    # detection helpers used by recovery/catching
    # ------------------------------------------------------------------ #
    def read_hp_state(self) -> Tuple[Optional[str], dict]:
        region = self.calib.region("hp_bar")
        return detect_hp(self.grab, region, self.calib.hp, attempts=2)

    def read_fighter_hp(self) -> Tuple[Optional[str], dict]:
        """HP state of OUR active Pokemon (bottom-right battle panel)."""
        region = self.calib.region("fighter_hp_bar")
        return detect_hp(self.grab, region, self.calib.hp, attempts=2)

    def mute_current_state(self) -> str:
        return self.mute.state

    def current_state(self) -> str:
        return self.mute.state

    def assert_encounter_still_active(self) -> None:
        frame = self.grab.frame(fresh=True)
        state = self.mute.update(frame, self.grab.frame_signature(frame))
        if state != "active":
            raise EncounterEnded("encounter no longer active during battle action")

    def await_capture_result(self, timeout_s: float) -> str:
        """'captured' | 'still_active' | 'stale_check'."""
        deadline = time.monotonic() + timeout_s
        msg_region = self.calib.optional_region("capture_message")
        while time.monotonic() < deadline:
            frame = self.grab.frame(fresh=True)
            sig = self.grab.frame_signature(frame)
            state = self.mute.update(frame, sig)
            if state == "idle":
                log.info("encounter ended -> capture assumed successful")
                return "captured"
            if msg_region:
                img = crop_region(frame, msg_region)
                res = read_text(img, psm=7)
                tail = normalize_text(res.raw)
                if any(w in tail for w in CAUGHT_WORDS):
                    return "stale_check"
            time.sleep(0.4)
        return "still_active"

    # ------------------------------------------------------------------ #
    # identification
    # ------------------------------------------------------------------ #
    def identify(self) -> Detected:
        d = Detected()
        log.info("IDENTIFYING_SPECIES / DETECTING_SHINY / IDENTIFYING_RESKIN / "
                 "IDENTIFYING_GRADIENT / DETECTING_TINT ...")

        retries = self.cfg.automation.max_ocr_retries
        # species OCR can burn a second of retries while the label is still
        # fading; when no enabled target uses a species criterion it never
        # matters for the decision -> skip it on the critical path (flee
        # speed) and read it for logging during the flee animation instead.
        species_needed = any(
            t.enabled and t.criteria.species for t in self.cfg.targets)
        self._species_needed = species_needed
        # the aspect readers are independent and each repeats internally with
        # short waits; running them concurrently brings the wall time down to
        # the slowest single reader (user: identify faster).
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="aspect") as ex:
            f_species = (ex.submit(self._read_species, retries)
                         if species_needed else None)
            f_shiny = ex.submit(
                detect_star_stable, self.grab,
                self.calib.region("shiny_indicator"), self.calib.shiny_star,
                retries, "shiny")
            f_tint = ex.submit(
                detect_star_stable, self.grab,
                self.calib.region("tint_indicator"), self.calib.tint_star,
                retries, "tint")
            f_chips = ex.submit(self._collect_chip_result)
            d.species = f_species.result() if f_species else None
            d.shiny, shiny_stats = f_shiny.result()
            d.tint, tint_stats = f_tint.result()
            d.reskin, d.gradient = f_chips.result()
        return d

    # ------------------------------------------------------------------ #
    # aspect-chip reader (early-started, adaptive window)
    # ------------------------------------------------------------------ #
    def start_chip_reader(self) -> None:
        """Start the aspect-chip poller right after the encounter pops.

        Reads can't race the chip fade-in (the old zero-match bug), so every
        encounter must observe the chip slots until the window closes no
        matter what. Starting at confirm time hides that mandatory window
        under the UI-stability wait and species OCR instead of stacking it
        on afterwards (flee speed). Once a live chip trip time has been
        observed the window tightens to trip + margin."""
        if self.calib.aspect_chips is None:
            return
        if self._chip_thread is not None and self._chip_thread.is_alive():
            return
        self._chip_result = None
        self._chip_error = None
        self._chip_started_at = time.monotonic()

        def job():
            spec = self.calib.aspect_chips
            observed = self._chip_trip_max_ms
            wait = spec.wait_ms
            if observed:
                # ratchet both ways: slowest chip render seen + margin,
                # clamped to [1500, 7000] ms
                wait = min(max(observed + 900, 1500), 7000)
            self._chip_result = self.ui.read_aspect_chips(wait_ms=wait)

        started = self._chip_started_at

        def guard():
            try:
                job()
            except BaseException as exc:  # noqa: BLE001 - surfaced on join
                self._chip_error = exc
                log.error("aspect chip reader crashed: %r", exc)

        self._chip_thread = threading.Thread(
            target=guard, name=f"aspect-chips-{started:.0f}", daemon=True)
        self._chip_thread.start()

    def _collect_chip_result(self) -> Tuple[str, str]:
        """Join the early-started chip poller; inline fallback when absent."""
        if self._chip_thread is None:
            return self._read_aspects()
        self._chip_thread.join()
        thread = self._chip_thread
        self._chip_thread = None
        if self._chip_error is not None:
            raise RuntimeError("aspect chip reader failed") from self._chip_error
        reskin, gradient, stats = self._chip_result or ("", "", {})
        trip = max(stats.get("reskin_trip_ms") or 0,
                   stats.get("gradient_trip_ms") or 0)
        if trip:
            old = self._chip_trip_max_ms
            self._chip_trip_max_ms = max(old, int(trip))
            if self._chip_trip_max_ms != old:
                log.info("chip window adapting: observed trip %d ms -> "
                         "new window %d ms", trip, self._chip_trip_max_ms + 900)
        return reskin, gradient

    def _read_aspects(self) -> Tuple[str, str]:
        """Chip-slot aspect reading with the poll window (never races the
        chip fade-in); falls back to the legacy single-shot readers when the
        calibration lacks aspect_chips."""
        if self.calib.aspect_chips is not None:
            return self._read_chips()
        retries = self.cfg.automation.max_ocr_retries
        return self._read_reskin(retries), self._read_gradient(retries)

    def _read_chips(self) -> Tuple[str, str]:
        reskin, gradient, stats = self.ui.read_aspect_chips()
        log.info("aspect chips: reskin=%r gradient=%r (reskin_px=%s,"
                 " gradient_reads=%s, trips %s/%s ms)",
                 reskin, gradient,
                 stats.get("reskin_pixels"), stats.get("gradient_reads"),
                 stats.get("reskin_trip_ms"), stats.get("gradient_trip_ms"))
        return reskin, gradient

    def _read_species(self, retries: int) -> Optional[str]:
        value, stable = self.ui.region_text("species_label", attempts=retries)
        if stable and value and not any(c.isalpha() for c in value):
            value = ""  # OCR noise only -> nothing readable
        log.debug("species read: %r (stable=%s)", value, stable)
        return value if stable else None

    def _start_species_log_read(self) -> None:
        """Log-only species OCR kicked off as the flee animation starts.

        The species label is still rendered while the run presentation
        plays, so this read hides in game animation time: the flee decision
        is already made and species is not part of it (flee speed)."""

        def job():
            try:
                value = self._read_species(self.cfg.automation.max_ocr_retries)
                if value:
                    log.info("species (during flee): %r", value)
            except BaseException as exc:  # noqa: BLE001 - logging only
                log.debug("background species read failed: %r", exc)

        threading.Thread(target=job, daemon=True, name="species-log").start()

    def _read_reskin(self, retries: int) -> Optional[str]:
        # reskin: readable-but-absent is "" ; unreadable is None
        value, stable = self.ui.region_text("reskin_label", attempts=retries)
        result = self._aspect_value(value, stable)
        log.debug("reskin read: %r raw stable=%s -> %r", value, stable, result)
        return result

    def _read_gradient(self, retries: int) -> Optional[str]:
        # gradient: presence comes from the visual band detector (chip letters
        # are always saturated, any colorway); OCR only attempts the NAME.
        present, _stats = detect_gradient_stable(
            self.grab, self.calib.region("gradient_label"),
            self.calib.gradient_band, attempts=retries, name="gradient")
        if present is True:
            g_value, _stable = self.ui.region_text("gradient_label", attempts=retries)
            text = (g_value or "").strip()
            clean = text if any(c.isalpha() for c in text) else ""
            return clean if clean else "gradient-present"
        if present is False:
            return ""
        return None

    @staticmethod
    def _aspect_value(value: Optional[str], stable: bool) -> Optional[str]:
        """Aspect text normalization: OCR noise ('|', 'yo') without a single
        letter is treated as empty (no aspect shown), not as an unknown value."""
        if not stable:
            return None
        if not value or not any(c.isalpha() for c in value):
            return ""
        return value

    def merge(self, a: Detected, b: Detected) -> Detected:
        """Combine two identification passes: prefer a known value for UNKNOWN slots."""
        def pick(x, y):
            if x is not None and ("?" not in str(x)):
                return x
            return y
        return Detected(
            species=pick(a.species, b.species),
            shiny=a.shiny if a.shiny is not None else b.shiny,
            reskin=pick(a.reskin, b.reskin),
            gradient=pick(a.gradient, b.gradient),
            tint=a.tint if a.tint is not None else b.tint,
        )

    # ------------------------------------------------------------------ #
    # workflows
    # ------------------------------------------------------------------ #
    def flee(self, reasons) -> None:
        """Non-target: click RUN once, verify the encounter actually ended."""
        self._start_species_log_read()
        log.info("FLEEING_NON_TARGET (reasons: %s)", ", ".join(reasons) if reasons else "-")
        max_retries = self.cfg.automation.max_action_retries
        for attempt in range(1, max_retries + 1):
            if self.mute.state == "idle":
                break
            try:
                self.assert_encounter_still_active()
            except EncounterEnded:
                break  # already over
            self.ui.flee_click()
            # short spacing only: wait_until("idle") is the real
            # verification and re-clicks if the run attempt failed
            self.ui.settle(200)
            if self.mute.wait_until("idle",
                                    timeout_s=self.cfg.automation.run_end_timeout_s):
                break
            log.info("RUN attempt %d did not end the encounter; re-checking screen "
                     "before another click", attempt)
        else:
            self.ui.debug_save("run_failed")
            raise SafeStop("RUN button did not end the encounter within retry limit")
        self.ui.settle(250)   # brief calm-down, idle already verified
        self.ui.park_mouse()
        log.info("encounter confirmed ended after flee")

    def wait_for_stability(self, max_s: float = 4.5) -> None:
        """Wait until the wild-card UI stops changing (entrance animations
        settle). Stability is judged ONLY over the regions identification
        reads: the battle background keeps animating for seconds longer and
        must not hold the hunt back (user: run from Pokemon faster)."""
        self.set_state("WAITING_FOR_UI_STABILITY")
        self.wait_until_stable(max_s, region=self._identification_bbox())
        log.debug("UI stable (identification region)")

    def _identification_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        # shiny_indicator is deliberately EXCLUDED: that region overlaps the
        # wild's animated sprite (sparkle/idle animation), so including it
        # pinned the stability wait at the full cap on animation-heavy wilds.
        # Star detection has its own multi-read retry and shiny is not a
        # required property of the hunt rule.
        names = ("species_label", "tint_indicator",
                 "reskin_label", "gradient_label")
        regions = [
            r for r in (self.calib.optional_region(n) for n in names)
            if r is not None
        ]
        if not regions:
            return None
        pad = 20
        x0 = min(r[0] for r in regions) - pad
        y0 = min(r[1] for r in regions) - pad
        x1 = max(r[0] + r[2] for r in regions) + pad
        y1 = max(r[1] + r[3] for r in regions) + pad
        return (max(0, x0), max(0, y0), x1 - x0, y1 - y0)

    def wait_until_stable(self, max_s: float,
                          region: Optional[Tuple[int, int, int, int]] = None) -> None:
        deadline = time.monotonic() + max_s
        last_sig = None
        stable_count = 0
        while time.monotonic() < deadline:
            sig = self.grab.frame_signature(self.grab.frame(fresh=True),
                                            region=region)
            if sig == last_sig and sig:
                stable_count += 1
                if stable_count >= 2:
                    break
            else:
                stable_count = 0
            last_sig = sig
            time.sleep(0.25)

    def weaken(self) -> None:
        """Fight menu -> weakening move -> catch handoff.

        Two styles (battle.weakening_style):
        - "damage" (False Swipe): repeat until the HP bar is red.
        - "sleep" (Spore, 15 PP, 100% accurate): the move does no damage but
          puts the wild to sleep — cast ONCE (verified turn consumed), then
          catch at full HP. A second cast on an already-asleep target would
          just burn PP.

        Self-care during weakening (user rules):
        - the fighter's OWN HP reaches the yellow/orange band -> Full Restore
        - the weakening move depletes -> Max Elixir restores the PP"""
        self.set_state("OPENING_FIGHT_MENU")
        target_state = self.cfg.battle.target_hp_state
        move = self.cfg.battle.weakening_move
        sleep_mode = self.cfg.battle.weakening_style == "sleep"
        automation = self.cfg.automation
        heal_state = self.cfg.battle.heal_hp_state
        attack = 0
        last_fractions: Optional[dict] = None
        unchanged = 0
        pp_flake_rescans = 0

        while attack < automation.max_weaken_attempts:
            hp_state, fractions = self.read_hp_state()
            log.info("CHECKING_TARGET_HP: %s (fractions=%s) attack %d/%d",
                     hp_state, fractions, attack + 1,
                     automation.max_weaken_attempts)
            if hp_state == target_state and not sleep_mode:
                log.info("HP bar reached target state '%s'; stop attacking",
                         target_state)
                return
            if attack >= 1 and hp_state is None and not sleep_mode:
                # the bar was readable during earlier attacks; a None now is
                # a transition/animation, not a heal (wilds never heal).
                # Attacking a 1-HP target again is pointless -> stop.
                log.info("HP bar unreadable after %d attacks; treating as "
                         "weakened", attack)
                return

            # full restore: our own HP bar hits the yellow/orange band
            fighter_state, _ = self.read_fighter_hp()
            if (fighter_state is not None
                    and (fighter_state in ("yellow", "red")
                         if heal_state == "yellow"
                         else fighter_state == "red")):
                if self.heal_cycles_used >= automation.max_recovery_cycles:
                    raise RecoveryExhausted(
                        f"heal limit {automation.max_recovery_cycles} "
                        f"reached (fighter HP {fighter_state})")
                self.heal_cycles_used += 1
                self.set_state("HEALING_MARSHADOW")
                self.recovery.execute()
                log.info("HEAL_COMPLETE (cycle %d)", self.heal_cycles_used)
                continue

            self.ui.open_fight_menu()
            self.set_state("USING_WEAKENING_MOVE")
            hit = self.ui.find_slot("move_slots", move, psm=PSM_BLOCK)
            if hit is None:
                # the move-grid OCR can flake for a single scan; re-verify
                # the grid is open (idempotent click) and re-scan with the
                # gray binarization that reads the SELECTED tile (white on
                # light-gray) before declaring the move unavailable
                for recheck in (1, 2):
                    self.ui.settle(700)
                    self.ui.open_fight_menu()
                    hit = self.ui.find_slot("move_slots", move, psm=PSM_BLOCK,
                                            binary_threshold=235)
                    if hit is not None:
                        log.info("move %r found on re-scan %d (OCR flake)",
                                 move, recheck)
                        break
            if hit is None and self.ui.fight_prompt_visible():
                # the move grid never opened (any-text verification passed on
                # the prompt bar) -> just reopen the fight menu
                log.info("move grid not open (prompt visible); reopening")
                continue
            if hit is None:
                # Before an expensive PP restore: if any slot still shows a
                # "N/M" fraction with N >= 1, the grid is readable but the
                # move-name OCR failed one scan -> reopen & retry instead of
                # burning a Max Elixir with full PP.
                # reading with PSM_SINGLE_LINE: move tiles are 2-line and PSM
                # 7 reads ZERO tokens on them (live-verified 18:45 Cursola
                # match); PSM_BLOCK gives 'name\npp/total type category',
                # binarized at 235 so the SELECTED tile reads too.
                texts = self.ui.scan_grid("move_slots", psm=PSM_BLOCK,
                                          binary_threshold=235)
                shown = " ".join(t for t, _ in texts if t)
                frac = re.search(r"(\d+)\s*/\s*\d+", shown)
                if frac and int(frac.group(1)) >= 1 and pp_flake_rescans < 3:
                    pp_flake_rescans += 1
                    log.info("move slots readable (%r) but %r not matched "
                             "-> OCR flake, reopening fight menu (%d/3)",
                             shown.strip(), move, pp_flake_rescans)
                    continue
                pp_flake_rescans = 0
                log.info("WEAKENING_MOVE_UNAVAILABLE (move %r not found) -> "
                         "restoring PP with %s", move,
                         self.cfg.battle.pp_item)
                if self.pp_cycles_used >= automation.max_recovery_cycles:
                    raise RecoveryExhausted(
                        f"PP-restore limit {automation.max_recovery_cycles} "
                        f"reached")
                self.pp_cycles_used += 1
                self.set_state("RESTORING_PP")
                self.recovery.execute_pp()
                log.info("PP_RESTORE_COMPLETE (cycle %d)", self.pp_cycles_used)
                continue
            pp_flake_rescans = 0
            attack += 1

            if not hit.exact:
                log.warning("move matched fuzzily (%r) for %r", hit.text, move)
            self.ui.click_move(hit)
            self.set_state("VERIFYING_HP")
            self.ui.settle(automation.attack_settle_ms)

            # verify the attack happened: the move grid shares its screen band
            # with the action prompt, so move text there after a click is the
            # PROMPT bleeding into the region (phantom "menu still open").
            # The prompt returning = the turn was consumed = attack executed.
            attack_landed = False
            polled = time.monotonic()
            while time.monotonic() - polled < 2.5:
                if self.ui.fight_prompt_visible():
                    attack_landed = True
                    break
                time.sleep(0.3)
            try:
                self.assert_encounter_still_active()
            except EncounterEnded:
                raise
            if not attack_landed:
                log.info("move grid still open after click (no prompt); "
                         "attack not consumed - retrying")
                continue
            if sleep_mode:
                log.info("SLEEP_MOVE_CAST: %r consumed the turn; target "
                         "should be asleep -> handing over to catch "
                         "(full HP is intended, sleep move does no damage)",
                         move)
                return

            hp_state, fractions = self.read_hp_state()
            if (hp_state is not None and fractions
                    and last_fractions is not None
                    and fractions == last_fractions):
                unchanged += 1
                if unchanged >= 2:
                    log.info(
                        "HP unchanged after %d verified attacks "
                        "(fractions=%s); wild appears immune to %r "
                        "(e.g. Ghost vs Normal move) -> skipping weakening, "
                        "catching at full HP",
                        unchanged, fractions, move)
                    self._immune_wild = True
                    return
            else:
                unchanged = 0
            if hp_state is not None:
                last_fractions = fractions
        if sleep_mode:
            raise SafeStop(f"{move!r} never consumed a turn within "
                           f"{automation.max_weaken_attempts} attempts "
                           "(sleep cast could not be verified)")
        raise SafeStop(f"HP did not reach '{target_state}' within "
                       f"{automation.max_weaken_attempts} attacks")

    def process_target(self, detected: Detected, matched_target_ids: list[str]) -> None:
        """Target match: notify, weaken, catch, notify."""
        self._immune_wild = False
        if not self._species_needed and detected.species is None:
            # species was skipped on the critical path (no target uses it):
            # read it now for the notification text — the catch flow runs for
            # tens of seconds and this is not a decision input anymore
            try:
                detected.species = self._read_species(
                    self.cfg.automation.max_ocr_retries) or ""
            except BaseException as exc:  # noqa: BLE001 - logging nicety
                log.debug("species backfill for notification failed: %r", exc)
            self._current_detected_text = detected.as_notification()
        detected_text = self._current_detected_text or "unknown Pokemon"
        if self.cfg.automation.notification_target_detected:
            self.set_state("TARGET_CONFIRMED")
            self.notifier.target_detected(
                detected_text,
                encounter_id=self.current_encounter_id,
                matched_target_ids=matched_target_ids,
            )
        self.notifier.capturing(
            detected_text,
            strategy=f"{self.cfg.battle.weakening_move} "
                     f"{'to sleep' if self.cfg.battle.weakening_style == 'sleep' else 'to red HP'}"
                     f", then {self.cfg.battle.catch_item}")
        self.weaken()
        self.set_state("OPENING_BAG")
        outcome = self.catching.execute()
        if outcome != "captured" and outcome == "hp_not_red":
            # target regained HP: re-weaken and try once more (impossible if
            # the wild is immune to the weakening move -> catch at full HP)
            if self._immune_wild:
                log.info("wild is immune to the weakening move; retrying "
                         "catch at current HP without re-weakening")
            else:
                self.weaken()
            outcome = self.catching.execute()
        if outcome == "captured":
            self.set_state("CATCH_CONFIRMED")
            self.notifier.captured(
                detected_text,
                encounter_id=self.current_encounter_id,
                matched_target_ids=matched_target_ids,
                save_status="SAVE_PENDING",
            )
            save_result = self.saver.execute(self.current_encounter_id)
            self.notifier.save_result(
                detected_text,
                encounter_id=self.current_encounter_id,
                save_status=save_result.status,
                matched_target_ids=matched_target_ids,
            )
        else:
            raise SafeStop(f"catch sequence ended with outcome '{outcome}'")

       # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def process_encounter(self) -> None:
        self.current_encounter_id = self.next_encounter_id()
        self.set_state("ENCOUNTER_CONFIRMED")
        self.recovery_cycles_used = 0
        self.heal_cycles_used = 0
        self.pp_cycles_used = 0
        self._matched_target = False   # a MATCHED target must never be fled
        self._matched_target_ids = []
        try:
            # aspect chips may start fading in while the panel is still
            # settling: start the poller NOW so its window overlaps the
            # stability wait and species OCR (flee speed)
            self.start_chip_reader()
            self.wait_for_stability()
            self.set_state("READING_ASPECTS")
            detected = self.identify()
            if self.cfg.automation.capture_each_encounter:
                # evidence taken AFTER the wild box has settled on screen
                self.set_state("CAPTURING_SCREENSHOT")
                self.ui.debug_save("encounter")

            self.set_state("MATCHING_TARGETS")
            multi = evaluate_all(detected, self.cfg.targets)
            per_target = []
            for evaluation in multi.evaluations:
                if not evaluation.enabled:
                    continue
                per_target.append(
                    f"{evaluation.target_id}={evaluation.result.overall.value} "
                    f"[failed={evaluation.result.failed or '-'} "
                    f"unknown={evaluation.result.unknown or '-'}]"
                )
            log.info("target evaluations: %s", "; ".join(per_target))
            if not multi.matched and any(
                evaluation.result.overall is Tri.UNKNOWN
                for evaluation in multi.evaluations if evaluation.enabled
            ):
                # 6.2: retry ambiguous identification once with a fresh frame
                self.set_state("VALIDATING_ASPECTS")
                unknown_ids = [
                    e.target_id for e in multi.evaluations
                    if e.enabled and e.result.overall is Tri.UNKNOWN
                ]
                log.info("UNKNOWN target evaluations %s - retrying identification", unknown_ids)
                # chips that were mid-fade on pass one are rendered by now:
                # the poller exits as soon as the gates see them
                self.start_chip_reader()
                self.grab.frame(fresh=True)
                time.sleep(0.7)
                detected2 = self.identify()
                detected = self.merge_unknown(detected, detected2)
                multi = evaluate_all(detected, self.cfg.targets)
                log.info("retry target evaluations: %s", "; ".join(
                    f"{e.target_id}={e.result.overall.value}" for e in multi.evaluations if e.enabled))

            self._current_detected_text = detected.as_notification()
            self._matched_target_ids = multi.matching_target_ids

            if multi.matched:
                self._matched_target = True
                self.process_target(detected, self._matched_target_ids)
            else:
                enabled = [e for e in multi.evaluations if e.enabled]
                unknown_fields = [f"{e.target_id}:{','.join(e.result.unknown)}"
                                  for e in enabled if e.result.overall is Tri.UNKNOWN]
                failed_fields = [f"{e.target_id}:{','.join(e.result.failed)}"
                                 for e in enabled if e.result.failed]
                self.set_state("RUNNING_FROM_NON_TARGET")
                self.flee(reasons=failed_fields + [f"UNKNOWN:{u}" for u in unknown_fields])
        except EncounterEnded:
            log.warning("EX 5. encounter ended mid-handling (was it caught/fled?)")
            raise
        except (RecognizedFailure, RecoveryExhausted, ItemNotAvailable, CatchAborted, SaveUnverified) as exc:
            log.error("recoverable failure during encounter: %s", exc)
            if getattr(self, "_matched_target", False):
                # NEVER flee a matched target: alert and stop hunting instead.
                # The encounter is left untouched so the user can act manually.
                log.error("MATCHED target - standing down without fleeing")
                raise SafeStop(
                    f"matched target, workflow failed: {exc}") from exc
            try:
                self.flee(reasons=[f"RECOVERY:{type(exc).__name__}"])
            except Exception as flee_exc:  # noqa: BLE001
                raise SafeStop(f"could not stabilize after {exc}") from flee_exc
            raise
        finally:
            self.set_state("RESETTING_ENCOUNTER")

    def merge_unknown(self, a: Detected, b: Detected) -> Detected:
        if a.species is None and b.species is not None:
            a.species = b.species
        if a.shiny is None and b.shiny is not None:
            a.shiny = b.shiny
        if a.reskin is None and b.reskin is not None:
            a.reskin = b.reskin
        if a.gradient is None and b.gradient is not None:
            a.gradient = b.gradient
        if a.tint is None and b.tint is not None:
            a.tint = b.tint
        return a

    def learn_idle_baseline(self) -> None:
        self.set_state("INITIALIZING")
        self.window.focus()
        self.set_state("CALIBRATING")
        log.info("learning MUTE baseline for %s; ensure NO encounter is active now",
                 self.instance_name)
        self.mute.learn_baseline()
        self.set_state("WAITING_FOR_ENCOUNTER")

    def _close_stray_backpack(self) -> None:
        """A crashed run can leave the Backpack modal open; it hides the mute
        button and would stall the WAITING loop forever (seen live once).
        Close it and verify it is gone."""
        value, stable = self.ui.region_text("backpack_title", attempts=1)
        if not (stable and value and "backpack" in value.lower()):
            return
        log.info("stray Backpack modal detected in WAITING; closing it")
        self.ui.click_region("close_bag")
        time.sleep(1.2)
        value, stable = self.ui.region_text("backpack_title", attempts=2)
        if stable and value and "backpack" in value.lower():
            self.ui.debug_save("stray_backpack_stuck")
            log.warning("Backpack still open after Close click")

    def run(self, stop_flag: Callable[[], bool]) -> None:
        self.learn_idle_baseline()
        interval = self.cfg.automation.loop_interval_ms / 1000.0
        failures = 0
        while not stop_flag():
            if self.state in ("WAITING_FOR_ENCOUNTER", "RESETTING_ENCOUNTER"):
                self._close_stray_backpack()
                self.ui.park_mouse()
                frame = self.grab.frame(fresh=True)
                sig = self.grab.frame_signature(frame)
                state = self.mute.update(frame, sig)
                if state == "active":
                    failures = 0
                    started = time.monotonic()
                    try:
                        self.process_encounter()
                    except EncounterEnded as exc:
                        log.info("resetting after unexpected encounter end: %s", exc)
                    except (RecoveryExhausted, ItemNotAvailable) as exc:
                        self.notifier.failure(str(exc), key="recovery")
                        raise SafeStop(str(exc))
                    except RecognizedFailure as exc:
                        failures += 1
                        log.error("encounter workflow failure (%d): %s", failures, exc)
                        self.set_state("WAITING_FOR_ENCOUNTER")
                        if failures >= 3:
                            raise SafeStop(f"workflow failed {failures}x: {exc}")
                    except CatchAborted as exc:
                        raise SafeStop(str(exc))
                    log.info("RESETTING_ENCOUNTER (encounter handled in %.1fs); "
                             "back to WAITING", time.monotonic() - started)
                    self.set_state("WAITING_FOR_ENCOUNTER")
            time.sleep(interval)


def window_grabber(window: GameWindow):
    from .capture import Grabber
    return Grabber(window)
