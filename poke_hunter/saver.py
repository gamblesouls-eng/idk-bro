"""Game save flow: send the /save chat command after each captured Pokemon.

Live-verified behavior (2026-10-22 session):
- the top-left chat bubble button opens/toggles the chat panel
- with the panel open the '/' key focuses the chat input (the panel chrome
  disappears) and never inserts a literal slash
- after a successful save the game prints '[Commands]: Successfully saved!'
  into the chat history and shows a top-center 'Saving, please wait...'
  toast while the server processes the save
- after Enter the input loses focus and the panel stays open with the
  'To chat click here or press / key' placeholder

ESC SAFETY RULE (user requirement): Escape is ONLY pressed while a chat
element is verifiably open. Escape with the chat closed/unknown reaches the
game and opens the Roblox system menu (People/Settings/Leave/Respawn),
which must never happen. The panel is closed via the bubble click (physical
cursor events land regardless of keyboard focus) and a final defocus click
into the safe letterbox area drops any lingering chat input focus.

Keyboard safety: keys go to the FOREGROUND window. Before any keystroke the
saver verifies the Roblox window is foreground and refuses to type
otherwise; clicks land on physical coordinates regardless of focus.

Failure policy: execute() NEVER raises. A catch that already happened must
not abort the hunt just because the save could not be confirmed - the status
string travels into the Discord notifications instead.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .inputio import press_key, type_text
from .logging_setup import get_logger
from .ocr import PSM_BLOCK, PSM_SINGLE_LINE, read_text
from .util import normalize_text

log = get_logger("saver")

TAB_WORDS = ("here", "global", "friends")
HISTORY_PHRASES = ("successfully saved", "saved successfully")


class SaveUnverified(Exception):
    """The /save command could not be confirmed on screen (kept for the
    orchestrator's shared failure handling; GameSaver.execute itself never
    raises it - it reports the outcome through SaveAttempt.status)."""


SAVE_VERIFIED = "SAVE_VERIFIED"
SAVE_UNVERIFIED = "SAVE_UNVERIFIED"
SAVE_REFUSED = "SAVE_REFUSED"


@dataclass
class SaveAttempt:
    """Outcome report of one full save cycle."""
    encounter_id: str = ""
    status: str = SAVE_UNVERIFIED
    detail: str = ""
    attempts_used: int = field(default=1)

    @property
    def ok(self) -> bool:
        return self.status == SAVE_VERIFIED


def _is_foreground(window) -> bool:
    try:
        return bool(window.is_foreground())
    except Exception:  # noqa: BLE001
        return False


class GameSaver:
    """Sends /save with on-screen verification and verified chat cleanup."""

    SAVE_ATTEMPTS = 2
    VERIFY_TIMEOUT_S = 6.0
    OPEN_SETTLE_S = 0.8
    VERIFY_STEP_S = 0.5
    PHRASES = HISTORY_PHRASES

    def __init__(self, ui):
        self.ui = ui
        # honor the configurable save policy when present
        save_cfg = getattr(ui.cfg, "save", None)
        if save_cfg is not None:
            self.SAVE_ATTEMPTS = max(1, int(save_cfg.max_attempts))
            self.VERIFY_TIMEOUT_S = float(save_cfg.confirmation_timeout_s)
            if save_cfg.confirmation_phrases:
                self.PHRASES = tuple(save_cfg.confirmation_phrases)

    # -- public -------------------------------------------------------------------
    def execute(self, encounter_id: str = "") -> SaveAttempt:
        """Send /save, verify the save happened, close the chat.

        Returns a SaveAttempt and never raises. With Roblox not foreground
        NO keystroke is sent at all (status SAVE_REFUSED)."""
        window = self.ui.window
        if not self._ensure_foreground(window):
            log.warning("GAME_SAVE skipped (refused: Roblox not foreground)")
            self._safe_debug("save_refused_no_focus")
            # nothing was opened by us here; do not fly into any cleanup keys
            return SaveAttempt(
                encounter_id=encounter_id, status=SAVE_REFUSED,
                detail="Roblox not foreground; keys refused")
        last_detail = "'/save' completion never visible"
        try:
            for attempt in range(1, self.SAVE_ATTEMPTS + 1):
                try:
                    if self._send_save():
                        log.info("GAME_SAVE confirmed (attempt %d)", attempt)
                        return SaveAttempt(
                            encounter_id=encounter_id, status=SAVE_VERIFIED,
                            attempts_used=attempt)
                    last_detail = ("'/save' completion was never visible "
                                   "(no 'Successfully saved!' history line "
                                   "and no 'Saving, please wait...' toast)")
                    log.warning("save attempt %d lacked on-screen "
                                "confirmation: %s", attempt, last_detail)
                except Exception as exc:  # noqa: BLE001
                    last_detail = f"error during save attempt: {exc}"
                    log.warning("save attempt %d errored: %s", attempt, exc)
        finally:
            try:
                closed = self.close_chat()
            except Exception as exc:  # noqa: BLE001
                log.warning("chat cleanup error suppressed: %s", exc)
            else:
                if not closed:
                    log.warning("chat panel still present after cleanup")
        self._safe_debug("save_unverified")
        return SaveAttempt(
            encounter_id=encounter_id, status=SAVE_UNVERIFIED,
            detail=last_detail, attempts_used=self.SAVE_ATTEMPTS)

    def _safe_debug(self, tag: str) -> None:
        try:
            self.ui.debug_save(tag)
        except Exception as exc:  # noqa: BLE001
            log.warning("debug image %s skipped: %s", tag, exc)

    # -- save attempt ----------------------------------------------------------------
    def _send_save(self) -> bool:
        window = self.ui.window
        if not self._open_chat_input(window):
            log.warning("chat input could not be focused; save attempt aborted")
            return False
        type_text(window, "/save")
        time.sleep(0.2)
        press_key(window, "enter")
        deadline = time.monotonic() + self.VERIFY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._save_confirmed():
                return True
            time.sleep(self.VERIFY_STEP_S)
        return False

    def _save_confirmed(self) -> bool:
        toast = self._read("chat_toast", PSM_SINGLE_LINE)
        # 'Saving, please wait...' while processing; 'Save successful' after
        if "saving" in toast or "successful" in toast:
            return True
        history = self._read("chat_history", PSM_BLOCK)
        return any(phrase in history for phrase in self.PHRASES)

    # -- chat opening / focusing -------------------------------------------------------
    def _open_chat_input(self, window) -> bool:
        """Open the panel via the bubble, focus the input via '/', and
        confirm focus before any typing happens."""
        for _ in range(2):
            self.ui.click_region("chat_button")      # physical, always lands
            time.sleep(self.OPEN_SETTLE_S)
            if self._tabs_visible():
                press_key(window, "slash")           # focus the chat input
                time.sleep(0.5)
                if self._hint_gone():
                    return True
                self.ui.click_region("chat_input_hint")
                time.sleep(0.8)
                if self._hint_gone():
                    return True
            if self._tabs_visible():
                # chat is verifiably open -> Escape is safe here
                press_key(window, "esc")
                time.sleep(0.5)
        return False

    # -- chat cleanup --------------------------------------------------------------------
    def close_chat(self) -> bool:
        """Close any open chat WITHOUT ever pressing Escape blindly.

        Order: verify closed -> click the chat bubble (toggle) -> only press
        Escape while the chat is verifiably open -> defocus click into the
        safe letterbox area -> verify closed."""
        window = self.ui.window
        for _ in range(3):
            if self._chat_closed():
                break
            self.ui.click_region("chat_button")
            time.sleep(self.OPEN_SETTLE_S)
            if self._chat_closed():
                break
            if _is_foreground(window) and self._tabs_visible():
                press_key(window, "esc")             # safe: chat visibly open
                time.sleep(0.5)
        self._defocus()
        return self._chat_closed()

    def _defocus(self) -> None:
        """Click the empty letterbox to drop any lingering chat input focus."""
        try:
            self.ui.click_region("chat_defocus")
        except Exception as exc:  # noqa: BLE001
            log.warning("defocus click skipped: %s", exc)
            return
        time.sleep(0.4)

    # -- markers ---------------------------------------------------------------------------
    def _read(self, region_name: str, psm: int = PSM_SINGLE_LINE) -> str:
        region = self.ui.calib.optional_region(region_name)
        if region is None:
            return ""
        img = self.ui.grab.frame(fresh=True)
        crop = self.ui.grab.crop(img, region)
        res = read_text(crop, psm=psm)
        return normalize_text(res.raw)

    def _tabs_visible(self) -> bool:
        text = self._read("chat_tabs", PSM_BLOCK)
        return any(word in text for word in TAB_WORDS)

    def _hint_gone(self) -> bool:
        return "to chat" not in self._read("chat_input_hint", PSM_SINGLE_LINE)

    def _chat_closed(self) -> bool:
        return not self._tabs_visible()

    # -- focus acquisition --------------------------------------------------------------
    def _ensure_foreground(self, window) -> bool:
        for _ in range(4):
            if _is_foreground(window):
                return True
            try:
                window.focus()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.3)
        return _is_foreground(window)
