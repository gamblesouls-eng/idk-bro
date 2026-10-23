"""GameSaver behavior tests.

Safety guarantees under test:
- execute() NEVER raises, even when clicks/reads explode
- no keystroke is ever sent when Roblox is not the foreground window
- Escape is pressed ONLY while the chat is verifiably open (blind ESC opens
  the Roblox system menu - user forbidden)
- the chat panel closes via physical bubble clicks + a letterbox defocus
"""
import numpy as np
import pytest
from types import SimpleNamespace

from poke_hunter import saver
from poke_hunter.config import SaveConfig
from poke_hunter.saver import (GameSaver, SaveAttempt, SaveUnverified,
                               SAVE_REFUSED, SAVE_UNVERIFIED, SAVE_VERIFIED)


class FakeWindow:
    def __init__(self):
        self.foreground = True
        self.focus_calls = 0

    def is_foreground(self):
        return self.foreground

    def focus(self):
        self.focus_calls += 1


class FakeCalib:
    REGIONS = ("chat_button", "chat_tabs", "chat_input_hint", "chat_toast",
               "chat_history", "chat_defocus")

    def optional_region(self, name):
        if name in self.REGIONS:
            return (0, 0, 20, 20)
        return None


class FakeGrab:
    def frame(self, fresh=False, min_age_ms=0):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def crop(self, frame, region):
        return frame


class FakeUi:
    def __init__(self):
        self.window = FakeWindow()
        self.calib = FakeCalib()
        self.grab = FakeGrab()
        self.cfg = SimpleNamespace(save=None)    # default policy (no override)
        self.clicks = []
        self.debugs = []
        self.fail_on_click = set()

    def click_region(self, name):
        if name in self.fail_on_click:
            raise RuntimeError(f"click {name} exploded")
        self.clicks.append(name)

    def debug_save(self, tag, *args, **kwargs):
        self.debugs.append(tag)


class Keys:
    def __init__(self):
        self.typed = []
        self.pressed = []

    def type_text(self, window, text, interval=0.0):
        self.typed.append(text)

    def press_key(self, window, key, times=1):
        self.pressed.extend([key] * times)


class FastClock:
    def __init__(self):
        self.t = 0.0

    def sleep(self, seconds):
        self.t += seconds

    def monotonic(self):
        return self.t


class RegionScript:
    """Per-region FIFO of OCR texts; pops once then falls back to default."""

    def __init__(self, script=None, default=""):
        (self.default, self.script) = (default, {k: list(v) for k, v in
                                                 (script or {}).items()})

    def as_read(self):
        script = self

        def fake_read(region_name, psm=0):  # noqa: ARG001
            queue = script.script.get(region_name) or []
            if queue:
                return queue.pop(0)
            return script.default
        return staticmethod(fake_read)


@pytest.fixture
def clock(monkeypatch):
    clk = FastClock()
    monkeypatch.setattr(saver.time, "sleep", clk.sleep)
    monkeypatch.setattr(saver.time, "monotonic", clk.monotonic)
    return clk


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    rec = Keys()
    monkeypatch.setattr(saver, "type_text", rec.type_text)
    monkeypatch.setattr(saver, "press_key", rec.press_key)
    monkeypatch.setattr(saver, "read_text", lambda *_a, **_k: None)
    return rec


def install_script(monkeypatch, script: RegionScript):
    monkeypatch.setattr(GameSaver, "_read", script.as_read())


# ---------------------------------------------------------------- outcomes /
def test_execute_verifies_save_and_closes_chat(monkeypatch, clock, keys):
    ui = FakeUi()
    install_script(
        monkeypatch,
        RegionScript({"chat_tabs": ["here global friends"],
                      "chat_input_hint": ["to chat click here"],
                      "chat_toast": ["saving, please wait"]}))
    assert GameSaver(ui).execute("enc-1").status == SAVE_VERIFIED
    assert keys.typed == ["/save"]
    assert "enter" in keys.pressed
    assert "esc" not in keys.pressed          # clean path: no escape needed
    assert "chat_button" in ui.clicks and "chat_defocus" in ui.clicks


def test_execute_refused_when_not_foreground(monkeypatch, clock, keys):
    ui = FakeUi()
    ui.window.foreground = False
    install_script(monkeypatch, RegionScript(default=""))
    result = GameSaver(ui).execute("enc-2")
    assert result.status == SAVE_REFUSED
    assert keys.typed == [] and keys.pressed == []   # NOTHING typed/pressed
    assert "save_refused_no_focus" in ui.debugs
    assert ui.window.focus_calls >= 1
    # chat was never opened by us: only focus attempts happened


def test_execute_never_raises_and_reports_unverified(monkeypatch, clock, keys):
    ui = FakeUi()
    install_script(monkeypatch, RegionScript(default=""))   # chat never opens
    result = GameSaver(ui).execute("enc-3")
    assert result.status == SAVE_UNVERIFIED
    assert keys.typed == []      # chat could not be opened -> nothing typed
    assert "save_unverified" in ui.debugs


def test_execute_survives_exploding_clicks(monkeypatch, clock, keys):
    ui = FakeUi()
    ui.fail_on_click.add("chat_button")
    install_script(monkeypatch, RegionScript(default=""))
    result = GameSaver(ui).execute("enc-4")
    assert result.status == SAVE_UNVERIFIED     # NOT raised


def test_verify_checks_history_when_no_toast(monkeypatch, clock, keys):
    ui = FakeUi()
    script = RegionScript({
        "chat_tabs": ["here global friends"],
        "chat_input_hint": [""],
        "chat_toast": ["", ""],                 # toast absent
        "chat_history": ["commands successfully saved"],
    })
    # history check: toast queue empties, then every read defaults ''
    script.default = ""
    monkeypatch.setattr(GameSaver, "_read", script.as_read())
    assert GameSaver(ui).execute("enc-5").status == SAVE_VERIFIED


def test_execute_unverified_after_failed_markers(monkeypatch, clock, keys):
    ui = FakeUi()
    script = RegionScript({
        "chat_tabs": ["here global friends"],
        "chat_input_hint": [""],
    })
    script.default = ""                          # no toast / no history
    monkeypatch.setattr(GameSaver, "_read", script.as_read())
    result = GameSaver(ui).execute("enc-6")
    assert result.status == SAVE_UNVERIFIED
    assert keys.typed == ["/save"]              # attempt 1 typed; attempt 2
                                                # was aborted before typing
                                                # (chat could not be reopened)
    assert keys.pressed.count("enter") == 1
    assert keys.pressed.count("esc") == 0


# ---------------------------------------------------------------- close_chat
def test_close_chat_clone_never_gets_escscribed(monkeypatch, clock, keys):
    ui = FakeUi()
    script = RegionScript(default="")            # chat always closed
    monkeypatch.setattr(GameSaver, "_read", script.as_read())
    assert GameSaver(ui).close_chat() is True
    assert keys.pressed == []                    # NO escape: none needed
    assert ui.clicks == ["chat_defocus"]         # defocus only


def test_close_chat_defocus_ignores_failures(monkeypatch, clock, keys):
    ui = FakeUi()
    ui.fail_on_click.add("chat_defocus")
    script = RegionScript(default="")
    monkeypatch.setattr(GameSaver, "_read", script.as_read())
    assert GameSaver(ui).close_chat() is True    # survives exploding click
    assert keys.pressed == []


# ------------------------------------------------------------------ open path
def test_open_chat_input_uses_bubble_then_slash_only_when_visible(
        monkeypatch, clock, keys):
    ui = FakeUi()
    script = RegionScript({
        "chat_tabs": [],                 # tabs NEVER visible
    })
    script.default = ""
    monkeypatch.setattr(GameSaver, "_read", script.as_read())
    assert GameSaver(ui)._open_chat_input(ui.window) is False
    # no opening detected -> the saver must not try to escape blindly
    assert "esc" not in keys.pressed


def test_open_chat_input_full_flow_with_fallback_click(monkeypatch, clock,
                                                       keys):
    ui = FakeUi()
    # tabs visible after bubble; slash did NOT remove the hint; the FALLBACK
    # click on the hint strip focuses the input (placeholder disappears)
    script = RegionScript({
        "chat_tabs": ["here global friends"],
        "chat_input_hint": ["to chat click here or press / key", ""],
        "chat_toast": ["saving, please wait"],
    })
    script.default = ""
    monkeypatch.setattr(GameSaver, "_read", script.as_read())
    assert GameSaver(ui).execute("enc-7").status == SAVE_VERIFIED
    assert keys.typed.count("/save") == 1
    assert keys.pressed.count("slash") == 1
    assert keys.pressed.count("esc") == 0
    assert "chat_input_hint" in ui.clicks          # fallback click used


# ---------------------------------------------------------------------- misc
def test_save_unverified_exception_still_exists():
    # orchestrator imports it defensively for its failure routing
    assert issubclass(SaveUnverified, Exception)


def test_save_attempt_ok_property():
    assert SaveAttempt(status=SAVE_VERIFIED).ok is True
    assert SaveAttempt(status=SAVE_UNVERIFIED).ok is False
    assert SaveAttempt(status=SAVE_REFUSED).ok is False


def test_save_config_policy_is_honored():
    ui = FakeUi()
    ui.cfg = SimpleNamespace(save=SaveConfig(max_attempts=3,
                                             confirmation_timeout_s=9.5,
                                             confirmation_phrases=(
                                                 "successfully saved",)))
    gs = GameSaver(ui)
    assert gs.SAVE_ATTEMPTS == 3
    assert gs.VERIFY_TIMEOUT_S == 9.5
    assert gs.PHRASES == ("successfully saved",)
