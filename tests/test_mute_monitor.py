"""MUTE-button encounter monitor tests: red-button presence + debounce."""
import numpy as np

from poke_hunter.detectors import MuteMonitor

REGION = (0, 0, 80, 40)


def battle_frame(w=200, h=100):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[0:40, 0:80] = (40, 40, 210)  # solid red button
    return frame


def idle_frame(w=200, h=100):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[0:40, 0:80] = (235, 235, 240)  # light panel
    return frame


class FakeGrabber:
    def __init__(self):
        self.current = idle_frame()

    def capture_region(self, region, fresh=False):
        return self.current


def make_monitor(**kw):
    defaults = dict(red_hsv_min=(0, 140, 120), red_hsv_max=(10, 255, 255),
                    min_red_fraction=0.45, active_frames=2, idle_frames=2)
    defaults.update(kw)
    return MuteMonitor(FakeGrabber(), REGION, **defaults)


def test_classify_active():
    mon = make_monitor()
    state, frac = mon.classify(battle_frame(200, 100))
    assert state == "active" and frac >= 0.45


def test_classify_idle():
    mon = make_monitor()
    state, frac = mon.classify(idle_frame(200, 100))
    assert state == "idle" and frac < 0.2


def test_single_active_frame_does_not_trigger():
    mon = make_monitor()
    assert mon.update(battle_frame(), "f1") != "active"
    assert mon.update(battle_frame(), "f2") == "active"


def test_alternating_noise_does_not_flip_state():
    mon = make_monitor()
    mon.update(battle_frame(), "f1")
    mon.update(idle_frame(), "f2")
    mon.update(battle_frame(), "f3")
    mon.update(idle_frame(), "f4")
    assert mon.state != "active"


def test_same_signature_frame_ignored():
    mon = make_monitor()
    assert mon.update(battle_frame(), "f1") != "active"
    assert mon.update(battle_frame(), "f1") != "active"   # duplicate: no progress
    assert mon.update(battle_frame(), "f2") == "active"


def test_active_to_idle_transition():
    mon = make_monitor()
    mon.update(battle_frame(), "f1")
    mon.update(battle_frame(), "f2")
    assert mon.state == "active"
    assert mon.update(idle_frame(), "f3") != "idle"    # one idle frame only
    assert mon.update(idle_frame(), "f4") == "idle"    # debounce confirms end


def test_real_battle_vs_playerlist_layout():
    # 1920x1080 simplified: Mute region at calibrated spot
    mon = MuteMonitor(FakeGrabber(), (1471, 100, 220, 70))
    battle = np.zeros((200, 1920, 3), np.uint8)
    battle[100:162, 1471:1680] = (40, 40, 210)
    state, _ = mon.classify(battle)
    assert state == "active"
    idle = np.zeros((200, 1920, 3), np.uint8)
    idle[30:170, 1500:1900] = (235, 235, 240)
    idle[104:112, 1520:1562] = (30, 30, 200)   # small red text like 'BATTLE'
    state2, _ = mon.classify(idle)
    assert state2 == "idle"
