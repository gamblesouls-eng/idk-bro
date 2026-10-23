"""Multi-instance supervisor (Section 14).

One thread per configured instance; a crash in one instance is contained and
reported via Discord but does not stop the others. Each instance owns its own
window reference, capture context, calibration-in-memory state, MUTE monitor,
OCR results and encounter state machine. Nothing is shared between instances.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Callable, List, Optional

from .config import AppConfig, Calibration, load_calibration
from .encounter import EncounterBot, SafeStop
from .logging_setup import get_logger, setup_logging
from .notify import Notifier
from .window import WindowNotFound, find_windows

log = get_logger("hunter")


class InstanceRunner(threading.Thread):
    def __init__(self, cfg: AppConfig, calib: Calibration, index: int):
        super().__init__(name=f"hunter:{cfg.instances[index].name}", daemon=True)
        self.cfg = cfg
        self.calib = calib
        self.index = index
        self.icfg = cfg.instances[index]
        self.stop_event = threading.Event()
        self.notifier = Notifier(cfg.discord, instance_name=self.icfg.name)
        self.setStatus: Optional[str] = None  # last failure text
        self.bot: Optional[EncounterBot] = None

    def stop(self) -> None:
        self.stop_event.set()

    def _stop_flag(self) -> bool:
        return self.stop_event.is_set() or self._stop_file_present()

    @staticmethod
    def _stop_file_present() -> bool:
        import os
        return os.path.exists(os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "STOP"))

    def run(self) -> None:
        try:
            window = find_windows(self.icfg.window_title_patterns,
                                  self.icfg.window_index, self.icfg.name)
        except WindowNotFound as exc:
            log.error("[%s] startup failed: %s", self.icfg.name, exc)
            self.notifier.failure(f"startup failed: {exc}", key="startup")
            return
        try:
            self.bot = EncounterBot(window, self.calib, self.cfg,
                                    self.notifier, self.icfg.name)
            self.bot.run(self._stop_flag)
            log.info("[%s] loop ended (stop requested or stop file found)",
                     self.icfg.name)
        except SafeStop as exc:
            log.error("[%s] SAFE STOP: %s", self.icfg.name, exc)
            self.notifier.failure(f"safe stop: {exc}", key="safestop")
            self.setStatus = str(exc)
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] unexpected fatal error: %s\n%s", self.icfg.name,
                      exc, traceback.format_exc())
            self.notifier.failure(f"fatal error: {type(exc).__name__}",
                                  key="fatal")
            self.setStatus = type(exc).__name__


def run_all(cfg: AppConfig, calib: Optional[Calibration] = None,
            single_instance_index: Optional[int] = None) -> List[InstanceRunner]:
    """Starts all configured instances (or one pinned one) and blocks until
    all threads end or Ctrl+C is pressed, then stops them cleanly."""
    calib = calib or load_calibration()
    indices = [single_instance_index] if single_instance_index is not None \
        else list(range(len(cfg.instances)))
    runners: List[InstanceRunner] = []
    for i in indices:
        r = InstanceRunner(cfg, calib, i)
        runners.append(r)
        r.start()
        time.sleep(1.5)  # stagger: players' windows fight for foreground focus
    try:
        while any(r.is_alive() for r in runners):
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Ctrl+C received: stopping instances...")
        for r in runners:
            r.stop()
        for r in runners:
            r.join(timeout=10.0)
    return runners
