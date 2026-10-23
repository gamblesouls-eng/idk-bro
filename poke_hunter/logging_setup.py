"""Logging setup with file + console output and instance identifiers."""
from __future__ import annotations

import logging
import os
import sys
import time
from logging import handlers

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_MAX_BYTES = 10 * 1024 * 1024
_BACKUPS = 5


def setup_logging(level: str = "INFO", instance_id: str = "") -> logging.Logger:
    """Configure project logging. Returns the base logger for this instance."""
    os.makedirs(_LOG_DIR, exist_ok=True)
    root = logging.getLogger("poke_hunter")
    root.setLevel(logging.DEBUG)  # handlers decide what is actually emitted
    root.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-7s]%(inst)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt.converter = time.gmtime

    inst_suffix = f" [{instance_id}]" if instance_id else ""
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s]%(inst)s %(message)s", datefmt="%H:%M:%S"
    )
    console_fmt.converter = time.gmtime

    class InstFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.inst = inst_suffix
            return True

    f = InstFilter()

    if not root.handlers:
        tag = f"{'_' + instance_id if instance_id else ''}"
        fh = handlers.RotatingFileHandler(
            os.path.join(_LOG_DIR, f"hunter{tag}.log"),
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUPS,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        fh.addFilter(f)
        root.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(console_fmt)
        ch.addFilter(f)
        root.addHandler(ch)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"poke_hunter.{name}")
