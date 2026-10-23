"""Development helper: capture the current game window for interactive
calibration by the assistant.

    python main.py shot idle       -> calib_shots/idle.png
    python main.py shot encounter  -> calib_shots/encounter.png
"""
from __future__ import annotations

import argparse
import os
import sys


def cmd_shot(args) -> int:
    from poke_hunter.capture import Grabber
    from poke_hunter.window import WindowNotFound, find_windows
    os.makedirs("calib_shots", exist_ok=True)
    patterns = args.window.split(",") if args.window else ["Roblox"]
    w = find_windows(patterns, args.index, "shot")
    w.focus()
    grabber = Grabber(w)
    frame = grabber.frame(fresh=True)
    import cv2
    path = os.path.join("calib_shots", f"{args.name}.png")
    cv2.imwrite(path, frame)
    print("saved", path, frame.shape[1], "x", frame.shape[0])
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shot")
    s.add_argument("--name", default="shot", help="output png stem")
    s.add_argument("--window", default=None)
    s.add_argument("--index", type=int, default=0)
    s.set_defaults(func=cmd_shot)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
