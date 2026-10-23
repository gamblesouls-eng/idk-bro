"""Command line entry point.

Usage:
    python main.py check-env      verify Python deps + Tesseract availability
    python main.py calibrate      interactive region editor (Tkinter)
    python main.py verify         live-verify all calibrated regions (OCR/colors)
    python main.py identify       one-shot aspect identification of current encounter
    python main.py run            start the automated hunt loop
    python main.py run --instance 1   start only the 2nd configured instance
    python main.py send-test      send a test message through the webhook
"""
from __future__ import annotations

import argparse
import sys

from poke_hunter.config import (ConfigError, load_app_config, load_calibration,
                                resolve_webhook)
from poke_hunter.logging_setup import get_logger, setup_logging
from poke_hunter.ocr import verify_tesseract

log = get_logger("main")


def cmd_check_env(_args) -> int:
    print("Python:", sys.version.splitlines()[0])
    try:
        print("Tesseract:", verify_tesseract())
        ok = True
    except Exception as exc:  # noqa: BLE001
        print("Tesseract problem:", exc)
        ok = False
    for mod in ("mss", "numpy", "cv2", "pyautogui", "pydirectinput", "requests",
                "pytesseract", "win32gui"):
        try:
            __import__(mod)
            print(f"module {mod}: OK")
        except ImportError as exc:
            print(f"module {mod}: MISSING ({exc})")
            ok = False
    try:
        cfg = load_app_config()
    except ConfigError as exc:
        cfg = None
        print("config:", exc)
    if cfg:
        print("config: OK")
        print(" target:", cfg.target.describe())
        print(" instances:", [i.name for i in cfg.instances])
    calib = load_calibration()
    missing = [k for k in REQUIRED_REGIONS if k not in calib.regions]
    print(f"calibration: {len(calib.regions)} regions, "
          f"{len(calib.grids)} grids; missing core regions: {missing or 'none'}")
    if cfg:
        url, src = resolve_webhook(cfg.discord)
        print("webhook:", "configured via " + src if url else "not configured")
    return 0 if ok else 2


REQUIRED_REGIONS = (
    "mute_button", "species_label", "reskin_label", "gradient_label",
    "hp_bar", "shiny_indicator", "tint_indicator", "run_button",
    "fight_button", "back_button", "bag_button", "backpack_title",
    "recovers_tab", "pokeballs_tab", "category_header", "search_field",
)


def cmd_calibrate(args) -> int:
    from poke_hunter.calibrate import run_editor
    run_editor()
    return 0


def cmd_verify(args) -> int:
    cfg = load_app_config()
    setup_logging("DEBUG", args.instance_name or "verify")
    from poke_hunter.window import WindowNotFound, find_windows
    patterns = cfg.instances[0].window_title_patterns if not args.window else [args.window]
    index = cfg.instances[0].window_index
    window = find_windows(patterns, index, "verify")
    window.focus()
    from poke_hunter.capture import Grabber
    from poke_hunter.detectors import detect_star, hp_state
    import mss
    grab = Grabber(window)
    calib = load_calibration()
    print("window:", window.describe())
    frame = grab.frame(fresh=True)
    print(f"frame size: {frame.shape[1]}x{frame.shape[0]}")
    for name, region in sorted(calib.regions.items()):
        from poke_hunter.util import crop_region
        crop = crop_region(frame, region)
        print(f"\n[{name}] region={region} size={crop.shape[1]}x{crop.shape[0]}")
        if name.endswith(("_indicator",)):
            if "shiny" in name and calib.shiny_star:
                ok, stats = detect_star(crop, calib.shiny_star)
                print(f"  shiny star: present={ok} stats={stats}")
            if "tint" in name and calib.tint_star:
                ok, stats = detect_star(crop, calib.tint_star)
                print(f"  tint star: present={ok} stats={stats}")
        elif name == "hp_bar":
            state, fr = hp_state(crop, calib.hp)
            print(f"  hp state: {state} fractions={fr}")
        elif name == "search_field":
            from poke_hunter.ocr import read_text
            res = read_text(crop, psm=7)
            print(f"  text: {res.value!r} conf={res.confidence:.0f} raw={res.raw!r}")
        else:
            from poke_hunter.ocr import read_text
            res = read_text(crop, psm=7)
            print(f"  text: {res.value!r} conf={res.confidence:.0f} raw={res.raw!r}")
    for gname, grid in calib.grids.items():
        print(f"\n[grid {gname}] {grid.count} slots step={grid.step}")
        from poke_hunter.ocr import read_text
        from poke_hunter.util import crop_region
        for i in range(grid.count):
            crop = crop_region(frame, grid.slot(i))
            res = read_text(crop, psm=7)
            print(f"  slot {i}: {res.value!r} conf={res.confidence:.0f}")
    return 0


def cmd_identify(args) -> int:
    """Run one full aspect identification against the live encounter."""
    cfg = load_app_config()
    setup_logging("DEBUG", "identify")
    window = None
    from poke_hunter.window import WindowNotFound, find_windows
    icfg = cfg.instances[0]
    window = find_windows(icfg.window_title_patterns, icfg.window_index, icfg.name)
    window.focus()
    import mss
    from poke_hunter.capture import Grabber
    from poke_hunter.encounter import EncounterBot
    from poke_hunter.matching import evaluate
    from poke_hunter.notify import Notifier
    calib = load_calibration()
    bot = EncounterBot(window, calib, cfg, Notifier(cfg.discord, "identify"), icfg.name)
    bot.wait_for_stability()
    detected = bot.identify()
    result = evaluate(detected, cfg.target)
    print("\n=== IDENTIFICATION RESULT ===")
    print(" species: ", detected.species)
    print(" shiny:   ", detected.shiny)
    print(" reskin:  ", detected.reskin)
    print(" gradient:", detected.gradient)
    print(" tint:    ", detected.tint)
    print(" overall: ", result.overall.value, "|", result.debug())
    return 0


def cmd_run(args) -> int:
    cfg = load_app_config()
    calib = load_calibration()
    from poke_hunter.ocr import OCRInitError
    try:
        setup_logging("INFO", "supervisor")
        print("Tesseract:", verify_tesseract())
    except Exception as exc:  # noqa: BLE001
        print("OCR initialization failed:", exc)
        print("continuing is still possible; notifications & GUI unaffected, "
              "but species/reskin/gradient cannot be identified.")
        if '--force' not in (sys.argv or []):
            return 2
    from poke_hunter.hunter import run_all
    runners = run_all(cfg, calib, single_instance_index=args.instance)
    code = 0
    for r in runners:
        if r.setStatus:
            print(f"[{r.icfg.name}] stopped: {r.setStatus}")
            code = 1
    return code


def cmd_send_test(_args) -> int:
    cfg = load_app_config()
    from poke_hunter.notify import Notifier
    n = Notifier(cfg.discord, "test")
    if not n.enabled:
        print("Webhook not configured. Set the environment variable "
              f"{cfg.discord.webhook_env_var} or create config/.webhook "
              "containing the URL.")
        return 1
    ok = n.send("poke_hunter test message ✅")
    print("delivered" if ok else "delivery failed (see logs)")
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pokemon encounter hunting automation")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("check-env").set_defaults(func=cmd_check_env)
    c = sub.add_parser("calibrate")
    c.set_defaults(func=cmd_calibrate)
    v = sub.add_parser("verify")
    v.add_argument("--window", default=None, help="override window title filter")
    v.add_argument("--instance-name", default=None)
    v.set_defaults(func=cmd_verify)
    ident = sub.add_parser("identify")
    ident.set_defaults(func=cmd_identify)
    r = sub.add_parser("run")
    r.add_argument("--instance", type=int, default=None,
                   help="only run this instance index from config.instances")
    r.set_defaults(func=cmd_run)
    sub.add_parser("send-test").set_defaults(func=cmd_send_test)
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nstopped by user")
        return 130
    except (ConfigError,) as exc:
        print("configuration error:", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
