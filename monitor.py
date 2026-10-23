"""Compact periodic monitor: new log lines + star-scan of new evidence."""
import os
import re
import subprocess
import sys
import time

OUT = (r"C:\Users\Seb\.local\share\opencode\shell"
       r"\3dd74ba10a054d54ee996323e27eb51cbcceeee6\sh_169315b10001P73Jjvj6Hx2P4e.out")
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "instance1.log")

ALERT = re.compile(r"already|MATCH\b|captur|throw|FAIL|ERROR|safe|STOP|stopped|"
                   r"Recovery|ItemNotAvailable|weaken", re.I)

def last_pos_file(path, key):
    lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_pos.json")
    import json
    try:
        with open(lp, "r", encoding="utf-8") as fh:
            pos = json.load(fh)
    except Exception:  # noqa: BLE001
        pos = {}
    v = pos.get(key, 0)
    # write back pointer in-band: return and let caller update? keep simple:
    return v, lp

def update_pos(path, key, v):
    import json
    lp = pos_path()
    try:
        with open(lp, "r", encoding="utf-8") as fh:
            pos = json.load(fh)
    except Exception:  # noqa: BLE001
        pos = {}
    pos[key] = v
    with open(lp, "w", encoding="utf-8") as fh:
        json.dump(pos, fh)


def pos_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_pos.json")


def newest_evidence():
    import glob
    import json
    files = sorted(glob.glob("logs/instance1_encounter_*_full.png"))
    pp = pos_path()
    try:
        with open(pp, "r", encoding="utf-8") as fh:
            pos = json.load(fh)
    except Exception:  # noqa: BLE001
        pos = {}
    seen = pos.get("newest")
    new = [f for f in files
           if (seen is None or os.path.basename(f) > os.path.basename(seen))]
    return files[-1] if files else None

def main():
    watch_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 43200.0
    interval = 20.0
    end = time.monotonic() + watch_seconds
    src = LOG if os.path.exists(LOG) else OUT
    pos_newest = None
    while time.monotonic() < end:
        print(f"--- {time.strftime('%H:%M:%S')} reading {os.path.basename(src)} ---")
        pos, lp = last_pos_file(src, "log")
        with open(src, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        update_pos(lp, "log", len(lines))
        new = lines[pos:]
        for ln in new:
            keep = ("MATCH_TARGET" in ln or ALERT.search(ln)
                    or "encounter state" in ln or "no encounters yet" in ln)
            if keep:
                print("  " + ln[-150:])
        # evidence: scan only screenshots newer than the newest one seen last time
        if pos_newest is not None:
            args = [sys.executable, "scan_encounters.py", pos_newest]
        else:
            args = [sys.executable, "scan_encounters.py"]
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=300)
            out_lines = [l for l in (r.stdout or "").splitlines() if "species=" in l]
            if out_lines:
                print("  evidence:")
                for l in out_lines:
                    print("   ", l[:150])
        except Exception as exc:  # noqa: BLE001
            print("  scan failed:", exc)
        pos_newest = newest_evidence() or pos_newest
        sys.stdout.flush()
        time.sleep(interval)

if __name__ == "__main__":
    main()
