# Pokémon Encounter Hunter

Automated encounter hunting with aspect detection, battle, catching and
Discord notifications, built with Python + Tesseract + mss + pydirectinput.

**Status:** unit-tested (83 tests) and live-verified against the real game:
encounter confirm → aspect reading (reskin/gradient chips) → flee / weaken /
catch flow → chat `/save` → Discord notifications all ran live on 2026-10-23
(full catch chain: Spore sleep → Repeat Ball throws → safe stop on empty
inventory). The gradient-chip presence gate is the dark chip box itself —
biome-invariant (the live grass biome tripped an older saturated-letters
fallback, which this replaced). Nothing is claimed to work without calibration:
see "Remaining real-game testing" below before using it on a new setup.

## Setup

1. Python 3.12+, Tesseract 5.x on PATH (verify: `python main.py check-env`).
2. Requirements (already installed on this machine):
   `pip install -r requirements.txt   (dev: requirements-dev.txt)`
3. Configuration: edit `config/config.json` (target, battle behavior,
   automation limits, instance list).
4. Discord webhook: environment variable `POKEMON_DISCORD_WEBHOOK` or file
   `config/.webhook` (git-ignored). Test with `python main.py send-test`.

## Calibration of the live game (REQUIRED before first run)

I cannot see your game window, so UI coordinates are NOT invented in this
repo — they must come from the actual interface:

1. `python main.py calibrate` - interactive editor over a live screenshot:
   - draw each region with its logical name:
     `mute_button, species_label, reskin_label, gradient_label, hp_bar,
     shiny_indicator, tint_indicator, run_button, fight_button, back_button,
     bag_button, backpack_tab, recovers_tab, pokeballs_tab, search_field,
     move_slots (grid), item_slots (grid), party_slots (grid), capture_message (optional)`
   - grids: draw the FIRST slot, then enter step (dx,dy) and count
     (moves = 4, party = 6, item slots = whatever the list shows).
   - use "Pixel analyzer" over the blue star, white star, and a filled HP bar
     to get HSV numbers for `detection.shiny_star`, `detection.tint_star`,
     `detection.hp_colors` in `config/calibration.json`.
2. `python main.py verify` - OCRs/detects every live region so you can fix
   any wrong rectangle before the hunt starts.
3. MUTE monitoring: the encounter state is detected by a learned fingerprint of
   the `[MUTE]` area. Baseline is learned at startup (10s with no encounter),
   the debounced difference then signals encounter active/idle. Watch the
   first full encounter and adjust `encounter_stability_frames` /
   `tolerance` in `calibrate.py` if needed.

## Running

```text
python main.py run                 # all configured instances, 1 thread each
python main.py run --instance 1    # only the 2nd instance entry
create file poke_hunter/STOP      # request a clean stop (or Ctrl+C)
```

## Commands

| command | purpose |
|---|---|
| `check-env` | python deps + tesseract + config/calibration sanity |
| `calibrate` | interactive region editor (Tkinter) |
| `verify` | live-verify every calibrated region (OCR/colors/stars) |
| `identify` | one-shot identification of the current encounter |
| `run` | start the automated hunt |
| `send-test` | webhook test |

## Design map

- `poke_hunter/window.py` - Win32 game-window identification; input is
  window-relative so multi-instance clicks cannot be misrouted.
- `poke_hunter/capture.py` - mss capture + frame signature (skips identical frames).
- `poke_hunter/ocr.py` - Tesseract wrapper: per-region PSM/whitelists,
  confidence, retry/stability consensus, debug screenshots.
- `poke_hunter/detectors.py` - `detect_star` (shiny blue / tint white),
  `detect_hp`, aspect-chip readers (`detect_reskin_letters` shape-filtered
  letter blobs; `detect_color_chip` gates gradient chips ONLY on the dark chip
  box — the box is part of the chip graphic and renders in every biome),
  `MuteMonitor` (red-button presence + debounce).
- `poke_hunter/matching.py` - tri-state MATCH / NO_MATCH / UNKNOWN; UNKNOWN never matches.
- `poke_hunter/actions.py` - verified UI actions (menus, tabs, search field).
- `poke_hunter/encounter.py` - orchestrator: flee / weaken / catch flows.
- `poke_hunter/recovery.py` - Full Restore recovery (Back->Bag->Backpack->
  Recovers->Full Restore->party: species AND level 150 -> verify).
- `poke_hunter/catching.py` - Repeat Ball procedure incl. mandatory search
  field clear + verification after selection.
- `poke_hunter/notify.py` - webhook with secret handling, retries, 429
  backoff, dedupe; failures never touch game state.
- `poke_hunter/hunter.py` - one contained thread per instance.

## Remaining real-game testing (explicitly not yet done)

1. All coordinates/steps above depend on the calibration you record; until
   then, OCR region expectations are untested against the real interface.
2. MUTE fingerprint separation: needs 2-3 example encounters to tune
   `tolerance` (logs show distances for calibration).
3. Fight-menu opening verification and move-slot OCR: depends on the real
   move list layout (PSM mode may need tuning in `ocr.py`).
4. Bag layout: Category tabs (Backpack/Recovers/Pokeballs), item slot order,
   search-field behavior after using an item — verify with `main.py verify`.
5. Party screen: level formatting (regex covers "Lv. 123" and "Lvl 123"); if
   the game renders levels differently adjust `recovery.LEVEL_RE`.
6. Full game-loop runs against the live game (flee/weaken/recover/catch).
   Run one flee-cycle at a time against separate encounters until each phase
   is verified, then enable full automation.
