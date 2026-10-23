"""Pokémon encounter hunting, aspect detection, automated battles and catching.

Package layout:
    config.py   - configuration loading / validation
    window.py   - game-window identification
    capture.py  - screen capture
    ocr.py      - Tesseract OCR wrapper
    detectors.py- shiny/tint stars, HP bar state, MUTE encounter monitor
    matching.py - three-valued target matching
    inputio.py  - window-targeted mouse / keyboard input
    actions.py  - verified UI actions (buttons, menus, tabs, items)
    recovery.py - Full Restore recovery state machine
    catching.py - Repeat Ball catching state machine
    encounter.py- encounter orchestrator (single-instance loop)
    notify.py   - Discord webhook notifications
    hunter.py   - multi-instance supervisor
"""

__version__ = "1.0.0"
