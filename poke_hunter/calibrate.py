"""Interactive calibration editor.

Shows a live screenshot of the game window, lets the user drag rectangles over
UI elements, name them, and saves everything window-relative to
config/calibration.json. Includes a pixel analyzer to derive the HSV specs for
the shiny/tint stars and the HP color ranges.

Coordinates are stored relative to the game client area, so moving or resizing
the window on the same display does not invalidate calibration.
"""
from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from .capture import Grabber
from .config import CALIBRATION_PATH, load_calibration
from .logging_setup import setup_logging
from .window import WindowNotFound, find_windows

setup_logging("INFO", "calibrate")

DEFAULT_REGION_HINTS = [
    "mute_button", "species_label", "shiny_indicator", "tint_indicator",
    "reskin_label", "gradient_label", "hp_bar", "run_button", "fight_button",
    "back_button", "bag_button", "backpack_tab", "recovers_tab",
    "pokeballs_tab", "search_field", "capture_message",
]
DEFAULT_GRIDS = ["move_slots", "item_slots", "party_slots"]


class CalibratorApp:
    def __init__(self, root: tk.Tk, patterns, index: int):
        self.root = root
        root.title("poke_hunter calibration")
        self.patterns, self.index = patterns, index
        self.window = find_windows(patterns, index, "calibrate")
        self.window.focus()
        self.grab = Grabber(self.window)

        # load existing calibration as the editable model
        try:
            calib = load_calibration()
            self.regions: dict = {k: list(v) for k, v in calib.regions.items()}
            self.grids: dict = {
                name: (list(g.region), list(g.step), g.count)
                for name, g in calib.grids.items()}
        except Exception:  # noqa: BLE001
            self.regions, self.grids = {}, {}

        buttons = tk.Frame(root)
        buttons.pack(side="top", fill="x")
        tk.Button(buttons, text="New region", command=self.new_region).pack(side="left")
        tk.Button(buttons, text="New grid slot", command=self.new_grid_slot).pack(side="left")
        tk.Button(buttons, text="Full-run grid", command=self.new_grid_full).pack(side="left")
        tk.Button(buttons, text="Region list", command=self.list_regions).pack(side="left")
        tk.Button(buttons, text="Pixel analyzer", command=self.pixel_analyzer).pack(side="left")
        tk.Button(buttons, text="Save", command=self.save, bg="#d8ffd8").pack(side="left")
        tk.Button(buttons, text="Reload screenshot", command=self.reload_image).pack(side="left")
        tk.Button(buttons, text="Delete last", command=self.delete_last).pack(side="left")

        self.status = tk.Label(root, anchor="w")
        self.status.pack(side="bottom", fill="x")
        self.canvas = None
        self.reload_image()
        self.notes = tk.Label(root, text=(
            "Workflow:\n"
            " 1. Ensure the game shows the element you are calibrating.\n"
            " 2. Reload screenshot -> New region -> draw box -> type name.\n"
            "    (Names: " + ", ".join(DEFAULT_REGION_HINTS) + ")\n"
            " 3. Grids: draw slot 1 (move_slots/item_slots/party_slots), then 'Full-run grid'\n"
            "    to auto-infer step+count across a row; verify the green boxes cover all slots.\n"
            " 4. Pixel analyzer: draw a tight box around the blue star, tint star, and HP bar\n"
            "    to measure HSV values; write them into config.calibration detection entries.\n"
            " 5. Save, then run: python main.py verify"
        ), justify="left")
        self.notes.pack(side="bottom", anchor="w")

    # ---------------- image handling ------------------------------------
    def refresh(self):
        self.frame = self.grab.frame(fresh=True)
        self.height, self.width = self.frame.shape[:2]
        self.scale = min(1.0, 1400.0 / self.width, 900.0 / self.height)
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize(
            (int(self.width * self.scale), int(self.height * self.scale)))
        self.photo = ImageTk.PhotoImage(pil_img)
        if self.canvas:
            self.canvas.destroy()
        self.canvas = tk.Canvas(self.root, width=self.photo.width(),
                                height=self.photo.height(), cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.bind("<Button-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.redraw_all()

    def reload_image(self):
        self.refresh()

    def redraw_all(self):
        for name, r in self.regions.items():
            self.draw_rect(name, r, "red")
        for name, (region, step, count) in self.grids.items():
            x, y, w, h = region
            for i in range(count):
                self.draw_rect(f"{name}[{i}]", [x + step[0] * i, y + step[1] * i, w, h],
                               "#00b140" if i else "orange")

    def draw_rect(self, name, region, outline):
        s = self.scale
        x, y, w, h = region
        self.canvas.create_rectangle(x * s, y * s, (x + w) * s, (y + h) * s,
                                     outline=outline, width=2, tags=f"reg:{name}")
        self.canvas.create_text(x * s + 2, (y + h) * s - 12, text=name, anchor="nw",
                                fill=outline, font=("Segoe UI", 8), tags=f"txt:{name}")

    # ---------------- region creation ------------------------------------
    def on_down(self, e):
        self.drag_start = (e.x, e.y)

    def on_drag(self, e):
        if self.drag_start:
            self.canvas.delete("preview")
            self.canvas.create_rectangle(*self.drag_start, e.x, e.y,
                                         outline="#ffe34d",
                                         width=2, tags="preview")

    def on_up(self, e):
        if not self.drag_start:
            return
        self.canvas.delete("preview")
        s = self.scale
        region = [int(min(self.drag_start[0], e.x) / s),
                  int(min(self.drag_start[1], e.y) / s),
                  int(abs(e.x - self.drag_start[0]) / s),
                  int(abs(e.y - self.drag_start[1]) / s)]
        self.drag_start = None
        if region[2] < 2 or region[3] < 2:
            return
        name = simpledialog.askstring("Name", "Region name:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()
        kind = messagebox.askyesno("Type", "Is this a grid's FIRST slot?\n"
                                   "Yes = grid slot (auto step/count needed later)\n"
                                   "No = simple region")
        if kind:
            step = simpledialog.askstring(
                "Step", "Step between slots as 'dx,dy'\n(example: 0,34):", parent=self.root)
            count = int(simpledialog.askstring("Count", "How many slots?", parent=self.root) or 1)
            step_x, step_y = 0, 0
            try:
                step_x, step_y = [int(v) for v in (step or "0,0").split(",")]
            except ValueError:
                pass
            self.grids[name] = (region, [step_x, step_y], count)
        else:
            self.regions[name] = region
        self.redraw_all()

    def new_region(self):
        messagebox.showinfo("Draw", "Drag a rectangle over a single UI element "
                                    "and enter its name.")

    def new_grid_slot(self):
        messagebox.showinfo("Draw", "Drag the FIRST slot of a grid and choose "
                                    "'Yes', then enter step dx,dy and count.")

    def new_grid_full(self):
        name = simpledialog.askstring("Grid", "Grid name :", parent=self.root)
        if not name or name.strip() not in self.grids:
            messagebox.showerror("Grid", "Define the first slot first "
                                         "('New grid slot').")
            return
        count = int(simpledialog.askstring(
            "Count", "Total number of slots in this grid:", parent=self.root) or 0)
        if count < 1:
            return
        region, step, _ = self.grids[name]
        if step == [0, 0]:
            step = [0, max(region[3], 8)]
            messagebox.showinfo("Step", f"Step defaulted to {step}; edit "
                                        f"config/calibration.json if wrong.")
        self.grids[name] = (region, step, count)
        self.redraw_all()

    def delete_last(self):
        if self.regions:
            name = sorted(self.regions)[-1]
            if messagebox.askyesno("Delete", f"Delete region {name}.", parent=self.root):
                self.regions.pop(name, None)
                self.refresh()

    def list_regions(self):
        lines = [f"{k}={v}" for k, v in sorted(self.regions.items())]
        lines += [f"grid {k}: region={v[0]} step={v[1]} count={v[2]}"
                  for k, v in sorted(self.grids.items())]
        messagebox.showinfo("Regions", "\n".join(lines) or "none yet")

    # ---------------- pixel analyzer --------------------------------------
    def pixel_analyzer(self):
        messagebox.showinfo("Analyze", "Drag a *tight* box over the element to "
                                       "measure its HSV pixel statistics.")
        s = self.scale
        self.pending = True

        def capture(e):
            region = [int(min(self.drag_start[0], e.x) / s) if self.drag_start else 0,
                      int(min(self.drag_start[1], e.y) / s) if self.drag_start else 0,
                      int(abs(e.x - self.drag_start[0]) / s) if self.drag_start else 1,
                      int(abs(e.y - self.drag_start[1]) / s) if self.drag_start else 1]
            self.drag_start = None
            crop = self.grab.capture_region(tuple(region), fresh=True)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            mins, maxs = np.percentile(hsv, 5, axis=0), np.percentile(hsv, 95, axis=0)
            txt = (f"region={region}\n"
                   f"HSV min(5%): {mins.astype(int).tolist()}\n"
                   f"HSV max(95%): {maxs.astype(int).tolist()}\n"
                   f"mean: {hsv.mean(axis=0).astype(int).tolist()}")
            top = np.bincount(hsv[:, 0] // 10).argmax() * 10
            txt += f"\nmost common hue range: {top}-{top + 10}"
            messagebox.showinfo("Pixel analysis (OpenCV HSV scale)", txt)

        self.canvas.bind("<Button-1>", self.on_down, add="+")
        self.canvas.bind("<ButtonRelease-1>", capture, add="+")

    # ---------------- persistence -----------------------------------------
    def save(self):
        data = {
            "regions": {k: [int(v) for v in r] for k, r in self.regions.items()},
            "grids": {
                name: {"region": region, "step": step, "count": count}
                for name, (region, step, count) in self.grids.items()
            },
        }
        # merge into existing detection block if present
        old = {}
        if os.path.exists(CALIBRATION_PATH):
            try:
                with open(CALIBRATION_PATH, "r", encoding="utf-8") as fh:
                    old = json.load(fh)
            except Exception:  # noqa: BLE001
                pass
        det = old.get("detection", {})
        data["detection"] = det
        os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
        with open(CALIBRATION_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        self.status.config(text=f"saved {len(self.regions)} regions and "
                                f"{len(self.grids)} grids to {CALIBRATION_PATH}")
        messagebox.showinfo("Saved", "Calibration saved.\nNext: python main.py verify")


def run_editor():
    patterns = ["Roblox"]
    try:
        with open(os.path.join(os.path.dirname(CALIBRATION_PATH),
                               "calibrate_window.txt"), "r", encoding="utf-8") as fh:
            patterns = [fh.read().strip()] or patterns
    except OSError:
        pass
    root = tk.Tk()
    try:
        app = CalibratorApp(root, patterns, window_index=0)
    except WindowNotFound as exc:
        messagebox.showerror("Window not found", str(exc))
        return
    root.mainloop()
