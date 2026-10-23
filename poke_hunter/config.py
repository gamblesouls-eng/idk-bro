"""Configuration loading and validation.

Files (in the project "config" directory):
    config.json      - targets, battle settings, automation, discord, instances
    calibration.json - per-window UI regions, slot grids and detection thresholds
    .webhook         - optional local file containing the Discord webhook URL
                       (never committed; also supported via env var)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
CALIBRATION_PATH = os.path.join(CONFIG_DIR, "calibration.json")
WEBHOOK_FILE = os.path.join(CONFIG_DIR, ".webhook")
log = logging.getLogger("poke_hunter.config")


class ConfigError(Exception):
    """Raised for invalid or missing configuration values."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Target:
    # None/"any" = don't care ; non-empty string = required value
    # reskin/gradient "any" = any non-empty value must be present
    species: Optional[str]
    shiny: Optional[bool]
    reskin: Optional[str]
    gradient: Optional[str]
    tint: Optional[bool]

    def describe(self) -> str:
        def txt(v):
            if v is None:
                return "any"
            if v == "any":
                return "must exist"
            return v
        def b(v):
            return ("Yes" if v else "No") if isinstance(v, bool) else "any"
        return (f"{txt(self.species)} | Shiny: {b(self.shiny)} | "
                f"Reskin: {txt(self.reskin)} | Gradient: {txt(self.gradient)} | "
                f"Tint: {b(self.tint)}")

    @property
    def species_any(self) -> bool:
        return not self.species

    @property
    def shiny_any(self) -> bool:
        return self.shiny is None

    @property
    def tint_any(self) -> bool:
        return self.tint is None

    @property
    def reskin_any(self) -> bool:
        return self.reskin == "any"

    @property
    def gradient_any(self) -> bool:
        return self.gradient == "any"

    @property
    def reskin_optional(self) -> bool:
        return self.reskin is None

    @property
    def gradient_optional(self) -> bool:
        return self.gradient is None


@dataclass(frozen=True)
class TargetDefinition:
    identifier: str
    enabled: bool
    criteria: Target

    def describe(self) -> str:
        state = "enabled" if self.enabled else "disabled"
        return f"{self.identifier} ({state}) -> {self.criteria.describe()}"


@dataclass(frozen=True)
class BattleConfig:
    target_hp_state: str = "red"
    weakening_move: str = "False Swipe"
    # "damage": repeat the move until target_hp_state is reached (False Swipe).
    # "sleep": the move does no damage but applies a status (Spore); cast once
    # per encounter, then catch at full HP — target_hp_state is ignored.
    weakening_style: str = "damage"
    recovery_species: str = "Marshadow"
    recovery_level: int = 150
    recovery_item: str = "Full Restore"
    pp_item: str = "Max Elixir"
    heal_hp_state: str = "yellow"
    catch_item: str = "Repeat Ball"


@dataclass(frozen=True)
class AutomationConfig:
    max_ocr_retries: int = 3
    max_action_retries: int = 3
    max_weaken_attempts: int = 40
    max_recovery_cycles: int = 2
    max_ball_attempts: int = 6
    encounter_stability_frames: int = 3
    encounter_end_frames: int = 3
    loop_interval_ms: int = 250
    post_click_settle_ms: int = 700
    ui_settle_ms: int = 800
    attack_settle_ms: int = 2200
    throw_settle_ms: int = 3500
    run_end_timeout_s: float = 8.0  # flee confirm: the flee presentation keeps
    # the red MUTE slot red for ~6.6-6.9s (live 2026-10-23: both flees went
    # idle only then), so 6.0 always expired and wasted the re-check detour;
    # 8s lets the first wait_until("idle") succeed while a genuinely missed
    # click still re-fires once within the retry loop.
    catch_result_timeout_s: float = 45.0
    notification_target_detected: bool = True
    stop_after_successful_capture: bool = False
    capture_each_encounter: bool = True
    save_after_capture: bool = True
    max_instances: int = 1


@dataclass(frozen=True)
class DiscordConfig:
    webhook_env_var: str = "POKEMON_DISCORD_WEBHOOK"
    webhook_file: str = WEBHOOK_FILE
    notify_capturing: bool = False
    notify_recovery: bool = True
    notify_failures: bool = True
    timeout_s: float = 10.0
    max_attempts: int = 3


@dataclass(frozen=True)
class SaveConfig:
    command: str = "/SAVE"
    confirmation_timeout_s: float = 10.0
    max_attempts: int = 2
    confirmation_phrases: Tuple[str, ...] = (
        "successfully saved",
        "saved successfully",
        "game saved successfully",
    )
    green_hsv_min: Tuple[int, int, int] = (38, 80, 80)
    green_hsv_max: Tuple[int, int, int] = (90, 255, 255)
    min_green_fraction: float = 0.002


@dataclass(frozen=True)
class InstanceConfig:
    name: str
    window_title_patterns: List[str]
    window_index: int = 0

    @staticmethod
    def parse(raw: Any, index: int) -> "InstanceConfig":
        if isinstance(raw, str):
            return InstanceConfig(name=raw, window_title_patterns=[raw])
        if isinstance(raw, dict):
            name = str(raw.get("name") or f"instance{index}")
            patterns = raw.get("window_title_patterns") or ["Roblox"]
            if isinstance(patterns, str):
                patterns = [patterns]
            return InstanceConfig(
                name=name,
                window_title_patterns=[str(p) for p in patterns],
                window_index=int(raw.get("window_index", 0)),
            )
        raise ConfigError(f"instances[{index}]: each entry must be a title string or an object")


@dataclass
class AppConfig:
    targets: List[TargetDefinition]
    battle: BattleConfig
    automation: AutomationConfig
    discord: DiscordConfig
    save: SaveConfig
    instances: List[InstanceConfig]

    @property
    def target(self) -> Target:
        if not self.targets:
            raise ConfigError("no configured targets")
        return self.targets[0].criteria


@dataclass(frozen=True)
class StarSpec:
    hsv_min: Tuple[int, int, int]
    hsv_max: Tuple[int, int, int]
    min_pixels: int = 8
    # background description (what the region looks like when the star is absent)
    bg_hsv_min: Tuple[int, int, int] = (0, 0, 0)
    bg_hsv_max: Tuple[int, int, int] = (179, 120, 140)
    bg_min_fraction: float = 0.85


@dataclass(frozen=True)
class HPSpec:
    colors: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    min_fraction: float = 0.30


@dataclass(frozen=True)
class GradientBandSpec:
    """Legacy gradient chip presence: chip lettering is always strongly
    saturated (red / blue / green gradient fills), the surrounding battle
    background is pastel. Fraction of saturated pixels inside the band
    decides presence.

    Superseded by AspectChipSpec (two-slot chip reader) but kept as the
    fallback when aspect_chips is absent from calibration.json."""
    sat_min: int = 120
    val_min: int = 80
    min_fraction: float = 0.03


@dataclass(frozen=True)
class AspectChipSpec:
    """Two-slot wild-card aspect chip reader (2026-10 live-calibrated).

    The wild card renders aspect badges in two stacked slots:
      * reskin slot (upper): bright italic lettering directly on the gray
        panel (e.g. 'MAGMA', 'DOODLE', 'FOREST GUARDIAN'), y ~365..395 --
        detected by shape-filtered letter blobs (streak lines dropped)
        and OCR'd via a clean letters-only canvas.
      * gradient slot (lower): a dark chip box (v ~121) that exists only
        while a chip is rendered, holding the gradient badge lettering --
        colored ('GLITCHED Rufescent', 'JACK-O-LANTERN'), dim ('Nebula')
        or desaturated ('Reaper') all included.

    Chips fade in seconds after the encounter panel settles, so both slots
    are poll-waited for up to ``wait_ms`` before concluding absent."""
    reskin_zone: Tuple[int, int, int, int]
    gradient_zone: Tuple[int, int, int, int]
    reskin_letter_v_min: int = 158
    reskin_letter_s_max: int = 80
    reskin_min_pixels: int = 180
    grad_sat_min: int = 100
    grad_val_min: int = 90
    grad_min_pixels: int = 430
    dark_box_v_min: int = 95
    dark_box_v_max: int = 145
    dark_box_s_max: int = 60
    dark_box_min_pixels: int = 6000
    wait_ms: int = 2500
    poll_ms: int = 300


@dataclass(frozen=True)
class FavoriteSpec:
    """Favorite ★ badge on a backpack item row: a bright yellow star drawn at
    the icon's bottom-left corner. `zone` is the in-crop pixel rect where the
    star is looked for; non-favorited rows show no saturated yellow there."""
    zone: Tuple[int, int, int, int] = (8, 45, 55, 33)
    hsv_min: Tuple[int, int, int] = (20, 110, 150)
    hsv_max: Tuple[int, int, int] = (48, 255, 255)
    min_fraction: float = 0.10


@dataclass(frozen=True)
class SlotGrid:
    """A repeated UI element: base region + step and count.

    Two-column panels (e.g. the party screen) set ``per_col_rows`` to the
    number of slots stacked in the first column; ``col_offset`` is the
    (dx, dy) jump applied to reach the top of the next column."""
    region: Tuple[int, int, int, int]
    step: Tuple[int, int]
    count: int
    per_col_rows: int = 0
    col_offset: Tuple[int, int] = (0, 0)
    # Optional (x, y, w, h) rectangle inside each slot that is OCR'd instead
    # of the full slot region. Used for list rows whose icon / star badge /
    # trailing quantity tag confuse Tesseract's line parser while the full
    # region remains the click geometry.
    text_zone: Optional[Tuple[int, int, int, int]] = None

    def slot(self, i: int) -> Tuple[int, int, int, int]:
        x, y, w, h = self.region
        dx, dy = self.step
        if self.per_col_rows and self.col_offset != (0, 0):
            col, row = divmod(i, self.per_col_rows)
            return (x + self.col_offset[0] * col, y + self.col_offset[1] * col + dy * row, w, h)
        return (x + dx * i, y + dy * i, w, h)


@dataclass
class Calibration:
    regions: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    grids: Dict[str, SlotGrid] = field(default_factory=dict)
    shiny_star: Optional[StarSpec] = None
    tint_star: Optional[StarSpec] = None
    gradient_band: Optional[GradientBandSpec] = None
    aspect_chips: Optional[AspectChipSpec] = None
    favorite_star: Optional[FavoriteSpec] = None
    hp: Optional[HPSpec] = None
    mute_baseline_path: str = ""

    # --- accessors ---------------------------------------------------------
    def region(self, name: str) -> Tuple[int, int, int, int]:
        if name not in self.regions:
            raise ConfigError(f"calibration.json: missing region '{name}' "
                              f"(run `python main.py calibrate` and add it)")
        return self.regions[name]

    def optional_region(self, name: str) -> Optional[Tuple[int, int, int, int]]:
        return self.regions.get(name)

    def grid(self, name: str) -> SlotGrid:
        if name not in self.grids:
            raise ConfigError(f"calibration.json: missing slot grid '{name}' "
                              f"(run `python main.py calibrate` and add it)")
        return self.grids[name]


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

def _require(raw: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in raw:
        raise ConfigError(f"{ctx}: missing required key '{key}'")
    return raw[key]


def _parse_target(raw: Any) -> Target:
    if not isinstance(raw, dict):
        raise ConfigError("target: must be an object")

    def _opt_str(key):
        v = raw.get(key, None)
        if v is None:
            return None
        if not isinstance(v, str):
            raise ConfigError(f"target.{key}: must be a string or null")
        v = v.strip()
        if v.lower() == "any":
            return "any"
        return v.lower()

    def _opt_bool(key):
        v = raw.get(key, None)
        if v is None:
            return None
        if not isinstance(v, bool):
            raise ConfigError(f"target.{key}: must be a boolean or null")
        return v

    species = _opt_str("species")
    if species == "any":
        species = None          # "any" and null both mean don't-care
    shiny = _opt_bool("shiny")
    reskin = _opt_str("reskin")
    gradient = _opt_str("gradient")
    tint = _opt_bool("tint")
    if all(v is None for v in (species, shiny, reskin, gradient, tint)):
        raise ConfigError("target: at least one criterion must be required")
    return Target(species=species, shiny=shiny, reskin=reskin,
                  gradient=gradient, tint=tint)


def _target_fingerprint(target: Target) -> Tuple[Any, ...]:
    return (target.species, target.shiny, target.reskin, target.gradient, target.tint)


def _parse_targets(raw: Dict[str, Any]) -> List[TargetDefinition]:
    if "targets" in raw:
        targets_raw = raw["targets"]
        if not isinstance(targets_raw, list) or not targets_raw:
            raise ConfigError("targets: must be a non-empty array")
        out: List[TargetDefinition] = []
        seen_ids: set[str] = set()
        seen_fingerprints: Dict[Tuple[Any, ...], str] = {}
        for index, entry in enumerate(targets_raw):
            ctx = f"targets[{index}]"
            if not isinstance(entry, dict):
                raise ConfigError(f"{ctx}: must be an object")
            enabled = entry.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ConfigError(f"{ctx}.enabled: must be a boolean")
            identifier = str(entry.get("id") or entry.get("name") or f"target_{index + 1}").strip()
            if not identifier:
                raise ConfigError(f"{ctx}: id/name must not be empty")
            if identifier in seen_ids:
                raise ConfigError(f"{ctx}: duplicate target identifier '{identifier}'")
            seen_ids.add(identifier)
            criteria = _parse_target(entry)
            fingerprint = _target_fingerprint(criteria)
            prior = seen_fingerprints.get(fingerprint)
            if prior:
                log.warning("%s duplicates the target definition used by %s", ctx, prior)
            else:
                seen_fingerprints[fingerprint] = identifier
            out.append(TargetDefinition(identifier=identifier, enabled=enabled, criteria=criteria))
        if not any(t.enabled for t in out):
            raise ConfigError("targets: at least one target must be enabled")
        return out
    if "target" not in raw:
        raise ConfigError("config.json: missing target/targets configuration")
    return [TargetDefinition(identifier="target_1", enabled=True, criteria=_parse_target(raw.get("target")))]


def _parse_battle(raw: Any) -> BattleConfig:
    if not isinstance(raw, dict):
        raise ConfigError("battle: must be an object")
    hp_state = str(raw.get("target_hp_state", "red")).lower()
    if hp_state not in ("red", "yellow"):
        raise ConfigError("battle.target_hp_state: must be 'red' or 'yellow'")
    heal_state = str(raw.get("heal_hp_state", "yellow")).lower()
    if heal_state not in ("yellow", "red"):
        raise ConfigError("battle.heal_hp_state: must be 'yellow' or 'red'")
    level = int(raw.get("recovery_pokemon_level", 150))
    if not (1 <= level <= 300):
        raise ConfigError("battle.recovery_pokemon_level: must be between 1 and 300")
    style = str(raw.get("weakening_style", "damage")).lower()
    if style not in ("damage", "sleep"):
        raise ConfigError(
            "battle.weakening_style: must be 'damage' or 'sleep'")
    return BattleConfig(
        target_hp_state=hp_state,
        weakening_move=str(_require(raw, "weakening_move", "battle")),
        weakening_style=style,
        recovery_species=str(raw.get("recovery_pokemon_species", "Marshadow")),
        recovery_level=level,
        recovery_item=str(raw.get("recovery_item", "Full Restore")),
        pp_item=str(raw.get("pp_item", "Max Elixir")),
        heal_hp_state=heal_state,
        catch_item=str(_require(raw, "catch_item", "battle")),
    )


def _parse_automation(raw: Any) -> AutomationConfig:
    if not isinstance(raw, dict):
        raise ConfigError("automation: must be an object")
    ints = {
        "max_ocr_retries": 3, "max_action_retries": 3, "max_weaken_attempts": 40,
        "max_recovery_cycles": 2, "max_ball_attempts": 6,
        "encounter_stability_frames": 3, "encounter_end_frames": 3,
        "loop_interval_ms": 250, "post_click_settle_ms": 700,
        "ui_settle_ms": 800, "attack_settle_ms": 2200, "throw_settle_ms": 3500,
        "run_end_timeout_s": 8, "catch_result_timeout_s": 45, "max_instances": 1,
    }
    vals: Dict[str, Any] = {}
    for k, default in ints.items():
        v = raw.get(k, default)
        if not isinstance(v, (int, float)) or v < 0 or int(v) != v:
            raise ConfigError(f"automation.{k}: must be a non-negative integer")
        vals[k] = int(v)
    if vals["loop_interval_ms"] < 50:
        raise ConfigError("automation.loop_interval_ms: minimum 50 ms")
    bools = {k: bool(raw.get(k, d)) for k, d in
             (("notification_target_detected", True),
              ("stop_after_successful_capture", False),
              ("capture_each_encounter", True),
              ("save_after_capture", True))}
    return AutomationConfig(**vals, **bools)


def _parse_discord(raw: Any) -> DiscordConfig:
    if not isinstance(raw, dict):
        raise ConfigError("discord: must be an object")
    return DiscordConfig(
        webhook_env_var=str(raw.get("webhook_env_var", "POKEMON_DISCORD_WEBHOOK")),
        webhook_file=str(raw.get("webhook_file", WEBHOOK_FILE)),
        notify_capturing=bool(raw.get("notify_capturing", False)),
        notify_recovery=bool(raw.get("notify_recovery", True)),
        notify_failures=bool(raw.get("notify_failures", True)),
        timeout_s=float(raw.get("timeout_s", 10.0)),
        max_attempts=int(raw.get("max_attempts", 3)),
    )


def _parse_save(raw: Any) -> SaveConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("save: must be an object")
    command = str(raw.get("command", "/SAVE")).strip()
    if command != "/SAVE":
        raise ConfigError("save.command: must be the exact string '/SAVE'")
    phrases = raw.get("confirmation_phrases", SaveConfig.confirmation_phrases)
    if not isinstance(phrases, (list, tuple)) or not phrases:
        raise ConfigError("save.confirmation_phrases: must be a non-empty array of strings")
    return SaveConfig(
        command=command,
        confirmation_timeout_s=float(raw.get("confirmation_timeout_s", 10.0)),
        max_attempts=int(raw.get("max_attempts", 2)),
        confirmation_phrases=tuple(str(p).strip().lower() for p in phrases if str(p).strip()),
        green_hsv_min=tuple(int(v) for v in raw.get("green_hsv_min", (38, 80, 80))),
        green_hsv_max=tuple(int(v) for v in raw.get("green_hsv_max", (90, 255, 255))),
        min_green_fraction=float(raw.get("min_green_fraction", 0.002)),
    )


def load_app_config(path: str = CONFIG_PATH) -> AppConfig:
    if not os.path.exists(path):
        raise ConfigError(
            f"Configuration file not found: {path}\n"
            "Copy config/config.example.json to config/config.json and edit it."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ConfigError("config.json: top level must be an object")
    instances_raw = raw.get("instances", ["Roblox"])
    if isinstance(instances_raw, str):
        instances_raw = [instances_raw]
    if not isinstance(instances_raw, list) or not instances_raw:
        raise ConfigError("instances: must be a non-empty array")
    instances = [InstanceConfig.parse(v, i) for i, v in enumerate(instances_raw)]
    targets = _parse_targets(raw)
    max_instances = int((raw.get("automation") or {}).get("max_instances", len(instances)))
    if max_instances < 1:
        raise ConfigError("automation.max_instances: must be at least 1")
    if len(instances) > max_instances:
        instances = instances[:max_instances]
        log.warning("instances trimmed to automation.max_instances=%d", max_instances)
    return AppConfig(
        targets=targets,
        battle=_parse_battle(raw.get("battle", {})),
        automation=_parse_automation(raw.get("automation", {})),
        discord=_parse_discord(raw.get("discord", {})),
        save=_parse_save(raw.get("save", {})),
        instances=instances,
    )


def _parse_region(v: Any, name: str) -> Tuple[int, int, int, int]:
    try:
        x, y, w, h = v
        return (int(x), int(y), int(w), int(h))
    except (TypeError, ValueError):
        raise ConfigError(f"calibration.json: region '{name}' must be [x, y, w, h]")

def _parse_star_spec(raw: Any) -> Optional[StarSpec]:
    if not isinstance(raw, dict):
        return None
    return StarSpec(
        hsv_min=tuple(int(v) for v in raw["hsv_min"]),
        hsv_max=tuple(int(v) for v in raw["hsv_max"]),
        min_pixels=int(raw.get("min_pixels", 8)),
        bg_hsv_min=tuple(int(v) for v in raw.get("bg_hsv_min", (0, 0, 0))),
        bg_hsv_max=tuple(int(v) for v in raw.get("bg_hsv_max", (179, 120, 140))),
        bg_min_fraction=float(raw.get("bg_min_fraction", 0.85)),
    )

def _parse_hp_spec(raw: Any) -> Optional[HPSpec]:
    if not isinstance(raw, dict):
        return None
    colors = {}
    for name in ("red", "yellow", "green"):
        if name in raw:
            colors[name] = (
                tuple(int(v) for v in raw[name]["hsv_min"]),
                tuple(int(v) for v in raw[name]["hsv_max"]),
            )
    return HPSpec(colors=colors, min_fraction=float(raw.get("min_fraction", 0.30)))


def _parse_gradient_spec(raw: Any) -> Optional[GradientBandSpec]:
    if not isinstance(raw, dict):
        return None
    return GradientBandSpec(
        sat_min=int(raw.get("sat_min", 120)),
        val_min=int(raw.get("val_min", 80)),
        min_fraction=float(raw.get("min_fraction", 0.03)),
    )


def _parse_aspect_chips(raw: Any) -> Optional[AspectChipSpec]:
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        reskin_zone = tuple(int(v) for v in raw["reskin_zone"])
        gradient_zone = tuple(int(v) for v in raw["gradient_zone"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(
            "calibration.json: detection.aspect_chips needs reskin_zone "
            "and gradient_zone as [x, y, w, h]") from exc
    if len(reskin_zone) != 4 or len(gradient_zone) != 4:
        raise ConfigError(
            "calibration.json: detection.aspect_chips zones must be [x, y, w, h]")
    return AspectChipSpec(
        reskin_zone=reskin_zone,
        gradient_zone=gradient_zone,
        reskin_letter_v_min=int(raw.get("reskin_letter_v_min", 158)),
        reskin_letter_s_max=int(raw.get("reskin_letter_s_max", 80)),
        reskin_min_pixels=int(raw.get("reskin_min_pixels", 180)),
        grad_sat_min=int(raw.get("grad_sat_min", 100)),
        grad_val_min=int(raw.get("grad_val_min", 90)),
        grad_min_pixels=int(raw.get("grad_min_pixels", 430)),
        dark_box_v_min=int(raw.get("dark_box_v_min", 95)),
        dark_box_v_max=int(raw.get("dark_box_v_max", 145)),
        dark_box_s_max=int(raw.get("dark_box_s_max", 60)),
        dark_box_min_pixels=int(raw.get("dark_box_min_pixels", 6000)),
        wait_ms=int(raw.get("wait_ms", 2500)),
        poll_ms=int(raw.get("poll_ms", 300)),
    )


def _parse_favorite_spec(raw: Any) -> Optional[FavoriteSpec]:
    if not isinstance(raw, dict) or not raw:
        return None
    zone = raw.get("zone", FavoriteSpec.zone)
    return FavoriteSpec(
        zone=(int(zone[0]), int(zone[1]), int(zone[2]), int(zone[3])),
        hsv_min=(int(raw["hsv_min"][0]), int(raw["hsv_min"][1]),
                 int(raw["hsv_min"][2])) if "hsv_min" in raw
        else FavoriteSpec.hsv_min,
        hsv_max=(int(raw["hsv_max"][0]), int(raw["hsv_max"][1]),
                 int(raw["hsv_max"][2])) if "hsv_max" in raw
        else FavoriteSpec.hsv_max,
        min_fraction=float(raw.get("min_fraction", 0.10)),
    )


def load_calibration(path: str = CALIBRATION_PATH) -> Calibration:
    cal = Calibration()
    if not os.path.exists(path):
        return cal  # tests / CLI tools may work without it
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ConfigError("calibration.json: top level must be an object")
    for name, v in (raw.get("regions") or {}).items():
        cal.regions[name] = _parse_region(v, name)
    for name, v in (raw.get("grids") or {}).items():
        if not isinstance(v, dict) or "region" not in v or "step" not in v or "count" not in v:
            raise ConfigError(f"calibration.json: grid '{name}' needs region, step, count")
        tz_raw = v.get("text_zone")
        tz = None
        if tz_raw is not None:
            if (not isinstance(tz_raw, list) or len(tz_raw) != 4
                    or not all(isinstance(n, (int, float)) for n in tz_raw)):
                raise ConfigError(
                    f"calibration.json: grid '{name}.text_zone' must be [x, y, w, h]")
            tz = (int(tz_raw[0]), int(tz_raw[1]), int(tz_raw[2]), int(tz_raw[3]))
        cal.grids[name] = SlotGrid(
            region=_parse_region(v["region"], f"{name}.region"),
            step=(int(v["step"][0]), int(v["step"][1])),
            count=int(v["count"]),
            per_col_rows=int(v.get("per_col_rows", 0) or 0),
            col_offset=(
                int(v.get("col_offset", [0, 0])[0]),
                int(v.get("col_offset", [0, 0])[1]),
            ),
            text_zone=tz,
        )
    det = raw.get("detection") or {}
    cal.shiny_star = _parse_star_spec(det.get("shiny_star"))
    cal.tint_star = _parse_star_spec(det.get("tint_star"))
    cal.gradient_band = _parse_gradient_spec(det.get("gradient_band"))
    cal.aspect_chips = _parse_aspect_chips(det.get("aspect_chips"))
    cal.favorite_star = _parse_favorite_spec(det.get("favorite_star"))
    cal.hp = _parse_hp_spec(det.get("hp_colors"))
    cal.mute_baseline_path = str(det.get("mute_baseline_path", ""))
    if not cal.shiny_star:
        raise ConfigError("calibration.json detection.shiny_star missing")
    if not cal.tint_star:
        raise ConfigError("calibration.json detection.tint_star missing")
    if not cal.gradient_band:
        raise ConfigError("calibration.json detection.gradient_band missing")
    if not cal.aspect_chips:
        raise ConfigError("calibration.json detection.aspect_chips missing")
    if not cal.hp or not cal.hp.colors:
        raise ConfigError("calibration.json detection.hp_colors missing")
    return cal


def resolve_webhook(dc: DiscordConfig) -> Tuple[Optional[str], str]:
    """Locate the webhook secret. Returns (url|None, source). Never logs it."""
    env = os.environ.get(dc.webhook_env_var, "").strip()
    if env:
        return env, f"env:{dc.webhook_env_var}"
    if dc.webhook_file and os.path.exists(dc.webhook_file):
        with open(dc.webhook_file, "r", encoding="utf-8") as fh:
            url = fh.read().strip()
        if url:
            return url, f"file:{os.path.basename(dc.webhook_file)}"
    return None, "none"


def get_calibration_dir() -> str:
    return CONFIG_DIR
