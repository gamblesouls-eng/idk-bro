"""Config / calibration validation tests."""
import json
import os

import pytest

from poke_hunter.config import (ConfigError, load_app_config, load_calibration,
                                InstanceConfig, Target)


def write(tmp_path, monkeypatch, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def full_config(**overrides):
    data = {
        "instances": [{"name": "a", "window_title_patterns": ["Roblox"], "window_index": 0}],
        "target": {"species": "Quaxly", "shiny": True, "reskin": "inferno",
                   "gradient": "mirage", "tint": False},
        "battle": {"weakening_move": "False Swipe", "catch_item": "Repeat Ball",
                   "recovery_pokemon_species": "Marshadow",
                   "recovery_pokemon_level": 150},
        "automation": {},
        "discord": {},
    }
    data.update(overrides)
    return data


def test_valid_config_loads(tmp_path):
    path = write(tmp_path, None, full_config())
    cfg = load_app_config(path)
    assert cfg.target.species == "quaxly"
    assert cfg.battle.recovery_level == 150
    assert cfg.battle.recovery_species == "Marshadow"
    assert cfg.battle.catch_item == "Repeat Ball"
    assert cfg.battle.pp_item == "Max Elixir"
    assert cfg.battle.heal_hp_state == "yellow"
    assert cfg.instances[0].name == "a"


def test_criteria_config_loads(tmp_path):
    data = full_config()
    data["target"] = {"species": None, "shiny": None, "reskin": "any",
                      "gradient": "any", "tint": None}
    cfg = load_app_config(write(tmp_path, None, data))
    assert cfg.target.species is None
    assert cfg.target.species_any
    assert cfg.target.reskin_any and cfg.target.gradient_any
    assert cfg.target.reskin_optional is False


def test_all_optional_target_errors(tmp_path):
    data = full_config()
    data["target"] = {"species": None, "shiny": None, "reskin": None,
                      "gradient": None, "tint": None}
    with pytest.raises(ConfigError):
        load_app_config(write(tmp_path, None, data))


def test_missing_species_is_optional(tmp_path):
    data = full_config()
    del data["target"]["species"]   # null/default: don't care is allowed now
    cfg = load_app_config(write(tmp_path, None, data))
    assert cfg.target.species is None


def test_bad_hp_state_errors(tmp_path):
    data = full_config()
    data["battle"]["target_hp_state"] = "orange"
    with pytest.raises(ConfigError):
        load_app_config(write(tmp_path, None, data))


def test_bad_level_errors(tmp_path):
    data = full_config()
    data["battle"]["recovery_pokemon_level"] = 999
    with pytest.raises(ConfigError):
        load_app_config(write(tmp_path, None, data))


def test_string_instance_accepted():
    icfg = InstanceConfig.parse("Roblox", 0)
    assert icfg.window_title_patterns == ["Roblox"]
    assert icfg.window_index == 0


def test_defaults_not_silent(tmp_path):
    cfg = load_app_config(write(tmp_path, None, full_config()))
    # battle defaults exist per spec
    assert cfg.battle.target_hp_state == "red"
    assert cfg.battle.recovery_item.lower().startswith("full restore") or \
        cfg.battle.recovery_item == "Full Restore"


def test_invalid_heal_hp_state_errors(tmp_path):
    data = full_config()
    data["battle"]["heal_hp_state"] = "green"
    with pytest.raises(ConfigError):
        load_app_config(write(tmp_path, None, data))


def test_heal_red_only_policy_loads(tmp_path):
    data = full_config()
    data["battle"]["heal_hp_state"] = "red"
    data["battle"]["pp_item"] = "Max Elixir"
    cfg = load_app_config(write(tmp_path, None, data))
    assert cfg.battle.heal_hp_state == "red"
    assert cfg.battle.pp_item == "Max Elixir"


def test_weakening_style_sleep_roundtrip(tmp_path):
    data = full_config()
    data["battle"]["weakening_move"] = "Spore"
    data["battle"]["weakening_style"] = "sleep"
    cfg = load_app_config(write(tmp_path, None, data))
    assert cfg.battle.weakening_style == "sleep"
    assert cfg.battle.weakening_move == "Spore"


def test_weakening_style_invalid_rejected(tmp_path):
    data = full_config()
    data["battle"]["weakening_style"] = "powder"
    with pytest.raises(ConfigError) as exc:
        load_app_config(write(tmp_path, None, data))
    assert "weakening_style" in str(exc.value)
