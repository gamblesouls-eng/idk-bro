"""Embed field-splitting: detected_text -> subject + labeled inline fields."""
from poke_hunter.notify import Notifier


IMPORTANT = ("Gastly | Reskin: fear | Gradient: whe | Tint: yes | Shiny: ?")


def _split(text):
    return Notifier._split_detected(text)


def test_split_subject_and_fields():
    subject, fields = _split(IMPORTANT)
    assert subject == "Gastly"
    labels = [name for name, _value, _inline in fields]
    assert labels == ["Reskin", "Gradient", "Tint", "Shiny"]
    values = [value for _name, value, _inline in fields]
    assert values == ["**fear**", "**whe**", "**yes**", "**?**"]
    assert all(inline for _n, _v, inline in fields)


def test_split_tolerates_garbage():
    subject, fields = _split("")
    assert subject == "Unknown Pokemon"
    assert fields == []
    subject, fields = _split("Gastly")
    assert subject == "Gastly" and fields == []
    subject, fields = _split("Gastly | (no label here)")
    assert subject == "Gastly" and fields == []
