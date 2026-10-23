"""Target matching (tri-state) tests: every property, single-property flips."""
import pytest

from poke_hunter.config import Target
from poke_hunter.detectors import Tri
from poke_hunter.matching import Detected, evaluate

TARGET = Target(species="Quaxly", shiny=True, reskin="inferno",
                gradient="mirage", tint=False)


def D(**kw):
    base = dict(species="quaxly", shiny=True, reskin="inferno",
                gradient="mirage", tint=False)
    base.update(kw)
    return Detected(**base)


def test_full_match():
    r = evaluate(D(), TARGET)
    assert r.matched is True
    assert r.overall is Tri.MATCH
    assert not r.failed and not r.unknown


def test_each_single_flip_rejects():
    flips = [
        D(species="azumarill"),
        D(shiny=False),
        D(reskin="glacier"),
        D(gradient="ember"),
        D(tint=True),
    ]
    for detected in flips:
        r = evaluate(detected, TARGET)
        assert r.overall is Tri.NO_MATCH, f"should reject {detected}"
        assert r.failed, "the flipped property must be recorded"


def test_unknown_never_matches():
    for prop in ("species", "shiny", "reskin", "gradient", "tint"):
        d = D(**{prop: None})
        r = evaluate(d, TARGET)
        assert r.overall is Tri.UNKNOWN
        assert prop in r.unknown
        assert r.matched is False


def test_partial_info_is_rejected_not_matched():
    # species known, everything else unreadable -> UNKNOWN -> not a match
    r = evaluate(D(species="quaxly", shiny=None, reskin=None,
                   gradient=None, tint=None), TARGET)
    assert r.overall is Tri.UNKNOWN
    assert r.matched is False


def test_case_and_whitespace_normalization():
    d = D(species="QUAXLY ", reskin=" Inferno  ", gradient=" MIRAGE")
    assert evaluate(d, TARGET).matched


def test_similar_species_name_is_not_substituted():
    # exact + small OCR jitter OK; clearly different species stays NO_MATCH
    r = evaluate(D(species="mudkip"), TARGET)
    assert r.overall is Tri.NO_MATCH
    # 'quaxlyy' is a one-letter OCR artifact of Quaxly -> accepted match
    assert evaluate(D(species="quaxlyy"), TARGET).matched


def test_notification_text_shapes():
    d = D()
    text = d.as_notification()
    assert "Quaxly" not in text  # notification uses detected, not target
    # clean labeled pieces: lowercase values, " | " separated
    assert "Shiny: yes" in text and "Tint: no" in text
    pieces = text.split(" | ")
    assert pieces[0] == "quaxly"
    assert "Reskin: inferno" in pieces[1] or pieces[1] == "Reskin: inferno"


# --- "reskin AND gradient present, everything else don't-care" rule --------- #

ANY_BOTH = Target(species=None, shiny=None, reskin="any", gradient="any",
                  tint=None)


def test_any_both_catches():
    # any species, shiny or not, tinted or not -> presence rules only
    d = D(species="gastly", shiny=False, reskin="christmas",
          gradient="starlight", tint=True)
    assert evaluate(d, ANY_BOTH).matched
    d2 = D(species="mewtwo", shiny=True, reskin="sorcerer",
           gradient="inferno", tint=False)
    assert evaluate(d2, ANY_BOTH).matched


def test_any_both_rejects_missing_reskin():
    r = evaluate(D(reskin=""), ANY_BOTH)
    assert r.overall is Tri.NO_MATCH and "reskin" in r.failed


def test_any_both_rejects_missing_gradient():
    r = evaluate(D(gradient=""), ANY_BOTH)
    assert r.overall is Tri.NO_MATCH and "gradient" in r.failed


def test_any_both_unreadable_is_not_catch():
    for prop in ("reskin", "gradient"):
        r = evaluate(D(**{prop: None}), ANY_BOTH)
        assert r.matched is False
        assert r.overall is Tri.UNKNOWN


def test_dontcare_shiny_unreadable_still_catches():
    # shiny star region uncalibrated / unreadable: must NOT block the catch
    d = D(species="litwick", shiny=None, reskin="christmas",
          gradient="starlight", tint=None)
    assert evaluate(d, ANY_BOTH).matched
