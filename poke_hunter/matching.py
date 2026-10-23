"""Three-valued target matching (Section 6).

    MATCH     - property confirmed equal to target
    NO_MATCH  - property confirmed present but different
    UNKNOWN   - property could not be read reliably; never counts as a match
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import Target, TargetDefinition
from .detectors import Tri
from .util import close_match, normalize_text


@dataclass
class Detected:
    """All five properties of one encounter, as far as they could be read.

    species / reskin / gradient: normalized string; "" = readable but absent;
    None = unreadable (unknown).
    shiny / tint: True / False / None.
    """
    species: Optional[str] = None
    shiny: Optional[bool] = None
    reskin: Optional[str] = None
    gradient: Optional[str] = None
    tint: Optional[bool] = None

    def as_notification(self) -> str:
        """Clean, labeled pieces separated by ' | ' — the notifier splits
        this into Discord embed fields.

        Order: Subject | Aspect: value | ..."""
        def yn(v):
            if isinstance(v, bool):
                return "yes" if v else "no"
            return "?" if v is None else v

        def aspect(v):
            return ("none" if v == "" else "?" if v is None else v)

        subject = (self.species or "").strip() or "Unknown"
        return (" | ".join([
            subject,
            f"Reskin: {aspect(self.reskin)}",
            f"Gradient: {aspect(self.gradient)}",
            f"Tint: {yn(self.tint)}",
            f"Shiny: {yn(self.shiny)}",
        ]))


@dataclass
class MatchResult:
    overall: Tri = Tri.UNKNOWN
    per_property: Dict[str, Tri] = field(default_factory=dict)
    failed: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return self.overall is Tri.MATCH

    def debug(self) -> str:
        parts = [f"{k}={v.value}" for k, v in self.per_property.items()]
        return ", ".join(parts)


@dataclass
class TargetEvaluation:
    target_id: str
    enabled: bool
    result: MatchResult


@dataclass
class MultiMatchResult:
    evaluations: List[TargetEvaluation] = field(default_factory=list)

    @property
    def matching_target_ids(self) -> List[str]:
        return [e.target_id for e in self.evaluations if e.enabled and e.result.matched]

    @property
    def matched(self) -> bool:
        return bool(self.matching_target_ids)


def _compare_text(observed: Optional[str], expected: Optional[str], key: str) -> Tri:
    """Text property with criterion semantics.

    expected None     -> don't care: readable or unreadable both MATCH
    expected "any"    -> present (non-empty) MATCH ; empty NO_MATCH ; unstable UNKNOWN
    expected "name"   -> exact/fuzzy match required ; empty->NO_MATCH ; None->UNKNOWN
    """
    if expected is None:
        return Tri.MATCH
    if observed is None:
        return Tri.UNKNOWN
    observed = normalize_text(observed)
    if expected == "any":
        return Tri.MATCH if observed else Tri.NO_MATCH
    expected = normalize_text(expected)
    if observed == expected or (expected and close_match(observed, expected)):
        return Tri.MATCH
    return Tri.NO_MATCH


def _compare_bool(observed: Optional[bool], expected: Optional[bool]) -> Tri:
    if expected is None:
        return Tri.MATCH          # don't care
    if observed is None:
        return Tri.UNKNOWN
    return Tri.MATCH if observed == expected else Tri.NO_MATCH


def evaluate(detected: Detected, target: Target, fuzzy_threshold: float = 0.88) -> MatchResult:
    """Compare a fully-detected encounter against the configured target.

    Don't-care criteria never block a catch. A REQUIRED value that could not be
    read reliably yields UNKNOWN, and UNKNOWN NEVER counts as a match (6.1).
    """
    res = MatchResult()
    tris = {
        "species": _compare_text(detected.species, None if target.species_any else target.species, "species"),
        "shiny": _compare_bool(detected.shiny, target.shiny),
        "reskin": _compare_text(detected.reskin, "any" if target.reskin_any else (None if target.reskin_optional else target.reskin), "reskin"),
        "gradient": _compare_text(detected.gradient, "any" if target.gradient_any else (None if target.gradient_optional else target.gradient), "gradient"),
        "tint": _compare_bool(detected.tint, target.tint),
    }
    res.per_property = tris
    for key, tri in tris.items():
        if tri is Tri.MATCH:
            continue
        if tri is Tri.NO_MATCH:
            res.failed.append(key)
        else:
            res.unknown.append(key)
    if res.failed:
        res.overall = Tri.NO_MATCH
    elif res.unknown:
        res.overall = Tri.UNKNOWN
    else:
        res.overall = Tri.MATCH
    return res


def evaluate_all(detected: Detected, targets: List[TargetDefinition]) -> MultiMatchResult:
    evaluations: List[TargetEvaluation] = []
    for target in targets:
        result = evaluate(detected, target.criteria)
        evaluations.append(TargetEvaluation(
            target_id=target.identifier,
            enabled=target.enabled,
            result=result,
        ))
    return MultiMatchResult(evaluations=evaluations)
