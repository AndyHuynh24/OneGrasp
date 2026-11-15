"""Parse EgoPressure / PressureVisionDB sequence names (both use the same gesture protocol)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_NAME_RE = re.compile(
    r"^(?:p_(?P<pid>\d{3})_)?"  # EgoPressure prefix; PVDB has the participant in the folder name
    r"(?P<gesture>.+?)"
    r"(?:_(?P<reps>x\d+|\d+x))?"
    r"(?:_(?P<take>\d+))?"
    r"_(?P<hand>left|right|both)"
    r"(?:\((?P<dup>\d+)\))?$"  # PVDB re-recordings, e.g. "..._both(1)"
)

# Last gesture token that is a variant of a gesture family (e.g. index_press + high).
_VARIANTS = {
    "high",
    "low",
    "no-contact",
    "pull",
    "push",
    "left",
    "right",
    "rotate-left",
    "rotate-right",
}


@dataclass(frozen=True)
class SequenceName:
    participant: int | None  # EgoPressure participant number; None for PVDB
    gesture: str  # e.g. "index_press_high"
    family: str  # e.g. "index_press"
    variant: str  # e.g. "high" ("" if the gesture has no variant)
    reps: str  # e.g. "x5" or "5x" ("" if absent)
    take: int  # 1 unless the name carries a take number ("_2") or duplicate marker ("(1)")
    hand: str  # "left" | "right" | "both"


def parse_sequence_name(name: str) -> SequenceName:
    """'p_001_index_press_high_x5_right'
    -> SequenceName(1, 'index_press_high', 'index_press', ...)."""
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"unrecognized sequence name: {name!r}")
    gesture = m["gesture"]
    family, _, last = gesture.rpartition("_")
    if not family or last not in _VARIANTS:
        family, last = gesture, ""
    take = int(m["take"]) if m["take"] else (int(m["dup"]) + 1 if m["dup"] else 1)
    return SequenceName(
        participant=int(m["pid"]) if m["pid"] else None,
        gesture=gesture,
        family=family,
        variant=last,
        reps=m["reps"] or "",
        take=take,
        hand=m["hand"],
    )
