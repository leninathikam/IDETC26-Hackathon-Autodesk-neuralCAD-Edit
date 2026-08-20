"""Instruction Parser stage: target, operation, dimensions, location, preserve."""

from __future__ import annotations

import re

from groundedcad.agents.classifier import classified_to_operation, classify_instruction
from groundedcad.agents.schemas import OperationType
from pydantic import BaseModel, Field


class InstructionSpec(BaseModel):
    raw: str
    target: str = ""
    operation: OperationType = OperationType.CUSTOM
    operations: list[OperationType] = Field(default_factory=list)
    dimensions: dict[str, float] = Field(default_factory=dict)
    location: str = ""
    preserve: list[str] = Field(default_factory=list)
    pattern: str = ""
    action: str = ""
    target_kind: str = ""
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    complete: bool = False


_LOCATION = re.compile(
    r"\b(front(?:\s+center)?|back|left|right|top|bottom|center|middle|"
    r"hole(?:\s+edges?)?|largest|smallest|inner|outer|vertical|horizontal)\b",
    re.I,
)


def parse_instruction(text: str) -> InstructionSpec:
    raw = (text or "").strip()
    edit = classify_instruction(raw)
    loc_m = _LOCATION.findall(raw)
    location = ", ".join(dict.fromkeys(x.lower() for x in loc_m))
    target = edit.target_kind or location or "existing referenced feature"
    op = classified_to_operation(edit)
    return InstructionSpec(
        raw=raw,
        target=target,
        operation=op,
        operations=[op],
        dimensions=edit.to_dimensions(),
        location=location,
        preserve=[
            "unrelated faces, edges, and holes",
            "existing dimensions that the instruction does not name",
        ],
        pattern=edit.edit_type.value,
        action=edit.action,
        target_kind=edit.target_kind,
        direction=edit.direction,
        complete=edit.complete,
    )
