"""Typed edit classifier: instruction → slots, not a bigger prompt.

Example:
  "Move the hole 5 mm to the right."
  → TRANSLATE_FEATURE, target=hole, distance=5, direction=+X
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from groundedcad.agents.schemas import EditPattern, OperationType


class ClassifiedEdit(BaseModel):
    edit_type: EditPattern
    target_kind: str = "feature"
    action: str = "edit"
    diameter_mm: Optional[float] = None
    radius_mm: Optional[float] = None
    distance_mm: Optional[float] = None
    groove_mm: Optional[float] = None
    count: Optional[int] = None
    factor: Optional[float] = None
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    complete: bool = False
    plan_status: str = "INCOMPLETE"  # COMPLETE | INCOMPLETE | AMBIGUOUS
    notes: str = ""
    dimensions: dict[str, float] = Field(default_factory=dict)
    # Keep every parsed request available to downstream planners.  ``edit_type``
    # remains the primary routing hint, but a compound instruction must not
    # lose its secondary operations merely because one was sorted first.
    operations: list[dict[str, Any]] = Field(default_factory=list)

    def to_dimensions(self) -> dict[str, float]:
        out = dict(self.dimensions)
        if self.diameter_mm is not None:
            out["diameter"] = self.diameter_mm
        if self.radius_mm is not None:
            out["radius"] = self.radius_mm
        if self.distance_mm is not None:
            out["distance"] = self.distance_mm
            out["value_mm"] = self.distance_mm
        if self.groove_mm is not None:
            out["groove"] = self.groove_mm
        if self.count is not None:
            out["count"] = float(self.count)
        if self.factor is not None:
            out["factor"] = self.factor
        return out


_DIR = {
    "right": (1.0, 0.0, 0.0),
    "+x": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    # Semantic CAD directions use cardinal axes.  Rendered front/back views
    # may add a small elevation so thin parts remain visible.
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "taller": (0.0, 0.0, 1.0),
    "top": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
    "bottom": (0.0, 0.0, -1.0),
    "forward": (0.0, 1.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
}


def _direction(text: str) -> tuple[float, float, float]:
    lower = text.lower()
    for key, vec in _DIR.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return vec
    if re.search(r"\+\s*x\b|\bin x\b", lower):
        return (1.0, 0.0, 0.0)
    if re.search(r"\+\s*y\b|\bin y\b", lower):
        return (0.0, 1.0, 0.0)
    if re.search(r"\+\s*z\b|\bin z\b", lower):
        return (0.0, 0.0, 1.0)
    return (0.0, 0.0, 0.0)


def _blend_verb(text: str) -> bool:
    """True only when the instruction *is* a blend, not 'round pins' / 'rounded end'."""
    if re.search(r"\b(chamfer|fillet)\b", text):
        return True
    if re.search(r"\badd rounds?\b|\badd radii\b|\bradii to\b|\bradius to\b", text):
        return True
    if re.search(r"\brounds? to all edges\b|\br\s*=", text):
        return True
    if re.search(r"\bradius\b", text) and re.search(r"\bremove\b|\breplace\b|\blargest\b|\bsmallest\b", text):
        return True
    return False


def _feature_tag(text: str) -> str:
    lower = text.lower()
    if "rib" in lower:
        return "rib"
    if "rod" in lower or "arm length" in lower:
        return "rod"
    if "screw" in lower:
        return "screw"
    if "port" in lower or "inlet" in lower or "outlet" in lower:
        return "port"
    if "button" in lower:
        return "button"
    if "switch" in lower:
        return "switch"
    if "filling" in lower or "pour" in lower or re.search(r"\bcap\b", lower):
        return "cap"
    return "feature"


class ParsedOp(BaseModel):
    type: str
    target_kind: str = "feature"
    diameter_mm: Optional[float] = None
    radius_mm: Optional[float] = None
    distance_mm: Optional[float] = None
    groove_mm: Optional[float] = None
    count: Optional[int] = None
    factor: Optional[float] = None
    angle_deg: Optional[float] = None


def parse_operations(text: str, dims: dict[str, float]) -> list[ParsedOp]:
    """Collect every requested op. SCALE wins over 'rounds' in the same sentence."""
    lower = text.lower()
    ops: list[ParsedOp] = []
    diameter = dims.get("diameter")
    radius = dims.get("radius")
    distance = dims.get("distance") or dims.get("value_mm")
    groove = dims.get("groove")
    count = int(dims["count"]) if dims.get("count") else None
    factor = dims.get("factor")
    if factor is None:
        m10 = re.search(r"\b(\d+(?:\.\d+)?)x\b", lower)
        if m10:
            factor = float(m10.group(1))
    angle = dims.get("angle")

    if re.search(r"\bscale\b|\b10x\b", lower):
        ops.append(ParsedOp(type="SCALE", target_kind="body", factor=factor))
    elif re.search(r"taller by|shallower|reduce overall height", lower):
        ops.append(ParsedOp(type="SCALE", target_kind="body", distance_mm=distance, factor=factor))
    if re.search(r"\bdraft", lower):
        ops.append(ParsedOp(type="DRAFT", angle_deg=angle or 2.0))
    if re.search(r"\b(move|shift|translate|prolong)\b", lower):
        if re.search(r"\bholes?\b", lower):
            tk = "hole"
        elif re.search(r"\bbody\b|\bthe part\b", lower):
            tk = "body"
        else:
            tk = "feature"
        ops.append(ParsedOp(type="TRANSLATE", target_kind=tk, distance_mm=distance))
    if re.search(
        r"\bpattern\b|\bmirror\b|\binstances?\b|\bmultiply\b|\bduplicate\b|\bcopies of\b|"
        r"third rotor blade|another 7|slot pattern",
        lower,
    ):
        c = count if count is not None else (2 if re.search(r"\bduplicate\b", lower) else None)
        ops.append(ParsedOp(type="PATTERN", count=c, distance_mm=distance))
    if re.search(r"hexagonal profile|flower type|flower-type", lower):
        ops.append(ParsedOp(type="HEX_PROFILE", diameter_mm=diameter, radius_mm=radius))
    if re.search(r"spur gear|gear teeth|into .{0,50}teeth", lower):
        ops.append(ParsedOp(type="GEAR_TEETH", count=count))
    if re.search(r"cut through|through complete body|\bcutouts?\b|\bopening\b", lower) and not _blend_verb(lower):
        ops.append(ParsedOp(type="CUT", distance_mm=distance))
    if re.search(r"remove the collision|\bdelete\b", lower) and not _blend_verb(lower):
        ops.append(ParsedOp(type="DELETE"))
    if re.search(
        r"connecting hole|\bdrill\b|\bbore\b|inscribed hexagonal|"
        r"\bholes?\b.*diameter|diameter.*\bholes?\b|add a .{0,40}\bholes?\b",
        lower,
    ):
        ops.append(ParsedOp(type="ADD_HOLE", target_kind="hole", diameter_mm=diameter, groove_mm=groove))
    skip_blend = bool(
        re.search(r"hexagonal profile|flower type|spur gear|gear teeth|into .{0,50}teeth", lower)
    )
    if (
        not skip_blend
        and (
            _blend_verb(lower)
            or (re.search(r"\brounds?\b|\bradii\b", lower) and not re.search(r"round pins?|rounded end", lower))
        )
    ):
        # Do not let SCALE+rounds collapse to hole. Blend is secondary.
        is_chamfer = bool(re.search(r"\bchamfer\b|\bgrooves?\b", lower))
        tk = "hole_edge" if re.search(r"\bholes?\b|circular|cylind", lower) else "edge"
        if re.search(r"\bslot\b", lower):
            tk = "slot"
        if re.search(r"all edges", lower):
            tk = "all_edges"
        if re.search(r"front center|front centre", lower):
            tk = "front_center"
        ops.append(
            ParsedOp(
                type="CHAMFER" if is_chamfer else "FILLET",
                target_kind=tk,
                distance_mm=distance,
                radius_mm=radius,
                groove_mm=groove,
            )
        )
    if re.search(r"\b(add|create|design|insert)\b", lower) and not any(
        o.type in {"ADD_HOLE", "FILLET", "CHAMFER", "PATTERN", "HEX_PROFILE", "GEAR_TEETH"} for o in ops
    ):
        ops.append(ParsedOp(type="ADD_FEATURE", distance_mm=distance, radius_mm=radius))
    return ops


_PRIMARY_ORDER = [
    "SCALE",
    "TRANSLATE",
    "HEX_PROFILE",
    "GEAR_TEETH",
    "PATTERN",
    "CUT",
    "ADD_HOLE",
    "CHAMFER",
    "FILLET",
    "DELETE",
    "ADD_FEATURE",
    "DRAFT",
]


def classify_instruction(text: str) -> ClassifiedEdit:
    from groundedcad.agents.grounder import _extract_dimensions, _normalize_instruction

    raw = _normalize_instruction(text or "")
    lower = raw.lower()
    dims = _extract_dimensions(raw)
    direction = _direction(lower)
    ops = parse_operations(raw, dims)
    diameter = dims.get("diameter")
    radius = dims.get("radius")
    distance = dims.get("distance") or dims.get("value_mm")
    groove = dims.get("groove")
    count = int(dims["count"]) if dims.get("count") else None
    factor = dims.get("factor")
    if factor is None:
        m10 = re.search(r"\b(\d+(?:\.\d+)?)x\b", lower)
        if m10:
            factor = float(m10.group(1))

    def _done(**kwargs) -> ClassifiedEdit:
        edit = ClassifiedEdit(dimensions=dims, direction=direction, **kwargs)
        edit.operations = [op.model_dump(exclude_none=True) for op in ops]
        if edit.diameter_mm is None:
            edit.diameter_mm = diameter
        if edit.radius_mm is None:
            edit.radius_mm = radius
        if edit.distance_mm is None:
            edit.distance_mm = distance
        if edit.groove_mm is None:
            edit.groove_mm = groove
        if edit.count is None:
            edit.count = count
        if edit.factor is None:
            edit.factor = factor
        # plan_status: COMPLETE when slots are filled; AMBIGUOUS for unknown type;
        # INCOMPLETE when dims/target weak.
        if "plan_status" not in kwargs:
            if edit.edit_type == EditPattern.AMBIGUOUS:
                edit.plan_status = "AMBIGUOUS"
            elif not edit.complete:
                edit.plan_status = "INCOMPLETE"
            elif edit.target_kind == "feature" and edit.edit_type == EditPattern.FILLET_CHAMFER:
                # Blend with size but vague target — still usable as COMPLETE for mutate.
                edit.plan_status = "COMPLETE"
            else:
                edit.plan_status = "COMPLETE"
        return edit

    if not ops:
        return _done(edit_type=EditPattern.AMBIGUOUS, complete=False, notes="no_strategy")

    ops.sort(key=lambda o: _PRIMARY_ORDER.index(o.type) if o.type in _PRIMARY_ORDER else 99)
    primary = ops[0]
    extra = ",".join(o.type for o in ops[1:])

    if primary.type == "SCALE":
        complete = factor is not None or (primary.factor is not None)
        return _done(
            edit_type=EditPattern.DIMENSION_CHANGE,
            target_kind="body",
            action="scale",
            factor=factor or primary.factor,
            complete=bool(complete),
            notes=f"ops=SCALE{','+extra if extra else ''}",
        )
    if primary.type == "TRANSLATE":
        return _done(
            edit_type=EditPattern.FEATURE_TRANSLATION,
            target_kind=primary.target_kind,
            action="move",
            complete=distance is not None and direction != (0.0, 0.0, 0.0),
            notes="translate_feature",
        )
    if primary.type == "HEX_PROFILE":
        return _done(
            edit_type=EditPattern.BOOLEAN_MODIFICATION,
            target_kind="hole",
            action="hex",
            complete=True,
            notes="hex_profile",
        )
    if primary.type == "GEAR_TEETH":
        return _done(
            edit_type=EditPattern.BOOLEAN_MODIFICATION,
            target_kind="profile",
            action="gear",
            complete=True,
            notes="gear_teeth",
        )
    if primary.type == "PATTERN":
        c = primary.count if primary.count is not None else count
        slot = bool(re.search(r"\bslot", lower))
        return _done(
            edit_type=EditPattern.PATTERN,
            action="pattern",
            count=c,
            complete=True,
            notes="slot_pattern" if slot else "pattern",
        )
    if primary.type == "CUT":
        return _done(edit_type=EditPattern.BOOLEAN_MODIFICATION, action="cut", complete=True, notes="boolean_cut")
    if primary.type == "DELETE":
        return _done(edit_type=EditPattern.FEATURE_DELETION, action="delete", complete=False, notes="deletion")
    if primary.type == "ADD_HOLE":
        return _done(
            edit_type=EditPattern.HOLE_EDIT,
            target_kind="hole",
            action="add",
            complete=diameter is not None,
            notes="add_hole" if diameter is not None else "add_hole_missing_diameter",
        )
    if primary.type in {"CHAMFER", "FILLET"}:
        complete = (dims.get("distance") is not None) or (radius is not None) or (distance is not None)
        return _done(
            edit_type=EditPattern.FILLET_CHAMFER,
            target_kind=primary.target_kind,
            action="chamfer" if primary.type == "CHAMFER" else "fillet",
            complete=complete,
            notes="blend",
        )
    if primary.type == "ADD_FEATURE":
        tag = _feature_tag(lower)
        return _done(
            edit_type=EditPattern.FEATURE_ADDITION,
            action="add",
            complete=True,
            notes=f"feature_add:{tag}",
        )
    return _done(edit_type=EditPattern.AMBIGUOUS, complete=False, notes="no_strategy")


def classified_to_operation(edit: ClassifiedEdit) -> OperationType:
    if edit.edit_type == EditPattern.FILLET_CHAMFER:
        return OperationType.CHAMFER if edit.action == "chamfer" else OperationType.FILLET
    return {
        EditPattern.HOLE_EDIT: OperationType.HOLE,
        EditPattern.PATTERN: OperationType.PATTERN,
        EditPattern.DIMENSION_CHANGE: OperationType.SCALE,
        EditPattern.FEATURE_TRANSLATION: OperationType.TRANSFORM,
        EditPattern.BOOLEAN_MODIFICATION: OperationType.EXTRUDE_CUT,
        EditPattern.FEATURE_ADDITION: OperationType.EXTRUDE_ADD,
        EditPattern.FEATURE_DELETION: OperationType.EXTRUDE_CUT,
        EditPattern.AMBIGUOUS: OperationType.CUSTOM,
    }[edit.edit_type]


def high_confidence_local(edit: ClassifiedEdit) -> bool:
    """True when a deterministic local tool is likely the whole edit.

    Hole chamfer / all-edge fillet with a parsed size already beat Autodesk
    on some of the 48; those should not spend a CadQuery visual loop.
    """
    if edit.edit_type == EditPattern.FILLET_CHAMFER:
        has_size = bool(edit.radius_mm or edit.distance_mm or edit.groove_mm)
        return has_size and edit.target_kind in {"hole", "hole_edge", "all_edges"}
    if edit.edit_type == EditPattern.HOLE_EDIT:
        return bool(edit.diameter_mm)
    return False


_RECONSTRUCT_HINT = re.compile(
    r"\b(handle|second|plug|europlug|collision|pin heads?|bearing|"
    r"mounting|flower|hexagon|profile|replace|redesign|design |"
    r"coffeepot|heatsink|threaded|platforms?|rib|boss|knob)\b",
    re.I,
)


def cadquery_strategy(edit: ClassifiedEdit, instruction: str = "") -> str:
    """Pick local / mutate / reconstruct from THIS instruction, never row IDs.

    local: sized all-edge fillet/chamfer or sized hole — keep the cheap OCC tool.
    mutate: import STEP and change only the named feature (typical blend/move).
    reconstruct: Autodesk-style CadQuery may add/cut/sketch/pattern and rebuild
    a local region, still starting from the imported STEP of whatever new file
    was given.
    """
    if edit.edit_type == EditPattern.FILLET_CHAMFER:
        has_size = bool(edit.radius_mm or edit.distance_mm or edit.groove_mm)
        if has_size and edit.target_kind == "all_edges":
            return "local"
        return "mutate"
    if edit.edit_type == EditPattern.HOLE_EDIT and edit.diameter_mm and not _RECONSTRUCT_HINT.search(
        instruction or ""
    ):
        return "mutate"
    if edit.edit_type in {
        EditPattern.FEATURE_ADDITION,
        EditPattern.FEATURE_DELETION,
        EditPattern.BOOLEAN_MODIFICATION,
        EditPattern.PATTERN,
        EditPattern.AMBIGUOUS,
        EditPattern.DIMENSION_CHANGE,
    }:
        return "reconstruct"
    if _RECONSTRUCT_HINT.search(instruction or ""):
        return "reconstruct"
    return "mutate"


def skip_visual_after_local(edit: ClassifiedEdit) -> bool:
    """Skip CadQuery when a sized local blend is the whole instruction."""
    return (
        edit.edit_type == EditPattern.FILLET_CHAMFER
        and edit.target_kind in {"all_edges", "hole", "hole_edge"}
        and bool(edit.radius_mm or edit.distance_mm or edit.groove_mm)
    )
