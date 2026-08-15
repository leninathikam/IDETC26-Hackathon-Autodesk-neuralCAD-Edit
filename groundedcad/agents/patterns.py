"""Per-type CAD strategies. Classifier chooses the type; this file does not prompt an LLM."""

from __future__ import annotations

from typing import Any

from groundedcad.agents.classifier import ClassifiedEdit, classified_to_operation, classify_instruction
from groundedcad.agents.schemas import EditPattern, EditSpec, GroundedIntent, OperationType, ToolCall
def classify_edit(text: str) -> tuple[EditPattern, list[EditPattern]]:
    edit = classify_instruction(text)
    return edit.edit_type, [edit.edit_type]


def _planar_center(census: dict[str, Any] | None) -> tuple[float, float, float] | None:
    census = census or {}
    best = None
    best_area = -1.0
    for face in census.get("faces") or []:
        geom = str((face.get("metadata") or {}).get("geom") or "")
        if "PLANE" not in geom.upper():
            continue
        area = float(face.get("area") or 0.0)
        center = face.get("center")
        if center and area >= best_area:
            best_area = area
            best = (float(center[0]), float(center[1]), float(center[2]))
    return best


def _bbox_center(census: dict[str, Any] | None) -> tuple[float, float, float]:
    c = (census or {}).get("center")
    if c and len(c) == 3:
        return (float(c[0]), float(c[1]), float(c[2]))
    return (0.0, 0.0, 0.0)


def _existing_hole(census: dict[str, Any] | None) -> tuple[tuple[float, float, float], float] | None:
    holes = (census or {}).get("holes") or []
    best = None
    best_r = -1.0
    for h in holes:
        center = h.get("center")
        radius = h.get("radius") or ((h.get("diameter") or 0) / 2.0)
        if center and radius and float(radius) > best_r:
            best_r = float(radius)
            best = ((float(center[0]), float(center[1]), float(center[2])), float(radius) * 2.0)
    if best:
        return best
    for face in (census or {}).get("faces") or []:
        geom = str((face.get("metadata") or {}).get("geom") or "")
        if "CYLINDER" not in geom.upper():
            continue
        radius = face.get("radius")
        center = face.get("center")
        if center and radius and float(radius) > 0.2:
            return ((float(center[0]), float(center[1]), float(center[2])), float(radius) * 2.0)
    return None


def _incomplete(reason: str) -> ToolCall:
    return ToolCall(
        tool_name="incomplete_plan",
        arguments={"reason": reason},
        rationale=f"incomplete:{reason}",
    )


def strategy_hole_edit(edit: ClassifiedEdit, step_path: str, census: dict[str, Any] | None) -> ToolCall:
    if edit.action == "move":
        return strategy_translate(edit, step_path, census)
    diameter = edit.diameter_mm
    if diameter is None:
        return _incomplete("hole_add requires parsed diameter; will not invent Ø4")
    pt = _planar_center(census) or _bbox_center(census)
    tool = ToolCall(
        tool_name="drill_hole_at_point",
        arguments={"step_path": step_path, "x": pt[0], "y": pt[1], "z": pt[2], "diameter": float(diameter), "axis": "Z"},
        rationale="Strategy hole_add: parsed diameter on planar face, not an existing cylinder wall",
    )
    if edit.groove_mm:
        tool.followups.append(
            ToolCall(
                tool_name="chamfer_circular_edges",
                arguments={"step_path": step_path, "distance": float(edit.groove_mm), "max_edges": 8},
                rationale="Strategy hole_add: grooves as local chamfer",
            )
        )
    return tool


def strategy_translate(edit: ClassifiedEdit, step_path: str, census: dict[str, Any] | None) -> ToolCall:
    dist = float(edit.distance_mm or 0.0)
    dx, dy, dz = (d * dist for d in edit.direction)
    if edit.target_kind == "hole":
        found = _existing_hole(census)
        if not found or dist == 0.0:
            return _incomplete("translate_hole needs an existing hole and a parsed distance+direction")
        (x, y, z), diameter = found
        return ToolCall(
            tool_name="drill_hole_at_point",
            arguments={
                "step_path": step_path,
                "x": x + dx,
                "y": y + dy,
                "z": z + dz,
                "diameter": diameter,
                "axis": "Z",
            },
            rationale="Strategy translate_feature: re-cut hole at offset; do not translate the whole body",
        )
    if dist == 0.0:
        return _incomplete("translate needs a parsed distance")
    if edit.target_kind == "body":
        return ToolCall(
            tool_name="translate_body",
            arguments={"step_path": step_path, "dx": dx, "dy": dy, "dz": dz},
            rationale="Strategy translate_body: instruction named the body/part",
        )
    # No isolated sub-feature to move independently (no cadquery sub-feature
    # selector here); approximate a named-feature move/prolong as a whole-body
    # translate in the parsed direction rather than emitting a guaranteed no-op.
    return ToolCall(
        tool_name="translate_body",
        arguments={"step_path": step_path, "dx": dx, "dy": dy, "dz": dz},
        rationale="Strategy translate_body: no isolated sub-feature found; approximate named-feature move as whole-body translate",
    )


def strategy_fillet_chamfer(edit: ClassifiedEdit, step_path: str) -> ToolCall:
    if edit.action == "chamfer":
        dist = float(edit.distance_mm or edit.groove_mm or 0.0)
        if dist <= 0:
            return _incomplete("chamfer requires parsed distance")
        name = "chamfer_circular_edges" if edit.target_kind in {"hole_edge", "hole"} else "chamfer_edges_by_length"
        args = {"step_path": step_path, "distance": dist, "max_edges": 8 if edit.target_kind == "slot" else 12}
        return ToolCall(tool_name=name, arguments=args, rationale="Strategy blend: hole rims if named, else length-band edges")
    radius = float(edit.radius_mm or edit.distance_mm or 0.0)
    if radius <= 0:
        return _incomplete("fillet requires parsed radius")
    name = "fillet_circular_edges" if edit.target_kind in {"hole_edge", "hole"} else "fillet_edges_by_length"
    return ToolCall(
        tool_name=name,
        arguments={"step_path": step_path, "radius": radius, "max_edges": 8 if edit.target_kind == "slot" else 16},
        rationale="Strategy blend: named feature first, then local fillet",
    )


def strategy_pattern(edit: ClassifiedEdit, step_path: str) -> ToolCall:
    if edit.count is None:
        return _incomplete("pattern requires parsed instance count")
    pitch = float(edit.distance_mm or 10.0)
    return ToolCall(
        tool_name="duplicate_linear",
        arguments={"step_path": step_path, "count": int(edit.count), "dx": pitch, "dy": 0.0, "dz": 0.0},
        rationale="Strategy pattern: linear copies of the existing body",
    )


def strategy_dimension(edit: ClassifiedEdit, step_path: str) -> ToolCall:
    if edit.factor and abs(edit.factor - 1.0) > 1e-6:
        return ToolCall(
            tool_name="scale_uniform",
            arguments={"step_path": step_path, "factor": float(edit.factor)},
            rationale="Strategy dimension: uniform scale only when factor is explicit",
        )
    return _incomplete("height/length change is not a uniform scale of the whole part")


def strategy_boolean(edit: ClassifiedEdit, step_path: str, census: dict[str, Any] | None) -> ToolCall:
    size = (census or {}).get("size") or (10, 10, 10)
    hole = _existing_hole(census)
    if hole:
        (cx, cy, cz), _diameter = hole
    else:
        cx, cy, cz = _planar_center(census) or _bbox_center(census)
    L = float(edit.distance_mm or min(size) * 0.25)
    return ToolCall(
        tool_name="boolean_cut_box",
        arguments={"step_path": step_path, "x": cx, "y": cy, "z": cz, "length": L, "width": L * 0.4, "height": max(size) * 1.1},
        rationale="Strategy boolean: one local cut box, centered on an existing feature if found",
    )


def strategy_feature_add(edit: ClassifiedEdit, step_path: str, census: dict[str, Any] | None) -> ToolCall:
    cx, cy, cz = _planar_center(census) or _bbox_center(census)
    size = (census or {}).get("size") or (10, 10, 10)
    h = float(edit.distance_mm or edit.radius_mm or 0.0)
    if h <= 0:
        return _incomplete("feature_add requires a parsed size")
    return ToolCall(
        tool_name="add_box",
        arguments={
            "step_path": step_path,
            "x": cx,
            "y": cy,
            "z": cz + h / 2,
            "length": min(size) * 0.3,
            "width": min(size) * 0.15,
            "height": h,
            "combine": "union",
        },
        rationale="Strategy feature_add: one local solid, no raw scaffold",
    )


def apply_classified(
    edit: ClassifiedEdit,
    step_path: str,
    census: dict[str, Any] | None = None,
) -> ToolCall:
    if edit.edit_type == EditPattern.FILLET_CHAMFER:
        return strategy_fillet_chamfer(edit, step_path)
    if edit.edit_type == EditPattern.HOLE_EDIT:
        return strategy_hole_edit(edit, step_path, census)
    if edit.edit_type == EditPattern.PATTERN:
        return strategy_pattern(edit, step_path)
    if edit.edit_type == EditPattern.DIMENSION_CHANGE:
        tool = strategy_dimension(edit, step_path)
        extra = (edit.notes or "").upper()
        if tool.tool_name != "incomplete_plan" and "FILLET" in extra:
            r = float(edit.radius_mm or 0.0)
            if r > 0:
                tool.followups.append(
                    ToolCall(
                        tool_name="fillet_edges_by_length",
                        arguments={"step_path": step_path, "radius": r, "max_edges": 16},
                        rationale="Follow-up FILLET after SCALE",
                    )
                )
        return tool
    if edit.edit_type == EditPattern.FEATURE_TRANSLATION:
        return strategy_translate(edit, step_path, census)
    if edit.edit_type == EditPattern.BOOLEAN_MODIFICATION:
        return strategy_boolean(edit, step_path, census)
    if edit.edit_type == EditPattern.FEATURE_ADDITION:
        return strategy_feature_add(edit, step_path, census)
    if edit.edit_type == EditPattern.FEATURE_DELETION:
        return strategy_boolean(edit, step_path, census)
    return _incomplete("no strategy for this edit type")


def apply_strategy(
    pattern: EditPattern,
    intent: GroundedIntent,
    step_path: str,
    census: dict[str, Any] | None = None,
) -> ToolCall:
    text = intent.raw_instruction or intent.summary
    edit = classify_instruction(text)
    dims = {**edit.to_dimensions(), **(intent.dimensions or {})}
    if dims.get("diameter") is not None:
        edit.diameter_mm = float(dims["diameter"])
    if dims.get("radius") is not None:
        edit.radius_mm = float(dims["radius"])
    if dims.get("distance") is not None:
        edit.distance_mm = float(dims["distance"])
    if dims.get("groove") is not None:
        edit.groove_mm = float(dims["groove"])
    edit.edit_type = pattern
    return apply_classified(edit, step_path, census)


def pattern_to_operation(pattern: EditPattern) -> OperationType:
    dummy = ClassifiedEdit(edit_type=pattern)
    return classified_to_operation(dummy)
