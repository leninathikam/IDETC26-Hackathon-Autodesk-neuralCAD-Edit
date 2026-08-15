"""Intent -> classified slots -> strategy tool call."""

from __future__ import annotations

import json
from typing import Any, Optional

from groundedcad.agents.classifier import classified_to_operation, classify_instruction
from groundedcad.agents.patterns import apply_classified
from groundedcad.agents.schemas import EditSpec, GroundedIntent, OperationType, SuccessCheck, ToolCall
from groundedcad.llm.base import LLMClient

PLANNER_SYSTEM = (
    "Fill missing numeric slots only (diameter_mm, distance_mm, direction). "
    "Do not write CadQuery. Do not choose raw_cadquery."
)


def _default_checks(op: OperationType) -> list[SuccessCheck]:
    checks = [
        SuccessCheck(name="valid_brep", description="Valid B-Rep", check_type="valid_brep"),
        SuccessCheck(
            name="connectivity",
            description="Not fragmented",
            check_type="connectivity",
            params={"max_bodies": 25},
        ),
        SuccessCheck(
            name="bbox_delta",
            description="Bounded bbox growth",
            check_type="bbox_delta",
            params={"max_growth": 2.5},
        ),
    ]
    if op in {OperationType.HOLE, OperationType.EXTRUDE_CUT, OperationType.CHAMFER, OperationType.FILLET}:
        checks.append(
            SuccessCheck(
                name="volume_delta",
                description="Local edit: small volume change (helps Chamfer/DINO)",
                check_type="volume_delta",
                params={"max_ratio": 0.25},
            )
        )
    elif op in {OperationType.EXTRUDE_ADD, OperationType.DUPLICATE, OperationType.PATTERN}:
        checks.append(
            SuccessCheck(
                name="volume_delta",
                description="Expect volume increase",
                check_type="volume_delta",
                params={"max_ratio": 3.0, "expect_increase": True},
            )
        )
    else:
        checks.append(
            SuccessCheck(
                name="volume_delta",
                description="Bounded volume change",
                check_type="volume_delta",
                params={"max_ratio": 0.9},
            )
        )
    return checks


def _census_center(census: dict[str, Any] | None) -> tuple[float, float, float]:
    census = census or {}
    best = None
    best_area = -1.0
    for face in census.get("faces") or []:
        radius = face.get("radius")
        area = float(face.get("area") or 0.0)
        center = face.get("center")
        if radius and center and area >= best_area:
            best_area = area
            best = tuple(float(x) for x in center)
    if best:
        return best  # type: ignore[return-value]
    c = census.get("center")
    if c and len(c) == 3:
        return (float(c[0]), float(c[1]), float(c[2]))
    return (0.0, 0.0, 0.0)


def _inferred_hole_diameter(dims: dict[str, float], census: dict[str, Any] | None) -> float | None:
    if dims.get("diameter"):
        return float(dims["diameter"])
    if dims.get("radius"):
        return 2.0 * float(dims["radius"])
    # Infer from largest existing cylinder only when enlarging; for add-hole leave None.
    return None


def heuristic_needs_llm(intent: GroundedIntent) -> bool:
    edit = classify_instruction(intent.raw_instruction or intent.summary)
    return (not edit.complete) and edit.edit_type.value == "ambiguous"


def heuristic_plan(intent: GroundedIntent, step_path: str, census: dict[str, Any] | None = None) -> tuple[EditSpec, ToolCall]:
    edit = classify_instruction(intent.raw_instruction or intent.summary)
    dims = {**edit.to_dimensions(), **(intent.dimensions or {})}
    if dims.get("diameter") is not None:
        edit.diameter_mm = float(dims["diameter"])
    if dims.get("radius") is not None:
        edit.radius_mm = float(dims["radius"])
    if dims.get("distance") is not None:
        edit.distance_mm = float(dims["distance"])
    if dims.get("groove") is not None:
        edit.groove_mm = float(dims["groove"])
    if dims.get("factor") is not None:
        edit.factor = float(dims["factor"])
    tool = apply_classified(edit, step_path, census)
    op = classified_to_operation(edit)
    spec = EditSpec(
        intent_summary=intent.summary,
        operation=op,
        targets=intent.targets,
        parameters=tool.arguments if tool.tool_name != "raw_cadquery" else intent.dimensions,
        invariants=intent.constraints + intent.preserve,
        success_checks=_default_checks(op),
        prefer_local=True,
        notes=f"edit_type={edit.edit_type.value} target={edit.target_kind} action={edit.action} complete={edit.complete}",
        fallback_to_raw_script=False,
    )
    return spec, tool
def llm_plan(
    client: LLMClient,
    intent: GroundedIntent,
    step_path: str,
    census: dict[str, Any],
    revision_advice: str = "",
) -> tuple[EditSpec, ToolCall]:
    seed_spec, seed_tool = heuristic_plan(intent, step_path, census)
    edit = classify_instruction(intent.raw_instruction or intent.summary)
    from groundedcad.geometry.inspect import geometry_brief

    payload = {
        "instruction": intent.raw_instruction or intent.summary,
        "slots": edit.model_dump(),
        "model": geometry_brief(census),
        "failed": (revision_advice or "")[:400],
    }
    try:
        data = client.complete_json(system=PLANNER_SYSTEM, user=json.dumps(payload, indent=2))
    except Exception:
        return seed_spec, seed_tool
    if data.get("diameter_mm") is not None:
        edit.diameter_mm = float(data["diameter_mm"])
    if data.get("distance_mm") is not None:
        edit.distance_mm = float(data["distance_mm"])
    if data.get("direction") and len(data["direction"]) == 3:
        edit.direction = tuple(float(x) for x in data["direction"])  # type: ignore[assignment]
    tool = apply_classified(edit, step_path, census)
    spec = seed_spec.model_copy(
        update={"parameters": tool.arguments, "notes": f"slot_fill {edit.notes}"}
    )
    return spec, tool


def plan_edit(
    intent: GroundedIntent,
    step_path: str,
    census: dict[str, Any],
    client: Optional[LLMClient] = None,
    revision_advice: str = "",
    use_llm: bool = False,
) -> tuple[EditSpec, ToolCall]:
    if (not use_llm) or client is None:
        return heuristic_plan(intent, step_path, census)
    return llm_plan(client, intent, step_path, census, revision_advice=revision_advice)
