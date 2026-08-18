"""Lightweight EditPlan: what should change, before CadQuery decides how."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from groundedcad.agents.classifier import ClassifiedEdit, cadquery_strategy


@dataclass
class EditPlan:
    classified: ClassifiedEdit
    target_candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_candidates: list[dict[str, Any]] = field(default_factory=list)
    expected_delta: dict[str, str] = field(default_factory=dict)
    locality_envelope: Optional[dict[str, Any]] = None
    plan_status: str = "COMPLETE"
    fallback_strategy: str = "mutate"
    confidence: float = 0.5


def expected_delta_for(edit: ClassifiedEdit) -> dict[str, str]:
    kind = edit.edit_type.value
    if kind == "fillet_chamfer":
        return {
            "volume": "decrease_small",
            "bbox": "unchanged",
            "faces": "increase",
            "locality": "target_rim",
        }
    if kind == "hole_edit":
        return {
            "volume": "decrease",
            "bbox": "unchanged",
            "cylindrical_faces": "+1",
            "locality": "target_face",
        }
    if kind == "feature_addition":
        return {
            "volume": "increase",
            "bbox": "possibly_increase",
            "faces": "increase",
            "locality": "target_face",
        }
    if kind == "boolean_modification":
        return {
            "volume": "decrease",
            "bbox": "unchanged_or_shrink",
            "faces": "change",
            "locality": "target_region",
        }
    return {"volume": "change", "bbox": "unknown", "faces": "change"}


def locality_envelope_for(
    edit: ClassifiedEdit,
    candidates: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not candidates:
        return None
    top = candidates[0]
    centers = top.get("centers") or []
    if not centers and top.get("center"):
        centers = [tuple(top["center"])]
    if not centers:
        return None
    r = float(top.get("radius") or (float(top.get("diameter") or 0) / 2.0) or 1.0)
    # Envelope: center ± 2r around each rim center (union bbox).
    xs = [float(c[0]) for c in centers]
    ys = [float(c[1]) for c in centers]
    zs = [float(c[2]) for c in centers]
    pad = 2.0 * r
    return {
        "xmin": min(xs) - pad,
        "xmax": max(xs) + pad,
        "ymin": min(ys) - pad,
        "ymax": max(ys) + pad,
        "zmin": min(zs) - pad,
        "zmax": max(zs) + pad,
        "radius": r,
        "kind": edit.target_kind,
    }


def build_edit_plan(
    edit: ClassifiedEdit,
    census: dict[str, Any] | None = None,
    instruction: str = "",
) -> EditPlan:
    census = census or {}
    candidates: list[dict[str, Any]] = []
    if edit.edit_type.value == "fillet_chamfer" and edit.target_kind in {"hole", "hole_edge"}:
        from groundedcad.geometry.inspect import rank_hole_rims

        blend = float(edit.distance_mm or edit.radius_mm or edit.groove_mm or 0.0) or None
        candidates = rank_hole_rims(census, blend_mm=blend)
    elif edit.edit_type.value == "hole_edit":
        candidates = list(census.get("hole_candidates") or [])[:4]

    status = edit.plan_status or ("COMPLETE" if edit.complete else "INCOMPLETE")
    if edit.edit_type.value == "ambiguous":
        status = "AMBIGUOUS"
    conf = 0.9 if status == "COMPLETE" and candidates else (0.6 if status == "COMPLETE" else 0.3)
    return EditPlan(
        classified=edit,
        target_candidates=candidates,
        selected_candidates=candidates[:1],
        expected_delta=expected_delta_for(edit),
        locality_envelope=locality_envelope_for(edit, candidates),
        plan_status=status,
        fallback_strategy=cadquery_strategy(edit, instruction),
        confidence=conf,
    )


def point_in_envelope(point: tuple[float, float, float], envelope: dict[str, Any]) -> bool:
    x, y, z = point
    return (
        float(envelope["xmin"]) <= x <= float(envelope["xmax"])
        and float(envelope["ymin"]) <= y <= float(envelope["ymax"])
        and float(envelope["zmin"]) <= z <= float(envelope["zmax"])
    )
