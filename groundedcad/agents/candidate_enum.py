"""Deterministic, census-grounded ToolCall candidate enumeration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from groundedcad.agents.classifier import ClassifiedEdit
from groundedcad.agents.schemas import ToolCall


@dataclass
class EnumeratedCandidate:
    tool: ToolCall
    anchor_id: str
    census_source: str
    safe: bool = True


def _dedupe_points(points) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for point in points:
        if not point or len(point) != 3:
            continue
        p = tuple(float(x) for x in point)
        key = tuple(round(x, 2) for x in p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _full_rim_families(
    census: dict[str, Any], *, blend_mm: float
) -> list[dict[str, Any]]:
    """Group near-full circular edges by diameter; partial arcs are not rims."""
    floor = max(0.8, 6.0 * abs(float(blend_mm)))
    groups: dict[float, dict[str, Any]] = {}
    for edge in census.get("circular_edges") or []:
        radius = float(edge.get("radius") or 0.0)
        length = float(edge.get("length") or 0.0)
        center = edge.get("center")
        if radius <= 0 or length <= 0 or not center:
            continue
        if length / (2.0 * math.pi * radius) < 0.85:
            continue
        diameter = round(2.0 * radius, 2)
        if diameter < floor:
            continue
        group = groups.setdefault(
            diameter,
            {"diameter": diameter, "centers": [], "n_rims": 0},
        )
        point = tuple(round(float(x), 3) for x in center)
        if point not in group["centers"]:
            group["centers"].append(point)
            group["n_rims"] = len(group["centers"])
    return sorted(
        groups.values(),
        key=lambda group: (-int(group["n_rims"]), float(group["diameter"])),
    )


def _incomplete(reason: str) -> EnumeratedCandidate:
    return EnumeratedCandidate(
        tool=ToolCall(
            tool_name="incomplete_plan",
            arguments={"reason": reason},
            rationale=f"candidate:incomplete:{reason}",
        ),
        anchor_id="incomplete",
        census_source="classified_slots",
        safe=True,
    )


def enumerate_tool_candidates(
    edit: ClassifiedEdit,
    step_path: str,
    census: dict[str, Any] | None,
    text: str,
    max_candidates: int = 4,
) -> list[EnumeratedCandidate]:
    """Return distinct local candidates, all rooted at ``step_path``."""
    from groundedcad.agents.patterns import strategy_feature_add, strategy_pattern
    from groundedcad.geometry.inspect import (
        bbox_corners,
        cavity_candidates,
        protrusion_candidates,
        top_planar_centers,
    )

    census = census or {}
    limit = max(1, int(max_candidates))
    kind = edit.edit_type.value
    out: list[EnumeratedCandidate] = []

    if kind == "fillet_chamfer":
        size = float(edit.distance_mm or edit.radius_mm or edit.groove_mm or 0.0)
        if size <= 0:
            return [_incomplete("blend requires parsed size")]
        if edit.target_kind in {"hole", "hole_edge"}:
            families = _full_rim_families(census, blend_mm=size)
            for index, family in enumerate(families[:limit]):
                is_chamfer = edit.action == "chamfer"
                args: dict[str, Any] = {
                    "step_path": step_path,
                    "max_edges": min(8, max(2, int(family["n_rims"]))),
                    "hole_diameters": [float(family["diameter"])],
                    "rim_centers": family["centers"],
                }
                args["distance" if is_chamfer else "radius"] = size
                out.append(
                    EnumeratedCandidate(
                        tool=ToolCall(
                            tool_name=(
                                "chamfer_circular_edges"
                                if is_chamfer
                                else "fillet_circular_edges"
                            ),
                            arguments=args,
                            rationale=(
                                f"candidate:full_rim_family_{index}:"
                                f"d={family['diameter']}"
                            ),
                        ),
                        anchor_id=f"rim:d={family['diameter']}",
                        census_source="circular_edges_near_full",
                        safe=True,
                    )
                )
            return out or [_incomplete("no near-full circular rim family")]

        is_chamfer = edit.action == "chamfer"
        requested = {
            "all_edges": ["all", "length", "front_center", "slot"],
            "slot": ["slot", "length", "front_center", "all"],
            "front_center": ["front_center", "length", "slot", "all"],
        }.get(edit.target_kind, ["length", "front_center", "slot", "all"])
        for region in requested[:limit]:
            out.append(
                EnumeratedCandidate(
                    tool=ToolCall(
                        tool_name=(
                            "chamfer_edges_by_length"
                            if is_chamfer
                            else "fillet_edges_by_length"
                        ),
                        arguments={
                            "step_path": step_path,
                            ("distance" if is_chamfer else "radius"): size,
                            "max_edges": 48 if region == "all" else 8,
                            "region": region,
                        },
                        rationale=f"candidate:blend_region:{region}",
                    ),
                    anchor_id=f"region:{region}",
                    census_source="classified_target",
                    safe=region != "all",
                )
            )
        return out

    if kind == "hole_edit":
        diameter = float(edit.diameter_mm or 0.0)
        if diameter <= 0:
            return [_incomplete("hole edit requires parsed diameter")]
        axis = str(census.get("shortest_axis") or "Z").upper()
        points = []
        for hole in census.get("hole_candidates") or []:
            if hole.get("center"):
                points.append(hole["center"])
        points.extend(top_planar_centers(census, limit))
        for index, (x, y, z) in enumerate(_dedupe_points(points)[:limit]):
            out.append(
                EnumeratedCandidate(
                    tool=ToolCall(
                        tool_name="drill_hole_at_point",
                        arguments={
                            "step_path": step_path,
                            "x": x,
                            "y": y,
                            "z": z,
                            "diameter": diameter,
                            "axis": axis,
                        },
                        rationale=f"candidate:hole_site_{index}",
                    ),
                    anchor_id=f"hole:{x:.2f},{y:.2f},{z:.2f}",
                    census_source="hole_candidates_or_planar_faces",
                    safe=True,
                )
            )
        return out or [_incomplete("no census-backed hole site")]

    if kind == "feature_addition":
        tagged = (edit.notes or "").split(":")[-1]
        recognized = tagged in {"rib", "rod", "screw", "port", "button", "switch", "cap"}
        diameter = float(edit.diameter_mm or 0.0)
        if diameter <= 0 and not recognized:
            return [_incomplete("undimensioned generic feature addition")]
        if recognized:
            return [
                EnumeratedCandidate(
                    tool=strategy_feature_add(edit, step_path, census, text=text),
                    anchor_id=f"tag:{tagged}",
                    census_source="feature_tag",
                    safe=False,
                )
            ]
        size = census.get("size") or (10.0, 10.0, 10.0)
        height = float(edit.distance_mm or max(0.12 * min(float(x) for x in size), diameter))
        points = [p["center"] for p in protrusion_candidates(census, limit) if p.get("center")]
        points.extend(top_planar_centers(census, limit))
        points.extend(bbox_corners(census, limit))
        for index, (x, y, z) in enumerate(_dedupe_points(points)[:limit]):
            out.append(
                EnumeratedCandidate(
                    tool=ToolCall(
                        tool_name="add_cylinder",
                        arguments={
                            "step_path": step_path,
                            "x": x,
                            "y": y,
                            "z": z + height / 2.0,
                            "diameter": diameter,
                            "height": height,
                            "axis": "Z",
                            "combine": "union",
                        },
                        rationale=f"candidate:addition_anchor_{index}",
                    ),
                    anchor_id=f"add:{x:.2f},{y:.2f},{z:.2f}",
                    census_source="protrusions_planar_corners",
                    safe=True,
                )
            )
        return out or [_incomplete("no census-backed feature anchor")]

    if kind in {"boolean_modification", "feature_deletion"}:
        size = tuple(float(x) for x in (census.get("size") or (10.0, 10.0, 10.0)))
        shortest = max(min(size), 1e-3)
        axis = str(census.get("shortest_axis") or "Z").upper()
        for index, cavity in enumerate(cavity_candidates(census, limit)[:limit]):
            center = cavity.get("center")
            if not center:
                continue
            diameter = float(cavity.get("diameter") or 0.0)
            width = max(0.9 * diameter, 0.08 * shortest)
            thickness = max(0.35 * width, 0.04 * shortest)
            x, y, z = (float(v) for v in center)
            out.append(
                EnumeratedCandidate(
                    tool=ToolCall(
                        tool_name="cut_through",
                        arguments={
                            "step_path": step_path,
                            "x": x,
                            "y": y,
                            "z": z,
                            "width": width,
                            "thickness": thickness,
                            "axis": axis,
                        },
                        rationale=f"candidate:cavity_cut_{index}",
                    ),
                    anchor_id=f"cavity:{x:.2f},{y:.2f},{z:.2f}",
                    census_source="cavity_candidates",
                    safe=False,
                )
            )
        return out or [_incomplete("no census-backed cavity for local deletion")]

    if kind == "pattern":
        lower = (text or "").lower()
        if "mirror" in lower:
            return [_incomplete("no mirror primitive; use hybrid fallback")]
        if "slot" in lower or "slot" in (edit.notes or ""):
            return [
                EnumeratedCandidate(
                    tool=strategy_pattern(edit, step_path, census),
                    anchor_id="pattern:slot",
                    census_source="bbox_pattern",
                    safe=True,
                )
            ]
        if edit.distance_mm and edit.count:
            return [
                EnumeratedCandidate(
                    tool=strategy_pattern(edit, step_path, census),
                    anchor_id="pattern:parsed_pitch",
                    census_source="parsed_slots",
                    safe=True,
                )
            ]
        return [_incomplete("pattern lacks parsed pitch or supported primitive")]

    return []
