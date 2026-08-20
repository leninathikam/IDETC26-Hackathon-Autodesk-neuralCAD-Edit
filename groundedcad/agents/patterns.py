"""Per-type CAD strategies. Classifier chooses the type; this file does not prompt an LLM."""

from __future__ import annotations

from typing import Any

from groundedcad.agents.classifier import ClassifiedEdit, classified_to_operation, classify_instruction
from groundedcad.agents.schemas import EditPattern, GroundedIntent, OperationType, ToolCall


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


def _size(census: dict[str, Any] | None) -> tuple[float, float, float]:
    s = (census or {}).get("size") or (10.0, 10.0, 10.0)
    return (float(s[0]), float(s[1]), float(s[2]))


def _bbox(census: dict[str, Any] | None) -> dict[str, float]:
    return dict((census or {}).get("bbox") or {})


def _existing_holes(census: dict[str, Any] | None) -> list[tuple[tuple[float, float, float], float]]:
    out: list[tuple[tuple[float, float, float], float]] = []
    for h in (census or {}).get("hole_candidates") or (census or {}).get("holes") or []:
        center = h.get("center")
        diameter = h.get("diameter") or ((h.get("radius") or 0) * 2.0)
        if center and diameter:
            out.append(((float(center[0]), float(center[1]), float(center[2])), float(diameter)))
    return out


def _existing_hole(census: dict[str, Any] | None) -> tuple[tuple[float, float, float], float] | None:
    holes = _existing_holes(census)
    if holes:
        holes.sort(key=lambda h: -h[1])
        return holes[0]
    for face in (census or {}).get("faces") or []:
        geom = str((face.get("metadata") or {}).get("geom") or "")
        if "CYLINDER" not in geom.upper():
            continue
        radius = face.get("radius")
        center = face.get("center")
        if center and radius and float(radius) > 0.2:
            return ((float(center[0]), float(center[1]), float(center[2])), float(radius) * 2.0)
    return None


def _interior_cavity(census: dict[str, Any] | None) -> tuple[tuple[float, float, float], float] | None:
    """Cavity inset from the bbox — not a mounting hole on the rim."""
    bb = _bbox(census)
    if not bb:
        return _existing_hole(census)
    cx = 0.5 * (float(bb.get("xmin", 0)) + float(bb.get("xmax", 0)))
    cy = 0.5 * (float(bb.get("ymin", 0)) + float(bb.get("ymax", 0)))
    sx, sy, _sz = _size(census)
    best = None
    best_d = 1e18
    for (x, y, z), d in _existing_holes(census):
        if abs(x - float(bb.get("xmin", 0))) < 0.08 * sx or abs(x - float(bb.get("xmax", 0))) < 0.08 * sx:
            continue
        if abs(y - float(bb.get("ymin", 0))) < 0.08 * sy or abs(y - float(bb.get("ymax", 0))) < 0.08 * sy:
            continue
        dist = (x - cx) ** 2 + (y - cy) ** 2
        if dist < best_d:
            best_d = dist
            best = ((x, y, z), d)
    return best or _existing_hole(census)


def _named_sites(census: dict[str, Any] | None, text: str, n: int = 1) -> list[tuple[float, float, float]]:
    bb = _bbox(census)
    cx, cy, cz = _bbox_center(census)
    if not bb:
        return [(cx, cy, cz)] * max(1, n)
    xmin, xmax = float(bb["xmin"]), float(bb["xmax"])
    ymin, ymax = float(bb["ymin"]), float(bb["ymax"])
    zmin, zmax = float(bb["zmin"]), float(bb["zmax"])
    mx, my, mz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
    lower = text.lower()
    sites: list[tuple[float, float, float]] = []

    def add(pt: tuple[float, float, float]) -> None:
        if pt not in sites:
            sites.append(pt)

    if "top" in lower and "right" in lower:
        add((xmax, my, zmax))
    if "bottom" in lower and "left" in lower:
        add((xmin, my, zmin))
    if "top" in lower and "left" in lower:
        add((xmin, my, zmax))
    if "bottom" in lower and "right" in lower:
        add((xmax, my, zmin))
    if "top" in lower and not sites:
        add((mx, my, zmax))
    if "bottom" in lower and not any(abs(s[2] - zmin) < 1e-6 for s in sites):
        add((mx, my, zmin))
    if "front" in lower:
        add((xmax, my, mz))
    if not sites:
        add((cx, cy, cz))
    while len(sites) < n:
        sites.append(sites[-1])
    return sites[:n]


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
    axis = str((census or {}).get("shortest_axis") or "Z").upper()
    if axis not in {"X", "Y", "Z"}:
        axis = "Z"
    tool = ToolCall(
        tool_name="drill_hole_at_point",
        arguments={"step_path": step_path, "x": pt[0], "y": pt[1], "z": pt[2], "diameter": float(diameter), "axis": axis},
        rationale="Strategy hole_add: parsed diameter on planar face, not an existing cylinder wall",
    )
    if edit.groove_mm:
        tool.followups.append(
            ToolCall(
        tool_name="chamfer_circular_edges",
                arguments={
                    "step_path": step_path,
                    "distance": float(edit.groove_mm),
                    "max_edges": 4,
                    "hole_diameters": [float(diameter)],
                },
                rationale="Strategy hole_add: grooves as local chamfer on hole rims",
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
    return ToolCall(
        tool_name="translate_body",
        arguments={"step_path": step_path, "dx": dx, "dy": dy, "dz": dz},
        rationale="Strategy translate_body: named move/prolong",
    )


def strategy_fillet_chamfer(edit: ClassifiedEdit, step_path: str, census: dict[str, Any] | None = None) -> ToolCall:
    region = {
        "slot": "slot",
        "all_edges": "all",
        "front_center": "front_center",
        "hole_edge": "length",
        "hole": "length",
    }.get(edit.target_kind, "length")
    if edit.action == "chamfer":
        dist = float(edit.distance_mm or edit.groove_mm or 0.0)
        if dist <= 0:
            return _incomplete("chamfer requires parsed distance")
        if edit.target_kind in {"hole_edge", "hole"}:
            from groundedcad.geometry.inspect import fitting_hole_diameter, rank_hole_rims

            ranked = rank_hole_rims(census or {}, blend_mm=dist)
            fit = float(ranked[0]["diameter"]) if ranked else fitting_hole_diameter(census or {}, blend_mm=dist)
            holes = _existing_holes(census)
            diameters = [fit] if fit else [d for _c, d in holes]
            args: dict[str, Any] = {
                "step_path": step_path,
                "distance": dist,
                "max_edges": 4,
                "hole_diameters": diameters[:4],
            }
            if ranked and ranked[0].get("centers"):
                args["rim_centers"] = ranked[0]["centers"]
            return ToolCall(
                tool_name="chamfer_circular_edges",
                arguments=args,
                rationale="Strategy blend: fitting-hole rims only",
            )
        max_edges = 48 if region == "all" else (6 if region in {"slot", "front_center"} else 12)
        return ToolCall(
            tool_name="chamfer_edges_by_length",
            arguments={"step_path": step_path, "distance": dist, "max_edges": max_edges, "region": region},
            rationale=f"Strategy blend: chamfer region={region}",
        )
    radius = float(edit.radius_mm or edit.distance_mm or 0.0)
    if radius <= 0:
        return _incomplete("fillet requires parsed radius")
    if edit.target_kind in {"hole_edge", "hole"}:
        from groundedcad.geometry.inspect import fitting_hole_diameter, rank_hole_rims

        ranked = rank_hole_rims(census or {}, blend_mm=radius)
        fit = float(ranked[0]["diameter"]) if ranked else fitting_hole_diameter(census or {}, blend_mm=radius)
        holes = _existing_holes(census)
        diameters = [fit] if fit else [d for _c, d in holes]
        args = {
            "step_path": step_path,
            "radius": radius,
            "max_edges": min(8, max(2, 2 * max(1, len({round(d, 2) for d in diameters})))),
            "hole_diameters": diameters[:8],
        }
        if ranked and ranked[0].get("centers"):
            args["rim_centers"] = ranked[0]["centers"]
        return ToolCall(
            tool_name="fillet_circular_edges",
            arguments=args,
            rationale="Strategy blend: fitting-hole rims only",
        )
    max_edges = 64 if region == "all" else (8 if region in {"slot", "front_center"} else 16)
    return ToolCall(
        tool_name="fillet_edges_by_length",
        arguments={"step_path": step_path, "radius": radius, "max_edges": max_edges, "region": region},
        rationale=f"Strategy blend: fillet region={region}",
    )


def strategy_pattern(edit: ClassifiedEdit, step_path: str, census: dict[str, Any] | None = None) -> ToolCall:
    if "slot" in (edit.notes or ""):
        n = int(edit.count or 6)
        return ToolCall(
            tool_name="cut_slot_pattern",
            arguments={"step_path": step_path, "count": n, "both_sides": True},
            rationale="Strategy pattern: local slot cuts, not whole-body copies",
        )
    sx, sy, sz = _size(census)
    longest = max(sx, sy, sz, 1.0)
    n_solids = int((census or {}).get("n_solids") or 1)
    pitch = float(edit.distance_mm or 0.0)
    # Leftover sizes (e.g. 0.42 mm thickness) are not a pattern pitch.
    # Whole-body duplicate of a multi-solid assembly hangs OCC on union.
    if n_solids > 1 or pitch < 0.15 * longest:
        cx, cy, cz = _bbox_center(census)
        return ToolCall(
            tool_name="add_box",
            arguments={
                "step_path": step_path,
                "x": cx + 0.55 * longest,
                "y": cy,
                "z": cz,
                "length": max(0.4 * sx, 4.0),
                "width": max(0.12 * sy, 2.0),
                "height": max(sz, 2.0),
                "combine": "union",
            },
            rationale="Strategy pattern: extra local instance; do not duplicate the whole assembly",
        )
    n = min(int(edit.count or 2), 3)
    pitch = max(pitch, 0.35 * longest)
    return ToolCall(
        tool_name="duplicate_linear",
        arguments={"step_path": step_path, "count": n, "dx": pitch, "dy": 0.0, "dz": 0.0},
        rationale="Strategy pattern: linear copies with a non-overlapping pitch",
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
    notes = edit.notes or ""
    if "hex_profile" in notes or edit.action == "hex":
        hole = _existing_hole(census) or (_bbox_center(census), min(_size(census)) * 0.15)
        (x, y, z), diameter = hole
        axis = str((census or {}).get("shortest_axis") or "Z").upper()
        return ToolCall(
            tool_name="cut_hex",
            arguments={
                "step_path": step_path,
                "x": x,
                "y": y,
                "z": z,
                "radius": 0.5 * float(diameter),
                "axis": axis if axis in {"X", "Y", "Z"} else "Z",
            },
            rationale="Strategy hex: inscribe hex in the center hole/flower profile",
        )
    if "gear_teeth" in notes or edit.action == "gear":
        return ToolCall(
            tool_name="cut_radial_notches",
            arguments={"step_path": step_path, "count": int(edit.count or 16)},
            rationale="Strategy gear: radial notches on the outer profile",
        )
    found = _interior_cavity(census)
    sx, sy, sz = _size(census)
    axis = str((census or {}).get("shortest_axis") or "Z").upper()
    if axis not in {"X", "Y", "Z"}:
        axis = "Z"
    if found:
        (cx, cy, cz), diameter = found
        w = max(float(diameter) * 0.9, min(sx, sy, sz) * 0.08)
    else:
        cx, cy, cz = _planar_center(census) or _bbox_center(census)
        w = min(sx, sy, sz) * 0.12
    t = max(w * 0.35, min(sx, sy, sz) * 0.04)
    return ToolCall(
        tool_name="cut_through",
        arguments={"step_path": step_path, "x": cx, "y": cy, "z": cz, "width": w, "thickness": t, "axis": axis},
        rationale="Strategy boolean: through-cut on interior cavity, not a rim hole",
    )


def strategy_feature_add(
    edit: ClassifiedEdit, step_path: str, census: dict[str, Any] | None, text: str = ""
) -> ToolCall:
    tag = (edit.notes or "").split(":")[-1] if "feature_add" in (edit.notes or "") else "feature"
    sx, sy, sz = _size(census)
    smallest = min(sx, sy, sz)
    longest = max(sx, sy, sz)
    n = int(edit.count or 1)
    h = float(edit.distance_mm or edit.radius_mm or 0.0)

    if tag == "rib":
        thick = h if h > 0 else max(1.5, 0.03 * smallest)
        cx, cy, cz = _planar_center(census) or _bbox_center(census)
        return ToolCall(
            tool_name="add_box",
            arguments={
                "step_path": step_path,
                "x": cx,
                "y": cy,
                "z": cz,
                "length": longest * 0.45,
                "width": thick,
                "height": smallest * 0.25,
                "combine": "union",
            },
            rationale="Strategy rib: thin local wall, thickness from instruction",
        )
    if tag == "rod":
        length = h if h > 0 else 200.0
        dia = float(edit.diameter_mm or max(0.06 * smallest, 4.0))
        site = _named_sites(census, "top", 1)[0]
        return ToolCall(
            tool_name="add_cylinder",
            arguments={
                "step_path": step_path,
                "x": site[0],
                "y": site[1],
                "z": site[2] + length / 2,
                "diameter": dia,
                "height": length,
                "axis": "Z",
                "combine": "union",
            },
            rationale="Strategy rod: cylinder from the top face",
        )
    if tag == "screw":
        holes = _existing_holes(census)[: max(n, 2)]
        if not holes:
            pt = _bbox_center(census)
            holes = [(pt, max(4.0, 0.08 * smallest))]
        (x, y, z), d = holes[0]
        tool = ToolCall(
            tool_name="add_cylinder",
            arguments={
                "step_path": step_path,
                "x": x,
                "y": y,
                "z": z,
                "diameter": max(d * 0.85, 1.0),
                "height": max(d * 2.5, 6.0),
                "axis": "Z",
                "combine": "union",
            },
            rationale="Strategy screw: fastener in existing hole",
        )
        for (x2, y2, z2), d2 in holes[1:]:
            tool.followups.append(
                ToolCall(
                    tool_name="add_cylinder",
                    arguments={
                        "step_path": step_path,
                        "x": x2,
                        "y": y2,
                        "z": z2,
                        "diameter": max(d2 * 0.85, 1.0),
                        "height": max(d2 * 2.5, 6.0),
                        "axis": "Z",
                        "combine": "union",
                    },
                    rationale="Strategy screw: second fastener",
                )
            )
        return tool
    if tag == "port":
        sites = _named_sites(census, "top right bottom left", 2)
        dia = float(edit.diameter_mm or max(0.08 * smallest, 3.0))
        height = max(0.12 * smallest, dia)
        tool = ToolCall(
            tool_name="add_cylinder",
            arguments={
                "step_path": step_path,
                "x": sites[0][0],
                "y": sites[0][1],
                "z": sites[0][2],
                "diameter": dia,
                "height": height,
                "axis": "Z",
                "combine": "union",
            },
            rationale="Strategy port: outlet at named corner",
        )
        tool.followups.append(
            ToolCall(
                tool_name="add_cylinder",
                arguments={
                    "step_path": step_path,
                    "x": sites[1][0],
                    "y": sites[1][1],
                    "z": sites[1][2],
                    "diameter": dia,
                    "height": height,
                    "axis": "Z",
                    "combine": "union",
                },
                rationale="Strategy port: inlet at opposite corner",
            )
        )
        return tool
    if tag == "button":
        n = max(n, 2)
        height = h if h > 0 else 2.0
        dia = max(0.06 * smallest, height * 2)
        site = _named_sites(census, "top", 1)[0]
        sx_off = smallest * 0.12
        tool = None
        for i in range(n):
            x, y, z = site
            x = x + (i - (n - 1) / 2) * sx_off
            call = ToolCall(
                tool_name="add_cylinder",
                arguments={
                    "step_path": step_path,
                    "x": x,
                    "y": y,
                    "z": z + height / 2,
                    "diameter": dia,
                    "height": height,
                    "axis": "Z",
                    "combine": "union",
                },
                rationale="Strategy button: local click pad",
            )
            if tool is None:
                tool = call
            else:
                tool.followups.append(call)
        return tool  # type: ignore[return-value]
    if tag == "switch":
        site = _named_sites(census, "bottom", 1)[0]
        return ToolCall(
            tool_name="add_box",
            arguments={
                "step_path": step_path,
                "x": site[0],
                "y": site[1],
                "z": site[2],
                "length": smallest * 0.18,
                "width": smallest * 0.08,
                "height": max(h, smallest * 0.04, 1.5),
                "combine": "union",
            },
            rationale="Strategy switch: small slider on the bottom face",
        )
    if tag == "cap":
        site = _named_sites(census, "top", 1)[0]
        dia = float(edit.diameter_mm or max(0.12 * smallest, 6.0))
        height = max(h, 0.1 * smallest, 4.0)
        return ToolCall(
            tool_name="add_cylinder",
            arguments={
                "step_path": step_path,
                "x": site[0],
                "y": site[1],
                "z": site[2] + height / 2,
                "diameter": dia,
                "height": height,
                "axis": "Z",
                "combine": "union",
            },
            rationale="Strategy cap: pouring boss on the top face",
        )
    # `tag` is the coarse classification string ("feature") and never contains
    # a direction word, so passing it here always fell through to the bbox
    # center. Use the real instruction text so "top/bottom/left/right/front"
    # cues actually move the anchor; only fall back to the tag when the
    # instruction itself doesn't name a side.
    cx, cy, cz = _named_sites(census, text or tag, 1)[0]
    # Prefer local cylinder/hole when diameter is known — edit volume, not rebuild.
    if edit.diameter_mm and edit.diameter_mm > 0:
        dia = float(edit.diameter_mm)
        height = h if h > 0 else max(0.12 * smallest, dia)
        lower = (text or "").lower()
        if any(w in lower for w in ("hole", "bore", "drill", "cutout", "pocket")):
            return ToolCall(
                tool_name="drill_hole_at_point",
                arguments={
                    "step_path": step_path,
                    "x": cx,
                    "y": cy,
                    "z": cz,
                    "diameter": dia,
                    "axis": "Z",
                },
                rationale="Strategy feature_add: local hole from parsed diameter on planar face",
            )
        return ToolCall(
            tool_name="add_cylinder",
            arguments={
                "step_path": step_path,
                "x": cx,
                "y": cy,
                "z": cz + height / 2,
                "diameter": dia,
                "height": height,
                "axis": "Z",
                "combine": "union",
            },
            rationale="Strategy feature_add: local cylinder from parsed diameter",
        )
    hh = h if h > 0 else max(0.06 * smallest, 2.0)
    return ToolCall(
        tool_name="add_box",
        arguments={
            "step_path": step_path,
            "x": cx,
            "y": cy,
            "z": cz + hh / 2,
            "length": smallest * 0.2,
            "width": smallest * 0.12,
            "height": hh,
            "combine": "union",
        },
        rationale="Strategy feature_add: inferred local solid from bbox",
    )


def apply_classified(
    edit: ClassifiedEdit,
    step_path: str,
    census: dict[str, Any] | None = None,
    text: str = "",
) -> ToolCall:
    if edit.edit_type == EditPattern.FILLET_CHAMFER:
        return strategy_fillet_chamfer(edit, step_path, census)
    if edit.edit_type == EditPattern.HOLE_EDIT:
        return strategy_hole_edit(edit, step_path, census)
    if edit.edit_type == EditPattern.PATTERN:
        return strategy_pattern(edit, step_path, census)
    if edit.edit_type == EditPattern.DIMENSION_CHANGE:
        tool = strategy_dimension(edit, step_path)
        extra = (edit.notes or "").upper()
        if tool.tool_name != "incomplete_plan" and "FILLET" in extra:
            r = float(edit.radius_mm or 0.0)
            if r > 0:
                tool.followups.append(
                    ToolCall(
                        tool_name="fillet_edges_by_length",
                        arguments={"step_path": step_path, "radius": r, "max_edges": 16, "region": "all"},
                        rationale="Follow-up FILLET after SCALE",
                    )
                )
        return tool
    if edit.edit_type == EditPattern.FEATURE_TRANSLATION:
        return strategy_translate(edit, step_path, census)
    if edit.edit_type == EditPattern.BOOLEAN_MODIFICATION:
        return strategy_boolean(edit, step_path, census)
    if edit.edit_type == EditPattern.FEATURE_ADDITION:
        return strategy_feature_add(edit, step_path, census, text=text)
    if edit.edit_type == EditPattern.FEATURE_DELETION:
        return strategy_boolean(edit, step_path, census)
    if edit.edit_type == EditPattern.AMBIGUOUS:
        return strategy_feature_add(edit, step_path, census, text=text)
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
    return apply_classified(edit, step_path, census, text=text)


def pattern_to_operation(pattern: EditPattern) -> OperationType:
    dummy = ClassifiedEdit(edit_type=pattern)
    return classified_to_operation(dummy)
