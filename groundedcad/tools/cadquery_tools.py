"""Constrained editing primitives with CadQuery or pure-Python fallback."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

from groundedcad.geometry.fallback import (
    AABB,
    SimpleSolid,
    cadquery_available,
    load_simple,
)

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}


def tool(name: str):
    def deco(fn: Callable[..., Any]):
        TOOL_REGISTRY[name] = fn
        fn.tool_name = name  # type: ignore[attr-defined]
        return fn

    return deco


def _norm_step(step_path: str) -> str:
    return str(Path(step_path).expanduser().resolve())


def _load_wp(step_path: str):
    step_path = _norm_step(step_path)
    if not cadquery_available():
        return load_simple(step_path)
    import cadquery as cq

    return cq.importers.importStep(step_path)


def _as_shape(wp):
    if isinstance(wp, SimpleSolid):
        return wp
    return wp.val() if hasattr(wp, "val") else wp


@tool("incomplete_plan")
def incomplete_plan(step_path: str, reason: str = "", **_: Any):
    """Do not invent geometry. Return the start solid unchanged."""
    return _load_wp(step_path)


@tool("inspect_model")
def inspect_model(step_path: str, **_: Any) -> dict[str, Any]:
    from groundedcad.geometry.inspect import inspect_step

    return inspect_step(step_path)


def _select_edges(shape, min_length: float, max_length: float, max_edges: int, region: str = "length"):
    bb = shape.BoundingBox()
    span = (
        max(float(bb.xlen), float(bb.ylen), float(bb.zlen), 1e-6),
        float(bb.xlen),
        float(bb.ylen),
        float(bb.zlen),
    )
    diag, xlen, ylen, zlen = span
    mx = 0.5 * (float(bb.xmin) + float(bb.xmax))
    my = 0.5 * (float(bb.ymin) + float(bb.ymax))
    mz = 0.5 * (float(bb.zmin) + float(bb.zmax))
    inset = 0.08 * diag
    scored = []
    for e in shape.Edges():
        try:
            length = float(e.Length())
        except Exception:
            continue
        if not (min_length <= length <= max_length):
            continue
        cx, cy, cz = _edge_center(e)
        scored.append((length, cx, cy, cz, e))
    region = (region or "length").lower()
    if region in {"inner", "slot"}:
        inner = [
            t
            for t in scored
            if (float(bb.xmin) + inset < t[1] < float(bb.xmax) - inset)
            or (float(bb.ymin) + inset < t[2] < float(bb.ymax) - inset)
            or (float(bb.zmin) + inset < t[3] < float(bb.zmax) - inset)
        ]
        # Slot: longer inner edges, not the outer silhouette.
        pool = inner or scored
        pool.sort(key=lambda t: t[0], reverse=True)
        return [t[4] for t in pool[: max(1, int(max_edges))]]
    if region in {"all", "all_edges"}:
        pool = sorted(scored, key=lambda t: t[0], reverse=True)
        return [t[4] for t in pool[: max(1, int(max_edges))]]
    if region in {"front", "front_center"}:
        pool = [t for t in scored if t[1] >= mx]
        pool.sort(key=lambda t: (abs(t[2] - my) + abs(t[3] - mz), -t[0]))
        return [t[4] for t in (pool or scored)[: max(1, int(max_edges))]]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [t[4] for t in scored[: max(1, int(max_edges))]]


def _blend_edges_one_by_one(shape, edges, kind: str, size: float):
    """Apply fillet/chamfer per edge so one BRep failure does not zero the solid."""
    import cadquery as cq

    current = shape
    applied = 0
    last_err = None
    for edge in edges:
        cx, cy, cz = _edge_center(edge)
        tol = max(0.05, 0.02 * max(float(edge.Length()), size, 1e-3))
        ok = False
        for scale in (1.0, 0.5, 0.25, 0.1):
            d = abs(float(size)) * scale
            if d < 1e-6:
                continue
            try:
                wp = cq.Workplane("XY").newObject([current]).edges(_near_selector(cq, cx, cy, cz, tol))
                nxt = wp.chamfer(d) if kind == "chamfer" else wp.fillet(d)
                current = nxt.val() if hasattr(nxt, "val") else nxt
                applied += 1
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if not ok:
            continue
    if applied == 0:
        raise last_err or ValueError(f"{kind} failed")
    return cq.Workplane("XY").newObject([current])


@tool("fillet_edges_by_length")
def fillet_edges_by_length(
    step_path: str,
    radius: float,
    min_length: float = 0.0,
    max_length: float = 1e9,
    max_edges: int = 20,
    region: str = "length",
    **_: Any,
):
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        solid.fillets.append(
            {"radius": float(radius), "min_length": min_length, "max_length": max_length, "max_edges": max_edges}
        )
        return solid

    shape = _as_shape(_load_wp(step_path))
    selected = _select_edges(shape, min_length, max_length, max_edges, region)
    if not selected:
        raise ValueError("No edges matched fillet filter")
    return _blend_edges_one_by_one(shape, selected, "fillet", float(radius))


@tool("chamfer_edges_by_length")
def chamfer_edges_by_length(
    step_path: str,
    distance: float,
    min_length: float = 0.0,
    max_length: float = 1e9,
    max_edges: int = 20,
    region: str = "length",
    **_: Any,
):
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        solid.chamfers.append(
            {
                "distance": float(distance),
                "min_length": min_length,
                "max_length": max_length,
                "max_edges": max_edges,
            }
        )
        return solid

    shape = _as_shape(_load_wp(step_path))
    selected = _select_edges(shape, min_length, max_length, max_edges, region)
    if not selected:
        raise ValueError("No edges matched chamfer filter")
    return _blend_edges_one_by_one(shape, selected, "chamfer", float(distance))


def _edge_center(edge) -> tuple[float, float, float]:
    bb = edge.BoundingBox()
    return (
        0.5 * (float(bb.xmin) + float(bb.xmax)),
        0.5 * (float(bb.ymin) + float(bb.ymax)),
        0.5 * (float(bb.zmin) + float(bb.zmax)),
    )


def _edge_radius(edge) -> float:
    length = max(float(edge.Length()), 1e-12)
    # Full circle: L = 2πr. Tiny bbox-z used to report ~0.02 on real ~0.75 holes.
    return length / (2.0 * math.pi)


def _is_circular(edge) -> bool:
    try:
        geom = str(edge.geomType()).upper()
    except Exception:
        return False
    return "CIRCLE" in geom or "ARC" in geom


def _pick_circular_edges(shape, min_length: float, max_length: float, max_edges: int, hole_rims: bool = False):
    circ = []
    if hole_rims:
        seen = set()
        for face in shape.Faces():
            try:
                geom = str(face.geomType()).upper()
            except Exception:
                continue
            if "CYLINDER" not in geom:
                continue
            for e in face.Edges():
                if not _is_circular(e):
                    continue
                try:
                    length = float(e.Length())
                except Exception:
                    continue
                if not (min_length <= length <= max_length):
                    continue
                cx, cy, cz = _edge_center(e)
                key = (round(cx, 3), round(cy, 3), round(cz, 3), round(length, 3))
                if key in seen:
                    continue
                seen.add(key)
                circ.append(e)
        if circ:
            circ.sort(key=lambda e: float(e.Length()))
            # Hole rims: skip decorative micro-arcs and the outer silhouette.
            if len(circ) > 4:
                lo = max(1, len(circ) // 5)
                hi = max(lo + 1, (4 * len(circ)) // 5)
                circ = circ[lo:hi] or circ
            return circ[: max(1, int(max_edges))]
    for e in shape.Edges():
        if not _is_circular(e):
            continue
        try:
            length = float(e.Length())
        except Exception:
            continue
        if min_length <= length <= max_length:
            circ.append(e)
    circ.sort(key=lambda e: float(e.Length()), reverse=True)
    if not circ:
        return []
    # Skip decorative micro-arcs unless the caller narrowed the length band.
    if min_length <= 0.0 and max_length >= 1e8:
        diag = shape.BoundingBox()
        span = max(float(diag.xlen), float(diag.ylen), float(diag.zlen), 1e-6)
        floor = max(0.08, 0.02 * span)
        longish = [e for e in circ if float(e.Length()) >= floor]
        if longish:
            circ = longish
    return circ[: max(1, int(max_edges))]


def _distance_candidates(requested: float, edges) -> list[float]:
    radii = [_edge_radius(e) for e in edges] or [0.5]
    rmin = min(radii)
    cap = 0.32 * rmin
    out: list[float] = []
    for raw in (requested, requested / 10.0, requested / 1000.0):
        d = min(abs(float(raw)), cap)
        if d >= 1e-5 and d not in out:
            out.append(d)
    if cap >= 1e-5 and cap not in out:
        out.append(cap)
    return out


def _near_selector(cq, x: float, y: float, z: float, tol: float):
    target = cq.Vector(x, y, z)

    class _Near(cq.Selector):
        def filter(self, objectList):  # noqa: N802
            picked = []
            for o in objectList:
                c = o.Center() if hasattr(o, "Center") else None
                if c is None:
                    continue
                if (cq.Vector(float(c.x), float(c.y), float(c.z)) - target).Length <= tol:
                    picked.append(o)
            return picked[:2]

    return _Near()


def _apply_circular_blend(shape, kind: str, size: float, min_length: float, max_length: float, max_edges: int, hole_rims: bool = False):
    import cadquery as cq

    picked = _pick_circular_edges(shape, min_length, max_length, max_edges, hole_rims=hole_rims)
    if not picked:
        raise ValueError("No circular edges")
    distances = _distance_candidates(size, picked)
    current = shape
    applied = 0
    last_err: Exception | None = None
    for edge in picked:
        cx, cy, cz = _edge_center(edge)
        tol = max(0.03, 0.15 * _edge_radius(edge))
        ok = False
        for d in distances:
            try:
                wp = cq.Workplane("XY").newObject([current]).edges(_near_selector(cq, cx, cy, cz, tol))
                nxt = wp.chamfer(d) if kind == "chamfer" else wp.fillet(d)
                current = nxt.val() if hasattr(nxt, "val") else nxt
                applied += 1
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if not ok:
            continue
    if applied == 0:
        raise last_err or ValueError(f"circular {kind} failed")
    return cq.Workplane("XY").newObject([current])


@tool("chamfer_circular_edges")
def chamfer_circular_edges(
    step_path: str,
    distance: float,
    min_length: float = 0.0,
    max_length: float = 1e9,
    max_edges: int = 16,
    **_: Any,
):
    """Chamfer hole/circle rims — local edit, better Chamfer/DINO vs rebuilding."""
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        solid.chamfers.append({"distance": float(distance), "circular": True, "max_edges": max_edges})
        return solid

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    return _apply_circular_blend(shape, "chamfer", float(distance), min_length, max_length, max_edges, hole_rims=True)


@tool("fillet_circular_edges")
def fillet_circular_edges(
    step_path: str,
    radius: float,
    min_length: float = 0.0,
    max_length: float = 1e9,
    max_edges: int = 16,
    **_: Any,
):
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        solid.fillets.append({"radius": float(radius), "circular": True, "max_edges": max_edges})
        return solid

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    return _apply_circular_blend(shape, "fillet", float(radius), min_length, max_length, max_edges, hole_rims=True)


@tool("drill_hole_at_point")
def drill_hole_at_point(
    step_path: str,
    x: float,
    y: float,
    z: float,
    diameter: float,
    depth: Optional[float] = None,
    axis: str = "Z",
    **_: Any,
):
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        bb = solid.bbox()
        d = depth if depth is not None else (bb.zmax - bb.zmin)
        solid.holes.append(
            {"x": float(x), "y": float(y), "z": float(z), "diameter": float(diameter), "depth": float(d), "axis": axis}
        )
        return solid

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    bb = shape.BoundingBox()
    radius = diameter / 2.0
    if axis.upper() == "Z":
        d = depth if depth is not None else (bb.zmax - bb.zmin) * 1.2
        cutter = (
            cq.Workplane("XY")
            .workplane(offset=bb.zmin - 0.1 * abs(d))
            .center(x, y)
            .circle(radius)
            .extrude(abs(d) * 1.2)
        )
    elif axis.upper() == "Y":
        d = depth if depth is not None else (bb.ymax - bb.ymin) * 1.2
        cutter = (
            cq.Workplane("XZ")
            .workplane(offset=bb.ymin - 0.1 * abs(d))
            .center(x, z)
            .circle(radius)
            .extrude(abs(d) * 1.2)
        )
    else:
        d = depth if depth is not None else (bb.xmax - bb.xmin) * 1.2
        cutter = (
            cq.Workplane("YZ")
            .workplane(offset=bb.xmin - 0.1 * abs(d))
            .center(y, z)
            .circle(radius)
            .extrude(abs(d) * 1.2)
        )
    return cq.Workplane("XY").newObject([shape]).cut(cutter)


@tool("translate_body")
def translate_body(step_path: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, **_: Any):
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        solid.bodies = [b.translated(dx, dy, dz) for b in solid.bodies]
        return solid

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    return cq.Workplane("XY").newObject([shape]).translate((dx, dy, dz))


@tool("rotate_body")
def rotate_body(step_path: str, axis: str = "Z", angle_deg: float = 90.0, **_: Any):
    if not cadquery_available():
        # Fallback: approximate rotate about Z by swapping/translating bbox (identity-ish keep volume)
        solid = load_simple(step_path).copy()
        if abs(float(angle_deg)) % 360 < 1e-6:
            return solid
        if axis.upper() == "Z" and abs(abs(float(angle_deg)) - 90) < 1e-3:
            new_bodies = []
            for b in solid.bodies:
                # 90 deg about Z: (x,y)->(-y,x)
                corners = [
                    (b.xmin, b.ymin),
                    (b.xmin, b.ymax),
                    (b.xmax, b.ymin),
                    (b.xmax, b.ymax),
                ]
                rot = [(-y, x) for x, y in corners]
                xs = [p[0] for p in rot]
                ys = [p[1] for p in rot]
                new_bodies.append(AABB(min(xs), min(ys), b.zmin, max(xs), max(ys), b.zmax))
            solid.bodies = new_bodies
        return solid

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    axis_u = axis.upper()
    origin = (0, 0, 0)
    end = (1, 0, 0) if axis_u == "X" else (0, 1, 0) if axis_u == "Y" else (0, 0, 1)
    return cq.Workplane("XY").newObject([shape]).rotate(origin, end, angle_deg)


@tool("scale_uniform")
def scale_uniform(step_path: str, factor: float, **_: Any):
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        solid.bodies = [b.scaled(float(factor)) for b in solid.bodies]
        return solid

    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Pnt, gp_Trsf

    shape = _as_shape(_load_wp(step_path))
    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(0, 0, 0), float(factor))
    transformed = BRepBuilderAPI_Transform(shape.wrapped, trsf, True).Shape()
    return cq.Shape.cast(transformed)


@tool("duplicate_linear")
def duplicate_linear(
    step_path: str,
    count: int,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    **_: Any,
):
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        base = list(solid.bodies)
        for i in range(1, int(count)):
            solid.bodies.extend([b.translated(dx * i, dy * i, dz * i) for b in base])
        return solid

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    pitch = (float(dx) ** 2 + float(dy) ** 2 + float(dz) ** 2) ** 0.5
    bb = shape.BoundingBox()
    span = max(float(bb.xlen), float(bb.ylen), float(bb.zlen), 1e-6)
    if pitch < 0.1 * span:
        raise ValueError(f"duplicate_linear pitch {pitch:.4f} is too small vs part span {span:.4f}")
    n = min(max(int(count), 1), 3)
    result = cq.Workplane("XY").newObject([shape])
    for i in range(1, n):
        moved = cq.Workplane("XY").newObject([shape]).translate((dx * i, dy * i, dz * i))
        try:
            result = result.union(moved)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"duplicate_linear union failed: {exc}") from exc
    return result


@tool("add_box")
def add_box(
    step_path: str,
    x: float,
    y: float,
    z: float,
    length: float,
    width: float,
    height: float,
    combine: str = "union",
    **_: Any,
):
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        hx, hy, hz = length / 2, width / 2, height / 2
        box = AABB(x - hx, y - hy, z - hz, x + hx, y + hy, z + hz)
        if combine == "cut":
            # Approximate cut as volume hole marker using bbox
            solid.holes.append(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "diameter": min(length, width),
                    "depth": height,
                    "axis": "Z",
                }
            )
        else:
            solid.bodies.append(box)
        return solid

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    box = cq.Workplane("XY").box(length, width, height).translate((x, y, z))
    base = cq.Workplane("XY").newObject([shape])
    if combine == "cut":
        return base.cut(box)
    return base.union(box)


@tool("boolean_cut_box")
def boolean_cut_box(
    step_path: str,
    x: float,
    y: float,
    z: float,
    length: float,
    width: float,
    height: float,
    **kwargs: Any,
):
    return add_box(
        step_path,
        x=x,
        y=y,
        z=z,
        length=length,
        width=width,
        height=height,
        combine="cut",
        **kwargs,
    )


@tool("add_cylinder")
def add_cylinder(
    step_path: str,
    x: float,
    y: float,
    z: float,
    diameter: float,
    height: float,
    axis: str = "Z",
    combine: str = "union",
    **_: Any,
):
    """Local boss / rod / button / screw — union or cut a cylinder on the imported STEP."""
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        r = float(diameter) / 2.0
        h = abs(float(height))
        solid.bodies.append(AABB(x - r, y - r, z - h / 2, x + r, y + r, z + h / 2))
        return solid

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    r = max(abs(float(diameter)) / 2.0, 1e-4)
    h = abs(float(height)) or 1e-3
    axis_u = (axis or "Z").upper()
    if axis_u == "Y":
        cyl = cq.Workplane("XZ").circle(r).extrude(h).translate((x, y - h / 2, z))
    elif axis_u == "X":
        cyl = cq.Workplane("YZ").circle(r).extrude(h).translate((x - h / 2, y, z))
    else:
        cyl = cq.Workplane("XY").circle(r).extrude(h).translate((x, y, z - h / 2))
    base = cq.Workplane("XY").newObject([shape])
    if combine == "cut":
        return base.cut(cyl)
    return base.union(cyl)


@tool("cut_through")
def cut_through(
    step_path: str,
    x: float,
    y: float,
    z: float,
    width: float,
    thickness: float,
    axis: str = "Z",
    **_: Any,
):
    """Extend a slot/cavity through the whole body along axis."""
    if not cadquery_available():
        return boolean_cut_box(step_path, x, y, z, width, thickness, abs(thickness) * 4)

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    bb = shape.BoundingBox()
    axis_u = (axis or "Z").upper()
    w = max(abs(float(width)), 1e-3)
    t = max(abs(float(thickness)), 1e-3)
    if axis_u == "Z":
        h = (float(bb.zmax) - float(bb.zmin)) * 1.4
        cz = 0.5 * (float(bb.zmin) + float(bb.zmax))
        box = cq.Workplane("XY").box(w, t, h).translate((x, y, cz))
    elif axis_u == "Y":
        h = (float(bb.ymax) - float(bb.ymin)) * 1.4
        cy = 0.5 * (float(bb.ymin) + float(bb.ymax))
        box = cq.Workplane("XY").box(w, h, t).translate((x, cy, z))
    else:
        h = (float(bb.xmax) - float(bb.xmin)) * 1.4
        cx = 0.5 * (float(bb.xmin) + float(bb.xmax))
        box = cq.Workplane("XY").box(h, w, t).translate((cx, y, z))
    return cq.Workplane("XY").newObject([shape]).cut(box)


@tool("cut_hex")
def cut_hex(
    step_path: str,
    x: float,
    y: float,
    z: float,
    radius: float,
    axis: str = "Z",
    **_: Any,
):
    """Inscribe a hexagonal through-cut at an existing hole / flower profile."""
    if not cadquery_available():
        d = abs(float(radius)) * 2
        return boolean_cut_box(step_path, x, y, z, d, d, d)

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    bb = shape.BoundingBox()
    r = max(abs(float(radius)), 1e-3)
    axis_u = (axis or "Z").upper()
    if axis_u == "Y":
        h = (float(bb.ymax) - float(bb.ymin)) * 1.4
        cutter = cq.Workplane("XZ").center(x, z).polygon(6, 2 * r).extrude(h).translate((0, float(bb.ymin) - 0.2 * h, 0))
    elif axis_u == "X":
        h = (float(bb.xmax) - float(bb.xmin)) * 1.4
        cutter = cq.Workplane("YZ").center(y, z).polygon(6, 2 * r).extrude(h).translate((float(bb.xmin) - 0.2 * h, 0, 0))
    else:
        h = (float(bb.zmax) - float(bb.zmin)) * 1.4
        cutter = (
            cq.Workplane("XY")
            .workplane(offset=float(bb.zmin) - 0.2 * h)
            .center(x, y)
            .polygon(6, 2 * r)
            .extrude(h * 1.4)
        )
    return cq.Workplane("XY").newObject([shape]).cut(cutter)


@tool("cut_radial_notches")
def cut_radial_notches(
    step_path: str,
    count: int = 16,
    depth: Optional[float] = None,
    **_: Any,
):
    """Approximate spur-gear teeth: local radial notches on the outer profile."""
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        return solid

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    bb = shape.BoundingBox()
    cx = 0.5 * (float(bb.xmin) + float(bb.xmax))
    cy = 0.5 * (float(bb.ymin) + float(bb.ymax))
    cz = 0.5 * (float(bb.zmin) + float(bb.zmax))
    r_out = 0.5 * min(float(bb.xlen), float(bb.ylen))
    n = max(6, min(int(count or 16), 48))
    d = float(depth) if depth and depth > 0 else max(0.04 * r_out, 0.3)
    w = max(0.35 * (2 * math.pi * r_out / n), 0.2)
    h = (float(bb.zmax) - float(bb.zmin)) * 1.3
    current = cq.Workplane("XY").newObject([shape])
    for i in range(n):
        ang = 360.0 * i / n
        rad = math.radians(ang)
        px = cx + (r_out - d / 2) * math.cos(rad)
        py = cy + (r_out - d / 2) * math.sin(rad)
        cutter = cq.Workplane("XY").box(w, d, h).rotate((0, 0, 0), (0, 0, 1), ang).translate((px, py, cz))
        try:
            current = current.cut(cutter)
        except Exception:
            continue
    return current


@tool("cut_slot_pattern")
def cut_slot_pattern(
    step_path: str,
    count: int = 6,
    both_sides: bool = True,
    **_: Any,
):
    """Cut a row of lightening slots on the top (and optionally bottom) face."""
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        return solid

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))
    bb = shape.BoundingBox()
    n = max(2, min(int(count or 6), 16))
    xlen, ylen, zlen = float(bb.xlen), float(bb.ylen), float(bb.zlen)
    along_x = xlen >= ylen
    span = xlen if along_x else ylen
    slot_len = 0.55 * (ylen if along_x else xlen)
    slot_w = max(0.04 * span, 0.4)
    depth = max(0.12 * zlen, 0.3)
    margin = 0.15 * span
    pitch = (span - 2 * margin) / max(n - 1, 1)
    cx = 0.5 * (float(bb.xmin) + float(bb.xmax))
    cy = 0.5 * (float(bb.ymin) + float(bb.ymax))
    zs = [float(bb.zmax) - depth / 2]
    if both_sides:
        zs.append(float(bb.zmin) + depth / 2)
    current = cq.Workplane("XY").newObject([shape])
    for zi in zs:
        for i in range(n):
            off = -0.5 * (n - 1) * pitch + i * pitch
            if along_x:
                box = cq.Workplane("XY").box(slot_w, slot_len, depth).translate((cx + off, cy, zi))
            else:
                box = cq.Workplane("XY").box(slot_len, slot_w, depth).translate((cx, cy + off, zi))
            try:
                current = current.cut(box)
            except Exception:
                continue
    return current


def tool_specs_for_prompt() -> str:
    specs = {
        "inspect_model": {"args": ["step_path"]},
        "chamfer_circular_edges": {
            "args": ["step_path", "distance", "min_length", "max_length", "max_edges"]
        },
        "fillet_circular_edges": {
            "args": ["step_path", "radius", "min_length", "max_length", "max_edges"]
        },
        "fillet_edges_by_length": {
            "args": ["step_path", "radius", "min_length", "max_length", "max_edges"]
        },
        "chamfer_edges_by_length": {
            "args": ["step_path", "distance", "min_length", "max_length", "max_edges"]
        },
        "drill_hole_at_point": {
            "args": ["step_path", "x", "y", "z", "diameter", "depth?", "axis?"]
        },
        "translate_body": {"args": ["step_path", "dx", "dy", "dz"]},
        "rotate_body": {"args": ["step_path", "axis", "angle_deg"]},
        "scale_uniform": {"args": ["step_path", "factor"]},
        "duplicate_linear": {"args": ["step_path", "count", "dx", "dy", "dz"]},
        "add_box": {"args": ["step_path", "x", "y", "z", "length", "width", "height", "combine"]},
        "add_cylinder": {"args": ["step_path", "x", "y", "z", "diameter", "height", "axis", "combine"]},
        "boolean_cut_box": {"args": ["step_path", "x", "y", "z", "length", "width", "height"]},
        "cut_through": {"args": ["step_path", "x", "y", "z", "width", "thickness", "axis"]},
        "cut_hex": {"args": ["step_path", "x", "y", "z", "radius", "axis"]},
        "cut_radial_notches": {"args": ["step_path", "count", "depth"]},
        "cut_slot_pattern": {"args": ["step_path", "count", "both_sides"]},
        "raw_cadquery": {
            "args": ["my_cad_function string"],
            "note": "Fallback only when CadQuery is available.",
        },
    }
    return json.dumps(specs, indent=2)


def execute_tool(tool_name: str, arguments: dict[str, Any]):
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {tool_name}. Available: {sorted(TOOL_REGISTRY)}")
    args = dict(arguments)
    size = None
    for key in ("distance", "radius", "chamfer_length", "fillet_radius", "size", "value", "value_mm", "d"):
        if args.get(key) is not None:
            size = args[key]
            break
    if size is not None:
        if "chamfer" in tool_name:
            args.setdefault("distance", size)
        if "fillet" in tool_name:
            args.setdefault("radius", size)
    return TOOL_REGISTRY[tool_name](**args)


RAW_FUNCTION_TEMPLATE = '''
def my_cad_function(args):
    import cadquery as cq
    import os
    input_file = os.path.expanduser(args["input_file"])
    shape = cq.importers.importStep(input_file)
    # TODO: edit shape
    return shape
'''.strip()
