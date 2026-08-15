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


@tool("fillet_edges_by_length")
def fillet_edges_by_length(
    step_path: str,
    radius: float,
    min_length: float = 0.0,
    max_length: float = 1e9,
    max_edges: int = 20,
    **_: Any,
):
    if not cadquery_available():
        solid = load_simple(step_path).copy()
        solid.fillets.append(
            {"radius": float(radius), "min_length": min_length, "max_length": max_length, "max_edges": max_edges}
        )
        return solid

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))

    class _LenSel(cq.Selector):
        def filter(self, objectList):  # noqa: N802
            out = [o for o in objectList if min_length <= float(o.Length()) <= max_length]
            return out[:max_edges]

    selected = _LenSel().filter(list(shape.Edges()))
    if not selected:
        raise ValueError("No edges matched fillet length filter")
    last_err = None
    for n in (min(max_edges, len(selected)), 4, 2, 1):
        for scale in (1.0, 0.5, 0.25):
            try:
                class _NSel(cq.Selector):
                    def filter(self, objectList, _n=n):  # noqa: N802
                        return [o for o in objectList if min_length <= float(o.Length()) <= max_length][:_n]

                return cq.Workplane("XY").newObject([shape]).edges(_NSel()).fillet(float(radius) * scale)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
    raise last_err or ValueError("fillet failed")


@tool("chamfer_edges_by_length")
def chamfer_edges_by_length(
    step_path: str,
    distance: float,
    min_length: float = 0.0,
    max_length: float = 1e9,
    max_edges: int = 20,
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

    import cadquery as cq

    shape = _as_shape(_load_wp(step_path))

    class _LenSel(cq.Selector):
        def filter(self, objectList):  # noqa: N802
            out = [o for o in objectList if min_length <= float(o.Length()) <= max_length]
            return out[:max_edges]

    selected = _LenSel().filter(list(shape.Edges()))
    if not selected:
        raise ValueError("No edges matched chamfer length filter")
    last_err = None
    for n in (min(max_edges, len(selected)), 4, 2, 1):
        for scale in (1.0, 0.5, 0.25):
            try:
                class _NSel(cq.Selector):
                    def filter(self, objectList, _n=n):  # noqa: N802
                        return [o for o in objectList if min_length <= float(o.Length()) <= max_length][:_n]

                return cq.Workplane("XY").newObject([shape]).edges(_NSel()).chamfer(float(distance) * scale)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
    raise last_err or ValueError("chamfer failed")


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
        circ.sort(key=lambda e: float(e.Length()), reverse=True)
        if circ:
            return circ[: max(1, int(max_edges))]
        hole_rims = False
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
    result = cq.Workplane("XY").newObject([shape])
    for i in range(1, int(count)):
        moved = cq.Workplane("XY").newObject([shape]).translate((dx * i, dy * i, dz * i))
        result = result.union(moved)
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
        "boolean_cut_box": {"args": ["step_path", "x", "y", "z", "length", "width", "height"]},
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
