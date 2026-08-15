"""STEP topology census and entity descriptors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from groundedcad.agents.schemas import BoundingBox, EntityKind, EntityRef
from groundedcad.geometry.fallback import cadquery_available, inspect_simple, load_simple


def _safe_import_cq():
    import cadquery as cq

    return cq


def _bbox_from_shape(shape) -> BoundingBox:
    bb = shape.BoundingBox()
    return BoundingBox(
        xmin=float(bb.xmin),
        ymin=float(bb.ymin),
        zmin=float(bb.zmin),
        xmax=float(bb.xmax),
        ymax=float(bb.ymax),
        zmax=float(bb.zmax),
    )


def _vec_tuple(v) -> tuple[float, float, float]:
    return (float(v.x), float(v.y), float(v.z))


def _round_tuple(t: tuple[float, float, float], nd: int = 4) -> tuple[float, float, float]:
    return tuple(round(x, nd) for x in t)  # type: ignore[return-value]


def load_step(step_path: str | Path):
    if not cadquery_available():
        return load_simple(step_path)
    cq = _safe_import_cq()
    path = Path(step_path)
    if not path.exists():
        raise FileNotFoundError(f"STEP file not found: {path}")
    return cq.importers.importStep(str(path))


def shape_from_workplane(wp):
    from groundedcad.geometry.fallback import SimpleSolid

    if isinstance(wp, SimpleSolid):
        return wp
    if hasattr(wp, "val"):
        return wp.val()
    return wp


def inspect_step(step_path: str | Path, max_faces: int = 80, max_edges: int = 120) -> dict[str, Any]:
    """Return a compact JSON-serializable geometry census."""
    if not cadquery_available():
        solid = load_simple(step_path)
        census = inspect_simple(solid)
        census["source"] = str(step_path)
        return census
    wp = load_step(step_path)
    shape = shape_from_workplane(wp)
    return inspect_shape(shape, max_faces=max_faces, max_edges=max_edges, source=str(step_path))


def inspect_shape(
    shape,
    max_faces: int = 80,
    max_edges: int = 120,
    source: str = "",
) -> dict[str, Any]:
    from groundedcad.geometry.fallback import SimpleSolid

    if isinstance(shape, SimpleSolid):
        census = inspect_simple(shape)
        census["source"] = source or census.get("source", "")
        return census

    solids = list(shape.Solids()) if hasattr(shape, "Solids") else []
    faces = list(shape.Faces()) if hasattr(shape, "Faces") else []
    edges = list(shape.Edges()) if hasattr(shape, "Edges") else []
    vertices = list(shape.Vertices()) if hasattr(shape, "Vertices") else []

    try:
        volume = float(shape.Volume())
        ocp_valid = bool(shape.isValid()) if hasattr(shape, "isValid") else True
        # OCCT isValid() is often false after chamfer/fillet even when the solid is usable.
        valid = bool(ocp_valid or (volume > 1e-9 and len(solids) >= 1))
    except Exception as exc:  # noqa: BLE001
        volume = 0.0
        valid = False
        validity_error = str(exc)
    else:
        validity_error = None

    bbox = _bbox_from_shape(shape)

    body_refs: list[EntityRef] = []
    for i, solid in enumerate(solids[:40]):
        sb = _bbox_from_shape(solid)
        try:
            vol = float(solid.Volume())
        except Exception:  # noqa: BLE001
            vol = 0.0
        body_refs.append(
            EntityRef(
                entity_id=f"body_{i}",
                kind=EntityKind.BODY,
                description=f"Solid body {i}",
                confidence=1.0,
                center=sb.center,
                metadata={"volume": vol, "bbox": sb.model_dump()},
            )
        )

    face_refs: list[EntityRef] = []
    for i, face in enumerate(faces[:max_faces]):
        fb = _bbox_from_shape(face)
        try:
            area = float(face.Area()) if hasattr(face, "Area") else None
        except Exception:
            area = None
        normal = None
        radius = None
        geom = "unknown"
        try:
            geom = str(face.geomType())
        except Exception:  # noqa: BLE001
            pass
        try:
            center = face.Center()
            if hasattr(face, "normalAt"):
                n = face.normalAt(center)
                normal = _round_tuple(_vec_tuple(n))
        except Exception:  # noqa: BLE001
            center = fb.center
        else:
            center = _round_tuple(_vec_tuple(center))

        if "CYLINDER" in geom.upper() or "CIRCLE" in geom.upper():
            sx, sy, sz = fb.size
            dims = sorted(x for x in (sx, sy, sz) if x > 1e-6)
            radius = 0.5 * dims[0] if len(dims) >= 2 else (0.5 * dims[-1] if dims else None)

        face_refs.append(
            EntityRef(
                entity_id=f"face_{i}",
                kind=EntityKind.FACE,
                description=f"{geom} face area={area:.3f}" if area is not None else f"{geom} face",
                confidence=1.0,
                center=center if isinstance(center, tuple) else None,
                area=area,
                radius=radius,
                normal=normal,
                metadata={"geom": geom, "bbox": fb.model_dump()},
            )
        )

    edge_refs: list[EntityRef] = []
    for i, edge in enumerate(edges[:max_edges]):
        eb = _bbox_from_shape(edge)
        length = float(edge.Length()) if hasattr(edge, "Length") else None
        geom = "unknown"
        radius = None
        try:
            geom = str(edge.geomType())
        except Exception:  # noqa: BLE001
            pass
        if "CIRCLE" in geom.upper() or "ARC" in geom.upper():
            if length and length > 1e-9:
                radius = length / (2.0 * math.pi)
            else:
                sx, sy, sz = eb.size
                dims = sorted(x for x in (sx, sy, sz) if x > 1e-6)
                radius = 0.5 * dims[-1] if dims else None
        edge_refs.append(
            EntityRef(
                entity_id=f"edge_{i}",
                kind=EntityKind.EDGE,
                description=f"{geom} edge len={length:.3f}" if length is not None else f"{geom} edge",
                confidence=1.0,
                center=eb.center,
                length=length,
                radius=radius,
                metadata={"geom": geom, "bbox": eb.model_dump()},
            )
        )

    sx, sy, sz = bbox.size
    axes = sorted([("x", sx), ("y", sy), ("z", sz)], key=lambda t: t[1])
    symmetry_hints = []
    if abs(sx - sy) / max(sx, sy, 1e-6) < 0.05:
        symmetry_hints.append("approx_square_xy")
    if abs(sx - sz) / max(sx, sz, 1e-6) < 0.05:
        symmetry_hints.append("approx_square_xz")
    if abs(sy - sz) / max(sy, sz, 1e-6) < 0.05:
        symmetry_hints.append("approx_square_yz")

    return {
        "source": source,
        "valid": valid,
        "validity_error": validity_error,
        "volume": volume,
        "bbox": bbox.model_dump(),
        "center": bbox.center,
        "size": bbox.size,
        "n_solids": len(solids),
        "n_faces": len(faces),
        "n_edges": len(edges),
        "n_vertices": len(vertices),
        "bodies": [b.model_dump() for b in body_refs],
        "faces": [f.model_dump() for f in face_refs],
        "edges": [e.model_dump() for e in edge_refs],
        "holes": [
            e.model_dump()
            for e in edge_refs
            if e.radius and e.length and e.length > 0.05
        ][:24],
        "planar_faces": [f.model_dump() for f in face_refs if "PLANE" in str((f.metadata or {}).get("geom", "")).upper()],
        "cylindrical_faces": [f.model_dump() for f in face_refs if "CYLINDER" in str((f.metadata or {}).get("geom", "")).upper()],
        "circular_edges": [e.model_dump() for e in edge_refs if e.radius],
        "hole_candidates": _hole_candidates(face_refs, edge_refs, bbox),
        "symmetry_hints": symmetry_hints,
        "longest_axis": axes[-1][0] if axes else None,
        "shortest_axis": axes[0][0] if axes else None,
        "backend": "cadquery",
    }


def _axis_from_size(size: tuple[float, float, float]) -> str:
    names = ("X", "Y", "Z")
    return names[min(range(3), key=lambda i: size[i])]


def _hole_candidates(face_refs: list[EntityRef], edge_refs: list[EntityRef], bbox: BoundingBox) -> list[dict[str, Any]]:
    """Cluster circular edges / cylinders into hole facts OCC can compute."""
    groups: dict[tuple, dict[str, Any]] = {}
    for e in edge_refs:
        if not e.radius or not e.center or e.radius < 0.15:
            continue
        key = (round(e.center[0], 1), round(e.center[1], 1), round(e.radius, 2))
        g = groups.setdefault(
            key,
            {
                "type": "hole",
                "diameter": round(2.0 * float(e.radius), 3),
                "center": (round(e.center[0], 3), round(e.center[1], 3), round(e.center[2], 3)),
                "axis": "Z",
                "depth": "blind",
                "n_circles": 0,
            },
        )
        g["n_circles"] += 1
        if g["n_circles"] >= 2:
            g["depth"] = "through"
    for f in face_refs:
        geom = str((f.metadata or {}).get("geom", "")).upper()
        if "CYLINDER" not in geom or not f.radius or f.radius < 0.15 or not f.center:
            continue
        bb = (f.metadata or {}).get("bbox") or {}
        size = (
            float(bb.get("xmax", 0) - bb.get("xmin", 0)),
            float(bb.get("ymax", 0) - bb.get("ymin", 0)),
            float(bb.get("zmax", 0) - bb.get("zmin", 0)),
        )
        axis = _axis_from_size(size)
        key = (round(f.center[0], 1), round(f.center[1], 1), round(f.radius, 2))
        g = groups.setdefault(
            key,
            {
                "type": "hole",
                "diameter": round(2.0 * float(f.radius), 3),
                "center": (round(f.center[0], 3), round(f.center[1], 3), round(f.center[2], 3)),
                "axis": axis,
                "depth": "unknown",
                "n_circles": 0,
            },
        )
        g["axis"] = axis
        if f.normal:
            g["face_normal"] = f.normal
    holes = sorted(groups.values(), key=lambda h: -float(h["diameter"]))[:16]
    return holes


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def cavity_candidates(census: dict[str, Any], max_n: int = 3) -> list[dict[str, Any]]:
    """Existing hole/cavity openings from the census — geometry only, no text
    heuristics. Blind holes are surfaced first: they are the most plausible
    "extend this through the body" targets for a cut-through instruction."""
    census = census or {}
    holes = list(census.get("hole_candidates") or [])
    holes.sort(key=lambda h: (h.get("depth") != "blind", -(h.get("diameter") or 0)))
    out = [
        {"center": h.get("center"), "diameter": h.get("diameter"), "depth": h.get("depth")}
        for h in holes[:max_n]
        if h.get("center")
    ]
    if out:
        return out
    for f in (census.get("cylindrical_faces") or [])[:max_n]:
        if f.get("center") and f.get("radius"):
            out.append({"center": f["center"], "diameter": round(2.0 * float(f["radius"]), 3), "depth": "unknown"})
    return out[:max_n]


def top_planar_centers(census: dict[str, Any], max_n: int = 4) -> list[tuple[float, float, float]]:
    """A few distinct, largest planar-face centers — candidate sites for
    separate local cuts (e.g. "several cutouts"), deduped by rounded position."""
    census = census or {}
    faces = sorted(
        (f for f in (census.get("planar_faces") or []) if f.get("center")),
        key=lambda f: -(f.get("area") or 0.0),
    )
    out: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for f in faces:
        c = tuple(round(float(x), 1) for x in f["center"])
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= max_n:
            break
    return out


def protrusion_candidates(census: dict[str, Any], max_n: int = 3) -> list[dict[str, Any]]:
    """Small-radius cylindrical faces (pins/bosses) — candidate anchors for a
    new small feature (e.g. a "locking mechanism" near a "hook")."""
    census = census or {}
    cyls = sorted(
        (f for f in (census.get("cylindrical_faces") or []) if f.get("radius") and f.get("center")),
        key=lambda f: float(f["radius"]),
    )
    return [
        {"center": tuple(round(float(x), 3) for x in f["center"]), "radius": round(float(f["radius"]), 3)}
        for f in cyls[:max_n]
    ]


def bbox_corners(census: dict[str, Any], max_n: int = 4) -> list[tuple[float, float, float]]:
    bbox = (census or {}).get("bbox") or {}
    if not bbox:
        return []
    corners = [
        (bbox["xmin"], bbox["ymin"], bbox["zmin"]),
        (bbox["xmax"], bbox["ymax"], bbox["zmax"]),
        (bbox["xmin"], bbox["ymax"], bbox["zmin"]),
        (bbox["xmax"], bbox["ymin"], bbox["zmax"]),
    ]
    return [tuple(round(float(v), 3) for v in c) for c in corners[:max_n]]


def cavity_span_along_axis(
    census: dict[str, Any], xy: tuple[float, float], axis: str, tol: Optional[float] = None
) -> float:
    """Span (max-min) along ``axis`` of edges near the (x, y) point ``xy`` —
    a cheap proxy for "how far this cavity currently reaches" without full
    feature-tree access. Returns 0.0 if nothing is found near ``xy``."""
    census = census or {}
    idx = _AXIS_INDEX.get((axis or "z").lower(), 2)
    other = [i for i in range(3) if i != idx]
    size = census.get("size") or (1.0, 1.0, 1.0)
    tol = tol if tol is not None else 0.1 * max(float(size[other[0]]), float(size[other[1]]), 1e-6)
    lo, hi = None, None
    for e in census.get("edges") or []:
        center = e.get("center")
        if not center:
            continue
        if abs(center[other[0]] - xy[0]) > tol or abs(center[other[1]] - xy[1]) > tol:
            continue
        bb = (e.get("metadata") or {}).get("bbox") or {}
        keys = (("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax"))[idx]
        if keys[0] not in bb:
            continue
        a, b = float(bb[keys[0]]), float(bb[keys[1]])
        lo = a if lo is None else min(lo, a)
        hi = b if hi is None else max(hi, b)
    if lo is None:
        return 0.0
    return float(hi - lo)


def geometry_brief(census: dict[str, Any], max_holes: int = 8) -> str:
    """Structured MODEL block for the LLM — facts from OCC, not from an image."""
    census = census or {}
    bb = census.get("bbox") or {}
    size = census.get("size") or (0, 0, 0)
    lines = [
        "MODEL:",
        "",
        "Bounding box:",
        f"X: {float(bb.get('xmin', 0)):.3f} → {float(bb.get('xmax', 0)):.3f} mm",
        f"Y: {float(bb.get('ymin', 0)):.3f} → {float(bb.get('ymax', 0)):.3f} mm",
        f"Z: {float(bb.get('zmin', 0)):.3f} → {float(bb.get('zmax', 0)):.3f} mm",
        "",
        f"Overall size: {float(size[0]):.3f} x {float(size[1]):.3f} x {float(size[2]):.3f} mm",
        f"Volume: {float(census.get('volume') or 0):.4f}",
        f"Solids: {census.get('n_solids')}  Faces: {census.get('n_faces')}  "
        f"Edges: {census.get('n_edges')}  Vertices: {census.get('n_vertices')}",
        f"Planar faces: {len(census.get('planar_faces') or [])}  "
        f"Cylindrical faces: {len(census.get('cylindrical_faces') or [])}  "
        f"Circular edges: {len(census.get('circular_edges') or [])}",
        f"Symmetry: {', '.join(census.get('symmetry_hints') or []) or 'none'}",
        f"Longest axis: {census.get('longest_axis')}  Shortest axis: {census.get('shortest_axis')}",
        "",
        "Cylindrical / hole features:",
    ]
    holes = census.get("hole_candidates") or []
    if not holes:
        lines.append("(none clustered)")
    for i, h in enumerate(holes[:max_holes], 1):
        c = h.get("center") or (0, 0, 0)
        lines.append(f"Feature {i}:")
        lines.append(f"  type: {h.get('type', 'hole')}")
        lines.append(f"  diameter: {h.get('diameter')} mm")
        lines.append(f"  center: ({c[0]}, {c[1]}, {c[2]})")
        lines.append(f"  axis: {h.get('axis')}")
        lines.append(f"  depth: {h.get('depth')}")
        if h.get("face_normal"):
            lines.append(f"  face_normal: {h['face_normal']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def census_to_prompt(census: dict[str, Any], max_entities: int = 8) -> str:
    """LLM-facing MODEL block from OCC facts."""
    brief = geometry_brief(census, max_holes=max_entities)
    extras = []
    for face in (census.get("planar_faces") or [])[:4]:
        n = face.get("normal")
        extras.append(f"planar {face.get('entity_id')} area={face.get('area')} n={n} c={face.get('center')}")
    if extras:
        return brief + "\n\nPlanar faces:\n" + "\n".join(extras)
    return brief


def find_entities_near_point(
    census: dict[str, Any],
    point: tuple[float, float, float],
    kinds: Optional[list[str]] = None,
    k: int = 5,
) -> list[EntityRef]:
    kinds = kinds or ["face", "edge", "body"]
    candidates: list[tuple[float, EntityRef]] = []
    for kind in kinds:
        for item in census.get(f"{kind}s", []) or []:
            center = item.get("center")
            if not center:
                continue
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(center, point)))
            candidates.append((dist, EntityRef.model_validate(item)))
    candidates.sort(key=lambda t: t[0])
    return [c[1] for c in candidates[:k]]


def volume_delta_ratio(before: dict[str, Any], after: dict[str, Any]) -> float:
    v0 = float(before.get("volume") or 0.0)
    v1 = float(after.get("volume") or 0.0)
    if abs(v0) < 1e-9:
        return 0.0 if abs(v1) < 1e-9 else 1.0
    return abs(v1 - v0) / abs(v0)
