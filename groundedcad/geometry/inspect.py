"""STEP topology census and entity descriptors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from groundedcad.agents.schemas import BoundingBox, EntityKind, EntityRef
from groundedcad.geometry.fallback import cadquery_available, inspect_simple, load_simple
from groundedcad.geometry import taxonomy


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
    """Peel CadQuery Workplanes until an OCC Shape/Solid (importStep returns a Workplane)."""
    from groundedcad.geometry.fallback import SimpleSolid

    if isinstance(wp, SimpleSolid):
        return wp
    obj = wp
    for _ in range(8):
        if obj is None:
            return obj
        name = type(obj).__name__
        if name in {"Workplane", "CQ"} or hasattr(obj, "newObject"):
            nxt = obj.val() if hasattr(obj, "val") else None
            if nxt is None or nxt is obj:
                break
            obj = nxt
            continue
        break
    return obj


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

    shape = shape_from_workplane(shape)

    solids = list(shape.Solids()) if hasattr(shape, "Solids") else []
    faces = list(shape.Faces()) if hasattr(shape, "Faces") else []
    edges = list(shape.Edges()) if hasattr(shape, "Edges") else []
    vertices = list(shape.Vertices()) if hasattr(shape, "Vertices") else []

    try:
        volume = float(shape.Volume())
        # Skip OCCT isValid() — it can hang for minutes on multi-solid assemblies
        # and is often false after chamfer/fillet even when the solid is usable.
        valid = bool(volume > 1e-9 and len(solids) >= 1)
        validity_error = None
    except Exception as exc:  # noqa: BLE001
        volume = 0.0
        valid = False
        validity_error = str(exc)

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
            try:
                radius = float(edge.radius())
            except Exception:  # noqa: BLE001
                radius = None
            if radius is None or radius < 1e-9:
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

    components: list[dict[str, Any]] = []
    if len(solids) > 1:
        components = component_breakdown(solids, max_components=8)

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
        "components": components,
    }


def component_breakdown(solids: list, max_components: int = 8) -> list[dict[str, Any]]:
    """Per-body census + ISO-10303-derived feature classification.

    Only meaningful for multi-solid assemblies — a single-solid part's
    "component" is just the whole part, so callers should skip this for
    ``n_solids <= 1`` (``inspect_shape`` already does).  Each solid is
    inspected with the same ``inspect_shape`` used for the whole part, so
    the per-component census has the identical shape (faces/edges/holes/
    bbox) the rest of the pipeline already knows how to read — just scoped
    to one body and tagged with an ISO 10303 feature class.
    """
    out: list[dict[str, Any]] = []
    for i, solid in enumerate(solids[:max_components]):
        try:
            sub = inspect_shape(solid, source=f"component_{i}")
        except Exception as exc:  # noqa: BLE001
            out.append({"component_id": f"component_{i}", "error": str(exc)})
            continue
        classification = taxonomy.classify_component(sub)
        out.append(
            {
                "component_id": f"component_{i}",
                "bbox": sub.get("bbox"),
                "center": sub.get("center"),
                "size": sub.get("size"),
                "volume": sub.get("volume"),
                "n_faces": sub.get("n_faces"),
                "n_edges": sub.get("n_edges"),
                "hole_candidates": sub.get("hole_candidates"),
                **classification,
            }
        )
    return out


def component_of_point(census: dict[str, Any], point: tuple[float, float, float]) -> Optional[str]:
    """Which component_id's bbox (expanded slightly) contains ``point``.

    Used by the grounder to scope a temporal-cue click/cursor point to a
    single body in a multi-solid assembly, instead of letting
    find_entities_near_point wander into a neighboring component.
    """
    components = (census or {}).get("components") or []
    if not components:
        return None
    best_id, best_dist = None, None
    for comp in components:
        bbox = comp.get("bbox") or {}
        if not bbox:
            continue
        pad = 0.02 * max(
            bbox.get("xmax", 0) - bbox.get("xmin", 0),
            bbox.get("ymax", 0) - bbox.get("ymin", 0),
            bbox.get("zmax", 0) - bbox.get("zmin", 0),
            1e-6,
        )
        inside = (
            bbox.get("xmin", 0) - pad <= point[0] <= bbox.get("xmax", 0) + pad
            and bbox.get("ymin", 0) - pad <= point[1] <= bbox.get("ymax", 0) + pad
            and bbox.get("zmin", 0) - pad <= point[2] <= bbox.get("zmax", 0) + pad
        )
        if inside:
            return comp.get("component_id")
        center = comp.get("center") or (0, 0, 0)
        dist = sum((a - b) ** 2 for a, b in zip(center, point)) ** 0.5
        if best_dist is None or dist < best_dist:
            best_dist, best_id = dist, comp.get("component_id")
    return best_id


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


def _circle_diameter_counts(census: dict[str, Any], *, near_full_only: bool = False) -> dict[float, int]:
    counts: dict[float, int] = {}
    for e in census.get("circular_edges") or []:
        r = e.get("radius")
        if not r:
            continue
        if near_full_only:
            length = e.get("length")
            if length is None:
                continue
            full = 2.0 * math.pi * float(r)
            if full < 1e-12 or float(length) / full < 0.85:
                continue
        d = round(2.0 * float(r), 2)
        if d < 0.4:
            continue
        counts[d] = counts.get(d, 0) + 1
    return counts


def through_hole_candidates(census: dict[str, Any]) -> list[dict[str, Any]]:
    """Cylinders that look like through-bores, not fillet/boss rounds."""
    census = census or {}
    holes = list(census.get("hole_candidates") or [])
    size = census.get("size") or (1.0, 1.0, 1.0)
    spans = [float(s) for s in size if float(s) > 1e-9]
    outer_cap = 0.7 * max(spans) if spans else 1e9
    inner = [
        h
        for h in holes
        if 0.4 <= float(h.get("diameter") or 0) <= outer_cap
    ]
    through = [
        h
        for h in inner
        if str(h.get("depth") or "") == "through" or int(h.get("n_circles") or 0) >= 2
    ]
    return through or inner


def fitting_hole_diameter(
    census: dict[str, Any],
    *,
    min_diameter_mm: Optional[float] = None,
    blend_mm: Optional[float] = None,
) -> Optional[float]:
    """Through-bore diameter large enough to take the blend, not micro-arcs."""
    census = census or {}
    size = census.get("size") or (1.0, 1.0, 1.0)
    spans = [float(s) for s in size if float(s) > 1e-9]
    outer_cap = 0.7 * max(spans) if spans else 1e9
    floor = 0.8
    if blend_mm:
        # OCC chamfer fails if distance ≳ 0.32 * radius → diameter ≳ 6 * blend
        floor = max(floor, 6.0 * float(blend_mm))
    if min_diameter_mm is not None:
        floor = max(floor, float(min_diameter_mm))
    counts = {
        d: n
        for d, n in _circle_diameter_counts(census, near_full_only=True).items()
        if floor <= d <= outer_cap
    }
    if not counts:
        counts = {
            d: n
            for d, n in _circle_diameter_counts(census).items()
            if floor <= d <= outer_cap
        }
    pairish = [(d, n) for d, n in counts.items() if 2 <= n <= 24]
    if pairish:
        return min(pairish, key=lambda kv: (kv[1], kv[0]))[0]
    ranked = rank_hole_rims(census, blend_mm=blend_mm, min_diameter_mm=min_diameter_mm)
    if ranked:
        return float(ranked[0]["diameter"])
    holes = through_hole_candidates(census)
    if not holes:
        return None
    score: dict[float, int] = {}
    for h in holes:
        d = round(float(h.get("diameter") or 0), 2)
        if d < floor:
            continue
        score[d] = score.get(d, 0) + int(h.get("n_circles") or 1)
    if not score:
        return None
    return max(score.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def rank_hole_rims(
    census: dict[str, Any] | None,
    *,
    blend_mm: Optional[float] = None,
    min_diameter_mm: Optional[float] = None,
    location_hint: Optional[tuple[float, float, float]] = None,
    max_families: int = 4,
) -> list[dict[str, Any]]:
    """Rank hole families from cylindrical faces + matching circular rim edges.

    Returns dicts with diameter, centers (rim centers), n_rims, score, axis.
    Prefers through-bores large enough for the blend and inset from the outer bbox.
    """
    census = census or {}
    size = census.get("size") or (1.0, 1.0, 1.0)
    spans = [float(s) for s in size if float(s) > 1e-9]
    outer_cap = 0.7 * max(spans) if spans else 1e9
    floor = 0.8
    if blend_mm:
        floor = max(floor, 6.0 * float(blend_mm))
    if min_diameter_mm is not None:
        floor = max(floor, float(min_diameter_mm))

    bbox = census.get("bbox") or {}
    mid = (
        0.5 * (float(bbox.get("xmin", 0)) + float(bbox.get("xmax", 0))),
        0.5 * (float(bbox.get("ymin", 0)) + float(bbox.get("ymax", 0))),
        0.5 * (float(bbox.get("zmin", 0)) + float(bbox.get("zmax", 0))),
    )
    half = (
        0.5 * max(float(bbox.get("xmax", 1)) - float(bbox.get("xmin", 0)), 1e-6),
        0.5 * max(float(bbox.get("ymax", 1)) - float(bbox.get("ymin", 0)), 1e-6),
        0.5 * max(float(bbox.get("zmax", 1)) - float(bbox.get("zmin", 0)), 1e-6),
    )

    # Family key: rounded (cx, cy or axis-aware) + radius
    families: dict[tuple, dict[str, Any]] = {}

    for f in census.get("cylindrical_faces") or []:
        r = f.get("radius")
        c = f.get("center")
        if not r or not c or float(r) < 0.15:
            continue
        d = round(2.0 * float(r), 2)
        if not (floor <= d <= outer_cap):
            continue
        cx, cy, cz = (float(c[0]), float(c[1]), float(c[2]))
        key = (round(cx, 1), round(cy, 1), round(float(r), 2))
        fam = families.setdefault(
            key,
            {
                "diameter": d,
                "radius": float(r),
                "centers": [],
                "n_rims": 0,
                "has_cylinder": False,
                "axis": "Z",
            },
        )
        fam["has_cylinder"] = True
        meta = f.get("metadata") or {}
        bb = meta.get("bbox") or {}
        if bb:
            size_f = (
                float(bb.get("xmax", 0) - bb.get("xmin", 0)),
                float(bb.get("ymax", 0) - bb.get("ymin", 0)),
                float(bb.get("zmax", 0) - bb.get("zmin", 0)),
            )
            fam["axis"] = _axis_from_size(size_f)

    for e in census.get("circular_edges") or []:
        r = e.get("radius")
        c = e.get("center")
        if not r or not c or float(r) < 0.15:
            continue
        d = round(2.0 * float(r), 2)
        if not (floor <= d <= outer_cap):
            continue
        cx, cy, cz = (float(c[0]), float(c[1]), float(c[2]))
        key = (round(cx, 1), round(cy, 1), round(float(r), 2))
        fam = families.setdefault(
            key,
            {
                "diameter": d,
                "radius": float(r),
                "centers": [],
                "n_rims": 0,
                "has_cylinder": False,
                "axis": "Z",
            },
        )
        center = (round(cx, 3), round(cy, 3), round(cz, 3))
        if center not in fam["centers"]:
            fam["centers"].append(center)
            fam["n_rims"] = len(fam["centers"])

    # Also fold hole_candidates that may lack face dumps
    for h in census.get("hole_candidates") or []:
        d = round(float(h.get("diameter") or 0), 2)
        c = h.get("center")
        if not c or not (floor <= d <= outer_cap):
            continue
        r = d / 2.0
        cx, cy, cz = (float(c[0]), float(c[1]), float(c[2]))
        key = (round(cx, 1), round(cy, 1), round(r, 2))
        fam = families.setdefault(
            key,
            {
                "diameter": d,
                "radius": r,
                "centers": [],
                "n_rims": int(h.get("n_circles") or 0),
                "has_cylinder": True,
                "axis": str(h.get("axis") or "Z"),
            },
        )
        fam["has_cylinder"] = True
        center = (round(cx, 3), round(cy, 3), round(cz, 3))
        if center not in fam["centers"]:
            fam["centers"].append(center)
            fam["n_rims"] = max(fam["n_rims"], len(fam["centers"]), int(h.get("n_circles") or 0))

    ranked: list[dict[str, Any]] = []
    for fam in families.values():
        if fam["n_rims"] < 1 and not fam["has_cylinder"]:
            continue
        if not fam["centers"]:
            # Cylinder without rim centers yet — use family key center
            continue
        # Representative center = mean of rim centers
        n = len(fam["centers"])
        mx = sum(c[0] for c in fam["centers"]) / n
        my = sum(c[1] for c in fam["centers"]) / n
        mz = sum(c[2] for c in fam["centers"]) / n
        # Inset score: distance from bbox mid relative to half-span (inner better)
        inset = min(
            abs(mx - mid[0]) / half[0],
            abs(my - mid[1]) / half[1],
            abs(mz - mid[2]) / half[2],
        )
        through = 1.0 if fam["n_rims"] >= 2 else 0.0
        cyl = 1.0 if fam["has_cylinder"] else 0.0
        loc = 0.0
        if location_hint is not None:
            hx, hy, hz = location_hint
            dist = math.sqrt((mx - hx) ** 2 + (my - hy) ** 2 + (mz - hz) ** 2)
            loc = 1.0 / (1.0 + dist)
        # Prefer through + cylinder + inset (small inset value) + location
        score = 3.0 * through + 2.0 * cyl + (1.0 - min(inset, 1.0)) + loc
        # Prefer smaller diameters among equal scores (fitting bore, not outer round)
        score -= 0.01 * float(fam["diameter"])
        ranked.append(
            {
                "diameter": float(fam["diameter"]),
                "radius": float(fam["radius"]),
                "centers": list(fam["centers"]),
                "n_rims": int(fam["n_rims"]),
                "axis": fam["axis"],
                "score": float(score),
            }
        )

    ranked.sort(key=lambda r: (-r["score"], r["diameter"]))
    # Collapse same diameter families: keep top family per diameter for tool args,
    # but also return top families overall for rim_centers.
    return ranked[: max(1, int(max_families))]


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


def _round_num(value: Any, nd: int = 2) -> Any:
    if isinstance(value, float):
        return round(value, nd)
    if isinstance(value, (list, tuple)):
        return [_round_num(v, nd) for v in value]
    if isinstance(value, dict):
        return {k: _round_num(v, nd) for k, v in value.items()}
    return value


def edit_context(
    census: dict[str, Any],
    *,
    edit_type: str = "",
    max_holes: int = 4,
) -> dict[str, Any]:
    """Compact census slice for LLM slot-fill. Not the full STEP / face list."""
    census = census or {}
    size = census.get("size") or (0, 0, 0)
    ctx: dict[str, Any] = {
        "size_mm": [round(float(s), 2) for s in size],
        "volume": round(float(census.get("volume") or 0), 1),
        "n_solids": census.get("n_solids"),
        "shortest_axis": census.get("shortest_axis"),
    }
    holes = []
    src = through_hole_candidates(census)
    fit = fitting_hole_diameter(census)
    if edit_type == "fillet_chamfer" and fit is not None:
        src = [h for h in src if abs(float(h.get("diameter") or 0) - fit) <= 0.02]
        src = src[:2]
    for h in src[:max_holes]:
        c = h.get("center") or (0, 0, 0)
        holes.append(
            {
                "d": round(float(h.get("diameter") or 0), 3),
                "c": [round(float(x), 2) for x in c],
                "axis": h.get("axis"),
            }
        )
    if holes and edit_type in {
        "",
        "hole_edit",
        "fillet_chamfer",
        "boolean_modification",
        "feature_deletion",
        "ambiguous",
    }:
        ctx["holes"] = holes
    return ctx


def geometry_brief(
    census: dict[str, Any],
    max_holes: int = 8,
    *,
    blend_mm: Optional[float] = None,
) -> str:
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
        "Through-holes only (ignore fillets, bosses, and other cylinders):",
    ]
    holes = through_hole_candidates(census)
    fit = fitting_hole_diameter(census, blend_mm=blend_mm)
    if fit is not None:
        lines.append(
            f"Fitting hole: diameter {fit:.3f} mm. "
            "Chamfer/fillet at most 4 circular rims of THIS diameter only. "
            "Do not OR together other diameters."
        )
        holes = [h for h in holes if abs(float(h.get("diameter") or 0) - fit) <= 0.02]
        holes = holes[:2]
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
    components = census.get("components") or []
    if components:
        lines.append(f"Components ({len(components)} bodies, ISO 10303 surface taxonomy):")
        for comp in components:
            if comp.get("error"):
                lines.append(f"  {comp.get('component_id')}: inspection error — {comp['error']}")
                continue
            lines.append("  " + taxonomy.component_summary_line(comp["component_id"], comp, comp))
        lines.append(
            "  (Scope edits to ONE component_id unless the instruction clearly spans the whole assembly.)"
        )
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