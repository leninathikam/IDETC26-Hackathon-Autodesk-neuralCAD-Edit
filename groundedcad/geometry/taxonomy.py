"""ISO 10303 (STEP) surface taxonomy and per-component feature classification.

STEP AP203/AP214/AP242 represents B-Rep geometry with a small, fixed
vocabulary of EXPRESS surface entities: every ADVANCED_FACE wraps exactly
one of PLANE, CYLINDRICAL_SURFACE, CONICAL_SURFACE, SPHERICAL_SURFACE,
TOROIDAL_SURFACE, SURFACE_OF_REVOLUTION, SURFACE_OF_LINEAR_EXTRUSION, or a
B_SPLINE_SURFACE_WITH_KNOTS. CadQuery/OCC's ``Face.geomType()`` string is a
near 1:1 analogue of that vocabulary (it comes from the same underlying
OCCT `GeomAbs_SurfaceType` enum STEP importers populate from). This module:

1. Maps OCC geom-type strings onto their ISO 10303 EXPRESS entity names,
   so the census speaks the file format's own taxonomy rather than an
   ad-hoc one.
2. Uses the resulting per-face histogram, together with volume/hole/face
   counts already computed in ``inspect.py``, to assign each *component*
   (solid body) a coarse feature class (plate_with_holes, boss_or_pin,
   cylindrical_body, bracket, shell, freeform, ...).

This is deliberately a lightweight heuristic classifier, not a full
feature-recognition engine — it exists to give the grounder/LLM a
structured, per-component description of multi-body assemblies instead of
one flattened whole-part census, so edits can be scoped to the right body
instead of drifting into neighboring geometry (the WRONG_SCOPE / OVER_EDIT
failure buckets the dual critic already flags).
"""

from __future__ import annotations

from typing import Any

# OCC geomType() substring -> ISO 10303-42 EXPRESS surface entity name.
# Order matters: checked as substring match against the upper-cased geom
# string, most specific first.
ISO_SURFACE_MAP: list[tuple[str, str]] = [
    ("PLANE", "PLANE"),
    ("CYLINDER", "CYLINDRICAL_SURFACE"),
    ("CONE", "CONICAL_SURFACE"),
    ("SPHERE", "SPHERICAL_SURFACE"),
    ("TORUS", "TOROIDAL_SURFACE"),
    ("REVOLUTION", "SURFACE_OF_REVOLUTION"),
    ("EXTRUSION", "SURFACE_OF_LINEAR_EXTRUSION"),
    ("OFFSET", "OFFSET_SURFACE"),
    ("BEZIER", "B_SPLINE_SURFACE_WITH_KNOTS"),
    ("BSPLINE", "B_SPLINE_SURFACE_WITH_KNOTS"),
    ("SPLINE", "B_SPLINE_SURFACE_WITH_KNOTS"),
]

# Coarse curve-side taxonomy for edges (ISO 10303-42 curve entities),
# used the same way as ISO_SURFACE_MAP but for EDGE_CURVE / geomType()
# on edges.
ISO_CURVE_MAP: list[tuple[str, str]] = [
    ("LINE", "LINE"),
    ("CIRCLE", "CIRCLE"),
    ("ARC", "CIRCLE"),
    ("ELLIPSE", "ELLIPSE"),
    ("HYPERBOLA", "HYPERBOLA"),
    ("PARABOLA", "PARABOLA"),
    ("BEZIER", "B_SPLINE_CURVE_WITH_KNOTS"),
    ("BSPLINE", "B_SPLINE_CURVE_WITH_KNOTS"),
    ("SPLINE", "B_SPLINE_CURVE_WITH_KNOTS"),
]


def iso_surface_entity(geom: str) -> str:
    """Map an OCC face geomType() string to its ISO 10303 entity name."""
    g = str(geom or "").upper()
    for needle, entity in ISO_SURFACE_MAP:
        if needle in g:
            return entity
    return "SURFACE"  # ISO 10303-42 generic supertype fallback


def iso_curve_entity(geom: str) -> str:
    g = str(geom or "").upper()
    for needle, entity in ISO_CURVE_MAP:
        if needle in g:
            return entity
    return "CURVE"


def iso_surface_histogram(component_census: dict[str, Any]) -> dict[str, int]:
    """Count faces of a (component) census by ISO 10303 surface entity."""
    hist: dict[str, int] = {}
    for f in (component_census or {}).get("faces") or []:
        geom = (f.get("metadata") or {}).get("geom", "")
        entity = iso_surface_entity(geom)
        hist[entity] = hist.get(entity, 0) + 1
    return hist


def iso_curve_histogram(component_census: dict[str, Any]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for e in (component_census or {}).get("edges") or []:
        geom = (e.get("metadata") or {}).get("geom", "")
        entity = iso_curve_entity(geom)
        hist[entity] = hist.get(entity, 0) + 1
    return hist


def classify_component(component_census: dict[str, Any]) -> dict[str, Any]:
    """Coarse feature-class + evidence for one solid body's census.

    Returns a small dict (not a heavy object) so it round-trips cleanly
    through the JSON census payload sent to the LLM.
    """
    c = component_census or {}
    surf_hist = iso_surface_histogram(c)
    n_faces = sum(surf_hist.values()) or int(c.get("n_faces") or 0) or 1
    planar = surf_hist.get("PLANE", 0)
    cyl = surf_hist.get("CYLINDRICAL_SURFACE", 0)
    freeform = surf_hist.get("B_SPLINE_SURFACE_WITH_KNOTS", 0) + surf_hist.get("SURFACE_OF_REVOLUTION", 0)

    planar_ratio = planar / n_faces
    cyl_ratio = cyl / n_faces
    freeform_ratio = freeform / n_faces

    n_holes = len(c.get("hole_candidates") or c.get("holes") or [])
    size = c.get("size") or (0.0, 0.0, 0.0)
    dims = sorted(float(s) for s in size)
    thin = dims[0] < 0.2 * max(dims[-1], 1e-6) if dims and dims[-1] > 1e-6 else False

    feature_class = "unknown"
    evidence: list[str] = []

    if freeform_ratio > 0.35:
        feature_class = "freeform"
        evidence.append(f"{freeform}/{n_faces} faces are B-spline/revolution surfaces")
    elif planar_ratio > 0.6 and n_holes >= 1 and thin:
        feature_class = "plate_with_holes"
        evidence.append(f"thin planar body ({dims[0]:.2f} vs {dims[-1]:.2f} mm) with {n_holes} hole(s)")
    elif planar_ratio > 0.6 and thin:
        feature_class = "bracket_or_plate"
        evidence.append(f"thin planar body, {planar}/{n_faces} planar faces")
    elif cyl_ratio > 0.5 and dims and dims[-1] > 0 and max(dims) / max(dims[0], 1e-6) > 2.0:
        feature_class = "boss_or_pin"
        evidence.append(f"slender mostly-cylindrical body, {cyl}/{n_faces} cylindrical faces")
    elif cyl_ratio > 0.5:
        feature_class = "cylindrical_body"
        evidence.append(f"{cyl}/{n_faces} cylindrical faces")
    elif planar_ratio > 0.75 and n_faces <= 8:
        feature_class = "prismatic_block"
        evidence.append(f"box-like, {n_faces} planar-dominant faces")
    elif n_faces > 20:
        feature_class = "shell_or_housing"
        evidence.append(f"high face count ({n_faces}), mixed surface types")
    else:
        evidence.append(f"planar={planar} cylindrical={cyl} other={n_faces - planar - cyl}")

    return {
        "feature_class": feature_class,
        "iso_surface_histogram": surf_hist,
        "n_faces": n_faces,
        "n_holes": n_holes,
        "evidence": evidence,
    }


def component_summary_line(component_id: str, component_census: dict[str, Any], classification: dict[str, Any]) -> str:
    """One compact line for the geometry_brief prompt block."""
    size = component_census.get("size") or (0.0, 0.0, 0.0)
    vol = float(component_census.get("volume") or 0.0)
    hist = classification.get("iso_surface_histogram") or {}
    hist_str = ", ".join(f"{k}x{v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:3])
    return (
        f"{component_id}: {classification.get('feature_class')} | "
        f"size {float(size[0]):.2f}x{float(size[1]):.2f}x{float(size[2]):.2f} mm | "
        f"vol {vol:.2f} | holes {classification.get('n_holes')} | "
        f"surfaces [{hist_str}]"
    )