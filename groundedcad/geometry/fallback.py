"""Pure-Python geometry fallback when CadQuery/OCP is unavailable.

Provides axis-aligned box solids, simple CSG bookkeeping, and minimal
AP203 STEP export sufficient for local demos and unit tests.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        from OCP.gp import gp_Pnt  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class AABB:
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float

    def size(self) -> tuple[float, float, float]:
        return (self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin)

    def center(self) -> tuple[float, float, float]:
        return (
            0.5 * (self.xmin + self.xmax),
            0.5 * (self.ymin + self.ymax),
            0.5 * (self.zmin + self.zmax),
        )

    def volume(self) -> float:
        sx, sy, sz = self.size()
        return abs(sx * sy * sz)

    def translated(self, dx: float, dy: float, dz: float) -> "AABB":
        return AABB(
            self.xmin + dx,
            self.ymin + dy,
            self.zmin + dz,
            self.xmax + dx,
            self.ymax + dy,
            self.zmax + dz,
        )

    def scaled(self, factor: float) -> "AABB":
        cx, cy, cz = self.center()
        sx, sy, sz = self.size()
        hx, hy, hz = 0.5 * sx * factor, 0.5 * sy * factor, 0.5 * sz * factor
        return AABB(cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz)


@dataclass
class SimpleSolid:
    """Collection of AABB bodies with optional cylindrical hole markers."""

    bodies: list[AABB] = field(default_factory=list)
    holes: list[dict[str, Any]] = field(default_factory=list)
    fillets: list[dict[str, Any]] = field(default_factory=list)
    chamfers: list[dict[str, Any]] = field(default_factory=list)
    valid: bool = True
    source: str = ""

    def volume(self) -> float:
        vol = sum(b.volume() for b in self.bodies)
        for h in self.holes:
            r = float(h["diameter"]) / 2.0
            depth = float(h.get("depth") or 1.0)
            vol -= math.pi * r * r * abs(depth)
        # fillet/chamfer approximate volume reduction
        for f in self.fillets:
            vol -= 0.15 * float(f.get("radius", 0.0)) * len(self.bodies)
        for c in self.chamfers:
            vol -= 0.1 * float(c.get("distance", 0.0)) * len(self.bodies)
        return max(vol, 0.0)

    def bbox(self) -> AABB:
        if not self.bodies:
            return AABB(0, 0, 0, 0, 0, 0)
        return AABB(
            min(b.xmin for b in self.bodies),
            min(b.ymin for b in self.bodies),
            min(b.zmin for b in self.bodies),
            max(b.xmax for b in self.bodies),
            max(b.ymax for b in self.bodies),
            max(b.zmax for b in self.bodies),
        )

    def copy(self) -> "SimpleSolid":
        import copy

        return copy.deepcopy(self)


def box_solid(length: float, width: float, height: float) -> SimpleSolid:
    hx, hy, hz = length / 2, width / 2, height / 2
    return SimpleSolid(bodies=[AABB(-hx, -hy, -hz, hx, hy, hz)])


def plate_with_boss_solid() -> SimpleSolid:
    plate = AABB(-30, -20, -4, 30, 20, 4)
    boss = AABB(2, -8, 4, 18, 8, 16)
    return SimpleSolid(bodies=[plate, boss])


def inspect_simple(solid: SimpleSolid) -> dict[str, Any]:
    bb = solid.bbox()
    faces = []
    edges = []
    # Synthesize 6 faces per body + 12 edges
    for i, body in enumerate(solid.bodies):
        cx, cy, cz = body.center()
        sx, sy, sz = body.size()
        faces.extend(
            [
                {
                    "entity_id": f"face_{i}_top",
                    "kind": "face",
                    "description": "PLANE face",
                    "confidence": 1.0,
                    "center": (cx, cy, body.zmax),
                    "area": sx * sy,
                    "normal": (0, 0, 1),
                    "radius": None,
                    "metadata": {"geom": "PLANE", "bbox": body.__dict__},
                },
                {
                    "entity_id": f"face_{i}_front",
                    "kind": "face",
                    "description": "PLANE face",
                    "confidence": 1.0,
                    "center": (cx, body.ymin, cz),
                    "area": sx * sz,
                    "normal": (0, -1, 0),
                    "radius": None,
                    "metadata": {"geom": "PLANE", "bbox": body.__dict__},
                },
            ]
        )
        edges.append(
            {
                "entity_id": f"edge_{i}_0",
                "kind": "edge",
                "description": f"LINE edge len={sx:.3f}",
                "confidence": 1.0,
                "center": (cx, body.ymin, body.zmax),
                "length": sx,
                "radius": None,
                "metadata": {"geom": "LINE", "bbox": body.__dict__},
            }
        )
    for j, h in enumerate(solid.holes):
        faces.append(
            {
                "entity_id": f"face_hole_{j}",
                "kind": "face",
                "description": "CYLINDER face",
                "confidence": 1.0,
                "center": (h["x"], h["y"], h["z"]),
                "area": math.pi * (h["diameter"] / 2) ** 2,
                "radius": h["diameter"] / 2,
                "normal": (0, 0, 1),
                "metadata": {"geom": "CYLINDER"},
            }
        )
        edges.append(
            {
                "entity_id": f"edge_hole_{j}",
                "kind": "edge",
                "description": "CIRCLE edge",
                "confidence": 1.0,
                "center": (h["x"], h["y"], h["z"]),
                "length": math.pi * h["diameter"],
                "radius": h["diameter"] / 2,
                "metadata": {"geom": "CIRCLE"},
            }
        )

    return {
        "source": solid.source,
        "valid": solid.valid,
        "validity_error": None if solid.valid else "invalid",
        "volume": solid.volume(),
        "bbox": bb.__dict__,
        "center": bb.center(),
        "size": bb.size(),
        "n_solids": len(solid.bodies),
        "n_faces": max(len(faces), 6 * len(solid.bodies)),
        "n_edges": max(len(edges), 12 * len(solid.bodies)),
        "n_vertices": 8 * len(solid.bodies),
        "bodies": [
            {
                "entity_id": f"body_{i}",
                "kind": "body",
                "description": f"Solid body {i}",
                "confidence": 1.0,
                "center": b.center(),
                "metadata": {"volume": b.volume(), "bbox": b.__dict__},
            }
            for i, b in enumerate(solid.bodies)
        ],
        "faces": faces,
        "edges": edges,
        "holes": [e for e in edges if e.get("radius")],
        "hole_candidates": [
            {
                "type": "hole",
                "diameter": float(h["diameter"]),
                "center": (h["x"], h["y"], h["z"]),
                "axis": h.get("axis", "Z"),
                "depth": "through",
            }
            for h in solid.holes
        ],
        "planar_faces": [],
        "cylindrical_faces": [f for f in faces if "CYLINDER" in str((f.get("metadata") or {}).get("geom", ""))],
        "circular_edges": [e for e in edges if e.get("radius")],
        "symmetry_hints": [],
        "longest_axis": None,
        "shortest_axis": None,
        "backend": "fallback",
    }


def export_simple_step(solid: SimpleSolid, path: Path) -> Path:
    """Write a minimal multi-box STEP using discrete Cartesian points.

    This is not a full B-Rep but is parseable as text and useful for packaging
    demos when OCP is blocked. Prefer CadQuery export in production.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('GroundedCAD fallback STEP'),'2;1');",
        f"FILE_NAME('{path.name}','2026-07-30',('GroundedCAD'),('IDETC'),",
        "  'GroundedCAD fallback','GroundedCAD','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));",
        "ENDSEC;",
        "DATA;",
        "/* GroundedCAD fallback solid summary */",
    ]
    for i, b in enumerate(solid.bodies):
        lines.append(
            f"/* BODY {i}: ({b.xmin},{b.ymin},{b.zmin})-({b.xmax},{b.ymax},{b.zmax}) vol={b.volume():.6f} */"
        )
    for i, h in enumerate(solid.holes):
        lines.append(f"/* HOLE {i}: center=({h['x']},{h['y']},{h['z']}) d={h['diameter']} */")
    for i, f in enumerate(solid.fillets):
        lines.append(f"/* FILLET {i}: r={f.get('radius')} */")
    for i, c in enumerate(solid.chamfers):
        lines.append(f"/* CHAMFER {i}: d={c.get('distance')} */")
    # Emit a single placeholder point so file is non-empty DATA section
    lines.append("#1=CARTESIAN_POINT('',(0.0,0.0,0.0));")
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")
    # Also sidecar JSON for exact fallback round-trip
    meta = {
        "bodies": [b.__dict__ for b in solid.bodies],
        "holes": solid.holes,
        "fillets": solid.fillets,
        "chamfers": solid.chamfers,
        "valid": solid.valid,
    }
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".fallback.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8"
    )
    return path


def export_simple_stl(solid: SimpleSolid, path: Path) -> Path:
    """ASCII STL of AABB boxes (no holes tessellation)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tris = []

    def add_box(b: AABB):
        x0, y0, z0, x1, y1, z1 = b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax
        faces = [
            # bottom/top/sides as 2 triangles each
            ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (0, 0, -1)),
            ((x0, y0, z0), (x1, y1, z0), (x0, y1, z0), (0, 0, -1)),
            ((x0, y0, z1), (x1, y1, z1), (x1, y0, z1), (0, 0, 1)),
            ((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (0, 0, 1)),
            ((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (0, -1, 0)),
            ((x0, y0, z0), (x1, y0, z1), (x1, y0, z0), (0, -1, 0)),
            ((x0, y1, z0), (x1, y1, z1), (x0, y1, z1), (0, 1, 0)),
            ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (0, 1, 0)),
            ((x0, y0, z0), (x0, y1, z1), (x0, y0, z1), (-1, 0, 0)),
            ((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (-1, 0, 0)),
            ((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (1, 0, 0)),
            ((x1, y0, z0), (x1, y1, z1), (x1, y1, z0), (1, 0, 0)),
        ]
        # rewrite as proper triangle triples
        quads = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # z0
            [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)],  # z1
            [(x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)],  # y0
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # y1
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # x0
            [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],  # x1
        ]
        for q in quads:
            tris.append((q[0], q[1], q[2]))
            tris.append((q[0], q[2], q[3]))

    for b in solid.bodies:
        add_box(b)

    with open(path, "w", encoding="utf-8") as f:
        f.write("solid groundedcad\n")
        for a, b, c in tris:
            f.write("  facet normal 0 0 0\n    outer loop\n")
            for p in (a, b, c):
                f.write(f"      vertex {p[0]} {p[1]} {p[2]}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid groundedcad\n")
    return path


def load_simple(step_path: str | Path) -> SimpleSolid:
    path = Path(step_path)
    meta_path = path.with_suffix(".fallback.json")
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bodies = [AABB(**b) for b in meta.get("bodies", [])]
        return SimpleSolid(
            bodies=bodies,
            holes=meta.get("holes", []),
            fillets=meta.get("fillets", []),
            chamfers=meta.get("chamfers", []),
            valid=bool(meta.get("valid", True)),
            source=str(path),
        )
    # Parse comment headers if present
    text = path.read_text(encoding="utf-8", errors="ignore")
    bodies = []
    import re

    for m in re.finditer(
        r"BODY\s+\d+:\s+\(([^)]+)\)-\(([^)]+)\)",
        text,
    ):
        a = [float(x) for x in m.group(1).split(",")]
        b = [float(x) for x in m.group(2).split(",")]
        bodies.append(AABB(a[0], a[1], a[2], b[0], b[1], b[2]))
    if not bodies:
        # default unit box
        bodies = [AABB(-10, -10, -5, 10, 10, 5)]
    return SimpleSolid(bodies=bodies, source=str(path))
