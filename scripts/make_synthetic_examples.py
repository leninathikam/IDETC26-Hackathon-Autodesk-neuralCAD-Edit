"""Generate synthetic STEP fixtures and grounded requests for local testing."""

from __future__ import annotations

import json
from pathlib import Path

from groundedcad.geometry.fallback import (
    box_solid,
    cadquery_available,
    export_simple_step,
    plate_with_boss_solid,
)


def _box_step(path: Path, size=(40, 30, 20)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cadquery_available():
        import cadquery as cq
        from cadquery import exporters

        result = cq.Workplane("XY").box(*size)
        exporters.export(result, str(path), exportType="STEP")
        return path
    export_simple_step(box_solid(*size), path)
    return path


def _plate_with_boss(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cadquery_available():
        import cadquery as cq
        from cadquery import exporters

        plate = cq.Workplane("XY").box(60, 40, 8)
        boss = cq.Workplane("XY").workplane(offset=4).center(10, 0).circle(8).extrude(12)
        result = plate.union(boss)
        exporters.export(result, str(path), exportType="STEP")
        return path
    export_simple_step(plate_with_boss_solid(), path)
    return path


def ensure_examples(root: Path) -> dict[str, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    created = {}

    d = root / "fillet_box"
    step = d / "input.step"
    _box_step(step)
    req = {
        "request": "synth_fillet_box",
        "instruction": "Add a 2mm fillet to the sharp edges of the box.",
        "transcript": "Please fillet the edges with a 2 mm radius.",
        "transcript_words": [
            {"word": "Please", "start": 0.0, "end": 0.3},
            {"word": "fillet", "start": 0.3, "end": 0.6},
            {"word": "the", "start": 0.6, "end": 0.7},
            {"word": "edges", "start": 0.7, "end": 1.0},
            {"word": "with", "start": 1.0, "end": 1.2},
            {"word": "a", "start": 1.2, "end": 1.3},
            {"word": "2", "start": 1.3, "end": 1.5},
            {"word": "mm", "start": 1.5, "end": 1.7},
            {"word": "radius", "start": 1.7, "end": 2.1},
        ],
        "events": [{"t": 0.5, "x": 0.55, "y": 0.5, "drawing": False}],
        "brep_start_path": [str(step.resolve())],
        "difficulty": "easy",
        "modality": "text",
        "root_dir": str(d.resolve()),
    }
    (d / "request.json").write_text(json.dumps(req, indent=2), encoding="utf-8")
    created["fillet_box"] = d

    d = root / "hole_plate"
    step = d / "input.step"
    _plate_with_boss(step)
    req = {
        "request": "synth_hole_plate",
        "instruction": "Drill a 6mm diameter hole through the top face near the boss.",
        "transcript": "Drill a six millimeter hole here through the top.",
        "transcript_words": [
            {"word": "Drill", "start": 0.0, "end": 0.3},
            {"word": "a", "start": 0.3, "end": 0.4},
            {"word": "six", "start": 0.4, "end": 0.7},
            {"word": "millimeter", "start": 0.7, "end": 1.2},
            {"word": "hole", "start": 1.2, "end": 1.5},
            {"word": "here", "start": 1.5, "end": 1.8},
        ],
        "events": [{"t": 1.6, "x": 0.65, "y": 0.5, "drawing": True}],
        "brep_start_path": [str(step.resolve())],
        "difficulty": "medium",
        "modality": "interactive",
        "root_dir": str(d.resolve()),
    }
    (d / "request.json").write_text(json.dumps(req, indent=2), encoding="utf-8")
    created["hole_plate"] = d

    d = root / "duplicate_box"
    step = d / "input.step"
    _box_step(step, size=(20, 20, 10))
    req = {
        "request": "synth_duplicate_box",
        "instruction": "Duplicate this part once, offset by 30mm in X.",
        "transcript": "Copy the part and move the copy 30 mm along X.",
        "brep_start_path": [str(step.resolve())],
        "difficulty": "easy",
        "modality": "text",
        "root_dir": str(d.resolve()),
    }
    (d / "request.json").write_text(json.dumps(req, indent=2), encoding="utf-8")
    created["duplicate_box"] = d

    d = root / "translate_box"
    step = d / "input.step"
    _box_step(step, size=(25, 15, 10))
    req = {
        "request": "synth_translate_box",
        "instruction": "Translate the body by 10mm in Y.",
        "transcript": "Move it 10 mm in the Y direction.",
        "brep_start_path": [str(step.resolve())],
        "difficulty": "easy",
        "modality": "text",
        "root_dir": str(d.resolve()),
    }
    (d / "request.json").write_text(json.dumps(req, indent=2), encoding="utf-8")
    created["translate_box"] = d

    d = root / "chamfer_box"
    step = d / "input.step"
    _box_step(step, size=(30, 30, 15))
    req = {
        "request": "synth_chamfer_box",
        "instruction": "Chamfer the edges by 1.5mm.",
        "transcript": "Apply a 1.5 mm chamfer on the edges.",
        "brep_start_path": [str(step.resolve())],
        "difficulty": "easy",
        "modality": "text",
        "root_dir": str(d.resolve()),
    }
    (d / "request.json").write_text(json.dumps(req, indent=2), encoding="utf-8")
    created["chamfer_box"] = d

    manifest = {
        "backend": "cadquery" if cadquery_available() else "fallback",
        "examples": {k: str(v) for k, v in created.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return created


if __name__ == "__main__":
    paths = ensure_examples(Path("examples/synthetic"))
    print(json.dumps({"backend": "cadquery" if cadquery_available() else "fallback", **{k: str(v) for k, v in paths.items()}}, indent=2))
