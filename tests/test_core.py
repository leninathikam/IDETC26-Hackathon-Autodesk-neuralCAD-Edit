"""Unit tests for GroundedCAD core (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundedcad.agents.grounder import heuristic_ground
from groundedcad.agents.instruction_parser import parse_instruction
from groundedcad.agents.planner import heuristic_plan
from groundedcad.agents.schemas import OperationType
from groundedcad.geometry.inspect import inspect_step, volume_delta_ratio
from groundedcad.ingest.request_parser import align_transcript_and_events, load_request_json, parse_request_from_row
from groundedcad.llm.base import MockLLMClient, parse_json_loose
from groundedcad.pipeline import GroundedCADPipeline
from groundedcad.runtime.sandbox import Sandbox
from groundedcad.tools.cadquery_tools import execute_tool
from groundedcad.verify.agent import deterministic_verification
from groundedcad.verify.geometric import verify_execution
from scripts.make_synthetic_examples import ensure_examples


@pytest.fixture(scope="session")
def examples(tmp_path_factory):
    root = tmp_path_factory.mktemp("examples")
    return ensure_examples(root)


def test_align_transcript_and_events():
    words = [
        {"word": "fillet", "start": 1.0, "end": 1.3},
        {"word": "this", "start": 1.3, "end": 1.5},
        {"word": "edge", "start": 1.5, "end": 1.8},
    ]
    events = [{"t": 1.4, "x": 0.5, "y": 0.5, "drawing": True}]
    cues = align_transcript_and_events(words, events)
    assert len(cues) == 1
    assert "fillet" in cues[0].text
    assert cues[0].drawing is True


def test_parse_json_loose():
    text = 'Sure.\n```json\n{"a": 1}\n```\n'
    assert parse_json_loose(text)["a"] == 1


def test_geometry_brief_and_edit_delta(examples):
    from groundedcad.geometry.inspect import geometry_brief, inspect_step
    from groundedcad.verify.edit_delta import check_edit_delta_unintended

    step = Path(examples["hole_plate"]) / "input.step"
    census = inspect_step(step)
    brief = geometry_brief(census)
    assert "Bounding box:" in brief
    assert "Solids:" in brief
    after = dict(census)
    size = list(census["size"])
    size[2] = size[2] * 0.7
    after["size"] = tuple(size)
    chk = check_edit_delta_unintended(
        census,
        after,
        instruction="Increase the center hole diameter from 8 mm to 12 mm.",
        edit_spec=None,
    )
    # Without edit_spec, thickness change is still listed in observed; local hole op needs spec.
    from groundedcad.agents.schemas import EditSpec, OperationType

    spec = EditSpec(intent_summary="enlarge hole", operation=OperationType.HOLE)
    chk = check_edit_delta_unintended(
        census,
        after,
        instruction="Increase the center hole diameter from 8 mm to 12 mm.",
        edit_spec=spec,
    )
    assert chk.passed is False
    assert any("thickness" in x or "size" in x for x in chk.measured["unintended"])
    step = Path(examples["fillet_box"]) / "input.step"
    before = inspect_step(step)
    assert before["valid"] is True
    assert before["n_solids"] >= 1

    result = execute_tool(
        "fillet_edges_by_length",
        {"step_path": str(step), "radius": 1.0, "max_edges": 8},
    )
    from groundedcad.geometry.inspect import inspect_shape, shape_from_workplane

    after = inspect_shape(shape_from_workplane(result))
    assert after["valid"] is True
    assert volume_delta_ratio(before, after) > 0


def test_hole_tool(examples):
    step = Path(examples["hole_plate"]) / "input.step"
    before = inspect_step(step)
    result = execute_tool(
        "drill_hole_at_point",
        {"step_path": str(step), "x": 0.0, "y": 0.0, "z": 0.0, "diameter": 4.0},
    )
    from groundedcad.geometry.inspect import inspect_shape, shape_from_workplane

    after = inspect_shape(shape_from_workplane(result))
    assert after["valid"] is True
    assert after["volume"] < before["volume"]


def test_classify_edit_patterns():
    from groundedcad.agents.patterns import EditPattern, classify_edit

    assert classify_edit("Add 0.2 mm chamfer to the hole edges")[0] == EditPattern.FILLET_CHAMFER
    assert classify_edit("Add a connecting hole of 1.7 millimetre diameter")[0] == EditPattern.HOLE_EDIT
    assert classify_edit("Create a slot pattern across the plastic part")[0] == EditPattern.PATTERN
    assert classify_edit("Scale 10x, 2 degree drafts, and rounds")[0] == EditPattern.DIMENSION_CHANGE
    from groundedcad.agents.classifier import parse_operations

    ops17 = parse_operations("Scale 10x, 2 degree drafts, and rounds", {"factor": 10.0, "angle": 2.0})
    assert {o.type for o in ops17} >= {"SCALE", "DRAFT", "FILLET"}
    assert classify_edit("Change that vertical slot to cut through complete body")[0] == EditPattern.BOOLEAN_MODIFICATION
    spec = parse_instruction(
        "Add a connecting hole of 1.7 millimetre diameter and apply 0.1 millimetre grooves"
    )
    assert spec.dimensions["diameter"] == 1.7
    assert spec.dimensions["groove"] == 0.1
    assert spec.operation.value == "hole"
    dims2 = parse_instruction("Add rounds to all edges of the part. R=0,2mm").dimensions
    assert dims2.get("radius") == 0.2 or dims2.get("value_mm") == 0.2

    from groundedcad.agents.classifier import classify_instruction

    moved = classify_instruction("Move the hole 5 mm to the right.")
    assert moved.edit_type == EditPattern.FEATURE_TRANSLATION
    assert moved.target_kind == "hole"
    assert moved.distance_mm == 5.0
    assert moved.direction == (1.0, 0.0, 0.0)
    assert moved.complete is True
    euro = classify_instruction("Change the current plug for Europlug with two round pins")
    assert euro.edit_type != EditPattern.FILLET_CHAMFER
    hole_plan_text = "Add a connecting hole of 1.7 millimetre diameter"
    from groundedcad.agents.planner import heuristic_plan
    from groundedcad.agents.schemas import GroundedIntent

    intent = GroundedIntent(summary=hole_plan_text, raw_instruction=hole_plan_text, dimensions={"diameter": 1.7})
    _spec, tool = heuristic_plan(intent, "in.step", {"center": (0, 0, 0), "faces": []})
    assert tool.tool_name == "drill_hole_at_point"
    assert tool.arguments["diameter"] == 1.7


def test_heuristic_ground_and_plan(examples):
    req = load_request_json(Path(examples["fillet_box"]) / "request.json")
    census = inspect_step(req.step_path)
    intent = heuristic_ground(req, census)
    assert OperationType.FILLET in intent.operation_hints
    assert intent.dimensions.get("radius") == 2.0 or intent.dimensions.get("value_mm") == 2.0
    spec, tool = heuristic_plan(intent, req.step_path)
    assert tool.tool_name == "fillet_edges_by_length"
    assert spec.success_checks


def test_sandbox_inprocess_and_verify(examples, tmp_path):
    step = Path(examples["translate_box"]) / "input.step"
    before = inspect_step(step)
    sb = Sandbox(render=False)
    exe = sb.run_inprocess_tool(
        "translate_body",
        {"step_path": str(step), "dx": 0, "dy": 10, "dz": 0},
        tmp_path / "out",
    )
    assert exe.success
    checks = verify_execution(before=before, execution=exe)
    assert any(c.name == "valid_brep" and c.passed for c in checks)
    decision = deterministic_verification(before=before, execution=exe)
    assert decision.topology_error is False
    assert decision.severity in {"none", "minor", "major"}
    assert set(decision.model_dump()) >= {
        "correct",
        "instruction_satisfied",
        "unintended_changes",
        "dimension_error",
        "position_error",
        "topology_error",
        "severity",
        "diagnosis",
        "specific_fix",
    }


def test_pipeline_demo(examples, tmp_path):
    req = load_request_json(Path(examples["duplicate_box"]) / "request.json")
    client = MockLLMClient()
    pipe = GroundedCADPipeline(
        grounding_client=client,
        planning_client=client,
        critic_client=client,
        max_iters=2,
        sandbox=Sandbox(render=False),
        inprocess=True,
        render=False,
    )
    result = pipe.run(req, tmp_path / "pipe")
    assert result.step_path
    assert Path(result.step_path).exists()
    settings = Path(result.step_path).parent / "settings.json"
    assert settings.exists()
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["edit_request_id"] == req.request_id
    assert data["isHuman"] is False


def test_identity_rejected():
    from groundedcad.verify.edit_delta import check_not_identity, is_identity

    before = {"volume": 10.0, "size": (1, 1, 1), "n_solids": 1, "n_faces": 6}
    assert is_identity(before, dict(before))
    chk = check_not_identity(before, dict(before), instruction="Add a 1.5 mm rib")
    assert chk.passed is False


def test_extract_llm_cadquery_script():
    from groundedcad.tools.llm_cadquery import extract_python_script, is_identity_scaffold

    fenced = "```python\ndef my_cad_function(args):\n    return 1\n```"
    assert "def my_cad_function" in extract_python_script(fenced)
    assert is_identity_scaffold("# TODO: edit shape\nreturn shape")


def test_parse_request_paths(tmp_path):
    step = tmp_path / "a.step"
    # minimal fake path reference; parser shouldn't require file exists
    row = {
        "request": "r1",
        "instruction": "make a hole",
        "brep_start_path": ["models/a.step"],
        "views": ["views/x_front.png"],
    }
    # create dummy files so absolute rewrite is meaningful
    (tmp_path / "models").mkdir()
    step.write_text("solid", encoding="utf-8")
    (tmp_path / "models" / "a.step").write_text("solid", encoding="utf-8")
    (tmp_path / "views").mkdir()
    (tmp_path / "views" / "x_front.png").write_bytes(b"png")
    req = parse_request_from_row(row, tmp_path)
    assert req.step_path.endswith("a.step")
    assert "front" in req.views
