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


def test_local_edit_tools_change_geometry(examples):
    step = Path(examples["fillet_box"]) / "input.step"
    before = inspect_step(step)
    from groundedcad.geometry.inspect import inspect_shape, shape_from_workplane

    through = execute_tool("cut_through", {"step_path": str(step), "x": 0.0, "y": 0.0, "z": 0.0, "width": 6.0, "thickness": 2.0, "axis": "Z"})
    after = inspect_shape(shape_from_workplane(through))
    assert after["valid"] is True
    assert after["volume"] < before["volume"]

    boss = execute_tool(
        "add_cylinder",
        {"step_path": str(step), "x": 0.0, "y": 0.0, "z": 12.0, "diameter": 6.0, "height": 8.0, "axis": "Z", "combine": "union"},
    )
    after_b = inspect_shape(shape_from_workplane(boss))
    assert after_b["valid"] is True
    assert after_b["volume"] > before["volume"]

    hexed = execute_tool("cut_hex", {"step_path": str(step), "x": 0.0, "y": 0.0, "z": 0.0, "radius": 4.0, "axis": "Z"})
    after_h = inspect_shape(shape_from_workplane(hexed))
    assert after_h["valid"] is True
    assert after_h["volume"] < before["volume"]


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

    from groundedcad.agents.planner import heuristic_plan
    from groundedcad.agents.schemas import GroundedIntent

    cases = [
        ("Add 0.2 mm chamfer to the hole edges to improve fitting.", "chamfer_circular_edges"),
        ("Add a connecting hole of 1.7 millimetre diameter and apply 0.1 millimetre grooves", "drill_hole_at_point"),
        ("Add a 1.5 millimetre rib to increase support and make the structure strong.", "add_box"),
        ("Hi , Please remove the fillet at the front center and add a 1 millimeter chamfer instead", "chamfer_edges_by_length"),
        ("Hi , Please change the center flower type profile into a hexagonal profile for better torque", "cut_hex"),
        ("Hi , Please convert the round edges of the gear into straight spur gear teeth.", "cut_radial_notches"),
        ("Change that vertical slot to cut through complete body to decrease weight", "cut_through"),
        ("Design a fixation rod mounted from the top with arm length 200mm to have ability to fix", "add_cylinder"),
        ("Insert screws to both holes to be able to fix tool on the table", "add_cylinder"),
        ("Hi , Please create an outlet port on the top right and an inlet port on the bottom left .", "add_cylinder"),
        ("Hi , Please create a filling system on the radiator , including a pouring section and a cap.", "add_cylinder"),
        ("Hi , Create a slot pattern across the plastic part surface on both the top and bottom sides.", "cut_slot_pattern"),
        ("Add a 2 millimetre fillet to the edges of the scrolling wheel slot to ensure a smooth surface.", "fillet_edges_by_length"),
        ("Add a sliding switch on the bottom to provide a convenient on and off function", "add_box"),
        ("Add two click buttons with the height of 2 millimetre to provide a comfortable feel", "add_cylinder"),
        ("Add rounds to all edges of the part. R=0,2mm", "fillet_edges_by_length"),
    ]
    census = {
        "center": (0, 0, 0),
        "size": (80.0, 40.0, 20.0),
        "bbox": {"xmin": -40, "xmax": 40, "ymin": -20, "ymax": 20, "zmin": -10, "zmax": 10},
        "shortest_axis": "z",
        "hole_candidates": [{"center": (0, 0, 0), "diameter": 10.0, "depth": "through"}],
        "faces": [],
    }
    for text, tool_name in cases:
        intent = GroundedIntent(summary=text, raw_instruction=text)
        _spec, tool = heuristic_plan(intent, "in.step", census)
        assert tool.tool_name == tool_name, f"{text!r} -> {tool.tool_name} (want {tool_name})"
        assert tool.tool_name != "incomplete_plan", text

    gear = classify_instruction("convert the round edges of the gear into straight spur gear teeth")
    assert gear.edit_type == EditPattern.BOOLEAN_MODIFICATION
    assert gear.notes == "gear_teeth"
    flower = classify_instruction("change the center flower type profile into a hexagonal profile")
    assert flower.notes == "hex_profile"
    rounds = classify_instruction("Add rounds to all edges of the part. R=0,2mm")
    assert rounds.target_kind == "all_edges"
    assert rounds.radius_mm == 0.2
    slot = classify_instruction("fillet to the edges of the scrolling wheel slot")
    assert slot.target_kind == "slot"

    hole_plan_text = "Add a connecting hole of 1.7 millimetre diameter"
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


def test_edit_context_is_compact():
    from groundedcad.geometry.inspect import edit_context, geometry_brief

    census = {
        "size": (100.0, 20.0, 8.0),
        "volume": 12345.67,
        "n_solids": 1,
        "shortest_axis": "z",
        "hole_candidates": [
            {"diameter": 8.0, "center": (1.111, 2.222, 3.333), "axis": "Z"},
            {"diameter": 4.0, "center": (0, 0, 0), "axis": "X"},
        ],
    }
    ctx = edit_context(census, edit_type="hole_edit")
    assert ctx["holes"][0]["d"] == 8.0
    assert len(json.dumps(ctx, separators=(",", ":"))) < len(geometry_brief(census))


def test_llm_local_edit_rejects_whole_body_ops():
    from groundedcad.llm.base import MockLLMClient
    from groundedcad.tools.llm_cadquery import generate_llm_local_edit, tool_from_llm_json

    assert tool_from_llm_json({"tool_name": "translate_body", "arguments": {"dx": 1}}, "a.step") is None
    assert tool_from_llm_json({"tool_name": "raw_cadquery", "arguments": {}}, "a.step") is None
    tool = tool_from_llm_json(
        {
            "tool_name": "add_box",
            "arguments": {"x": 1, "y": 2, "z": 3, "length": 4, "width": 5, "height": 6, "combine": "union"},
            "rationale": "port",
        },
        "a.step",
    )
    assert tool is not None
    assert tool.tool_name == "add_box"
    assert tool.arguments["step_path"] == "a.step"
    assert "step_path" in tool.arguments

    client = MockLLMClient()
    filled = generate_llm_local_edit(
        client,
        instruction="Add a 2 mm boss on the top face",
        model={"size_mm": [10, 10, 4]},
        classified={"edit_type": "feature_addition", "action": "add"},
        step_path="a.step",
    )
    assert filled is not None
    assert filled.tool_name in {"add_box", "drill_hole_at_point", "boolean_cut_box"}
    assert client.calls[-1]["max_tokens"] == 1024
    user = client.calls[-1]["user"]
    assert "Add a 2 mm boss" in user
    assert "def my_cad_function" not in user


def test_generate_llm_raw_local_edit_returns_script():
    from groundedcad.llm.base import MockLLMClient
    from groundedcad.tools.llm_cadquery import generate_llm_raw_local_edit

    client = MockLLMClient()
    script = generate_llm_raw_local_edit(
        client,
        instruction="Add a boss next to the mounting hole",
        model={"size_mm": [10, 10, 4]},
        classified={"edit_type": "feature_addition", "action": "add"},
        failure="FAILED_LOCATION: generic cut box does not satisfy the instruction.",
        location_hint={"protrusions": [], "existing_holes": [{"center": [1, 2, 3], "diameter": 4.0}]},
        step_path="a.step",
    )
    assert script is not None
    assert "def my_cad_function" in script
    assert "importStep" in script
    # Payload sent upstream must carry the richer location_hint, not just the instruction.
    user = client.calls[-1]["user"]
    assert "existing_holes" in user
    assert client.calls[-1]["max_tokens"] == 2048


def test_strategy_feature_add_uses_instruction_text_for_generic_site():
    from groundedcad.agents.classifier import ClassifiedEdit
    from groundedcad.agents.patterns import strategy_feature_add
    from groundedcad.agents.schemas import EditPattern

    census = {
        "center": (0.0, 0.0, 10.0),
        "size": (20.0, 10.0, 20.0),
        "bbox": {"xmin": -10, "xmax": 10, "ymin": -5, "ymax": 5, "zmin": 0, "zmax": 20},
        "faces": [],
    }
    edit = ClassifiedEdit(edit_type=EditPattern.FEATURE_ADDITION, action="add", notes="feature_add:feature")

    # No instruction text available -> old behaviour: falls back to bbox center.
    blind = strategy_feature_add(edit, "in.step", census)
    assert blind.arguments["z"] == pytest.approx(10.0 + 1.0)

    # Instruction names a side -> anchor should move toward that side instead
    # of sitting dead-center, since the tag alone ("feature") never carries a
    # direction word.
    grounded = strategy_feature_add(
        edit, "in.step", census, text="Add a small boss near the top of the part"
    )
    assert grounded.arguments["z"] == pytest.approx(20.0 + 1.0)


def test_jsonable_flattens_ragged_numpy_arrays():
    """scripts/run_full48.py serializes a parquet row through _jsonable() to
    hand it to an isolated subprocess. This dataset's brep_start_path column
    comes back as a ragged object array (one sub-array per candidate file,
    e.g. .f3d and .step logged separately) — falling through to str(obj)
    produced an unparseable repr, silently breaking step_path resolution for
    every row that hit the subprocess path fresh (not already cached on disk)."""
    import numpy as np

    from scripts.run_full48 import _jsonable

    ragged = np.array(
        [np.array(["breps/a.f3d"], dtype=object), np.array(["breps/a.step"], dtype=object)],
        dtype=object,
    )
    out = _jsonable({"brep_start_path": ragged})
    assert out == {"brep_start_path": [["breps/a.f3d"], ["breps/a.step"]]}
    # Must round-trip through json + back unchanged, since that's the actual
    # subprocess handoff path.
    assert json.loads(json.dumps(out)) == out


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


def test_high_confidence_local_and_dual_critic():
    from groundedcad.agents.classifier import ClassifiedEdit, classify_instruction, high_confidence_local
    from groundedcad.agents.schemas import EditPattern
    from groundedcad.verify.edit_delta import dual_critic_accept, dual_critic_failures

    hole_ch = classify_instruction("Add 0.2 mm chamfer to the hole edges to improve fitting.")
    assert high_confidence_local(hole_ch)
    all_r = classify_instruction("Add rounds to all edges of the part. R=0,2mm")
    assert high_confidence_local(all_r)
    rib = classify_instruction("Add a 1.5 millimetre rib to increase support.")
    assert not high_confidence_local(rib)

    before = {"volume": 100.0, "size": (10, 10, 10), "n_solids": 1, "n_faces": 6, "center": (0, 0, 0)}
    assert dual_critic_failures(before, dict(before))[0].startswith("IDENTITY")
    extra = dict(before, n_solids=2, volume=101.0, n_faces=12)
    blend = ClassifiedEdit(edit_type=EditPattern.FILLET_CHAMFER, action="chamfer", distance_mm=0.2)
    assert any("EXTRA_BODIES" in f for f in dual_critic_failures(before, extra, classified=blend))
    add = ClassifiedEdit(edit_type=EditPattern.FEATURE_ADDITION, action="add")
    assert not any("EXTRA_BODIES" in f for f in dual_critic_failures(before, extra, classified=add))
    huge = dict(before, volume=10.0, size=(10, 10, 10), n_faces=8)
    assert any("OVERSIZED_CUT" in f for f in dual_critic_failures(before, huge, classified=blend))
    assert dual_critic_accept([], iteration=0, visual=True) is False
    assert dual_critic_accept([], iteration=1, visual=True) is True
    assert dual_critic_accept([], iteration=0, cheap_high_conf=True, visual=False) is True
    assert dual_critic_accept(["IDENTITY"], iteration=1, visual=True) is False


def test_generate_grounded_cadquery_uses_occ_facts_and_blocks_first_complete():
    from groundedcad.llm.base import MockLLMClient
    from groundedcad.tools.llm_cadquery import GROUNDED_CQ_MAX_TOKENS, generate_grounded_cadquery

    client = MockLLMClient()
    out = generate_grounded_cadquery(
        client,
        instruction="Add a 1.5 mm rib",
        geometry_brief="MODEL:\nBounding box:\nX: 0 → 10 mm",
        classified={"edit_type": "feature_addition", "action": "add", "distance_mm": 1.5},
        location_hint={"protrusions": [{"center": [1, 2, 3]}]},
        failure="IDENTITY_OUTPUT: pred matches start",
        last_script="def my_cad_function(args):\n    return shape\n",
        stdout="Exported shape",
        images=None,
        iteration=0,
        visual_iters_remaining=5,
    )
    assert out["complete"] is False
    assert "def my_cad_function" in out["my_cad_function"]
    assert "importStep" in out["my_cad_function"]
    payload = client.calls[-1]["user"]
    assert "1.5 mm rib" in payload
    assert "protrusions" in payload
    assert "IDENTITY_OUTPUT" in payload
    assert "Bounding box" in payload
    assert client.calls[-1]["max_tokens"] == GROUNDED_CQ_MAX_TOKENS
    out2 = generate_grounded_cadquery(
        client,
        instruction="Add a 1.5 mm rib",
        geometry_brief="MODEL",
        classified={"edit_type": "feature_addition"},
        iteration=1,
        visual_iters_remaining=4,
    )
    # Mock always returns complete=false; first-iter force is the important lock.
    assert "complete" in client.calls[-1]["system"].lower()
