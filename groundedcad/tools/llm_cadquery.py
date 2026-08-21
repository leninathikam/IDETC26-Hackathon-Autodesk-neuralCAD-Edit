"""LLM fallback: fill a local tool call. Never rebuild the STEP from scratch."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from groundedcad.agents.schemas import ToolCall
from groundedcad.llm.base import LLMClient

# Whole-body ops are disallowed: they rewrite unrelated geometry and burn tokens.
LOCAL_TOOLS = frozenset(
    {
        "add_box",
        "add_cylinder",
        "boolean_cut_box",
        "cut_through",
        "cut_hex",
        "cut_radial_notches",
        "cut_slot_pattern",
        "drill_hole_at_point",
        "fillet_circular_edges",
        "fillet_edges_by_length",
        "chamfer_circular_edges",
        "chamfer_edges_by_length",
    }
)

LOCAL_EDIT_MAX_TOKENS = 1024

LOCAL_EDIT_SYSTEM = """Pick ONE local tool to apply to the imported STEP.
Do not write CadQuery. Do not rebuild or re-export the part from primitives.
The original STEP is imported for you; only the requested feature is mutated.

Allowed tool_name:
- add_box: x,y,z,length,width,height,combine (union|cut)
- add_cylinder: x,y,z,diameter,height,axis,combine
- boolean_cut_box: x,y,z,length,width,height
- cut_through: x,y,z,width,thickness,axis
- cut_hex: x,y,z,radius,axis
- cut_radial_notches: count,depth
- cut_slot_pattern: count,both_sides
- drill_hole_at_point: x,y,z,diameter,axis
- fillet_circular_edges / fillet_edges_by_length: radius, max_edges, region
- chamfer_circular_edges / chamfer_edges_by_length: distance, max_edges, region

Rules:
- Use coordinates from model/location_hint, not an invented origin.
- Keep new features small vs bbox. Never cut >15% of volume.
- Optional followups: at most 2 extra local tools (e.g. second port).
- Never choose translate_body, scale_uniform, duplicate_linear, rotate_body, or raw_cadquery.
- When `operations` contains multiple items, choose the first operation that
  has not already been completed; do not reinterpret it as a whole-body edit.

JSON only:
{"tool_name":"...","arguments":{...},"rationale":"...","followups":[]}
"""


def extract_python_script(text: str) -> str:
    raw = (text or "").strip()
    try:
        from groundedcad.llm.base import parse_json_loose

        data = parse_json_loose(raw)
        if isinstance(data, dict) and data.get("my_cad_function"):
            raw = str(data["my_cad_function"]).strip()
    except Exception:
        pass
    if "```" in raw:
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw, flags=re.I)
        if m:
            raw = m.group(1).strip()
    if "def my_cad_function" not in raw:
        raw = (
            "def my_cad_function(args):\n"
            "    import cadquery as cq\n"
            "    import os\n"
            "    shape = cq.importers.importStep(os.path.expanduser(args['input_file']))\n"
            "    return shape\n"
        )
    return raw


def is_identity_scaffold(script: str) -> bool:
    s = script or ""
    return "# TODO: edit shape" in s or (
        "importStep" in s and "return shape" in s and "TODO" in s
    )


def repair_common_cadquery_api(script: str) -> str:
    """Repair only unambiguous CadQuery 2.8 compatibility slips.

    The model occasionally emits an otherwise valid local edit that cannot run
    because it uses the old ``Workplane.sortBy`` spelling, or passes a locally
    constructed Workplane directly to a Shape boolean.  Both repairs preserve
    the intended operation and avoid spending an entire visual retry on a
    deterministic type/API error.  This intentionally does *not* infer
    geometry, selectors, dimensions, or boolean direction.
    """
    repaired = (script or "").replace(".sortBy(", ".sort(")
    workplane_names = {
        match.group(1)
        for match in re.finditer(
            r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*cq\.Workplane\s*\(", repaired
        )
    }
    for name in workplane_names:
        # Only an identifier that was directly assigned a Workplane is
        # rewritten.  Expressions and already-converted operands are left
        # untouched, so this cannot silently alter arbitrary generated code.
        repaired = re.sub(
            rf"(\b[A-Za-z_]\w*\.(?:cut|fuse)\(\s*){re.escape(name)}(\s*\))",
            rf"\1{name}.val()\2",
            repaired,
        )
    return repaired


def _clean_args(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k != "step_path" and v is not None}


def tool_from_llm_json(data: dict[str, Any], step_path: str) -> Optional[ToolCall]:
    name = str(data.get("tool_name") or "")
    if name not in LOCAL_TOOLS:
        return None
    args = _clean_args(data.get("arguments"))
    args["step_path"] = step_path
    followups: list[ToolCall] = []
    for item in (data.get("followups") or [])[:2]:
        if not isinstance(item, dict):
            continue
        fn = str(item.get("tool_name") or "")
        if fn not in LOCAL_TOOLS:
            continue
        fa = _clean_args(item.get("arguments"))
        fa["step_path"] = step_path
        followups.append(ToolCall(tool_name=fn, arguments=fa, rationale="llm_followup"))
    return ToolCall(
        tool_name=name,
        arguments=args,
        rationale=str(data.get("rationale") or "llm_local_edit"),
        followups=followups,
    )


def generate_llm_local_edit(
    client: LLMClient,
    *,
    instruction: str,
    model: dict[str, Any] | str,
    classified: dict[str, Any],
    failure: str = "",
    location_hint: dict[str, Any] | None = None,
    step_path: str = "",
) -> Optional[ToolCall]:
    slots = {
        k: classified.get(k)
        for k in (
            "edit_type",
            "target_kind",
            "action",
            "diameter_mm",
            "radius_mm",
            "distance_mm",
            "count",
            "direction",
        )
        if classified.get(k) not in (None, "", [], {})
    }
    payload = {
        "instruction": instruction,
        "slots": slots,
        "operations": classified.get("operations", []),
        "model": model,
        "failure": (failure or "")[:240],
    }
    if location_hint:
        payload["location_hint"] = location_hint
    data = client.complete_json(
        system=LOCAL_EDIT_SYSTEM,
        user=json.dumps(payload, separators=(",", ":")),
        max_tokens=LOCAL_EDIT_MAX_TOKENS,
    )
    return tool_from_llm_json(data, step_path)


RAW_LOCAL_EDIT_MAX_TOKENS = 2048

RAW_LOCAL_EDIT_SYSTEM = """Write ONE local CadQuery edit as Python code — the
fixed set of local tools was not expressive enough for this instruction, so you
get real CadQuery selectors instead of a JSON tool call.

def my_cad_function(args):
    import cadquery as cq
    import os
    shape = cq.importers.importStep(os.path.expanduser(args["input_file"]))
    # ... exactly one local add/cut/fillet/chamfer built from `shape` ...
    return result  # a cq.Workplane or cq.Shape/Solid

Ground every coordinate in the MODEL/location_hint facts given to you (holes,
cavities, protrusions, planar_sites, bbox_corners) or in a CadQuery selector
relative to the imported shape (e.g. `.faces(">Z")`, `.edges("|Z")`,
`.vertices(cq.NearestToPointSelector(...))`) — never invent an unrelated origin.

Rules:
- Import and start from the given STEP; never rebuild the part from scratch.
- Exactly one local feature (plus at most one follow-up cut/fillet on the same
  feature). No global scale/rotate/translate/mirror of the whole body.
- A cut must remove well under 15% of total volume; a new add must be small
  relative to the part's own size (see MODEL size_mm) — a boss/rib/port, not a
  second body.
- No network, filesystem, subprocess, or thread access beyond the snippet above.
- Output ONLY a single ```python fenced code block defining my_cad_function.
  No prose outside the fence.
- `operations` is the complete parsed instruction.  Implement its next unmet
  local operation, not merely the primary `slots` classification.
"""


def generate_llm_raw_local_edit(
    client: LLMClient,
    *,
    instruction: str,
    model: dict[str, Any] | str,
    classified: dict[str, Any],
    failure: str = "",
    location_hint: dict[str, Any] | None = None,
    step_path: str = "",
) -> Optional[str]:
    """Last-resort escalation: let the model write real CadQuery selectors
    instead of filling a fixed local-tool slot. Still one local edit on the
    imported STEP — never a from-scratch rebuild. Returns None on API failure;
    a returned script may still be geometrically identity — the caller
    verifies that the same way it verifies every other attempt."""
    slots = {
        k: classified.get(k)
        for k in (
            "edit_type",
            "target_kind",
            "action",
            "diameter_mm",
            "radius_mm",
            "distance_mm",
            "count",
            "direction",
        )
        if classified.get(k) not in (None, "", [], {})
    }
    payload = {
        "instruction": instruction,
        "slots": slots,
        "operations": classified.get("operations", []),
        "model": model,
        "failure": (failure or "")[:400],
    }
    if location_hint:
        payload["location_hint"] = location_hint
    try:
        resp = client.complete(
            system=RAW_LOCAL_EDIT_SYSTEM,
            user=json.dumps(payload, separators=(",", ":")),
            max_tokens=RAW_LOCAL_EDIT_MAX_TOKENS,
        )
    except Exception:
        return None
    return extract_python_script(resp.text or "")


def generate_llm_cadquery(
    client: LLMClient,
    *,
    instruction: str,
    geometry_brief: str,
    classified: dict[str, Any],
    failure: str = "",
    location_hint: dict[str, Any] | None = None,
) -> str:
    """Deprecated: kept for tests. Pipeline uses generate_llm_local_edit."""
    tool = generate_llm_local_edit(
        client,
        instruction=instruction,
        model=geometry_brief[:800],
        classified=classified,
        failure=failure,
        location_hint=location_hint,
    )
    if tool is None:
        return extract_python_script("")
    from groundedcad.tools.cadquery_gen import generate_cadquery

    return generate_cadquery(tool, "")


GROUNDED_CQ_MAX_TOKENS = 4096

GROUNDED_CQ_SYSTEM = """You write CadQuery that EDITS an imported STEP. You do not rebuild the part.

cq.importers.importStep returns a Workplane. Do not wrap it again with newObject([shape]).
Workplane uses .edges() / .val() (lowercase). OCC Solid uses .Edges() / .BoundingBox().

CadQuery 2.8 compatibility (mandatory):
- Do NOT call `cq.Face.makePlane(..., normal=...)`; this version does not
  accept a `normal` keyword. Prefer a Workplane on XY/XZ/YZ plus `rect`,
  `circle`, `extrude`, then boolean it with `solid`.
- For a mirror, use one of the named planes only, e.g.
  `solid.mirror(mirrorPlane="YZ", basePoint=(x, y, z))`. Do NOT pass a
  normal-vector tuple such as `(1, 0, 0)` as the mirror plane.
- Preserve the imported solid: form `edited = solid.fuse(feature)` or
  `edited = solid.cut(feature)` and return it in a Workplane.
- Never call `solid.chamfer(distance)` or `solid.fillet(radius)`: those Shape
  APIs require an explicit edge list and fail when called with one argument.
  For a selector-based blend, use a Workplane, e.g.
  `cq.Workplane("XY").newObject([solid]).edges(selector).chamfer(distance)`.
- Type discipline is mandatory: `solid` is a Shape; `feature_wp` is a
  Workplane. Never pass a Workplane into `solid.cut(...)` or `solid.fuse(...)`.
  Use `solid.cut(feature_wp.val())` / `solid.fuse(feature_wp.val())`, or do
  both booleans as Workplane operations. Never access `.wrapped` on a
  Workplane; use `.val()` for one Shape and `.vals()` for a shape list.
- CadQuery Workplane has `.sort(...)`, not `.sortBy(...)`.

def my_cad_function(args):
    import cadquery as cq
    import os
    wp = cq.importers.importStep(os.path.expanduser(args["input_file"]))
    solid = wp.val()
    # mutate solid using MODEL hole diameters; return a Workplane
    return cq.Workplane("XY").newObject([solid])

Hard rules:
- Always import args["input_file"]. Never rebuild the part with Workplane().box / .cylinder.
- Hole-edge chamfer/fillet: use ONLY MODEL "Fitting hole" diameter. Match |2*radius - that diameter| < 0.05 mm. Chamfer at most 4 rims. Never loop a list of Feature diameters.
- Change only the requested feature. Preserve unrelated faces and envelope.
- No network, subprocess, threads, or extra filesystem use.
- `operations` lists every requested operation.  Apply the next unmet one;
  do not silently drop secondary operations in a compound instruction.

Return JSON only, no markdown:
{"complete": false, "my_cad_function": "def my_cad_function(args):\\n ..."}

complete=true means the LAST executed solid already satisfies the instruction.
The first iteration can NEVER be complete. If complete is true, omit my_cad_function.
"""

RECONSTRUCT_CQ_SYSTEM = """You write CadQuery that implements the FULL instruction on an imported STEP.

The input file is whatever STEP the user supplied (a new part is normal). Always:

def my_cad_function(args):
    import cadquery as cq
    import os
    wp = cq.importers.importStep(os.path.expanduser(args["input_file"]))
    solid = wp.val()
    # Reconstruct the EDIT VOLUME only: one local add/cut/boolean positioned from
    # MODEL bbox / holes / cavities / location_hint — then fuse or cut with solid.
    return cq.Workplane("XY").newObject([solid])

importStep returns a Workplane. Do not newObject([wp]). Use .val() for the solid, .edges() on a Workplane.

CadQuery 2.8 compatibility (mandatory):
- Do NOT use `cq.Face.makePlane(..., normal=...)`. Build planar features on
  named XY/XZ/YZ Workplanes and extrude them instead.
- For reflection use a named plane, e.g.
  `solid.mirror(mirrorPlane="YZ", basePoint=(x, y, z))`; a normal-vector
  tuple is not a supported `mirrorPlane` value.
- Never call `solid.chamfer(distance)` or `solid.fillet(radius)` without an
  edge list. Use `cq.Workplane("XY").newObject([solid]).edges(selector)`
  followed by `.chamfer(distance)` or `.fillet(radius)`.
- Keep Shape and Workplane operands separate: before `solid.cut(...)` or
  `solid.fuse(...)`, convert a constructed Workplane with `.val()`. Do not use
  `.wrapped` on a Workplane, and use `.sort(...)` rather than `.sortBy(...)`.

Allowed (this is the reconstructive path):
- New sketches, extrudes, lofts, holes, bosses, handles, pin heads, hex profiles.
- Fuse or cut the new feature with the imported solid.
- Rebuild a LOCAL region from primitives, then boolean it onto the import.
- Polar/linear patterns of a feature you created.

Forbidden:
- Discard the import and replace the WHOLE part with a box/cylinder approximation.
- Invent an origin unrelated to MODEL bbox / holes / cavities / location_hint.
- Network, subprocess, threads, or extra filesystem use.
- `operations` lists every requested operation.  Apply the next unmet one;
  do not silently drop secondary operations in a compound instruction.

Preserve overall envelope unless the instruction changes size. Extra solids are
OK only if the instruction adds a body/feature; otherwise fuse into one solid.

Return JSON only, no markdown:
{"complete": false, "my_cad_function": "def my_cad_function(args):\\n ..."}

complete=true means the LAST executed solid already satisfies the instruction.
The first iteration can NEVER be complete.
"""


def generate_grounded_cadquery(
    client: LLMClient,
    *,
    instruction: str,
    geometry_brief: str,
    classified: dict[str, Any],
    location_hint: dict[str, Any] | None = None,
    failure: str = "",
    last_script: str = "",
    stdout: str = "",
    images: Optional[list[str]] = None,
    iteration: int = 0,
    visual_iters_remaining: int = 5,
    mode: str = "mutate",
    prior_candidates: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Autodesk-style script+render loop, grounded on OCC facts + edit-delta."""
    from groundedcad.llm.base import parse_json_loose

    system = RECONSTRUCT_CQ_SYSTEM if mode == "reconstruct" else GROUNDED_CQ_SYSTEM

    slots = {
        k: classified.get(k)
        for k in (
            "edit_type",
            "target_kind",
            "action",
            "diameter_mm",
            "radius_mm",
            "distance_mm",
            "count",
            "direction",
        )
        if classified.get(k) not in (None, "", [], {})
    }
    user = {
        "instruction": instruction,
        "slots": slots,
        "operations": classified.get("operations", []),
        "MODEL": geometry_brief,
        "location_hint": location_hint or {},
        "edit_delta": (failure or "")[:2400],
        "iteration": iteration,
        "iterations_remaining": visual_iters_remaining,
        "last_script": (last_script or "")[:4000],
        "program_output": (stdout or "")[:1500],
        "prior_candidates": (prior_candidates or [])[:2],
        "image_order": (
            "When images are attached, the canonical views of the original "
            "input model come first. Any images after those show the current "
            "best partial edit. Use the original views to identify the named "
            "feature and preserve it; use the current-edit views only to "
            "continue or correct work already applied."
        ),
        "note": (
            "First iteration complete must be false. "
            "mode=" + ("reconstruct" if mode == "reconstruct" else "mutate") + ". "
            "Always import the given STEP. It may be the original model or a "
            "validated partial edit from an earlier iteration; preserve and "
            "continue that partial edit rather than starting over. "
            + (
                "You may add/cut/sketch/pattern and rebuild a local region; do not replace the whole part."
                if mode == "reconstruct"
                else "If edit_delta is IDENTITY/OVERSIZED/EXTRA_BODIES, fix the script; do not redraw the part. "
                "For hole-edge blends, chamfer only the Fitting hole diameter (max 4 rims)."
            )
        ),
    }
    resp = client.complete(
        system=system,
        user=json.dumps(user, separators=(",", ":")),
        images=images or None,
        max_tokens=GROUNDED_CQ_MAX_TOKENS,
    )
    data: dict[str, Any] = {}
    try:
        parsed = parse_json_loose(resp.text or "")
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        data = {}
    script = str(data.get("my_cad_function") or "").strip() or extract_python_script(resp.text or "")
    script = repair_common_cadquery_api(script)
    complete = bool(data.get("complete"))
    if iteration == 0:
        complete = False
    return {"complete": complete, "my_cad_function": script, "raw": resp.text}
