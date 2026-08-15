"""LLM CadQuery only after a deterministic strategy fails. Keep the prompt small."""

from __future__ import annotations

import json
import re
from typing import Any

from groundedcad.llm.base import LLMClient

CADQUERY_SYSTEM = """You write CadQuery 2 that edits an existing STEP. Output Python only.

Required signature:
def my_cad_function(args):
    import cadquery as cq
    import os
    shape = cq.importers.importStep(os.path.expanduser(args["input_file"]))
    # mutate shape
    return shape

Rules:
- The imported solid MUST change. Returning it unchanged is failure.
- Use only cadquery, math, os. No network, no files except input_file.
- Apply numeric values from the instruction exactly (mm unless stated).
- Prefer transforms, holes, fillets, chamfers, and local booleans on the imported solid.
- Do not rebuild the part from scratch unless the instruction requires a new body.
- If the instruction refers to an EXISTING slot/hole/pocket/edge, find it in the
  geometry brief below (its face/edge center, radius, or bbox) and cut/extend
  exactly there. Do not place a new feature at an arbitrary/default location.
- location_hint (if present) gives real candidate coordinates from the STEP
  census: cavities to extend/cut, planar_sites for separate local cuts, or
  protrusions/bbox_corners to anchor a new small feature near. Use them
  instead of guessing a location. Never remove more than ~15% of total
  volume in a single cut.
"""


def extract_python_script(text: str) -> str:
    raw = (text or "").strip()
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


def generate_llm_cadquery(
    client: LLMClient,
    *,
    instruction: str,
    geometry_brief: str,
    classified: dict[str, Any],
    failure: str = "",
    location_hint: dict[str, Any] | None = None,
) -> str:
    payload = {
        "instruction": instruction,
        "classified": classified,
        "model": geometry_brief[:1800],
        "failure": (failure or "")[:500],
        "required_action": "Generate an actual geometric modification.",
    }
    if location_hint:
        payload["location_hint"] = location_hint
    resp = client.complete(
        system=CADQUERY_SYSTEM,
        user=json.dumps(payload, indent=2),
    )
    return extract_python_script(resp.text)
