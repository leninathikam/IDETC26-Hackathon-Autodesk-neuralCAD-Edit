"""Verification agent: instruction, preservation, dims, position, count, topology, magnitude."""

from __future__ import annotations

import json
from typing import Any, Optional

from groundedcad.agents.policy import CRITIC_SYSTEM
from groundedcad.agents.schemas import (
    Critique,
    EditSpec,
    ExecutionResult,
    GroundedIntent,
    OperationType,
    VerificationDecision,
)
from groundedcad.geometry.inspect import volume_delta_ratio
from groundedcad.llm.base import LLMClient
from groundedcad.verify.geometric import verify_execution


def _check_map(checks) -> dict[str, Any]:
    return {c.name: c for c in checks}


def deterministic_verification(
    *,
    before: dict[str, Any],
    execution: ExecutionResult,
    edit_spec: Optional[EditSpec] = None,
    intent: Optional[GroundedIntent] = None,
) -> VerificationDecision:
    checks = verify_execution(before=before, execution=execution, edit_spec=edit_spec)
    by_name = _check_map(checks)
    after = execution.geometry_summary or {}
    ratio = volume_delta_ratio(before, after)
    n0 = int(before.get("n_solids") or 0)
    n1 = int(after.get("n_solids") or 0)
    h0 = len(before.get("holes") or [])
    h1 = len(after.get("holes") or [])

    topology_error = (not execution.success) or (
        by_name.get("valid_brep") is not None and not by_name["valid_brep"].passed
    ) or (by_name.get("connectivity") is not None and not by_name["connectivity"].passed)
    unintended = by_name.get("only_requested_changed") is not None and not by_name["only_requested_changed"].passed
    if by_name.get("unintended_geometry") is not None and not by_name["unintended_geometry"].passed:
        unintended = True
    if ratio > 0.5:
        unintended = True

    dim_error = False
    pos_error = False
    count_error = False
    dims = (intent.dimensions if intent else {}) or {}
    op = edit_spec.operation if edit_spec else None

    if op == OperationType.HOLE and "diameter" in dims and execution.success:
        # Count: adding one hole should increase circular features, not explode solids.
        if abs(n1 - n0) > 1:
            count_error = True
        if h1 + 2 < h0:
            count_error = True
        drilled = next((t for t in execution.tool_calls if t.tool_name == "drill_hole_at_point"), None)
        if drilled:
            d = float(drilled.arguments.get("diameter") or 0)
            if abs(d - float(dims["diameter"])) > 0.05 * max(float(dims["diameter"]), 1e-6) + 0.05:
                dim_error = True
            bbox = before.get("bbox") or {}
            if bbox:
                cx = 0.5 * (float(bbox["xmin"]) + float(bbox["xmax"]))
                cy = 0.5 * (float(bbox["ymin"]) + float(bbox["ymax"]))
                x = float(drilled.arguments.get("x") or 0)
                y = float(drilled.arguments.get("y") or 0)
                span = max(abs(float(bbox["xmax"]) - float(bbox["xmin"])), 1e-6)
                if (x - cx) ** 2 + (y - cy) ** 2 > (0.75 * span) ** 2:
                    pos_error = True

    instruction_satisfied = bool(execution.success and not topology_error)
    if op in {OperationType.HOLE, OperationType.CHAMFER, OperationType.FILLET, OperationType.EXTRUDE_CUT} and ratio < 1e-8:
        instruction_satisfied = False
    if dim_error or count_error:
        instruction_satisfied = False
    if by_name.get("not_identity") is not None and not by_name["not_identity"].passed:
        instruction_satisfied = False
        unintended = True

    if topology_error or ratio > 0.5:
        severity: str = "major"
    elif dim_error or pos_error or unintended or count_error or not instruction_satisfied:
        severity = "minor"
    else:
        severity = "none"

    correct = instruction_satisfied and not unintended and not dim_error and not pos_error and not topology_error and not count_error
    if correct:
        severity = "none"

    failed = [c for c in checks if not c.passed]
    diagnosis_parts = [c.detail for c in failed]
    if dim_error:
        diagnosis_parts.append("Requested dimension does not match the tool/measurement.")
    if pos_error:
        diagnosis_parts.append("Edited feature is far from the original part/reference.")
    if count_error:
        diagnosis_parts.append("Affected feature count does not match the instruction.")
    diagnosis = " ".join(diagnosis_parts) or "Local edit matches instruction within geometric checks."

    if correct:
        specific_fix = ""
    elif topology_error:
        specific_fix = "Keep the original solid valid; apply a smaller local boolean/blend. Do not rebuild the part."
    elif dim_error and "diameter" in dims:
        specific_fix = f"Re-apply the hole/feature using diameter={dims['diameter']} from the instruction."
    elif by_name.get("not_identity") is not None and not by_name["not_identity"].passed:
        specific_fix = json.dumps(
            {
                "execution_status": "success",
                "edit_status": "FAILED",
                "failure_type": "IDENTITY_OUTPUT",
                "message": "The generated CAD is geometrically identical to the starting model. The requested edit was not applied.",
                "required_action": "Generate an actual geometric modification. Do not return the imported starting STEP unchanged.",
            }
        )
    else:
        specific_fix = "Apply the smallest parameter tweak that satisfies the instruction."

    return VerificationDecision(
        correct=correct,
        instruction_satisfied=instruction_satisfied,
        unintended_changes=unintended,
        dimension_error=dim_error,
        position_error=pos_error,
        topology_error=topology_error,
        severity=severity,  # type: ignore[arg-type]
        diagnosis=diagnosis,
        specific_fix=specific_fix,
    )


def decision_to_critique(decision: VerificationDecision, execution: ExecutionResult, checks) -> Critique:
    passed = [c for c in checks if c.passed]
    failed = [c for c in checks if not c.passed]
    accept = bool(decision.correct and execution.success)
    advice = decision.specific_fix if not accept else ""
    if decision.diagnosis and not accept:
        advice = f"{decision.diagnosis} | Fix: {decision.specific_fix}".strip(" |")
    # Scores follow edit hierarchy, not visual resemblance.
    instr = 1.0
    if decision.instruction_satisfied:
        instr = 4.0
    if decision.instruction_satisfied and not decision.unintended_changes:
        instr = 5.5
    if accept:
        instr = 6.0
    if decision.unintended_changes:
        instr = min(instr, 3.0)
    qual = 5.0 if accept else (2.0 if decision.severity == "major" else 3.0)
    if decision.unintended_changes:
        qual = min(qual, 2.5)
    return Critique(
        accept=accept,
        instruction_score=instr,
        quality_score=qual,
        failed_checks=failed,
        passed_checks=passed,
        revision_advice=advice,
        evidence=[
            "hierarchy: instruction > feature > preserve > dims/location > topology > visual",
            decision.diagnosis,
            decision.specific_fix,
        ],
        verification=decision,
    )


def run_verification_agent(
    *,
    before: dict[str, Any],
    execution: ExecutionResult,
    edit_spec: Optional[EditSpec] = None,
    intent: Optional[GroundedIntent] = None,
    instruction: str = "",
    client: Optional[LLMClient] = None,
    use_llm: bool = False,
) -> Critique:
    checks = verify_execution(before=before, execution=execution, edit_spec=edit_spec)
    decision = deterministic_verification(before=before, execution=execution, edit_spec=edit_spec, intent=intent)
    critique = decision_to_critique(decision, execution, checks)
    if not use_llm or client is None:
        return critique
    return llm_verification(
        client,
        instruction=instruction or (intent.raw_instruction if intent else ""),
        intent=intent,
        edit_spec=edit_spec,
        execution=execution,
        before=before,
        base=critique,
        decision=decision,
    )


def llm_verification(
    client: LLMClient,
    *,
    instruction: str,
    intent: Optional[GroundedIntent],
    edit_spec: Optional[EditSpec],
    execution: ExecutionResult,
    before: dict[str, Any],
    base: Critique,
    decision: VerificationDecision,
) -> Critique:
    payload = {
        "instruction": instruction,
        "parsed": {
            "target": intent.target_text if intent else "",
            "location": intent.location if intent else "",
            "dimensions": intent.dimensions if intent else {},
            "preserve": intent.preserve if intent else [],
        },
        "original": {
            "volume": before.get("volume"),
            "size": before.get("size"),
            "n_solids": before.get("n_solids"),
            "n_faces": before.get("n_faces"),
            "n_holes": len(before.get("holes") or []),
        },
        "edited": {
            "volume": (execution.geometry_summary or {}).get("volume"),
            "size": (execution.geometry_summary or {}).get("size"),
            "n_solids": (execution.geometry_summary or {}).get("n_solids"),
            "n_faces": (execution.geometry_summary or {}).get("n_faces"),
            "n_holes": len((execution.geometry_summary or {}).get("holes") or []),
            "valid": (execution.geometry_summary or {}).get("valid"),
        },
        "deterministic_decision": decision.model_dump(),
        "views": list(execution.image_paths.keys()),
        "rule": (
            "EDITS not generations. Hierarchy: instruction > feature > preserve > dims/location > topology > visual. "
            "Views must not override preservation or dimension failures. "
            "If deterministic unintended_changes or topology_error is true, correct must be false."
        ),
    }
    try:
        resp = client.complete_json(
            system=CRITIC_SYSTEM,
            user=json.dumps(payload, indent=2),
            images=list(execution.image_paths.values())[:7] or None,
        )
        merged = VerificationDecision(
            correct=bool(resp.get("correct", decision.correct)),
            instruction_satisfied=bool(resp.get("instruction_satisfied", decision.instruction_satisfied)),
            unintended_changes=bool(resp.get("unintended_changes", decision.unintended_changes)),
            dimension_error=bool(resp.get("dimension_error", decision.dimension_error)),
            position_error=bool(resp.get("position_error", decision.position_error)),
            topology_error=bool(resp.get("topology_error", decision.topology_error)),
            severity=str(resp.get("severity") or decision.severity),  # type: ignore[arg-type]
            diagnosis=str(resp.get("diagnosis") or decision.diagnosis),
            specific_fix=str(resp.get("specific_fix") or decision.specific_fix),
        )
        if decision.topology_error:
            merged.correct = False
            merged.topology_error = True
        if decision.unintended_changes:
            merged.correct = False
            merged.unintended_changes = True
        if decision.dimension_error:
            merged.dimension_error = True
            merged.correct = False
        if merged.severity not in {"none", "minor", "major"}:
            merged.severity = decision.severity
        out = decision_to_critique(merged, execution, base.failed_checks + base.passed_checks)
        return out
    except Exception as exc:  # noqa: BLE001
        extra = base.model_copy()
        extra.revision_advice = (extra.revision_advice + f" | critic_error={exc}").strip()
        return extra
