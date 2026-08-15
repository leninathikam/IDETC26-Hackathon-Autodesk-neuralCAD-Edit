"""Edit-delta validator: START → PRED vs requested change.

Rejects extra envelope/thickness/body changes even when the part still 'looks ok'.
"""

from __future__ import annotations

from typing import Any, Optional

from groundedcad.agents.classifier import ClassifiedEdit, classify_instruction
from groundedcad.agents.schemas import CheckResult, EditSpec, OperationType
from groundedcad.geometry.inspect import volume_delta_ratio


def is_identity(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """True when pred is geometrically the start solid (no-op)."""
    if not after:
        return True
    if volume_delta_ratio(before, after) > 1e-5:
        return False
    s0, s1 = _size(before), _size(after)
    for a, b in zip(s0, s1):
        if abs(b - a) > 1e-3 + 1e-4 * max(abs(a), 1e-6):
            return False
    if int(before.get("n_solids") or 0) != int(after.get("n_solids") or 0):
        return False
    f0, f1 = int(before.get("n_faces") or 0), int(after.get("n_faces") or 0)
    if f0 and f1 and abs(f1 - f0) > 2:
        return False
    return True


def identity_failure_report(instruction: str) -> dict[str, str]:
    return {
        "execution_status": "success",
        "edit_status": "FAILED",
        "failure_type": "IDENTITY_OUTPUT",
        "message": (
            "The generated CAD is geometrically identical to the starting model. "
            "The requested edit was not applied."
        ),
        "required_action": (
            "Generate an actual geometric modification. "
            "Do not return the imported starting STEP unchanged."
        ),
        "instruction": instruction[:240],
    }


def _size(census: dict[str, Any]) -> tuple[float, float, float]:
    s = census.get("size") or (0.0, 0.0, 0.0)
    return (float(s[0]), float(s[1]), float(s[2]))


def _thickness(census: dict[str, Any]) -> float:
    return min(_size(census))


def observed_changes(before: dict[str, Any], after: dict[str, Any], frac: float = 0.02) -> list[str]:
    out: list[str] = []
    n0 = int(before.get("n_solids") or 0)
    n1 = int(after.get("n_solids") or 0)
    if n0 != n1:
        out.append(f"solid_count {n0} → {n1}")
    s0, s1 = _size(before), _size(after)
    names = ("X", "Y", "Z")
    for i, name in enumerate(names):
        a, b = s0[i], s1[i]
        if abs(a) < 1e-9:
            continue
        if abs(b - a) / abs(a) > frac and abs(b - a) > 0.15:
            out.append(f"{name} size {a:.3f} → {b:.3f}")
    t0, t1 = _thickness(before), _thickness(after)
    if t0 > 1e-6 and abs(t1 - t0) / t0 > frac and abs(t1 - t0) > 0.15:
        out.append(f"plate thickness {t0:.3f} → {t1:.3f}")
    h0 = before.get("hole_candidates") or before.get("holes") or []
    h1 = after.get("hole_candidates") or after.get("holes") or []
    if abs(len(h1) - len(h0)) >= 1:
        out.append(f"hole_count {len(h0)} → {len(h1)}")
    d0 = sorted(float(h.get("diameter") or 2 * float(h.get("radius") or 0)) for h in h0 if h.get("diameter") or h.get("radius"))
    d1 = sorted(float(h.get("diameter") or 2 * float(h.get("radius") or 0)) for h in h1 if h.get("diameter") or h.get("radius"))
    if d0 and d1 and abs(max(d1) - max(d0)) > 0.2:
        out.append(f"max hole diameter {max(d0):.3f} → {max(d1):.3f}")
    v0 = float(before.get("volume") or 0)
    v1 = float(after.get("volume") or 0)
    if v0 > 1e-9 and abs(v1 - v0) / v0 > 0.005:
        out.append(f"volume {v0:.4f} → {v1:.4f}")
    return out


def requested_changes(edit: Optional[ClassifiedEdit], edit_spec: Optional[EditSpec]) -> list[str]:
    if edit is None and edit_spec is None:
        return ["unspecified"]
    tags: list[str] = []
    op = edit_spec.operation if edit_spec else None
    if edit:
        if edit.edit_type.value == "hole_edit":
            tags.append("hole diameter" if edit.action != "add" else "add hole")
            if edit.groove_mm:
                tags.append("groove/chamfer")
        elif edit.edit_type.value == "fillet_chamfer":
            tags.append("blend edges")
        elif edit.edit_type.value == "feature_translation":
            tags.append("move feature")
        elif edit.edit_type.value == "dimension_change":
            tags.append("overall size")
        elif edit.edit_type.value == "pattern":
            tags.append("repeat body")
        elif edit.edit_type.value == "feature_addition":
            tags.append("add feature")
        elif edit.edit_type.value == "boolean_modification":
            tags.append("local cut")
        else:
            tags.append(edit.edit_type.value)
    elif op:
        tags.append(op.value)
    return tags or ["unspecified"]


def unintended_changes(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    instruction: str = "",
    edit_spec: Optional[EditSpec] = None,
) -> list[str]:
    edit = classify_instruction(instruction) if instruction else None
    observed = observed_changes(before, after)
    requested = requested_changes(edit, edit_spec)
    op = edit_spec.operation if edit_spec else None
    local_ops = {
        OperationType.HOLE,
        OperationType.CHAMFER,
        OperationType.FILLET,
        OperationType.EXTRUDE_CUT,
        OperationType.BOOLEAN_CUT,
    }
    bad: list[str] = []
    allow_size = op in {OperationType.SCALE, OperationType.TRANSFORM, OperationType.PATTERN, OperationType.DUPLICATE, OperationType.EXTRUDE_ADD}
    if edit and edit.edit_type.value in {"dimension_change", "pattern", "feature_translation", "feature_addition"}:
        allow_size = True
    for ch in observed:
        if ch.startswith("plate thickness") and op in local_ops:
            bad.append(ch)
        elif ch.startswith(("X size", "Y size", "Z size")) and not allow_size:
            bad.append(ch)
        elif ch.startswith("solid_count") and op in {OperationType.HOLE, OperationType.CHAMFER, OperationType.FILLET}:
            bad.append(ch)
    return bad


def check_edit_delta_unintended(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    instruction: str = "",
    edit_spec: Optional[EditSpec] = None,
) -> CheckResult:
    bad = unintended_changes(before, after, instruction=instruction, edit_spec=edit_spec)
    obs = observed_changes(before, after)
    req = requested_changes(classify_instruction(instruction) if instruction else None, edit_spec)
    passed = not bad
    detail = (
        "Unintended geometry modification detected: " + "; ".join(bad)
        if bad
        else "No unintended envelope/thickness/body change"
    )
    return CheckResult(
        name="unintended_geometry",
        passed=passed,
        detail=detail,
        measured={"requested": req, "observed": obs, "unintended": bad},
    )


def check_not_identity(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    instruction: str = "",
) -> CheckResult:
    ident = is_identity(before, after)
    needs_edit = bool((instruction or "").strip())
    passed = not (ident and needs_edit)
    return CheckResult(
        name="not_identity",
        passed=passed,
        detail="Identity/no-op vs start" if ident else "Geometry differs from start",
        measured={"identity": ident},
    )
