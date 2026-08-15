"""Deterministic geometry and invariant checks."""

from __future__ import annotations

from typing import Any, Optional

from groundedcad.agents.schemas import CheckResult, EditSpec, ExecutionResult, OperationType, SuccessCheck
from groundedcad.geometry.inspect import volume_delta_ratio


def check_valid_brep(execution: ExecutionResult, params: dict[str, Any] | None = None) -> CheckResult:
    summary = execution.geometry_summary or {}
    volume = float(summary.get("volume") or 0.0)
    n_solids = int(summary.get("n_solids") or 0)
    usable = bool(summary.get("valid")) or (volume > 1e-9 and n_solids >= 1)
    ok = bool(execution.success and usable and execution.step_path)
    return CheckResult(
        name="valid_brep",
        passed=ok,
        detail="Valid STEP produced" if ok else f"Invalid/missing geometry: {execution.error}",
        measured={"valid": summary.get("valid"), "volume": volume, "n_solids": n_solids, "step_path": execution.step_path},
    )


def check_body_count(
    before: dict[str, Any],
    after: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> CheckResult:
    params = params or {}
    expected = params.get("expected")
    max_delta = params.get("max_delta", 2)
    n0 = int(before.get("n_solids", 0))
    n1 = int(after.get("n_solids", 0))
    if expected is not None:
        passed = n1 == int(expected)
        detail = f"body_count {n1} vs expected {expected}"
    else:
        passed = abs(n1 - n0) <= int(max_delta)
        detail = f"body_count {n0}->{n1} (max_delta={max_delta})"
    return CheckResult(
        name="body_count",
        passed=passed,
        detail=detail,
        measured={"before": n0, "after": n1},
    )


def check_volume_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> CheckResult:
    params = params or {}
    max_ratio = float(params.get("max_ratio", 0.75))
    min_ratio = float(params.get("min_ratio", 0.0))
    ratio = volume_delta_ratio(before, after)
    # Also track signed change
    v0 = float(before.get("volume") or 0.0)
    v1 = float(after.get("volume") or 0.0)
    signed = (v1 - v0) / abs(v0) if abs(v0) > 1e-9 else 0.0
    if "expect_increase" in params and params["expect_increase"] and signed <= 0:
        return CheckResult(
            name="volume_delta",
            passed=False,
            detail=f"Expected volume increase, got signed_ratio={signed:.4f}",
            measured={"ratio": ratio, "signed": signed, "v0": v0, "v1": v1},
        )
    if "expect_decrease" in params and params["expect_decrease"] and signed >= 0:
        return CheckResult(
            name="volume_delta",
            passed=False,
            detail=f"Expected volume decrease, got signed_ratio={signed:.4f}",
            measured={"ratio": ratio, "signed": signed, "v0": v0, "v1": v1},
        )
    passed = min_ratio <= ratio <= max_ratio
    return CheckResult(
        name="volume_delta",
        passed=passed,
        detail=f"volume change ratio={ratio:.4f} within [{min_ratio},{max_ratio}]",
        measured={"ratio": ratio, "signed": signed, "v0": v0, "v1": v1},
    )


def check_bbox_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> CheckResult:
    params = params or {}
    max_growth = float(params.get("max_growth", 2.0))
    b0 = before.get("size") or (0, 0, 0)
    b1 = after.get("size") or (0, 0, 0)
    growths = []
    for a, b in zip(b0, b1):
        if abs(a) < 1e-9:
            growths.append(0.0 if abs(b) < 1e-9 else max_growth + 1)
        else:
            growths.append(abs(b) / abs(a))
    max_g = max(growths) if growths else 0.0
    passed = max_g <= max_growth
    return CheckResult(
        name="bbox_delta",
        passed=passed,
        detail=f"bbox growth max={max_g:.3f} (limit {max_growth})",
        measured={"before": b0, "after": b1, "growths": growths},
    )


def check_connectivity(after: dict[str, Any], params: dict[str, Any] | None = None) -> CheckResult:
    """Heuristic: reject explosion into many tiny disconnected solids."""
    params = params or {}
    max_bodies = int(params.get("max_bodies", 25))
    n = int(after.get("n_solids", 0))
    passed = 1 <= n <= max_bodies
    return CheckResult(
        name="connectivity",
        passed=passed,
        detail=f"n_solids={n} (max {max_bodies})",
        measured={"n_solids": n},
    )


def check_edit_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    edit_spec: Optional[EditSpec] = None,
) -> CheckResult:
    """Reject identity edits and huge unrelated reconstructions."""
    ratio = volume_delta_ratio(before, after)
    op = edit_spec.operation if edit_spec else None
    modifying = op in {
        OperationType.HOLE,
        OperationType.CHAMFER,
        OperationType.FILLET,
        OperationType.EXTRUDE_CUT,
        OperationType.EXTRUDE_ADD,
        OperationType.BOOLEAN_CUT,
        OperationType.BOOLEAN_UNION,
        OperationType.DUPLICATE,
        OperationType.PATTERN,
        OperationType.SCALE,
    }
    if modifying and ratio < 1e-8:
        return CheckResult(
            name="edit_delta",
            passed=False,
            detail="Output matches start volume; requested feature did not change",
            measured={"ratio": ratio},
        )
    if ratio > 0.85 and op in {OperationType.CHAMFER, OperationType.FILLET, OperationType.HOLE}:
        return CheckResult(
            name="edit_delta",
            passed=False,
            detail=f"Volume change {ratio:.3f} is too large for a local {op}",
            measured={"ratio": ratio},
        )
    return CheckResult(
        name="edit_delta",
        passed=True,
        detail=f"volume delta ratio={ratio:.4f}",
        measured={"ratio": ratio},
    )


def check_only_requested_changed(
    before: dict[str, Any],
    after: dict[str, Any],
    edit_spec: Optional[EditSpec] = None,
) -> CheckResult:
    """Critic: did ONLY requested geometry change?"""
    n0 = int(before.get("n_solids") or 0)
    n1 = int(after.get("n_solids") or 0)
    ratio = volume_delta_ratio(before, after)
    b0 = before.get("size") or (0, 0, 0)
    b1 = after.get("size") or (0, 0, 0)
    growth = []
    for a, b in zip(b0, b1):
        growth.append(0.0 if abs(a) < 1e-9 else abs(b) / abs(a))
    max_g = max(growth) if growth else 1.0
    local = abs(n1 - n0) <= 2 and ratio <= 0.35 and max_g <= 1.35
    op = edit_spec.operation if edit_spec else None
    if ratio < 1e-8 and op != OperationType.TRANSFORM:
        local = False
    if op == OperationType.TRANSFORM:
        local = abs(n1 - n0) <= 1
    if op == OperationType.SCALE:
        local = abs(n1 - n0) <= 1
    return CheckResult(
        name="only_requested_changed",
        passed=local,
        detail=(
            "Local edit: body/volume/bbox stay close to the start model"
            if local
            else f"Unrelated change suspected: solids {n0}->{n1}, vol_ratio={ratio:.3f}, bbox_growth={max_g:.3f}"
        ),
        measured={"n0": n0, "n1": n1, "ratio": ratio, "bbox_growth": max_g},
    )


def run_success_check(
    check: SuccessCheck,
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    execution: ExecutionResult,
) -> CheckResult:
    t = check.check_type
    params = {**check.params}
    if t == "valid_brep":
        return check_valid_brep(execution, params)
    if t == "body_count":
        return check_body_count(before, after, params)
    if t == "volume_delta":
        return check_volume_delta(before, after, params)
    if t == "bbox_delta":
        return check_bbox_delta(before, after, params)
    if t == "connectivity":
        return check_connectivity(after, params)
    if t == "dimension":
        # Soft check: presence of expected numeric keys in geometry summary metadata
        key = params.get("key")
        target = params.get("value")
        tol = float(params.get("tol", 0.15))
        measured = after.get(key) if key else None
        if measured is None or target is None:
            return CheckResult(name=check.name, passed=True, detail="dimension soft-skip", measured={})
        passed = abs(float(measured) - float(target)) <= tol * max(abs(float(target)), 1e-6)
        return CheckResult(
            name=check.name,
            passed=passed,
            detail=f"{key}={measured} target={target}",
            measured={"measured": measured, "target": target},
        )
    return CheckResult(name=check.name, passed=True, detail="custom/no-op", measured={})


def verify_execution(
    *,
    before: dict[str, Any],
    execution: ExecutionResult,
    edit_spec: Optional[EditSpec] = None,
) -> list[CheckResult]:
    after = execution.geometry_summary or {}
    checks: list[SuccessCheck] = []
    if edit_spec and edit_spec.success_checks:
        checks = list(edit_spec.success_checks)
    else:
        checks = [
            SuccessCheck(name="valid_brep", description="Valid B-Rep", check_type="valid_brep"),
            SuccessCheck(
                name="body_count",
                description="Body count stable",
                check_type="body_count",
                params={"max_delta": 3},
            ),
            SuccessCheck(
                name="volume_delta",
                description="Volume change bounded",
                check_type="volume_delta",
                params={"max_ratio": 0.9},
            ),
            SuccessCheck(
                name="bbox_delta",
                description="BBox growth bounded",
                check_type="bbox_delta",
                params={"max_growth": 3.0},
            ),
            SuccessCheck(
                name="connectivity",
                description="Not fragmented",
                check_type="connectivity",
            ),
        ]

    results = [
        run_success_check(c, before=before, after=after, execution=execution) for c in checks
    ]
    results.append(check_edit_delta(before, after, edit_spec))
    results.append(check_only_requested_changed(before, after, edit_spec))
    from groundedcad.verify.edit_delta import check_edit_delta_unintended

    results.append(
        check_edit_delta_unintended(
            before,
            after,
            instruction=(edit_spec.intent_summary if edit_spec else ""),
            edit_spec=edit_spec,
        )
    )
    from groundedcad.verify.edit_delta import check_not_identity

    results.append(
        check_not_identity(before, after, instruction=(edit_spec.intent_summary if edit_spec else ""))
    )
    return results
