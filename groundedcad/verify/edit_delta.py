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
    c0, c1 = before.get("center"), after.get("center")
    if c0 and c1 and len(c0) == 3 and len(c1) == 3:
        if any(abs(float(a) - float(b)) > 0.05 for a, b in zip(c0, c1)):
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


def dual_critic_failures(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    classified: Optional[ClassifiedEdit] = None,
) -> list[str]:
    """Geometric critic Autodesk's PNG loop does not have.

    identity → fail; huge volume/bbox change → fail; extra bodies when the
    instruction did not ask for a new solid → fail.
    """
    failures: list[str] = []
    if not after:
        return ["NO_GEOMETRY: execution produced no inspectable solid"]
    if is_identity(before, after):
        failures.append("IDENTITY_OUTPUT: pred matches the start solid; the edit was not applied")

    v0 = float(before.get("volume") or 0.0)
    v1 = float(after.get("volume") or 0.0)
    drop = (v0 - v1) / v0 if v0 > 1e-9 else 0.0
    if drop > 0.35:
        failures.append(f"OVERSIZED_CUT: removed {drop * 100:.0f}% of volume (rebuild, not an edit)")
    elif drop > 0.15:
        kind = classified.edit_type.value if classified else ""
        if kind in {"fillet_chamfer", "hole_edit", "feature_addition"}:
            failures.append(f"OVERSIZED_CUT: removed {drop * 100:.0f}% of volume on a local {kind}")

    s0, s1 = _size(before), _size(after)
    for name, a, b in zip("XYZ", s0, s1):
        if float(b) > 3.0 * max(float(a), 1e-6):
            failures.append(f"OVERSIZED_BBOX: {name} grew {a:.3f} → {b:.3f} (>3x); looks like a rebuild")

    n0 = int(before.get("n_solids") or 0)
    n1 = int(after.get("n_solids") or 0)
    allow_bodies = False
    if classified is not None:
        allow_bodies = classified.edit_type.value in {
            "feature_addition",
            "pattern",
            "feature_translation",
        }
    if n1 > n0 and not allow_bodies:
        failures.append(f"EXTRA_BODIES: solid count {n0} → {n1} but the instruction did not request a new body")
    failures.extend(slot_mismatch_failures(before, after, classified=classified))
    return failures


def locality_violation_failures(
    before: dict[str, Any],
    after: dict[str, Any],
    envelope: Optional[dict[str, Any]],
    *,
    classified: Optional[ClassifiedEdit] = None,
) -> list[str]:
    """Flag edits whose bbox change far exceeds the target locality envelope."""
    if not envelope or not after or not before:
        return []
    if is_identity(before, after):
        return []
    kind = classified.edit_type.value if classified else ""
    if kind not in {"fillet_chamfer", "hole_edit"}:
        return []
    s0, s1 = _size(before), _size(after)
    env_span = max(
        float(envelope.get("xmax", 0)) - float(envelope.get("xmin", 0)),
        float(envelope.get("ymax", 0)) - float(envelope.get("ymin", 0)),
        float(envelope.get("zmax", 0)) - float(envelope.get("zmin", 0)),
        1e-6,
    )
    # Local blend/hole should keep global bbox nearly fixed.
    for a, b in zip(s0, s1):
        if abs(float(b) - float(a)) > max(0.5 * env_span, 0.5):
            return [
                f"LOCALITY_VIOLATION: bbox change {s0}→{s1} exceeds target envelope span {env_span:.3f}"
            ]
    return []


def score_expected_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    expected_delta: dict[str, str],
) -> int:
    """Count instruction-specific geometric delta predicates that are satisfied."""
    if not before or not after or not expected_delta:
        return 0
    score = 0
    v0 = float(before.get("volume") or 0.0)
    v1 = float(after.get("volume") or 0.0)
    signed = (v1 - v0) / v0 if v0 > 1e-9 else 0.0
    volume_rule = expected_delta.get("volume")
    if volume_rule == "decrease_small" and -0.05 < signed < -1e-8:
        score += 1
    elif volume_rule == "decrease" and signed < -1e-8:
        score += 1
    elif volume_rule == "increase" and signed > 1e-8:
        score += 1
    elif volume_rule == "change" and abs(signed) > 1e-8:
        score += 1

    s0, s1 = _size(before), _size(after)
    ratios = [
        float(b) / max(float(a), 1e-9)
        for a, b in zip(s0, s1)
    ]
    bbox_rule = expected_delta.get("bbox")
    if bbox_rule == "unchanged" and all(abs(r - 1.0) < 0.01 for r in ratios):
        score += 1
    elif bbox_rule == "unchanged_or_shrink" and all(r <= 1.01 for r in ratios):
        score += 1
    elif bbox_rule == "possibly_increase" and all(r < 2.0 for r in ratios):
        score += 1

    f0 = int(before.get("n_faces") or 0)
    f1 = int(after.get("n_faces") or 0)
    face_rule = expected_delta.get("faces")
    if face_rule == "increase" and f1 > f0:
        score += 1
    elif face_rule == "change" and f1 != f0:
        score += 1
    return score


def _hole_diameters(census: dict[str, Any]) -> list[float]:
    holes = census.get("hole_candidates") or census.get("holes") or []
    out: list[float] = []
    for h in holes:
        if h.get("diameter"):
            out.append(float(h["diameter"]))
        elif h.get("radius"):
            out.append(2.0 * float(h["radius"]))
    return out


def slot_mismatch_failures(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    classified: Optional[ClassifiedEdit] = None,
) -> list[str]:
    """OCC vs typed slots — retry signal, never a visual/LLM accept."""
    if classified is None or not after:
        return []
    fails: list[str] = []
    kind = classified.edit_type.value
    d0, d1 = _hole_diameters(before), _hole_diameters(after)
    want_d = classified.diameter_mm
    if want_d is not None:
        close = any(abs(d - want_d) <= 0.25 for d in d1)
        if kind in {"hole_edit", "feature_addition"} and classified.action == "add":
            if len(d1) <= len(d0) and not close:
                fails.append(
                    f"SLOT_MISMATCH: requested new hole Ø{want_d} mm; "
                    f"hole_count {len(d0)}→{len(d1)} max_d={max(d1) if d1 else None}"
                )
        elif d1 and not close:
            still = max(d1)
            prev = max(d0) if d0 else None
            if prev is None or abs(still - prev) < 0.05:
                fails.append(
                    f"SLOT_MISMATCH: hole diameter still {still:.3f} mm, instruction asked Ø{want_d} mm"
                )
    want_r = classified.radius_mm or classified.distance_mm
    if kind == "fillet_chamfer" and want_r and not is_identity(before, after):
        f0, f1 = int(before.get("n_faces") or 0), int(after.get("n_faces") or 0)
        if f0 and f1 and f1 <= f0:
            fails.append(
                f"SLOT_MISMATCH: blend {want_r} mm did not add faces ({f0}→{f1}); wrong edges or no-op blend"
            )
    if kind == "dimension_change" and classified.factor and classified.factor > 1.05:
        s0, s1 = _size(before), _size(after)
        grew = any(float(b) > 1.5 * max(float(a), 1e-6) for a, b in zip(s0, s1))
        if not grew:
            fails.append(
                f"SLOT_MISMATCH: scale factor {classified.factor} but bbox stayed {s0} → {s1}"
            )
    if kind == "feature_addition" and is_identity(before, after):
        fails.append("SLOT_MISMATCH: feature_addition produced identity; add/cut/fuse on the imported STEP")
    return fails


def failure_bucket(failures: list[str]) -> str:
    """Map dual-critic codes to actionable rewrite buckets."""
    joined = " | ".join(failures or [])
    upper = joined.upper()
    if "INCOMPLETE_PLAN" in upper:
        return "INCOMPLETE"
    if "IDENTITY_OUTPUT" in upper or "NO_GEOMETRY" in upper:
        return "NO_OP"
    if "OVERSIZED_CUT" in upper or "OVERSIZED_BBOX" in upper or "EXTRA_BODIES" in upper:
        return "OVER_EDIT"
    if "SLOT_MISMATCH" in upper or "LOCALITY_VIOLATION" in upper or "WRONG_SCOPE" in upper:
        return "WRONG_SCOPE"
    if "OCC" in upper or "NO CIRCULAR" in upper or "CHAMFER" in upper or "FILLET" in upper:
        if "FAILED" in upper or "ERROR" in upper or "NO CIRCULAR" in upper:
            return "OCC_FAIL"
    if any("error" in (f or "").lower() for f in (failures or [])):
        return "OCC_FAIL"
    return "NO_OP" if failures else "OK"


def _delta_lines(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> list[str]:
    if not before or not after:
        return []
    lines: list[str] = []
    f0, f1 = int(before.get("n_faces") or 0), int(after.get("n_faces") or 0)
    if f0 or f1:
        lines.append(f"faces {f0}→{f1}")
    v0, v1 = float(before.get("volume") or 0), float(after.get("volume") or 0)
    if v0 > 1e-9:
        lines.append(f"volume_delta {(v1 - v0) / v0 * 100:.2f}%")
    s0, s1 = _size(before), _size(after)
    if any(abs(a - b) > 1e-3 for a, b in zip(s0, s1)):
        lines.append(f"bbox {s0}→{s1}")
    else:
        lines.append("bbox unchanged")
    return lines


def delta_summary_dict(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Structured face/volume/bbox delta for one attempt — same facts as
    _delta_lines() but as a dict, for attempt-history logging/prompting
    instead of a formatted string."""
    if not before or not after:
        return {}
    f0, f1 = int(before.get("n_faces") or 0), int(after.get("n_faces") or 0)
    v0, v1 = float(before.get("volume") or 0), float(after.get("volume") or 0)
    s0, s1 = _size(before), _size(after)
    return {
        "faces_before": f0,
        "faces_after": f1,
        "volume_delta_pct": round((v1 - v0) / v0 * 100, 2) if v0 > 1e-9 else 0.0,
        "bbox_before": [round(x, 3) for x in s0],
        "bbox_after": [round(x, 3) for x in s1],
    }


def format_retry_feedback(
    *,
    instruction: str,
    failures: list[str],
    observed: Optional[list[str]] = None,
    history: Optional[list[str]] = None,
    attempt: int = 0,
    before: Optional[dict[str, Any]] = None,
    after: Optional[dict[str, Any]] = None,
    classified: Optional[ClassifiedEdit] = None,
) -> str:
    """Text the CadQuery writer sees on the next loop. Failures stay in the loop."""
    bucket = failure_bucket(failures)
    target = ""
    if classified is not None:
        target = f"{classified.edit_type.value}/{classified.target_kind}/{classified.action}"
    lines = [
        f"RETRY attempt={attempt}: previous solid FAILED. complete must be false. Rewrite my_cad_function.",
        "instruction: " + (instruction or "")[:240],
        f"failure_bucket: {bucket}",
        "failures: " + " | ".join(failures[:8]),
    ]
    if target:
        lines.append(f"classified_target: {target}")
    deltas = _delta_lines(before, after)
    if deltas:
        lines.append("geometry_delta: " + "; ".join(deltas))
    if observed:
        lines.append("observed_vs_start: " + "; ".join(observed[:8]))
    if history:
        lines.append("prior_attempts: " + " || ".join(history[-4:]))
    if bucket == "NO_OP":
        lines.append(
            "required_action: no-op detected — must alter geometry near the classified target; "
            "import args['input_file']; do not return the start STEP unchanged."
        )
    elif bucket == "OVER_EDIT":
        lines.append(
            "required_action: narrow scope / preserve unrelated geometry; "
            "one local boolean or blend on the imported STEP only."
        )
    elif bucket == "WRONG_SCOPE":
        lines.append(
            "required_action: rebind to the correct hole/edge candidates from the census; "
            "do not chamfer unrelated circular edges."
        )
    elif bucket == "OCC_FAIL":
        lines.append(
            "required_action: reduce blend distance under 0.28× rim radius or select fewer legal edges; "
            "keep the same target family."
        )
    elif bucket == "INCOMPLETE":
        lines.append(
            "required_action: use census fitting-hole diameter and parsed dims; "
            "do not invent a whole-part rebuild."
        )
    else:
        lines.append(
            "required_action: import args['input_file']; apply the instruction; "
            "do not copy the start STEP; do not replace the whole part with a box."
        )
    return "\n".join(lines)


def dual_critic_accept(
    failures: list[str],
    *,
    iteration: int,
    cheap_high_conf: bool = False,
    visual: bool = False,
) -> bool:
    """Cheap high-confidence local tools may accept on iter 0.
    The CadQuery visual loop cannot accept on its first executed script
    (same rule as Autodesk: first iter is never complete)."""
    if failures:
        return False
    if visual:
        return iteration > 0
    if iteration == 0:
        return cheap_high_conf
    return True


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
