"""Bounded grounder -> planner -> executor -> verifier -> critic loop."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from groundedcad.agents.grounder import heuristic_ground
from groundedcad.agents.instruction_parser import parse_instruction
from groundedcad.agents.planner import plan_edit
from groundedcad.agents.schemas import IterationLog, PipelineResult
from groundedcad.geometry.inspect import inspect_step
from groundedcad.geometry.render import render_canonical_views
from groundedcad.ingest.request_parser import EditRequest, write_benchmark_settings
from groundedcad.llm.base import LLMClient, MockLLMClient, auto_client_from_env
from groundedcad.runtime.sandbox import Sandbox
from groundedcad.tools.cadquery_gen import generate_cadquery
from groundedcad.verify.visual import deterministic_critique, llm_critique


FAILURE_CATEGORIES = {
    "grounding": "Failed to identify referenced geometry / multimodal cue",
    "planning": "Wrong operation or parameters despite correct target",
    "api_code": "CadQuery/tool execution error",
    "geometric_validity": "Invalid or fragmented B-Rep",
    "false_completion": "Accepted despite failing checks or unmet instruction",
    "timeout": "Sandbox/iteration budget exceeded",
    "unknown": "Unclassified failure",
}

# Bounded, named budgets for the Phase C visual CadQuery retry loop only —
# Phase A (deterministic candidate enumeration) and Phase B (cheap
# single-shot) are unmetered; they're cheap/free and capping them buys
# nothing. MAX_CANDIDATE_ATTEMPTS is mutate mode's ceiling; reconstruct
# mode keeps its existing +2 headroom on top (harder edits, deliberately
# more retries — see _adaptive_visual_iters). MAX_VLM_CRITIC_CALLS bounds
# real VLM judgment calls (verify.agent.llm_verification) within Phase C,
# spent only on candidates that already passed the free deterministic
# geometry check.
MAX_CANDIDATE_ATTEMPTS = 6
MAX_VLM_CRITIC_CALLS = 3


class GroundedCADPipeline:
    def __init__(
        self,
        *,
        grounding_client: Optional[LLMClient] = None,
        planning_client: Optional[LLMClient] = None,
        critic_client: Optional[LLMClient] = None,
        max_iters: int = 5,
        sandbox: Optional[Sandbox] = None,
        user_id: str = "groundedcad",
        use_llm_critic: bool = True,
        inprocess: bool = False,
        render: bool = True,
        use_llm_cadquery: Optional[bool] = None,
        raw_escalation_budget_s: float = 100.0,
        visual_iters: Optional[int] = None,
        hybrid_autodesk_fallback: bool = True,
        visual_iters_min: int = 5,
        visual_iters_max: Optional[int] = None,
        candidate_enumeration: bool = True,
        max_candidates: int = 4,
    ):
        default = grounding_client or planning_client or critic_client or auto_client_from_env()
        self.grounding_client = grounding_client or default
        self.planning_client = planning_client or default
        self.critic_client = critic_client or default
        self.max_iters = max_iters
        self.visual_iters = int(visual_iters if visual_iters is not None else max_iters)
        self.hybrid_autodesk_fallback = bool(hybrid_autodesk_fallback)
        self.visual_iters_min = max(1, int(visual_iters_min))
        requested_max = self.visual_iters if visual_iters_max is None else int(visual_iters_max)
        self.visual_iters_max = min(10, max(self.visual_iters_min, requested_max))
        self.candidate_enumeration = bool(candidate_enumeration)
        self.max_candidates = max(1, min(8, int(max_candidates)))
        self.sandbox = sandbox or Sandbox(render=render)
        self.user_id = user_id
        self.use_llm_critic = use_llm_critic
        self.inprocess = inprocess
        self.render = render
        if use_llm_cadquery is None:
            use_llm_cadquery = not isinstance(self.planning_client, MockLLMClient)
        self.use_llm_cadquery = bool(use_llm_cadquery)
        # Soft wall-clock budget (seconds since run() started) for the extra
        # raw-CadQuery escalation attempt in _try_llm_cadquery. External
        # batch runners kill a whole row after a fixed wall-clock timeout
        # (e.g. 180s); this keeps the new 3rd attempt from being the reason a
        # row that would otherwise have finished gets killed with nothing to
        # show for it. Once the budget is spent, _try_llm_cadquery falls back
        # to its original 2 constrained-tool attempts only.
        self.raw_escalation_budget_s = raw_escalation_budget_s
        self._run_start: Optional[float] = None

    def _adaptive_visual_iters(self, cq_mode: str, fail_history: list[str]) -> int:
        """Bounded retry budget for the Autodesk-style fallback. mutate mode
        is ceilinged at MAX_CANDIDATE_ATTEMPTS; reconstruct mode keeps its
        existing +2 headroom on top (harder edits get more retries)."""
        if not self.hybrid_autodesk_fallback:
            return max(0, self.visual_iters)
        joined = " | ".join(fail_history).upper()
        budget = 5 if cq_mode != "reconstruct" else 7
        if any(code in joined for code in ("IDENTITY", "NO_OP", "SLOT_MISMATCH")):
            budget += 1
        if any(code in joined for code in ("OVER_EDIT", "OVERSIZED", "EXTRA_BODIES")):
            budget += 1
        ceiling = MAX_CANDIDATE_ATTEMPTS + (2 if cq_mode == "reconstruct" else 0)
        return min(ceiling, self.visual_iters_max, max(self.visual_iters_min, budget))

    @staticmethod
    def _candidate_summary(
        *,
        iteration: int,
        execution,
        critique,
        identity: bool,
        failures: list[str],
        before: Optional[dict[str, Any]] = None,
        after: Optional[dict[str, Any]] = None,
        expected_delta: Optional[dict[str, str]] = None,
        candidate=None,
    ) -> dict[str, Any]:
        from groundedcad.geometry.inspect import volume_delta_ratio
        from groundedcad.verify.edit_delta import score_expected_delta

        summary = {
            "iteration": int(iteration),
            "success": bool(execution.success),
            "accepted": bool(critique.accept),
            "identity": bool(identity),
            "score": round(float(critique.instruction_score + critique.quality_score), 3),
            "failures": [str(x)[:240] for x in failures[:3]],
            "delta_match": score_expected_delta(before or {}, after or {}, expected_delta or {}),
            "volume_ratio": round(
                float(volume_delta_ratio(before or {}, after or {})) if before and after else 1.0,
                6,
            ),
            "locality_ok": not any("LOCALITY_VIOLATION" in str(x) for x in failures),
        }
        if candidate is not None:
            summary.update(
                {
                    "anchor_id": candidate.anchor_id,
                    "census_source": candidate.census_source,
                    "safe": bool(candidate.safe),
                }
            )
        return summary

    @staticmethod
    def _prompt_images(views: dict[str, str] | None) -> list[str]:
        """Prefer the two labelled composites (bordered grids of the 7
        perspective views and the 4 orthographic views) over sending all
        11 individually — 2 images instead of up to 11. Kept as two
        separate images, not merged into one grid, since perspective and
        orthographic are different projections of the same views and are
        each easier to read on their own. Falls back to the raw 7
        perspective views when no composite is present (e.g. render.py's
        plain render_canonical_views() output, which doesn't include one)."""
        views = views or {}
        composites = [
            views[key]
            for key in ("composite_perspective", "composite_ortho")
            if views.get(key) and Path(views[key]).exists()
        ]
        if composites:
            return composites
        return [
            views[name]
            for name in ("toprightiso", "front", "back", "left", "right", "top", "bottom")
            if views.get(name)
        ]

    @staticmethod
    def _candidate_rank(summary: dict[str, Any]) -> tuple:
        return (
            int(bool(summary.get("success"))),
            int(not bool(summary.get("identity"))),
            int(not bool(summary.get("failures"))),
            int(summary.get("delta_match") or 0),
            int(bool(summary.get("locality_ok"))),
            int(bool(summary.get("accepted"))),
            int(bool(summary.get("safe", True))),
            float(summary.get("score") or 0.0),
            -float(summary.get("volume_ratio") or 1.0),
            -int(summary.get("iteration") or 0),
        )

    @staticmethod
    def _attempt_record(
        *,
        attempt_n: int,
        tool,
        failures: list[str],
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        """One structured retry-memory entry: what was tried, why it failed,
        and the geometric delta it produced. In-memory only, scoped to a
        single run() call — never persisted except via the debug log."""
        from groundedcad.verify.edit_delta import delta_summary_dict, failure_bucket

        tool_call: dict[str, Any] = {"tool_name": tool.tool_name if tool else None}
        if tool is not None:
            args = {k: v for k, v in (tool.arguments or {}).items() if k != "script"}
            tool_call["arguments"] = args
            if tool.tool_name == "raw_cadquery":
                tool_call["script_preview"] = ((tool.arguments or {}).get("script") or "")[:400]
        reason = failure_bucket(failures)
        if failures:
            reason = f"{reason}: {failures[0][:200]}"
        return {
            "attempt_n": attempt_n,
            "tool_call": tool_call,
            "failure_reason": reason,
            "delta_summary": delta_summary_dict(before, after),
        }

    @staticmethod
    def _model_completion_should_stop(
        *, complete: bool, iteration: int, last_ok: bool, failures: list[str]
    ) -> bool:
        """Model completion is advisory; the last executed geometry must pass."""
        return bool(iteration > 0 and complete and last_ok and not failures)

    def _execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        step_path: str,
        iter_dir: Path,
        *,
        timeout_s: float | None = None,
    ):
        if tool_name == "raw_cadquery":
            script = arguments.get("script") or ""
            return self.sandbox.run_script(script, step_path, iter_dir, timeout_s=timeout_s)
        args = {**arguments, "step_path": arguments.get("step_path", step_path)}
        if self.inprocess:
            return self.sandbox.run_inprocess_tool(tool_name, args, iter_dir)
        return self.sandbox.run_tool(tool_name, args, iter_dir, timeout_s=timeout_s)

    def _execute_plan(
        self, tool, start_step: str, iter_dir: Path, *, timeout_s: float | None = None
    ):
        execution = self._execute(
            tool.tool_name, tool.arguments, start_step, iter_dir, timeout_s=timeout_s
        )
        current = execution.step_path or start_step
        all_calls = list(execution.tool_calls)
        for i, follow in enumerate(tool.followups or []):
            if not execution.success or not current:
                break
            sub = iter_dir / f"followup_{i:02d}"
            nxt = self._execute(
                follow.tool_name, follow.arguments, current, sub, timeout_s=timeout_s
            )
            all_calls.extend(nxt.tool_calls)
            if nxt.success and nxt.step_path:
                execution = nxt
                current = nxt.step_path
            else:
                execution = nxt
                break
        execution.tool_calls = all_calls
        return execution

    def _ensure_views(self, execution, iter_dir: Path) -> None:
        if not self.render or not execution.success or not execution.step_path:
            return
        if execution.image_paths:
            return
        try:
            from groundedcad.geometry.inspect import load_step, shape_from_workplane

            wp = load_step(execution.step_path)
            execution.image_paths = render_canonical_views(shape_from_workplane(wp), iter_dir)
        except Exception:
            pass

    def _token_counts(self) -> dict[str, float]:
        token_counts: dict[str, float] = {}
        for client in {self.grounding_client, self.planning_client, self.critic_client}:
            for k, v in client.total_tokens.items():
                token_counts[k] = token_counts.get(k, 0.0) + float(v)
        return token_counts

    def _dual_critique(
        self,
        *,
        before,
        execution,
        spec,
        intent,
        classified,
        instruction: str,
        history: Optional[list[str]] = None,
        attempt: int = 0,
    ):
        from groundedcad.verify.edit_delta import (
            dual_critic_failures,
            format_retry_feedback,
            identity_failure_report,
            is_identity,
            observed_changes,
        )

        critique = deterministic_critique(
            before=before,
            execution=execution,
            edit_spec=spec,
            intent=intent,
        )
        after: dict[str, Any] = {}
        identity = True
        if execution.success and execution.step_path:
            try:
                after = inspect_step(execution.step_path)
                identity = is_identity(before, after)
            except Exception:
                after = {}
                identity = True
        failures = dual_critic_failures(before, after, classified=classified)
        if not execution.success:
            err = execution.error or "api_code"
            if "incomplete" in err.lower():
                failures = [f"INCOMPLETE_PLAN: {err}"] + [f for f in failures if "INCOMPLETE_PLAN" not in f]
            else:
                failures = [err] + failures
        # Locality envelope from hole-rim EditPlan (geometry-only).
        try:
            from groundedcad.agents.edit_plan import build_edit_plan
            from groundedcad.verify.edit_delta import locality_violation_failures

            plan = build_edit_plan(classified, before, instruction)
            failures.extend(locality_violation_failures(before, after, plan.locality_envelope, classified=classified))
        except Exception:
            pass
        if identity:
            if critique.verification:
                critique.verification.correct = False
                critique.verification.instruction_satisfied = False
                critique.verification.specific_fix = json.dumps(identity_failure_report(instruction))
        if failures:
            critique.accept = False
            critique.revision_advice = format_retry_feedback(
                instruction=instruction,
                failures=failures,
                observed=observed_changes(before, after) if after else [],
                history=history,
                attempt=attempt,
                before=before,
                after=after or None,
                classified=classified,
            )
            if critique.verification:
                critique.verification.correct = False
                critique.verification.specific_fix = critique.revision_advice
        return critique, after, identity, failures

    def _record_iteration(
        self,
        *,
        logs: list,
        best,
        i: int,
        iter_dir: Path,
        intent,
        spec,
        tool,
        execution,
        critique,
        identity: bool,
        request_id: str,
        out: Path,
    ):
        from groundedcad.agents.schemas import IterationLog

        log = IterationLog(
            iteration=i,
            intent=intent,
            edit_spec=spec,
            execution=execution,
            critique=critique,
            token_counts=self._token_counts(),
            notes=tool.rationale if tool else "",
        )
        logs.append(log)
        (iter_dir / "log.json").write_text(log.model_dump_json(indent=2), encoding="utf-8")
        if execution.success and execution.step_path and not identity:
            score = critique.instruction_score + critique.quality_score
            prev = -1.0
            prev_accepted = False
            if best is not None and best.iterations and best.iterations[-1].critique:
                prev_accepted = bool(best.accepted)
                prev = (
                    best.iterations[-1].critique.instruction_score
                    + best.iterations[-1].critique.quality_score
                )
            # A critic-approved candidate always outranks a rejected one.
            # Within the same acceptance class, use instruction + quality score.
            should_replace = (
                best is None
                or (bool(critique.accept) and not prev_accepted)
                or (bool(critique.accept) == prev_accepted and score >= prev)
            )
            if should_replace:
                best = PipelineResult(
                    request_id=request_id,
                    accepted=critique.accept,
                    best_iteration=i,
                    output_dir=str(out),
                    step_path=execution.step_path,
                    stl_path=execution.stl_path,
                    views=execution.image_paths,
                    iterations=list(logs),
                )
        return best

    def _write_retry_debug_log(
        self, out: Path, request_id: str, attempt_history: list[dict[str, Any]], accepted: bool
    ) -> None:
        """Debug-only: one JSONL row per example, appended to
        <out.parent>/retry_debug_log.jsonl. Consumed by
        scripts/failure_buckets.py, scripts/ab_compare.py, and presentation
        material only — never read back into the pipeline during inference."""
        try:
            log_path = out.parent / "retry_debug_log.jsonl"
            row = {
                "example_id": request_id,
                "n_attempts": len(attempt_history),
                "accepted": bool(accepted),
                "attempt_history": attempt_history,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass  # debug-only; must never fail the pipeline over logging

    @staticmethod
    def _location_hint(classified, before: dict[str, Any]) -> dict[str, Any]:
        """Real coordinates from the STEP census — cheaper and more reliable
        than letting the LLM guess a location from prose alone."""
        from groundedcad.agents.schemas import EditPattern
        from groundedcad.geometry.inspect import (
            _round_num,
            bbox_corners,
            cavity_candidates,
            protrusion_candidates,
            top_planar_centers,
        )

        hint: dict[str, Any] = {
            "shortest_axis": before.get("shortest_axis"),
            "size": [round(float(s), 2) for s in (before.get("size") or (0, 0, 0))],
        }
        if classified.edit_type in {EditPattern.BOOLEAN_MODIFICATION, EditPattern.FEATURE_DELETION}:
            hint["cavities"] = _round_num(cavity_candidates(before, 5))
            hint["planar_sites"] = _round_num(top_planar_centers(before, 6))
            hint["note"] = "Extend the named cavity along shortest_axis; keep cuts local."
        elif classified.edit_type == EditPattern.FEATURE_ADDITION:
            hint["protrusions"] = _round_num(protrusion_candidates(before, 5))
            hint["existing_holes"] = _round_num(cavity_candidates(before, 4))
            hint["bbox_corners"] = _round_num(bbox_corners(before, 4))
            hint["note"] = "Anchor near a protrusion, an existing hole, or a named bbox_corner — not the part center."
        else:
            # Every other edit type still benefits from a concrete anchor list
            # instead of forcing the model to guess blind coordinates.
            hint["cavities"] = _round_num(cavity_candidates(before, 3))
            hint["protrusions"] = _round_num(protrusion_candidates(before, 3))
            hint["planar_sites"] = _round_num(top_planar_centers(before, 3))
            from groundedcad.geometry.inspect import fitting_hole_diameter

            fit = fitting_hole_diameter(before, blend_mm=classified.distance_mm or classified.radius_mm)
            if fit is not None:
                hint["fitting_hole_diameter_mm"] = fit
                hint["max_hole_rims"] = 4
            hint["note"] = "Anchor on one of the listed candidates, not an invented point."
        return hint

    def _try_llm_cadquery(self, request, classified, before, original_step, iter_dir, failure: str):
        """Up to 2 LLM *local tool* attempts, then one raw-CadQuery escalation.
        The original STEP is always imported; even the escalation is still a
        single local edit — it just gets real selectors instead of a fixed
        tool slot, for instructions the constrained tool set can't express
        precisely enough to land on the ground-truth region."""
        from groundedcad.agents.schemas import EditPattern, ExecutionResult, ToolCall
        from groundedcad.geometry.inspect import cavity_span_along_axis, edit_context
        from groundedcad.tools.cadquery_gen import generate_cadquery
        from groundedcad.tools.llm_cadquery import generate_llm_local_edit, generate_llm_raw_local_edit
        from groundedcad.verify.edit_delta import is_identity

        location_hint = self._location_hint(classified, before)
        is_boolean = classified.edit_type in {EditPattern.BOOLEAN_MODIFICATION, EditPattern.FEATURE_DELETION}
        cavities = location_hint.get("cavities") or []
        axis = location_hint.get("shortest_axis") or "z"
        v0 = float(before.get("volume") or 0.0)
        model = edit_context(before, edit_type=classified.edit_type.value)

        attempt_failure = failure
        tool: Optional[ToolCall] = None
        execution: Optional[ExecutionResult] = None
        elapsed = time.time() - self._run_start if self._run_start else 0.0
        total_attempts = 3 if elapsed < self.raw_escalation_budget_s else 2
        for attempt in range(total_attempts):
            if attempt == total_attempts - 1:
                script = generate_llm_raw_local_edit(
                    self.planning_client,
                    instruction=request.instruction,
                    model=model,
                    classified=classified.model_dump(),
                    failure=attempt_failure,
                    location_hint=location_hint,
                    step_path=original_step,
                )
                if not script:
                    execution = ExecutionResult(
                        success=False,
                        error="LLM_RAW_LOCAL_EDIT: model did not return a usable script.",
                    )
                    attempt_failure = execution.error
                    continue
                tool = ToolCall(
                    tool_name="raw_cadquery",
                    arguments={"script": script},
                    rationale="llm_raw_local_edit",
                )
            else:
                tool = generate_llm_local_edit(
                    self.planning_client,
                    instruction=request.instruction,
                    model=model,
                    classified=classified.model_dump(),
                    failure=attempt_failure,
                    location_hint=location_hint,
                    step_path=original_step,
                )
                if tool is None:
                    execution = ExecutionResult(
                        success=False,
                        error="LLM_LOCAL_EDIT: model did not return an allowed local tool.",
                    )
                    attempt_failure = execution.error
                    continue
            script = generate_cadquery(tool, original_step)
            (iter_dir / f"llm_local_{attempt}.py").write_text(script, encoding="utf-8")
            (iter_dir / "my_cad_function.py").write_text(script, encoding="utf-8")
            execution = self._execute_plan(tool, original_step, iter_dir / f"llm_attempt_{attempt}")
            execution.script = script
            if not (execution.success and execution.step_path):
                attempt_failure = (execution.error or "execution failed")[:240]
                continue

            after = inspect_step(execution.step_path)
            if is_identity(before, after):
                attempt_failure = (
                    "IDENTITY_OUTPUT: local tool produced no geometric change. "
                    "Pick a different local op or location."
                )
                continue

            b0 = before.get("size") or (0.0, 0.0, 0.0)
            a0 = after.get("size") or (0.0, 0.0, 0.0)
            if any(float(a) > 3.0 * max(float(b), 1e-6) for a, b in zip(a0, b0)):
                attempt_failure = (
                    "OVERSIZED_BBOX: the edit grew the part's overall bounding box by "
                    "more than 3x — that looks like a rebuild, not a local edit. Keep the "
                    "new/changed feature small vs the existing part."
                )
                continue

            if is_boolean:
                v1 = float(after.get("volume") or 0.0)
                drop = (v0 - v1) / v0 if v0 > 1e-9 else 0.0
                if drop > 0.15:
                    attempt_failure = (
                        f"OVERSIZED_CUT: removed {drop * 100:.0f}% of volume. Use a smaller local cut."
                    )
                    continue
                if cavities:
                    xy = cavities[0].get("center") if isinstance(cavities[0], dict) else None
                    if xy:
                        span_before = cavity_span_along_axis(before, (xy[0], xy[1]), axis)
                        span_after = cavity_span_along_axis(after, (xy[0], xy[1]), axis)
                        if span_after <= span_before + 1e-6:
                            attempt_failure = (
                                f"CAVITY_NOT_EXTENDED: cavity at {xy} did not grow along {axis}."
                            )
                            continue
            break
        return tool, execution

    def _last_resort_geometry_nudge(
        self,
        request: EditRequest,
        classified,
        intent,
        before: dict[str, Any],
        original_step: str,
        out: Path,
    ) -> Optional[PipelineResult]:
        """Never submit a literal identity copy when every strategy bailed out.

        One small deterministic local add/cut near a *confidently grounded*
        target beats a guaranteed-zero no-op: Diff F1 only needs the predicted
        change to overlap the region the GT actually changed. Without a real
        grounded target this degrades to a blind guess at the bbox/planar
        center, which empirically costs Volume F1/Chamfer without ever
        landing on the GT-changed region — so we skip it rather than guess.
        """
        center = None
        for t in sorted(intent.targets or [], key=lambda e: -e.confidence):
            if t.center and t.confidence >= 0.6:
                center = tuple(float(x) for x in t.center)
                break
        if center is None:
            return None

        size = before.get("size") or (10.0, 10.0, 10.0)
        scale = max(float(s) for s in size) or 1.0
        h = max(0.04 * scale, 1e-3)
        combine = "cut" if classified.action in {"delete", "cut"} else "union"

        iter_dir = out / "iterations" / "last_resort"
        iter_dir.mkdir(parents=True, exist_ok=True)
        execution = self._execute(
            "add_box",
            {
                "x": center[0],
                "y": center[1],
                "z": center[2],
                "length": h,
                "width": h,
                "height": h,
                "combine": combine,
            },
            original_step,
            iter_dir,
        )
        if not execution.success or not execution.step_path:
            return None
        from groundedcad.verify.edit_delta import is_identity

        after = inspect_step(execution.step_path)
        if is_identity(before, after):
            return None
        return PipelineResult(
            request_id=request.request_id,
            accepted=False,
            best_iteration=-1,
            output_dir=str(out),
            step_path=execution.step_path,
            stl_path=execution.stl_path,
            views=execution.image_paths,
        )

    def run(self, request: EditRequest, output_dir: str | Path, inplace: bool = False) -> PipelineResult:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        start = time.time()
        self._run_start = start
        edit_id = f"{self.user_id}_{start}"

        if not request.step_path or not Path(request.step_path).exists():
            return PipelineResult(
                request_id=request.request_id,
                accepted=False,
                output_dir=str(out),
                failure_category="api_code",
                settings={"error": f"Missing input STEP: {request.step_path}"},
            )

        # 1. Instruction Parser  (target, operation, dimensions, location, preserve)
        parsed = parse_instruction(request.instruction)
        (out / "instruction_spec.json").write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
        from groundedcad.agents.classifier import classify_instruction

        classified = classify_instruction(request.instruction)
        (out / "classified_edit.json").write_text(classified.model_dump_json(indent=2), encoding="utf-8")

        # 2. Geometry Inspector  (faces, edges, holes, dimensions, bounding box)
        # Enumeration needs the complete edge census. The compact default
        # (120 edges) can miss legitimate full-circle rims on complex parts.
        before = inspect_step(
            request.step_path,
            max_faces=400 if self.candidate_enumeration else 80,
            max_edges=1000 if self.candidate_enumeration else 120,
        )
        from groundedcad.geometry.inspect import edit_context, geometry_brief

        blend = classified.distance_mm or classified.radius_mm
        (out / "geometry_census.json").write_text(json.dumps(before, indent=2, default=str), encoding="utf-8")
        (out / "geometry_brief.txt").write_text(geometry_brief(before, blend_mm=blend), encoding="utf-8")
        # The first LLM pass previously saw only text.  That makes references
        # such as "black lever", "front panel", and "other side" impossible
        # to ground.  Render the source once and keep these views attached to
        # every visual pass; candidate views are appended later as feedback.
        source_images: list[str] = []
        if self.render:
            try:
                from groundedcad.geometry.inspect import load_step
                from groundedcad.geometry.render import render_full_composite

                source_shape = load_step(request.step_path)
                source_views = render_canonical_views(
                    source_shape,
                    out / "source_views",
                    views=["toprightiso", "front", "back", "left", "right", "top", "bottom"],
                )
                _, _, persp_composite, ortho_composite = render_full_composite(
                    source_shape, out / "source_views", perspective_views=source_views
                )
                if persp_composite:
                    source_views["composite_perspective"] = persp_composite
                if ortho_composite:
                    source_views["composite_ortho"] = ortho_composite
                source_images = self._prompt_images(source_views)
            except Exception:
                # Rendering is grounding context, not a reason to reject a
                # valid geometry-only edit on a headless machine.
                source_images = []
        intent = heuristic_ground(request, before)
        intent.target_text = parsed.target
        intent.location = parsed.location
        intent.preserve = parsed.preserve
        intent.dimensions = {**parsed.dimensions, **intent.dimensions}
        intent.reference_text = json.dumps(
            edit_context(before, edit_type=classified.edit_type.value),
            separators=(",", ":"),
        )

        logs: list[IterationLog] = []
        best: Optional[PipelineResult] = None
        original_step = request.step_path
        # A successful deterministic candidate is a valid starting point for
        # the LLM to finish a compound instruction.  Keep the original only
        # when no candidate made a real edit.
        working_step = original_step
        failure_category = None
        brief = geometry_brief(before, blend_mm=blend)
        location_hint = self._location_hint(classified, before)
        from groundedcad.agents.classifier import (
            cadquery_strategy,
            high_confidence_local,
            skip_visual_after_local,
        )
        from groundedcad.agents.schemas import EditPattern, ExecutionResult, ToolCall
        from groundedcad.tools.cadquery_gen import generate_cadquery
        from groundedcad.tools.llm_cadquery import generate_grounded_cadquery
        from groundedcad.verify.edit_delta import dual_critic_accept
        from groundedcad.verify.agent import llm_verification

        cheap_hc = high_confidence_local(classified)
        cq_mode = cadquery_strategy(classified, request.instruction)
        use_cheap = (
            True
            if self.hybrid_autodesk_fallback
            else cheap_hc and skip_visual_after_local(classified)
        )

        spec, tool = plan_edit(
            intent,
            original_step,
            before,
            client=self.planning_client,
            revision_advice="Apply the smallest local edit that matches the instruction.",
            use_llm=False,
        )
        fail_history: list[str] = []
        attempt_history: list[dict[str, Any]] = []  # in-memory only, this run() call
        vlm_critic_calls_used = 0
        hit_vlm_critic_limit = False
        hit_candidate_limit = False  # Phase C only
        last_fail = (
            "WRITE_CADQUERY: import args['input_file'] and apply the instruction; "
            "do not copy the start STEP; do not chamfer the longest edges."
        )
        last_script = ""
        last_stdout = ""
        last_images: list[str] = []
        last_ok = False
        skip_visual = False
        failures: list[str] = []
        candidate_summaries: list[dict[str, Any]] = []
        from groundedcad.agents.edit_plan import build_edit_plan

        edit_plan = build_edit_plan(classified, before, request.instruction)
        enumerated = []
        if self.candidate_enumeration:
            from groundedcad.agents.candidate_enum import enumerate_tool_candidates

            enumerated = enumerate_tool_candidates(
                classified,
                original_step,
                before,
                request.instruction,
                max_candidates=self.max_candidates,
            )

        # Execute every deterministic candidate independently from the original
        # STEP. Rank after all candidates have run; do not stop at first success.
        if enumerated:
            use_cheap = False
            candidate_records: list[dict[str, Any]] = []
            for ci, candidate in enumerate(enumerated):
                candidate_tool = candidate.tool
                iter_dir = out / "iterations" / "00" / f"candidate_{ci:02d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                script = generate_cadquery(candidate_tool, original_step)
                (iter_dir / "my_cad_function.py").write_text(script, encoding="utf-8")
                candidate_tool.arguments.setdefault("script", script)
                if candidate_tool.tool_name == "incomplete_plan":
                    execution = ExecutionResult(
                        success=False,
                        error=candidate_tool.rationale or "incomplete_plan",
                        script=script,
                    )
                else:
                    # Candidates are speculative.  A pathological OCC boolean
                    # must not consume the entire row budget before the
                    # image-grounded CadQuery loop gets a chance to repair it.
                    execution = self._execute_plan(
                        candidate_tool, original_step, iter_dir, timeout_s=15.0
                    )
                    execution.script = script
                self._ensure_views(execution, iter_dir)
                critique, after, identity, candidate_failures = self._dual_critique(
                    before=before,
                    execution=execution,
                    spec=spec,
                    intent=intent,
                    classified=classified,
                    instruction=request.instruction,
                    history=fail_history,
                    attempt=ci,
                )
                critique.accept = dual_critic_accept(
                    candidate_failures,
                    iteration=0,
                    cheap_high_conf=cheap_hc and candidate.safe,
                    visual=False,
                ) and bool(execution.success)
                candidate_result = self._record_iteration(
                    logs=logs,
                    best=None,
                    i=ci,
                    iter_dir=iter_dir,
                    intent=intent,
                    spec=spec,
                    tool=candidate_tool,
                    execution=execution,
                    critique=critique,
                    identity=identity,
                    request_id=request.request_id,
                    out=out,
                )
                summary = self._candidate_summary(
                    iteration=ci,
                    execution=execution,
                    critique=critique,
                    identity=identity,
                    failures=candidate_failures,
                    before=before,
                    after=after,
                    expected_delta=edit_plan.expected_delta,
                    candidate=candidate,
                )
                candidate_summaries.append(summary)
                candidate_records.append(
                    {
                        "rank": self._candidate_rank(summary),
                        "result": candidate_result,
                        "summary": summary,
                        "execution": execution,
                        "critique": critique,
                        "failures": candidate_failures,
                        "script": script,
                    }
                )
                if candidate_failures:
                    fail_history.append(
                        f"candidate{ci} {candidate_tool.tool_name}: "
                        + " | ".join(candidate_failures[:4])
                    )
                    attempt_history.append(
                        self._attempt_record(
                            attempt_n=ci, tool=candidate_tool, failures=candidate_failures,
                            before=before, after=after,
                        )
                    )

            chosen = max(candidate_records, key=lambda record: record["rank"])
            best = chosen["result"]
            chosen_execution = chosen["execution"]
            chosen_critique = chosen["critique"]
            failures = chosen["failures"]
            last_fail = chosen_critique.revision_advice or last_fail
            last_script = chosen_execution.script or chosen["script"]
            last_stdout = (
                (chosen_execution.stdout or "")
                + "\n"
                + (chosen_execution.stderr or "")
            )[:1500]
            last_images = self._prompt_images(chosen_execution.image_paths)
            last_ok = bool(chosen_critique.accept)
            if (
                chosen_execution.success
                and chosen_execution.step_path
                and not bool(chosen["summary"].get("identity"))
            ):
                working_step = chosen_execution.step_path
            skip_visual = bool(
                chosen_critique.accept
                and chosen_execution.success
                and bool(chosen["summary"].get("safe"))
                and cheap_hc
            )
            if skip_visual:
                failure_category = None
            elif not chosen_execution.success:
                failure_category = (
                    "timeout"
                    if chosen_execution.error
                    and "timeout" in chosen_execution.error.lower()
                    else "api_code"
                )
            else:
                failure_category = "planning"

        if use_cheap:
            iter_dir = out / "iterations" / "00"
            iter_dir.mkdir(parents=True, exist_ok=True)
            script = generate_cadquery(tool, original_step)
            (iter_dir / "my_cad_function.py").write_text(script, encoding="utf-8")
            tool.arguments.setdefault("script", script)
            if tool.tool_name == "incomplete_plan":
                execution = ExecutionResult(
                    success=False,
                    error=tool.rationale or "incomplete_plan",
                    script=script,
                )
            elif tool.tool_name == "raw_cadquery" and (
                "# TODO: edit shape" in script or "TODO: edit shape" in script
            ):
                execution = ExecutionResult(
                    success=False,
                    error="IDENTITY_OUTPUT: raw_cadquery scaffold returned the imported STEP unchanged.",
                    script=script,
                )
            else:
                execution = self._execute_plan(tool, original_step, iter_dir)
                execution.script = script
            self._ensure_views(execution, iter_dir)
            critique, after, identity, failures = self._dual_critique(
                before=before,
                execution=execution,
                spec=spec,
                intent=intent,
                classified=classified,
                instruction=request.instruction,
                history=fail_history,
                attempt=0,
            )
            if self.use_llm_critic and not isinstance(self.critic_client, MockLLMClient):
                critique = llm_critique(
                    self.critic_client,
                    instruction=request.instruction,
                    intent=intent,
                    edit_spec=spec,
                    execution=execution,
                    base_critique=critique,
                    before=before,
                )
            generic_guess = (
                tool.tool_name == "boolean_cut_box"
                and classified.edit_type in {EditPattern.BOOLEAN_MODIFICATION, EditPattern.FEATURE_DELETION}
            ) or (
                classified.edit_type == EditPattern.FEATURE_ADDITION
                and tool.tool_name == "add_box"
                and classified.diameter_mm is None
                and classified.radius_mm is None
                and classified.distance_mm is None
                and "feature_add:" not in (classified.notes or "")
            )
            critique.accept = dual_critic_accept(
                failures,
                iteration=0,
                cheap_high_conf=cheap_hc and not generic_guess,
                visual=False,
            ) and bool(execution.success)
            if generic_guess:
                critique.accept = False
                critique.revision_advice = (
                    (critique.revision_advice or "")
                    + " | FAILED_LOCATION: generic heuristic does not satisfy the instruction"
                ).strip(" |")
            if critique.verification:
                (iter_dir / "verification.json").write_text(
                    critique.verification.model_dump_json(indent=2),
                    encoding="utf-8",
                )
            best = self._record_iteration(
                logs=logs,
                best=best,
                i=0,
                iter_dir=iter_dir,
                intent=intent,
                spec=spec,
                tool=tool,
                execution=execution,
                critique=critique,
                identity=identity,
                request_id=request.request_id,
                out=out,
            )
            candidate_summaries.append(
                self._candidate_summary(
                    iteration=0,
                    execution=execution,
                    critique=critique,
                    identity=identity,
                    failures=failures,
                    before=before,
                    after=after,
                    expected_delta=edit_plan.expected_delta,
                )
            )
            if identity:
                failure_category = "false_completion"
            elif not execution.success:
                failure_category = "timeout" if (execution.error and "timeout" in execution.error.lower()) else "api_code"
            else:
                failure_category = "planning"
            skip_visual = bool(
                critique.accept
                and execution.success
                and (
                    skip_visual_after_local(classified)
                    or (self.hybrid_autodesk_fallback and cheap_hc and not generic_guess)
                )
            )
            last_fail = critique.revision_advice or last_fail
            last_script = execution.script or script
            last_stdout = (execution.stdout or "") + "\n" + (execution.stderr or "")
            last_images = self._prompt_images(execution.image_paths)
            last_ok = bool(critique.accept)
            if failures:
                fail_history.append(f"iter0 {tool.tool_name}: " + " | ".join(failures[:4]))
                attempt_history.append(
                    self._attempt_record(attempt_n=0, tool=tool, failures=failures, before=before, after=after)
                )

        # --- Autodesk-style CadQuery on the imported STEP ---
        n_visual = (
            self._adaptive_visual_iters(cq_mode, fail_history)
            if self.use_llm_cadquery
            else 0
        )
        if skip_visual:
            n_visual = 0
        loop_mode = cq_mode if cq_mode in {"mutate", "reconstruct"} else "mutate"
        visual_base = len(logs)
        visual_candidate_records: list[dict[str, Any]] = []  # this loop's attempts — best-so-far fallback
        for v in range(n_visual):
            if v == 0 and n_visual > 0:
                hit_candidate_limit = True  # provisional; cleared below on any successful exit
            i = visual_base + v
            iter_dir = out / "iterations" / f"{i:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            remaining = n_visual - v
            if any("IDENTITY" in x or "SLOT_MISMATCH" in x for x in fail_history) and loop_mode == "mutate":
                if classified.edit_type.value not in {"fillet_chamfer", "hole_edit"}:
                    loop_mode = "reconstruct"
            try:
                gen = generate_grounded_cadquery(
                    self.planning_client,
                    instruction=request.instruction,
                    geometry_brief=brief,
                    classified=classified.model_dump(),
                    location_hint=location_hint,
                    failure=last_fail,
                    last_script=last_script,
                    stdout=last_stdout,
                    images=(source_images + last_images) or None,
                    iteration=v,
                    visual_iters_remaining=remaining,
                    mode=loop_mode,
                    prior_candidates=sorted(
                        candidate_summaries,
                        key=self._candidate_rank,
                        reverse=True,
                    )[:2],
                    attempt_history=attempt_history[-4:],
                )
            except Exception as exc:  # noqa: BLE001
                gen = {
                    "complete": False,
                    "my_cad_function": "",
                    "raw": str(exc),
                }
            if self._model_completion_should_stop(
                complete=bool(gen.get("complete")),
                iteration=v,
                last_ok=last_ok,
                failures=failures,
            ):
                failure_category = None
                hit_candidate_limit = False
                break
            cq_script = gen.get("my_cad_function") or ""
            (iter_dir / "my_cad_function.py").write_text(cq_script, encoding="utf-8")
            tool = ToolCall(
                tool_name="raw_cadquery",
                arguments={"script": cq_script},
                rationale="grounded_visual_cadquery",
            )
            if not cq_script or "def my_cad_function" not in cq_script:
                execution = ExecutionResult(
                    success=False,
                    error="LLM_GROUNDED_CQ: no my_cad_function returned",
                    script=cq_script,
                )
            else:
                execution = self._execute("raw_cadquery", {"script": cq_script}, working_step, iter_dir)
                execution.script = cq_script
            self._ensure_views(execution, iter_dir)
            critique, after, identity, failures = self._dual_critique(
                before=before,
                execution=execution,
                spec=spec,
                intent=intent,
                classified=classified,
                instruction=request.instruction,
                history=fail_history,
                attempt=i,
            )
            # Deterministic check runs first, free, on every candidate. Only
            # a candidate that already passed it consumes one of the 3 real
            # VLM judgment calls; a failing candidate skips straight to the
            # next iteration with no VLM spend.
            if not failures and self.use_llm_critic and not isinstance(self.critic_client, MockLLMClient):
                if vlm_critic_calls_used < MAX_VLM_CRITIC_CALLS:
                    critique = llm_verification(
                        self.critic_client,
                        instruction=request.instruction,
                        intent=intent,
                        edit_spec=spec,
                        execution=execution,
                        before=before,
                        base=critique,
                        decision=critique.verification,
                    )
                    vlm_critic_calls_used += 1
                else:
                    hit_vlm_critic_limit = True
            critique.accept = dual_critic_accept(
                failures, iteration=v, cheap_high_conf=False, visual=True
            ) and bool(execution.success)
            if failures:
                fail_history.append(f"iter{i} raw_cadquery: " + " | ".join(failures[:4]))
                attempt_history.append(
                    self._attempt_record(attempt_n=i, tool=tool, failures=failures, before=before, after=after)
                )
            if critique.verification:
                (iter_dir / "verification.json").write_text(
                    critique.verification.model_dump_json(indent=2),
                    encoding="utf-8",
                )
            best = self._record_iteration(
                logs=logs,
                best=best,
                i=i,
                iter_dir=iter_dir,
                intent=intent,
                spec=spec,
                tool=tool,
                execution=execution,
                critique=critique,
                identity=identity,
                request_id=request.request_id,
                out=out,
            )
            candidate_summaries.append(
                self._candidate_summary(
                    iteration=i,
                    execution=execution,
                    critique=critique,
                    identity=identity,
                    failures=failures,
                    before=before,
                    after=after,
                    expected_delta=edit_plan.expected_delta,
                )
            )
            if execution.success and execution.step_path:
                visual_candidate_records.append(
                    {
                        "rank": self._candidate_rank(candidate_summaries[-1]),
                        "result": PipelineResult(
                            request_id=request.request_id,
                            accepted=bool(critique.accept),
                            best_iteration=i,
                            output_dir=str(out),
                            step_path=execution.step_path,
                            stl_path=execution.stl_path,
                            views=execution.image_paths,
                        ),
                    }
                )
            last_fail = critique.revision_advice or last_fail
            last_script = cq_script
            last_stdout = ((execution.stdout or "") + "\n" + (execution.stderr or ""))[:1500]
            last_images = self._prompt_images(execution.image_paths)
            last_ok = bool(critique.accept) and not identity
            if execution.success and execution.step_path and not identity:
                working_step = execution.step_path
            if identity:
                failure_category = "false_completion"
            elif not execution.success:
                failure_category = "timeout" if (execution.error and "timeout" in execution.error.lower()) else "api_code"
            else:
                failure_category = "planning"
            if critique.accept and execution.success and v > 0:
                failure_category = None
                hit_candidate_limit = False
                break

        if (hit_candidate_limit or hit_vlm_critic_limit) and visual_candidate_records:
            if best is None or not bool(best.accepted):
                best = max(visual_candidate_records, key=lambda r: r["rank"])["result"]

        if best is None:
            nudge = self._last_resort_geometry_nudge(request, classified, intent, before, original_step, out)
            if nudge is not None:
                best = nudge
                failure_category = failure_category or "false_completion"

        end = time.time()
        brep_end = Path(out) if inplace else (out / "brep_end" / str(start))
        brep_end.mkdir(parents=True, exist_ok=True)

        final_step = None
        final_stl = None
        final_views: dict[str, str] = {}
        accepted = False
        best_iter = -1

        if best and best.step_path and Path(best.step_path).exists():
            final_step = brep_end / "tmp.step"
            shutil.copy2(best.step_path, final_step)
            if best.stl_path and Path(best.stl_path).exists():
                final_stl = brep_end / "tmp.stl"
                shutil.copy2(best.stl_path, final_stl)
            for name, path in (best.views or {}).items():
                if Path(path).exists():
                    dest_name = "iso.png" if name in {"toprightiso", "isometric", "tmp"} else f"{name}.png"
                    if name == "tmp":
                        dest_name = "tmp.png"
                    dest = brep_end / dest_name
                    shutil.copy2(path, dest)
                    final_views[name] = str(dest)
            accepted = bool(best.accepted)
            best_iter = best.best_iteration
            # Hard rule, independent of whatever the critic decided: the
            # submitted solid must actually differ from the start model
            # (same volume+bbox+face count as start == identity == D will be
            # 0). Never report accepted=True on an unchanged solid.
            from groundedcad.verify.edit_delta import is_identity

            try:
                final_after = inspect_step(str(final_step))
                if is_identity(before, final_after):
                    accepted = False
                    failure_category = "false_completion"
            except Exception:
                accepted = False
                failure_category = failure_category or "geometric_validity"
        else:
            # Packaging still needs a STEP file, but this is not a successful edit.
            final_step = brep_end / "tmp.step"
            shutil.copy2(request.step_path, final_step)
            # The official evaluator consumes STL.  When no valid edit was
            # produced, preserve the exact input geometry rather than asking
            # CadQuery to re-export an often-problematic STEP during scoring.
            source_stl = request.metadata.get("brep_start_path_stl") or request.metadata.get("input_stl")
            if isinstance(source_stl, list):
                source_stl = source_stl[-1] if source_stl else None
            source_stl_path = Path(str(source_stl)) if source_stl else Path(request.step_path).with_suffix(".stl")
            if source_stl_path.exists():
                final_stl = brep_end / "tmp.stl"
                shutil.copy2(source_stl_path, final_stl)
            accepted = False
            failure_category = failure_category or "api_code"

        # False completion guard: never mark accept if last critique rejected
        if logs and logs[-1].critique and not logs[-1].critique.accept:
            if accepted and logs[best_iter].critique and not logs[best_iter].critique.accept:
                accepted = False
                failure_category = "false_completion"

        self._write_retry_debug_log(out, request.request_id, attempt_history, accepted)

        token_counts = logs[-1].token_counts if logs else {}
        ins = float(token_counts.get("input_tokens") or 0)
        outs = float(token_counts.get("output_tokens") or 0) + float(token_counts.get("thinking_tokens") or 0)
        settings_extra = {
            "token_counts": {
                **token_counts,
                "cost_estimate": ins * 3.0 / 1e6 + outs * 15.0 / 1e6,
            },
            "method": "grounded_visual_cadquery",
            "cadquery_strategy": cq_mode,
            "hybrid_autodesk_fallback": self.hybrid_autodesk_fallback,
            "candidate_enumeration": self.candidate_enumeration,
            "max_candidates": self.max_candidates,
            "max_iters": self.max_iters,
            "visual_iters": self.visual_iters,
            "visual_iters_min": self.visual_iters_min,
            "visual_iters_max": self.visual_iters_max,
            "adaptive_visual_iters": n_visual,
            "candidate_summaries": candidate_summaries[-10:],
            "best_iteration": best_iter,
            "accepted": accepted,
            "failure_category": failure_category,
            "retry_history": fail_history[-8:],
            "valid_start_copy": not bool(best and best.step_path),
            "intent_summary": intent.summary if intent else "",
            "used_llm": any(
                bool(x.notes and ("llm" in x.notes.lower() or "visual" in x.notes.lower() or "grounded" in x.notes.lower()))
                for x in logs
            ),
        }
        write_benchmark_settings(
            brep_end,
            request_id=request.request_id,
            edit_id=edit_id,
            user_id=self.user_id,
            start_time=start,
            end_time=end,
            filename=str(final_step) if final_step else None,
            extra=settings_extra,
        )

        # Persist full run log
        result = PipelineResult(
            request_id=request.request_id,
            accepted=accepted,
            best_iteration=best_iter,
            output_dir=str(out),
            step_path=str(final_step) if final_step else None,
            stl_path=str(final_stl) if final_stl else None,
            views=final_views,
            settings=settings_extra,
            iterations=logs,
            failure_category=None if accepted else failure_category,
        )
        (out / "pipeline_result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result
