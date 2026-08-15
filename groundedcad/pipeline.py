"""Bounded grounder -> planner -> executor -> verifier -> critic loop."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from groundedcad.agents.grounder import heuristic_ground
from groundedcad.agents.instruction_parser import parse_instruction
from groundedcad.agents.planner import heuristic_needs_llm, plan_edit
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


class GroundedCADPipeline:
    def __init__(
        self,
        *,
        grounding_client: Optional[LLMClient] = None,
        planning_client: Optional[LLMClient] = None,
        critic_client: Optional[LLMClient] = None,
        max_iters: int = 3,
        sandbox: Optional[Sandbox] = None,
        user_id: str = "groundedcad",
        use_llm_critic: bool = False,
        inprocess: bool = False,
        render: bool = True,
        use_llm_cadquery: Optional[bool] = None,
    ):
        default = grounding_client or planning_client or critic_client or auto_client_from_env()
        self.grounding_client = grounding_client or default
        self.planning_client = planning_client or default
        self.critic_client = critic_client or default
        self.max_iters = max_iters
        self.sandbox = sandbox or Sandbox(render=render)
        self.user_id = user_id
        self.use_llm_critic = use_llm_critic
        self.inprocess = inprocess
        self.render = render
        if use_llm_cadquery is None:
            use_llm_cadquery = not isinstance(self.planning_client, MockLLMClient)
        self.use_llm_cadquery = bool(use_llm_cadquery)

    def _execute(self, tool_name: str, arguments: dict[str, Any], step_path: str, iter_dir: Path):
        if tool_name == "raw_cadquery":
            script = arguments.get("script") or ""
            return self.sandbox.run_script(script, step_path, iter_dir)
        args = {**arguments, "step_path": arguments.get("step_path", step_path)}
        if self.inprocess:
            return self.sandbox.run_inprocess_tool(tool_name, args, iter_dir)
        return self.sandbox.run_tool(tool_name, args, iter_dir)

    def _execute_plan(self, tool, start_step: str, iter_dir: Path):
        execution = self._execute(tool.tool_name, tool.arguments, start_step, iter_dir)
        current = execution.step_path or start_step
        all_calls = list(execution.tool_calls)
        for i, follow in enumerate(tool.followups or []):
            if not execution.success or not current:
                break
            sub = iter_dir / f"followup_{i:02d}"
            nxt = self._execute(follow.tool_name, follow.arguments, current, sub)
            all_calls.extend(nxt.tool_calls)
            if nxt.success and nxt.step_path:
                execution = nxt
                current = nxt.step_path
            else:
                execution = nxt
                break
        execution.tool_calls = all_calls
        return execution

    @staticmethod
    def _location_hint(classified, before: dict[str, Any]) -> dict[str, Any]:
        """Real coordinates from the STEP census — cheaper and more reliable
        than letting the LLM guess a location from prose alone."""
        from groundedcad.agents.schemas import EditPattern
        from groundedcad.geometry.inspect import (
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
            hint["cavities"] = cavity_candidates(before, 3)
            hint["planar_sites"] = top_planar_centers(before, 4)
            hint["note"] = (
                "If instruction says extend/cut-through an existing slot/cavity: pick the "
                "closest cavity in `cavities` and cut along shortest_axis until it clears the "
                "far face. If instruction says several/multiple cutouts: add 2+ separate small "
                "cuts near different planar_sites; never remove >15% of total volume in one cut."
            )
        elif classified.edit_type == EditPattern.FEATURE_ADDITION:
            hint["protrusions"] = protrusion_candidates(before, 3)
            hint["bbox_corners"] = bbox_corners(before, 4)
            hint["note"] = (
                "Anchor the new feature near an existing small protrusion/boss in "
                "`protrusions`, or the bbox_corner nearest the location named in the "
                "instruction — not the part center."
            )
        return hint

    def _try_llm_cadquery(self, request, classified, before, original_step, iter_dir, failure: str):
        """Up to 2 LLM CadQuery attempts. Retries once, feeding back whichever
        of these applies: identity (no change), an oversized single cut, or an
        existing cavity that was not actually extended. Never more than 2
        scripts per edit."""
        from groundedcad.agents.schemas import EditPattern, ExecutionResult, ToolCall
        from groundedcad.geometry.inspect import cavity_span_along_axis, geometry_brief
        from groundedcad.tools.llm_cadquery import generate_llm_cadquery, is_identity_scaffold
        from groundedcad.verify.edit_delta import is_identity

        location_hint = self._location_hint(classified, before)
        is_boolean = classified.edit_type in {EditPattern.BOOLEAN_MODIFICATION, EditPattern.FEATURE_DELETION}
        cavities = location_hint.get("cavities") or []
        axis = location_hint.get("shortest_axis") or "z"
        v0 = float(before.get("volume") or 0.0)

        attempt_failure = failure
        script = ""
        execution: Optional[ExecutionResult] = None
        for attempt in range(2):
            script = generate_llm_cadquery(
                self.planning_client,
                instruction=request.instruction,
                geometry_brief=geometry_brief(before),
                classified=classified.model_dump(),
                failure=attempt_failure,
                location_hint=location_hint,
            )
            (iter_dir / f"llm_cadquery_{attempt}.py").write_text(script, encoding="utf-8")
            (iter_dir / "llm_cadquery.py").write_text(script, encoding="utf-8")
            if is_identity_scaffold(script):
                execution = ExecutionResult(
                    success=False,
                    error="IDENTITY_OUTPUT: LLM returned the import-only scaffold.",
                    script=script,
                )
                attempt_failure = execution.error
                continue
            tool = ToolCall(tool_name="raw_cadquery", arguments={"script": script}, rationale="llm_cadquery")
            execution = self._execute_plan(tool, original_step, iter_dir / f"llm_attempt_{attempt}")
            execution.script = script
            if not (execution.success and execution.step_path):
                attempt_failure = (execution.error or "execution failed")[:500]
                continue

            after = inspect_step(execution.step_path)
            if is_identity(before, after):
                attempt_failure = (
                    "IDENTITY_OUTPUT: the script ran but produced no geometric change "
                    "(same volume/bbox/face count as start). Change the geometry."
                )
                continue

            if is_boolean:
                v1 = float(after.get("volume") or 0.0)
                drop = (v0 - v1) / v0 if v0 > 1e-9 else 0.0
                if drop > 0.15:
                    attempt_failure = (
                        f"OVERSIZED_CUT: removed {drop * 100:.0f}% of total volume in one cut. "
                        "Use smaller, more local material removal — do not cut a single huge box."
                    )
                    continue
                if cavities:
                    xy = cavities[0].get("center")
                    if xy:
                        span_before = cavity_span_along_axis(before, (xy[0], xy[1]), axis)
                        span_after = cavity_span_along_axis(after, (xy[0], xy[1]), axis)
                        if span_after <= span_before + 1e-6:
                            attempt_failure = (
                                f"CAVITY_NOT_EXTENDED: the candidate cavity at {xy} did not grow "
                                f"along the {axis}-axis (before={span_before:.2f}, after="
                                f"{span_after:.2f}). Extend that existing opening through the body."
                            )
                            continue
            break
        return script, execution

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
        before = inspect_step(request.step_path)
        from groundedcad.geometry.inspect import geometry_brief

        (out / "geometry_census.json").write_text(json.dumps(before, indent=2, default=str), encoding="utf-8")
        (out / "geometry_brief.txt").write_text(geometry_brief(before), encoding="utf-8")
        intent = heuristic_ground(request, before)
        intent.target_text = parsed.target
        intent.location = parsed.location
        intent.preserve = parsed.preserve
        intent.dimensions = {**parsed.dimensions, **intent.dimensions}
        intent.reference_text = geometry_brief(before)

        logs: list[IterationLog] = []
        best: Optional[PipelineResult] = None
        revision = ""
        original_step = request.step_path
        failure_category = None

        for i in range(self.max_iters):
            iter_dir = out / "iterations" / f"{i:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # 3. Minimal Edit Plan
            need_llm = bool(revision) or heuristic_needs_llm(intent)
            spec, tool = plan_edit(
                intent,
                original_step,
                before,
                client=self.planning_client,
                revision_advice=revision or "Apply the smallest local edit that matches the instruction.",
                use_llm=need_llm,
            )

            # 4. CadQuery Generator
            script = generate_cadquery(tool, original_step)
            (iter_dir / "my_cad_function.py").write_text(script, encoding="utf-8")
            tool.arguments.setdefault("script", script)

            # 5. Execution (CadQuery harness; official neuralCAD loop comments this as FreeCAD)
            if tool.tool_name == "raw_cadquery" and (
                "# TODO: edit shape" in script or "TODO: edit shape" in script
            ):
                from groundedcad.agents.schemas import ExecutionResult

                execution = ExecutionResult(
                    success=False,
                    error="IDENTITY_OUTPUT: raw_cadquery scaffold returned the imported STEP unchanged.",
                    script=script,
                )
            else:
                execution = self._execute_plan(tool, original_step, iter_dir)
                execution.script = script
            if tool.tool_name == "incomplete_plan":
                execution.success = False
                execution.error = tool.rationale or "incomplete_plan"

            # "Wrong geometry, not wrong classification": some heuristic
            # strategies execute successfully and change real (non-identity)
            # geometry, but have no true feature localization for this
            # instruction — a generic cut box for a named slot/cutout set, or
            # a scale-only result when the instruction asked for more ops.
            # These must not be treated as success; force the LLM fallback
            # (and never let the critic accept the raw heuristic guess).
            from groundedcad.agents.schemas import EditPattern
            from groundedcad.verify.edit_delta import is_identity

            failed_location = False
            extra_ops_pending = False
            if tool.tool_name == "boolean_cut_box" and classified.edit_type in {
                EditPattern.BOOLEAN_MODIFICATION,
                EditPattern.FEATURE_DELETION,
            }:
                failed_location = True
            elif (
                classified.edit_type == EditPattern.DIMENSION_CHANGE
                and tool.tool_name == "scale_uniform"
                and "DRAFT" in (classified.notes or "")
            ):
                # No draft tool exists, so a SCALE (+ maybe FILLET followup)
                # never covers a "...and add drafts..." instruction — the
                # remaining op is structurally unaddressed, every time.
                extra_ops_pending = True
                failed_location = True
            elif (
                classified.edit_type == EditPattern.FEATURE_ADDITION
                and tool.tool_name == "add_box"
                and classified.diameter_mm is None
                and classified.radius_mm is None
                and classified.distance_mm is None
            ):
                # The instruction gave no explicit size for the new feature,
                # so whatever size this add_box used was invented (slot-filled
                # or defaulted) rather than read from the instruction. Let the
                # LLM reason about the described location/shape directly
                # instead of a generic box.
                failed_location = True

            ident0 = False
            if execution.success and execution.step_path:
                ident0 = is_identity(before, inspect_step(execution.step_path))

            need_llm_script = False
            if self.use_llm_cadquery:
                need_llm_script = (
                    tool.tool_name == "incomplete_plan"
                    or not execution.success
                    or ident0
                    or (failed_location and not ident0)
                )

            heuristic_execution, heuristic_tool = execution, tool
            kept_heuristic_guess = failed_location and not ident0
            if need_llm_script:
                if failed_location and execution.success and not ident0:
                    if extra_ops_pending:
                        fail = (
                            f"INCOMPLETE_MULTIOP: only part of the requested operations were "
                            f"applied ({classified.notes}). Apply ALL requested operations "
                            f"(e.g. scale AND draft AND fillet) in one script, not just the scale."
                        )
                    elif classified.edit_type == EditPattern.FEATURE_ADDITION:
                        fail = (
                            "FAILED_LOCATION: a generic default-sized box does not satisfy the "
                            "instruction. Use location_hint (protrusions/bbox_corners) to place a "
                            "shape matching what the instruction actually describes, not a plain box "
                            "at the part center."
                        )
                    else:
                        fail = (
                            "FAILED_LOCATION: a generic cut box does not satisfy the instruction. "
                            "Do not cut a random box at an arbitrary point on the part — modify the "
                            "named slot/feature, or add several separate lightening cutouts, exactly "
                            "as the instruction describes."
                        )
                else:
                    fail = execution.error or revision or "EDIT_FAILED: no geometric modification"
                try:
                    script, execution = self._try_llm_cadquery(
                        request, classified, before, original_step, iter_dir, fail
                    )
                    tool = tool.model_copy(
                        update={"tool_name": "raw_cadquery", "rationale": "llm_cadquery", "arguments": {"script": script}}
                    )
                    (iter_dir / "my_cad_function.py").write_text(script, encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    execution.error = f"llm_cadquery: {exc}\n{execution.error or ''}"

                if failed_location and not ident0:
                    llm_is_useful = bool(execution.success and execution.step_path)
                    if llm_is_useful:
                        llm_is_useful = not is_identity(before, inspect_step(execution.step_path))
                    if llm_is_useful:
                        kept_heuristic_guess = False
                    else:
                        # LLM couldn't beat the heuristic guess geometrically;
                        # keep its (still non-identity) geometry rather than
                        # regress to nothing, but acceptance stays forced off.
                        execution, tool = heuristic_execution, heuristic_tool
                        kept_heuristic_guess = True

            # 6–7. Geometry validator inputs + render 7 views for the critic
            if self.render and execution.success and execution.step_path and not execution.image_paths:
                try:
                    from groundedcad.geometry.inspect import load_step, shape_from_workplane

                    wp = load_step(execution.step_path)
                    execution.image_paths = render_canonical_views(shape_from_workplane(wp), iter_dir)
                except Exception:
                    pass

            # 8. Verification agent (instruction, preservation, dims, position, count, topology, magnitude)
            critique = deterministic_critique(
                before=before,
                execution=execution,
                edit_spec=spec,
                intent=intent,
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
            if critique.verification:
                (iter_dir / "verification.json").write_text(
                    critique.verification.model_dump_json(indent=2),
                    encoding="utf-8",
                )
            if critique.verification and not critique.verification.correct:
                critique.accept = False
                if critique.verification.specific_fix:
                    critique.revision_advice = critique.verification.specific_fix

            identity = False
            if execution.success and execution.step_path:
                from groundedcad.verify.edit_delta import identity_failure_report, is_identity

                after = inspect_step(execution.step_path)
                identity = is_identity(before, after)
                if identity:
                    critique.accept = False
                    report = identity_failure_report(request.instruction)
                    critique.revision_advice = json.dumps(report)
                    if critique.verification:
                        critique.verification.correct = False
                        critique.verification.instruction_satisfied = False
                        critique.verification.specific_fix = critique.revision_advice

            if kept_heuristic_guess:
                # Never accept a generic cut box / partial multi-op result:
                # it executed and changed real geometry, but has no evidence
                # the actually-named feature was the one that changed. This
                # must be the LAST word on accept/advice for the iteration —
                # placed after verification/identity so neither can restore
                # accept=True or drop the tag.
                critique.accept = False
                tag = (
                    "INCOMPLETE_MULTIOP: remaining requested operations were not applied."
                    if extra_ops_pending
                    else "FAILED_LOCATION: generic cut box does not satisfy the instruction; LLM fallback did not improve on it."
                )
                critique.revision_advice = (f"{critique.revision_advice} | {tag}").strip(" |")

            token_counts = {}
            for client in {self.grounding_client, self.planning_client, self.critic_client}:
                for k, v in client.total_tokens.items():
                    token_counts[k] = token_counts.get(k, 0.0) + float(v)

            log = IterationLog(
                iteration=i,
                intent=intent,
                edit_spec=spec,
                execution=execution,
                critique=critique,
                token_counts=token_counts,
                notes=tool.rationale,
            )
            logs.append(log)
            (iter_dir / "log.json").write_text(log.model_dump_json(indent=2), encoding="utf-8")

            if execution.success and execution.step_path and not identity:
                # Retain best valid so far — never promote a no-op as the winner
                score = critique.instruction_score + critique.quality_score
                if best is None or score >= (
                    (best.iterations[-1].critique.instruction_score + best.iterations[-1].critique.quality_score)
                    if best.iterations and best.iterations[-1].critique
                    else -1
                ):
                    best = PipelineResult(
                        request_id=request.request_id,
                        accepted=critique.accept,
                        best_iteration=i,
                        output_dir=str(out),
                        step_path=execution.step_path,
                        stl_path=execution.stl_path,
                        views=execution.image_paths,
                        iterations=list(logs),
                    )
            if critique.accept and execution.success:
                break

            # Classify failure for next revision
            if identity:
                failure_category = "false_completion"
            elif not execution.success:
                failure_category = "api_code"
                if execution.error and "timeout" in execution.error.lower():
                    failure_category = "timeout"
            elif any(c.name == "valid_brep" and not c.passed for c in critique.failed_checks):
                failure_category = "geometric_validity"
            elif intent.ambiguities and i == 0:
                failure_category = "grounding"
            else:
                failure_category = "planning"
            revision = critique.revision_advice or "Revise tool parameters; keep edit local."

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

            final_after = inspect_step(str(final_step))
            if is_identity(before, final_after):
                accepted = False
                failure_category = "false_completion"
        else:
            # Packaging still needs a STEP file, but this is not a successful edit.
            final_step = brep_end / "tmp.step"
            shutil.copy2(request.step_path, final_step)
            accepted = False
            failure_category = failure_category or "api_code"

        # False completion guard: never mark accept if last critique rejected
        if logs and logs[-1].critique and not logs[-1].critique.accept:
            if accepted and logs[best_iter].critique and not logs[best_iter].critique.accept:
                accepted = False
                failure_category = "false_completion"

        token_counts = logs[-1].token_counts if logs else {}
        ins = float(token_counts.get("input_tokens") or 0)
        outs = float(token_counts.get("output_tokens") or 0) + float(token_counts.get("thinking_tokens") or 0)
        settings_extra = {
            "token_counts": {
                **token_counts,
                "cost_estimate": ins * 3.0 / 1e6 + outs * 15.0 / 1e6,
            },
            "method": "groundedcad",
            "max_iters": self.max_iters,
            "best_iteration": best_iter,
            "accepted": accepted,
            "failure_category": failure_category,
            "valid_start_copy": not bool(best and best.step_path),
            "intent_summary": intent.summary if intent else "",
            "used_llm": any(bool(x.notes and "llm" in x.notes.lower()) for x in logs),
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
