"""Create an evidence-first diagnostic report for a completed 48-row run.

This script is deliberately read-only: it never executes CAD or changes a
prediction.  It joins the pipeline's per-iteration logs with the official
metric report, then makes conservative failure-stage labels.  Fields are
``unknown`` where the stored artifacts cannot prove instruction satisfaction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from groundedcad.geometry.inspect import inspect_step
from groundedcad.ingest.request_parser import parse_request_from_row
from groundedcad.verify.edit_delta import is_identity
from scripts.score_groundedcad_baseline import DATA, PARQUET, classify


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _short(value: Any, limit: int = 1200) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _scripts(row_dir: Path) -> list[dict[str, str]]:
    out = []
    for path in sorted(row_dir.rglob("my_cad_function.py")):
        out.append({"path": str(path), "code": path.read_text(encoding="utf-8", errors="replace")})
    return out


def _view_hashes(row_dir: Path | None) -> dict[str, Any]:
    if row_dir is None:
        return {"count": 0, "unique_hashes": 0, "duplicate_views": False}
    views = list((row_dir / "source_views").glob("*.png"))
    hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest() for path in views if path.is_file()
    }
    return {
        "count": len(views),
        "unique_hashes": len(hashes),
        "duplicate_views": len(views) > 1 and len(hashes) == 1,
    }


def _iteration_view_audit(row_dir: Path | None) -> list[dict[str, Any]]:
    if row_dir is None:
        return []
    out = []
    for directory in sorted((row_dir / "iterations").glob("**/*")):
        if not directory.is_dir():
            continue
        views = list(directory.glob("*.png"))
        if not views:
            continue
        hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in views}
        out.append({
            "path": str(directory),
            "count": len(views),
            "unique_hashes": len(hashes),
            "all_duplicate": len(views) > 1 and len(hashes) == 1,
        })
    return out


def _best_execution(result: dict[str, Any]) -> dict[str, Any]:
    best_iteration = int(result.get("best_iteration", -1))
    logs = result.get("iterations") or []
    for log in logs:
        if int(log.get("iteration", -2)) == best_iteration:
            return log.get("execution") or {}
    return (logs[-1].get("execution") or {}) if logs else {}


def _operation_stage(result: dict[str, Any], changed: bool | None) -> str:
    """Choose one actual failure stage; do not infer a semantic miss from D."""
    settings = result.get("settings") or {}
    history = " | ".join(str(x) for x in settings.get("retry_history") or []).upper()
    logs = result.get("iterations") or []
    errors = " | ".join(
        _short((x.get("execution") or {}).get("error"), 400).upper() for x in logs
    )
    joined = f"{history} | {errors}"
    if changed is False or settings.get("valid_start_copy"):
        return "geometry_unchanged"
    if "NO MY_CAD_FUNCTION" in joined or "MODEL DID NOT RETURN" in joined:
        return "cadquery_generation_failure"
    if "TIMEOUT" in joined or "TRACEBACK" in joined or "STANDARD_FAILURE" in joined or "BREP_API" in joined:
        return "cadquery_execution_failure"
    if "EXPORT" in joined:
        return "step_export_failure"
    if result.get("accepted") and not changed:
        return "validation_incorrect_success"
    if changed is True and not result.get("accepted"):
        return "geometry_changed_edit_unverified"
    if changed is True:
        return "geometry_changed_requires_semantic_review"
    return "other_or_insufficient_artifacts"


def _semantic_evidence(result: dict[str, Any], changed: bool | None) -> str:
    logs = result.get("iterations") or []
    if not logs:
        return "unknown_no_iterations"
    best = int(result.get("best_iteration", -1))
    log = next((x for x in logs if int(x.get("iteration", -2)) == best), logs[-1])
    critique = log.get("critique") or {}
    verify = critique.get("verification") or {}
    if verify.get("instruction_satisfied") is True:
        return "pipeline_claimed_satisfied"
    if changed is False:
        return "not_changed"
    if critique.get("accept") is False:
        return "pipeline_rejected_or_unverified"
    return "unknown_no_feature_level_oracle"


def _geometry_change(start_step: Path | None, output_step: Path | None) -> tuple[bool | None, str]:
    if not start_step or not output_step or not start_step.exists() or not output_step.exists():
        return None, "missing_start_or_output_step"
    try:
        return (not is_identity(inspect_step(str(start_step)), inspect_step(str(output_step)))), "inspected"
    except Exception as exc:  # noqa: BLE001
        return None, f"inspection_error:{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only per-row diagnosis for a completed GroundedCAD run")
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    score_rows = _read_json(args.scores)
    if not isinstance(score_rows, list):
        raise SystemExit(f"Expected score JSON list: {args.scores}")
    scores = {str(row.get("id")): row for row in score_rows}
    input_rows = {str(row["request"]): row for _, row in __import__("pandas").read_parquet(PARQUET).iterrows()}
    results = {str(data.get("request_id")): (path.parent, data) for path in args.outputs.rglob("pipeline_result.json") if (data := _read_json(path)).get("request_id")}

    rows: list[dict[str, Any]] = []
    for rid, source in input_rows.items():
        row_dir, result = results.get(rid, (None, {}))
        score = scores.get(rid, {})
        try:
            start_step = Path(parse_request_from_row(source.to_dict(), DATA).step_path)
        except Exception:
            start_step = None
        output_step = Path(str(result.get("step_path"))) if result.get("step_path") else (row_dir / "tmp.step" if row_dir else None)
        changed, change_evidence = _geometry_change(start_step, output_step)
        iterations = result.get("iterations") or []
        executions = [x.get("execution") or {} for x in iterations]
        best_execution = _best_execution(result)
        best_geometry = best_execution.get("geometry_summary") or {}
        settings = result.get("settings") or {}
        errors = [
            _short(execution.get("error")) for execution in executions if execution.get("error")
        ]
        duration = sum(float(execution.get("duration_s") or 0.0) for execution in executions)
        scripts = _scripts(row_dir) if row_dir else []
        source_view_audit = _view_hashes(row_dir)
        iteration_view_audit = _iteration_view_audit(row_dir)
        stage = _operation_stage(result, changed) if result else "missing_pipeline_result"
        false_completion = str(result.get("failure_category") or settings.get("failure_category") or "") == "false_completion"
        rows.append({
            "row_id": rid,
            "provider": (args.outputs / "run_manifest.json"),
            "model": "unknown",
            "input_step": str(start_step) if start_step else "",
            "instruction": str(source.get("request_text") or ""),
            "edit_type": classify(str(source.get("request_text") or "")),
            "generated_scripts": scripts,
            "source_view_audit": source_view_audit,
            "iteration_view_audit": iteration_view_audit,
            "renderer_failure": source_view_audit["count"] == 0,
            "missing_or_invalid_visual_feedback": source_view_audit["count"] < 2
            or any(audit["count"] > 1 and audit["all_duplicate"] for audit in iteration_view_audit),
            "execution_successes": [bool(x.get("success")) for x in executions],
            "execution_success": any(bool(x.get("success")) for x in executions),
            "output_step": str(output_step) if output_step else "",
            "output_step_exists": bool(output_step and output_step.exists()),
            "output_step_valid": best_geometry.get("topology_valid"),
            "output_step_inspectable": change_evidence == "inspected",
            "topology_validity_error": best_geometry.get("topology_validity_error", ""),
            "geometry_changed": changed,
            "geometry_change_evidence": change_evidence,
            "requested_feature_changed": _semantic_evidence(result, changed),
            "accepted": bool(result.get("accepted")),
            "d_score": score.get("diff_f1"),
            "chamfer": score.get("chamfer"),
            "volume_f1": score.get("volume_f1"),
            "unchanged_start": bool(settings.get("valid_start_copy")) or changed is False,
            "false_completion": false_completion,
            "failure_stage": stage,
            "failure_category": result.get("failure_category") or settings.get("failure_category"),
            "errors": errors,
            "retry_history": settings.get("retry_history") or [],
            "execution_duration_s": round(duration, 3),
            "pipeline_result": str(row_dir / "pipeline_result.json") if row_dir else "",
        })

    manifest = _read_json(args.outputs / "run_manifest.json")
    for row in rows:
        row["provider"] = manifest.get("provider", "unknown")
        row["model"] = manifest.get("model", "unknown")
    taxonomy = Counter(row["failure_stage"] for row in rows)
    operations: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        operations[row["edit_type"]]["total"] += 1
        operations[row["edit_type"]]["d_positive"] += int(float(row["d_score"] or 0.0) > 0.0)
        operations[row["edit_type"]]["accepted"] += int(bool(row["accepted"]))
        operations[row["edit_type"]]["unchanged"] += int(bool(row["unchanged_start"]))
    report = {
        "manifest": manifest,
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "d_mean": sum(float(row["d_score"] or 0.0) for row in rows) / max(len(rows), 1),
            "d_zero": sum(float(row["d_score"] or 0.0) == 0.0 for row in rows),
            "unchanged_start": sum(bool(row["unchanged_start"]) for row in rows),
            "false_completion": sum(bool(row["false_completion"]) for row in rows),
            "topology_invalid": sum(row["output_step_valid"] is False for row in rows),
            "topology_unknown": sum(row["output_step_valid"] is None for row in rows),
            "renderer_failures": sum(bool(row["renderer_failure"]) for row in rows),
            "missing_or_invalid_visual_feedback": sum(
                bool(row["missing_or_invalid_visual_feedback"]) for row in rows
            ),
            "taxonomy": dict(taxonomy),
            "operations": {name: dict(values) for name, values in operations.items()},
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    flat = [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]) if flat else [])
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.json}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
