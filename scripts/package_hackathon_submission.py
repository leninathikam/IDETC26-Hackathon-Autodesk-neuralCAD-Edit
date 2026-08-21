"""Assemble the Night_Owls hackathon submission folder.

This script does not run inference. It packages an already completed benchmark
run into the structure requested by the public neuralCAD-Edit instructions:
CAD artifacts, scores, qualitative examples, and the presentation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_VIEWS = ("iso", "top", "bottom", "front", "back", "left", "right")
QUALITATIVE_IDS = [
    "SUJ2G2UMJQR7PMBX_1759209987.785593",
    "3YH2WFSRM22W7DKT_1769773335.525203",
    "B7A2N74ZJBF9MZHU_1770174133.012106",
    "F332D3FXML85WLR2_1769607142.566352",
    "ZK22J6VYRKQ2RTFD_1758874422.1403751",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or src.stat().st_size <= 0:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(path: tuple[str, str]) -> float:
        vals = []
        for row in rows:
            cur: Any = row
            for key in path:
                cur = cur.get(key, {}) if isinstance(cur, dict) else {}
            vals.append(float(cur or 0.0))
        return round(sum(vals) / max(len(vals), 1), 4)

    buckets: dict[str, int] = {}
    for row in rows:
        bucket = str(row.get("failure_bucket") or "unknown")
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return {
        "n_rows": len(rows),
        "mean_chamfer": mean(("ours", "chamfer")),
        "mean_volume_f1": mean(("ours", "volume_f1")),
        "mean_diff_f1": mean(("ours", "diff_f1")),
        "valid": sum(1 for row in rows if row.get("ours", {}).get("valid")),
        "failure_buckets": dict(sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def _render_missing_views(row_dir: Path, timeout_s: int) -> None:
    if not (row_dir / "tmp.step").exists():
        return
    missing = [
        f"{name}.png"
        for name in REQUIRED_VIEWS
        if not (row_dir / f"{name}.png").exists()
    ]
    if not missing:
        return
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--render-one",
        str(row_dir),
    ]
    try:
        subprocess.run(cmd, cwd=ROOT, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        print(f"render timeout: {row_dir.name}", flush=True)


def _render_one(row_dir: Path) -> None:
    from groundedcad.geometry.inspect import load_step, shape_from_workplane
    from groundedcad.geometry.render import render_canonical_views

    shape = shape_from_workplane(load_step(row_dir / "tmp.step"))
    render_canonical_views(
        shape,
        row_dir,
        views=["toprightiso", "front", "back", "left", "right", "top", "bottom"],
    )


def package(args: argparse.Namespace) -> int:
    outputs = (ROOT / args.outputs).resolve()
    scores_path = (ROOT / args.scores).resolve()
    dest = (ROOT / args.dest).resolve()
    presentation = (ROOT / args.presentation).resolve()
    presentation_pdf = (ROOT / args.presentation_pdf).resolve() if args.presentation_pdf else None

    if not outputs.exists():
        raise SystemExit(f"Missing outputs folder: {outputs}")
    if not scores_path.exists():
        raise SystemExit(f"Missing scores JSON: {scores_path}")

    rows = _read_json(scores_path)
    if not isinstance(rows, list):
        raise SystemExit(f"Scores JSON must be a list: {scores_path}")
    if len(rows) != 48 and not args.allow_partial:
        raise SystemExit(
            f"Refusing to package partial run: {len(rows)}/48 rows. "
            "Use --allow-partial only for a draft package."
        )

    dest.mkdir(parents=True, exist_ok=True)
    model_dest = dest / "model_outputs"
    model_dest.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    score_ids = {str(row.get("id")) for row in rows}
    for row in rows:
        rid = str(row.get("id"))
        src = outputs / rid
        if args.render_missing_views:
            _render_missing_views(src, args.render_timeout_s)
        dst = model_dest / rid
        copied: list[str] = []
        missing: list[str] = []
        for name in ["tmp.step", "tmp.stl", "settings.json"]:
            if _copy_if_exists(src / name, dst / name):
                copied.append(name)
            else:
                missing.append(name)
        for view in REQUIRED_VIEWS:
            name = f"{view}.png"
            if _copy_if_exists(src / name, dst / name):
                copied.append(name)
            else:
                missing.append(name)
        _copy_if_exists(src / "orthographic_sheet.png", dst / "orthographic_sheet.png")
        _copy_if_exists(src / "pipeline_result.json", dst / "pipeline_result.json")
        artifacts.append(
            {
                "request_id": rid,
                "type": row.get("type"),
                "diff_f1": row.get("ours", {}).get("diff_f1"),
                "execution_status": row.get("execution_status"),
                "failure_bucket": row.get("failure_bucket"),
                "copied": copied,
                "missing": missing,
            }
        )

    scores_dest = dest / "scores"
    scores_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scores_path, scores_dest / scores_path.name)
    summary = _score_summary(rows)
    (scores_dest / "score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    examples_dest = dest / "qualitative_examples"
    examples_dest.mkdir(parents=True, exist_ok=True)
    for rid in QUALITATIVE_IDS:
        src = outputs / rid
        dst = examples_dest / rid
        dst.mkdir(parents=True, exist_ok=True)
        for name in ["tmp.step", "tmp.stl", "settings.json", "orthographic_sheet.png"]:
            _copy_if_exists(src / name, dst / name)
        for view in REQUIRED_VIEWS:
            _copy_if_exists(src / f"{view}.png", dst / f"{view}.png")
        if rid not in score_ids:
            (dst / "MISSING_FROM_SCORE_RUN.txt").write_text(
                "This required qualitative request is not present in the packaged score run.\n",
                encoding="utf-8",
            )

    if presentation.exists():
        shutil.copy2(presentation, dest / presentation.name)
    if presentation_pdf and presentation_pdf.exists():
        shutil.copy2(presentation_pdf, dest / presentation_pdf.name)

    manifest = {
        "team": args.team,
        "source_outputs": str(outputs),
        "source_scores": str(scores_path),
        "presentation": str(presentation) if presentation.exists() else None,
        "presentation_pdf": str(presentation_pdf) if presentation_pdf and presentation_pdf.exists() else None,
        "score_summary": summary,
        "required_qualitative_request_ids": QUALITATIVE_IDS,
        "artifact_rows": artifacts,
    }
    (dest / "submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    missing_rows = [row for row in artifacts if row["missing"]]
    readme = [
        "# Night_Owls Submission",
        "",
        "Contents prepared for the IDETC 2026 Autodesk neuralCAD-Edit hackathon.",
        "",
        f"- Rows packaged: {len(rows)}/48",
        f"- Chamfer similarity: {summary['mean_chamfer']}",
        f"- Volume F1: {summary['mean_volume_f1']}",
        f"- Diff F1: {summary['mean_diff_f1']}",
        f"- Valid outputs: {summary['valid']}/{summary['n_rows']}",
        "",
        "Official required qualitative request IDs are copied under `qualitative_examples/` when available.",
        "",
        "Known packaging warnings:",
    ]
    if missing_rows:
        readme.extend(
            f"- {row['request_id']}: missing {', '.join(row['missing'])}"
            for row in missing_rows[:80]
        )
    else:
        readme.append("- None.")
    readme.append("")
    (dest / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(json.dumps({"dest": str(dest), "summary": summary, "missing_rows": len(missing_rows)}, indent=2))
    return 0 if not missing_rows and len(rows) == 48 else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="Night_Owls")
    parser.add_argument("--outputs", default="data/full48_openai-gpt-5-2_fresh_rerun")
    parser.add_argument("--scores", default="docs/full48_openai-gpt-5-2_fresh_rerun_scores.json")
    parser.add_argument("--dest", default="submissions/Night_Owls")
    parser.add_argument("--presentation", default="Night_Owls_project_edited.pptx")
    parser.add_argument("--presentation-pdf", default="docs/Night_Owls_presentation.pdf")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--render-missing-views", action="store_true")
    parser.add_argument("--render-timeout-s", type=int, default=45)
    parser.add_argument("--render-one", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.render_one is not None:
        _render_one(args.render_one)
        return
    raise SystemExit(package(args))


if __name__ == "__main__":
    main()
