"""Score GroundedCAD outputs with Autodesk official metrics.

Uses the repo functions (voxel divisor 128, Open3D occupancy, bbox-normalized
Chamfer similarity). Missing predictions score 0.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # `python scripts/score_official_metrics.py` sets sys.path to scripts/,
    # not the repository root.  Keep direct invocation working as documented.
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.score_groundedcad_baseline import (
    DATA,
    PARQUET,
    brep_stl,
    classify,
    load_collection,
)

DEFAULT_OUTPUTS = ROOT / "data" / "model_outputs_official48"
REPORT = ROOT / "docs" / "official48_scores.json"


class _PathDB:
    def __init__(self, root_dir: str | Path):
        self.root_dir = str(root_dir)


def _require_official_runtime() -> None:
    """Open3D wheels used by the benchmark are available for CPython 3.12."""
    if sys.version_info[:2] != (3, 12):
        found = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "Official scoring requires the project Python 3.12 environment "
            f"because Open3D is unavailable for Python {found}. "
            "Create it with `uv venv --python 3.12` and install with "
            "`.venv\\Scripts\\python -m pip install -e .[eval]`."
        )


def find_latest_preds(output_dir: Path) -> dict[str, tuple[Path, bool]]:
    """Find each final artifact and whether it is an explicit start-copy."""
    latest: dict[str, tuple[float, Path, bool]] = {}
    for settings in output_dir.rglob("settings.json"):
        data = json.loads(settings.read_text(encoding="utf-8"))
        rid = data.get("edit_request_id")
        stl = settings.parent / "tmp.stl"
        step = settings.parent / "tmp.step"
        pred = stl if stl.exists() else step
        if not rid or not pred.exists():
            continue
        stamp = float(data.get("start_time") or 0.0)
        # The official-harness adapter serializes selected pipeline metadata
        # at the top level.  A false-completion with only tmp.step is the
        # pipeline's intentional start-copy fallback (a real successful edit
        # always arrives with tmp.stl from export_all_artifacts).
        start_copy = bool(data.get("valid_start_copy")) or (
            not stl.exists()
            and step.exists()
            and str(data.get("failure_category") or "") == "false_completion"
        )
        prev = latest.get(rid)
        if prev is None or stamp >= prev[0]:
            latest[rid] = (stamp, pred, start_copy)
    return {rid: (path, start_copy) for rid, (_, path, start_copy) in latest.items()}


def _rel(path: Path, db: _PathDB) -> str:
    return str(path.resolve().relative_to(Path(db.root_dir).resolve()))


def _mesh_path(pred: Path) -> Path:
    """Return the harness-produced STL required by the official metrics."""
    if pred.suffix.lower() == ".stl":
        return pred
    raise RuntimeError(
        f"Missing harness STL for official scoring: {pred}. "
        "Regenerate this row so it writes tmp.stl; converting failed STEP "
        "artifacts during scoring can hang CadQuery and is not benchmark-equivalent."
    )


def score_pair(start_p: Path, gt_p: Path, pred_p: Path) -> tuple[float, float, float]:
    try:
        from src.utils.evals_diff import diff_f1, volumetric_f1
        from src.utils.evals_feature_geometric import chamfer_similarity_norm
    except ModuleNotFoundError as exc:
        if exc.name == "open3d":
            raise RuntimeError(
                "Official scoring requires open3d. Install the evaluation "
                "dependencies with `uv pip install --python .\\.venv\\Scripts\\python.exe "
                "-e \".[official-score]\"`."
            ) from exc
        raise

    db = _PathDB(ROOT)
    gt_rel = _rel(gt_p, db)
    pred_rel = _rel(_mesh_path(pred_p), db)
    start_rel = _rel(start_p, db)
    chamfer = float(chamfer_similarity_norm(gt_rel, pred_rel, db) or 0.0)
    vol = volumetric_f1(gt_rel, pred_rel, db, voxel_divisor=128)
    diff = diff_f1(gt_rel, pred_rel, db, voxel_divisor=128, start_rel=start_rel)
    vol = 0.0 if vol != vol else float(vol)
    diff = 0.0 if diff != diff else float(diff)
    return chamfer, vol, diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()

    _require_official_runtime()

    requests = load_collection("requests")
    edits = load_collection("edits")
    df = pd.read_parquet(PARQUET)
    preds = find_latest_preds(args.outputs)
    print(f"parquet={len(df)} preds={len(preds)} outputs={args.outputs}", flush=True)

    rows = []
    for _, rec in df.iterrows():
        rid = rec["request"]
        text = str(rec.get("request_text") or "")
        etype = classify(text)
        pred_info = preds.get(rid)
        req = requests.get(rid)
        chamfer = vol_f1 = diff = 0.0
        valid = False
        note = ""
        if pred_info is None:
            note = "no_pred"
        elif req is None:
            note = "request_not_in_db"
        else:
            gt_user = req.get("user")
            start_id = req.get("brep_start")
            gt_edit = next(
                (e for e in edits.values() if e.get("request") == rid and e.get("user") == gt_user),
                None,
            )
            start_p = brep_stl(str(start_id) if start_id else None)
            gt_p = brep_stl(str(gt_edit.get("brep_end")) if gt_edit else None)
            if not start_p or not gt_p:
                note = "missing_gt_or_start"
            else:
                try:
                    pred, start_copy = pred_info
                    # Older pipeline runs copied the unchanged start STEP but
                    # omitted tmp.stl.  The geometry is known exactly: use the
                    # input STL rather than attempting a potentially hanging
                    # STEP re-export. New runs package tmp.stl directly.
                    metric_pred = start_p if start_copy and pred.suffix.lower() != ".stl" else pred
                    chamfer, vol_f1, diff = score_pair(start_p, gt_p, metric_pred)
                    valid = metric_pred.exists() and metric_pred.stat().st_size > 0
                    if metric_pred == start_p and pred != start_p:
                        note = "start_copy_from_input_stl"
                except Exception as exc:  # noqa: BLE001
                    note = f"metric_error:{exc}"
        rows.append(
            {
                "id": rid,
                "type": etype,
                "chamfer": round(chamfer, 4),
                "volume_f1": round(vol_f1, 4),
                "diff_f1": round(diff, 4),
                "valid": valid,
                "instruction": " ".join(text.split())[:120],
                "note": note,
            }
        )
        print(
            f"{len(rows):02d}/{len(df)} {etype:22s} C={chamfer:.3f} V={vol_f1:.3f} D={diff:.3f} "
            f"valid={valid} {note}",
            flush=True,
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    scored = [r for r in rows if r["note"] != "no_pred"]
    print("\n=== official metrics (missing=0) ===")
    print(
        f"ours n={len(rows)} scored={len(scored)} "
        f"chamfer={np.mean([r['chamfer'] for r in rows]):.4f} "
        f"volf1={np.mean([r['volume_f1'] for r in rows]):.4f} "
        f"diff_f1={np.mean([r['diff_f1'] for r in rows]):.4f} "
        f"valid={sum(r['valid'] for r in rows)}/{len(rows)}"
    )
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
