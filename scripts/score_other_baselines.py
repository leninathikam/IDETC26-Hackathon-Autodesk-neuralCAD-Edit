"""Score gpt-5.2 / gemini-3-pro / claude-sonnet-4.5's own cadquery-script
submissions on our 48 text-only edits, with the SAME local scorer
(voxel divisor 64, Chamfer/VolF1/DiffF1) used for GroundedCAD in
score_groundedcad_baseline.py and scripts/ab_compare.py.

These are the actual baseline model outputs shipped in the hackathon data
(edits collection, user == "<model>_cadquery-script", geometry in
data/edit_192_external/breps/). Not the official leaderboard scorer — same
caveat as score_groundedcad_baseline.py: voxel divisor 64 vs official 128,
use to RANK, not to quote as the official leaderboard number.

    python scripts/score_other_baselines.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.score_groundedcad_baseline import (
    DATA,
    PARQUET,
    chamfer_sim,
    classify,
    load_collection,
    load_geom,
    sample_points,
    voxel_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "baseline_models_scores.json"

MODEL_USERS = [
    "gpt-5.2_cadquery-script",
    "gemini-3-pro_cadquery-script",
    "claude-sonnet-4.5_cadquery-script",
]


def brep_stl(brep_id: str | None) -> Path | None:
    if not brep_id:
        return None
    p = DATA / "breps" / f"{brep_id}.stl"
    return p if p.exists() else None


def main() -> None:
    df = pd.read_parquet(PARQUET)
    requests = load_collection("requests")
    edits = load_collection("edits")

    # index edits by (request, user) for O(1) lookup
    by_req_user: dict[tuple[str, str], dict] = {}
    for e in edits.values():
        by_req_user[(str(e.get("request")), str(e.get("user")))] = e

    rows = []
    for _, rec in df.iterrows():
        rid = str(rec["request"])
        text = str(rec.get("request_text") or "")
        etype = classify(text)
        req = requests.get(rid)
        if req is None:
            continue
        gt_user = req.get("user")
        start_id = req.get("brep_start")
        start_p = brep_stl(str(start_id) if start_id else None)
        gt_edit = next(
            (e for e in edits.values() if e.get("request") == rid and e.get("user") == gt_user),
            None,
        )
        gt_p = brep_stl(str(gt_edit.get("brep_end")) if gt_edit else None)
        if not start_p or not gt_p:
            print(f"{rid}: missing start/gt, skipping", flush=True)
            continue
        start_m = load_geom(start_p)
        gt_m = load_geom(gt_p)

        row = {"id": rid, "type": etype, "instruction": " ".join(text.split())[:100]}
        for user in MODEL_USERS:
            model_edit = by_req_user.get((rid, user))
            pred_p = brep_stl(str(model_edit.get("brep_end"))) if model_edit else None
            if not pred_p:
                row[user] = {"chamfer": 0.0, "volume_f1": 0.0, "diff_f1": 0.0, "valid": False, "note": "no_pred"}
                continue
            try:
                pred_m = load_geom(pred_p)
                valid = len(pred_m[0]) > 0
                chamfer = chamfer_sim(sample_points(*gt_m), sample_points(*pred_m))
                vol_f1, diff = voxel_metrics(start_m, gt_m, pred_m)
                row[user] = {
                    "chamfer": round(float(chamfer), 4),
                    "volume_f1": round(float(vol_f1), 4),
                    "diff_f1": round(float(diff), 4),
                    "valid": bool(valid),
                }
            except Exception as exc:  # noqa: BLE001
                row[user] = {"chamfer": 0.0, "volume_f1": 0.0, "diff_f1": 0.0, "valid": False, "note": f"error:{exc}"}
        rows.append(row)
        print(
            f"{len(rows):02d} {etype:22s} "
            + " ".join(f"{u.split('_')[0]:>7s} D={row[u]['diff_f1']:.3f}" for u in MODEL_USERS)
            + f"  {text[:50]}",
            flush=True,
        )

    REPORT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== mean over scored rows (missing pred = 0) ===")
    print(f"{'model':30s} {'n':>3s} {'Chamfer':>8s} {'VolF1':>8s} {'DiffF1':>8s} {'valid':>6s}")
    for user in MODEL_USERS:
        vals = [r[user] for r in rows]
        n = len(vals)
        print(
            f"{user:30s} {n:3d} "
            f"{np.mean([v['chamfer'] for v in vals]):8.4f} "
            f"{np.mean([v['volume_f1'] for v in vals]):8.4f} "
            f"{np.mean([v['diff_f1'] for v in vals]):8.4f} "
            f"{sum(v['valid'] for v in vals):3d}/{n}"
        )
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
