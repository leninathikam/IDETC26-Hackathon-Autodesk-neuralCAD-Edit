"""A/B: frozen baseline (A) vs current pipeline (B) on the same text edits.

A = docs/baseline_scores.json (previous GroundedCAD STEPs already scored).
B = re-run GroundedCADPipeline (inspector + classifier + edit-delta) in-process.

  python scripts/ab_compare.py --n-rows 8
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from groundedcad.llm.base import MockLLMClient
from groundedcad.pipeline import GroundedCADPipeline
from groundedcad.runtime.sandbox import Sandbox
from groundedcad.ingest.request_parser import parse_request_from_row
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
A_JSON = ROOT / "docs" / "baseline_scores.json"
OUT_B = ROOT / "data" / "ab_b"
REPORT = ROOT / "docs" / "ab_compare.json"


def _score_one(start_p: Path, gt_p: Path, pred_p: Path) -> tuple[float, float, float, bool]:
    start_m = load_geom(start_p)
    gt_m = load_geom(gt_p)
    pred_m = load_geom(pred_p)
    valid = len(pred_m[0]) > 0
    chamfer = chamfer_sim(sample_points(*gt_m), sample_points(*pred_m))
    vol_f1, diff = voxel_metrics(start_m, gt_m, pred_m)
    return chamfer, vol_f1, diff, valid


def _means(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"chamfer": 0.0, "volume_f1": 0.0, "diff_f1": 0.0}
    return {
        "chamfer": float(np.mean([r["chamfer"] for r in rows])),
        "volume_f1": float(np.mean([r["volume_f1"] for r in rows])),
        "diff_f1": float(np.mean([r["diff_f1"] for r in rows])),
    }


def main(n_rows: int = 8, indices: list[int] | None = None, use_llm: bool = False) -> None:
    a_all = {r["id"]: r for r in json.loads(A_JSON.read_text(encoding="utf-8"))}
    df_all = pd.read_parquet(PARQUET)
    if indices:
        df = df_all.iloc[list(indices)]
    else:
        df = df_all.head(n_rows)
    requests = load_collection("requests")
    edits = load_collection("edits")
    if use_llm:
        from groundedcad.llm.base import auto_client_from_env

        client = auto_client_from_env()
        max_iters = 2
        llm_cq = True
    else:
        client = MockLLMClient()
        max_iters = 1
        llm_cq = False
    pipe = GroundedCADPipeline(
        grounding_client=client,
        planning_client=client,
        critic_client=client,
        max_iters=max_iters,
        sandbox=Sandbox(render=False),
        inprocess=True,
        render=False,
        use_llm_critic=False,
        use_llm_cadquery=llm_cq,
        user_id="groundedcad_b",
    )
    paired = []
    for _, rec in df.iterrows():
        rid = rec["request"]
        text = str(rec.get("request_text") or "")
        row = rec.to_dict()
        req = parse_request_from_row(row, DATA)
        out = OUT_B / rid
        result = pipe.run(req, out, inplace=True)
        pred = Path(result.step_path) if result.step_path else None
        db_req = requests.get(rid)
        gt_user = db_req.get("user") if db_req else None
        start_id = db_req.get("brep_start") if db_req else None
        gt_edit = next(
            (e for e in edits.values() if e.get("request") == rid and e.get("user") == gt_user),
            None,
        )
        start_p = DATA / "breps" / f"{start_id}.stl" if start_id else None
        gt_id = gt_edit.get("brep_end") if gt_edit else None
        gt_p = DATA / "breps" / f"{gt_id}.stl" if gt_id else None
        a = a_all.get(rid, {"chamfer": 0, "volume_f1": 0, "diff_f1": 0, "valid": False})
        b = {"chamfer": 0.0, "volume_f1": 0.0, "diff_f1": 0.0, "valid": False}
        if pred and pred.exists() and start_p and start_p.exists() and gt_p and gt_p.exists():
            c, v, d, ok = _score_one(start_p, gt_p, pred)
            b = {"chamfer": round(c, 4), "volume_f1": round(v, 4), "diff_f1": round(d, 4), "valid": ok}
        paired.append(
            {
                "id": rid,
                "type": classify(text),
                "instruction": " ".join(text.split())[:100],
                "A": {k: a.get(k) for k in ("chamfer", "volume_f1", "diff_f1", "valid")},
                "B": b,
                "B_accepted": result.accepted,
                "B_tool": (result.iterations[-1].notes if result.iterations else ""),
            }
        )
        print(
            f"{len(paired):02d} {classify(text):22s} "
            f"A D={float(a.get('diff_f1') or 0):.3f}  B D={b['diff_f1']:.3f}  {text[:60]}",
            flush=True,
        )

    a_rows = [p["A"] for p in paired]
    b_rows = [p["B"] for p in paired]
    summary = {"n": len(paired), "A": _means(a_rows), "B": _means(b_rows)}
    report = ROOT / "docs" / ("ab_compare_llm.json" if use_llm else "ab_compare.json")
    report.write_text(json.dumps({"summary": summary, "rows": paired}, indent=2), encoding="utf-8")
    print("\n=== A (frozen baseline) vs B ===")
    print(f"{'metric':12s} {'A':8s} {'B':8s}")
    for k in ("chamfer", "volume_f1", "diff_f1"):
        print(f"{k:12s} {summary['A'][k]:.3f}    {summary['B'][k]:.3f}")
    print(f"wrote {report}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-rows", type=int, default=8)
    p.add_argument("--indices", type=str, default="", help="Comma-separated 0-based parquet rows")
    p.add_argument("--llm", action="store_true", help="On tool/identity failure, generate CadQuery with GROUNDEDCAD_MODEL")
    args = p.parse_args()
    idx = [int(x) for x in args.indices.split(",") if x.strip()] if args.indices else None
    main(n_rows=args.n_rows, indices=idx, use_llm=args.llm)
