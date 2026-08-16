"""Run GroundedCAD on all 48 text-only edits with the real LLM client, scoring
each row immediately against GT and against gpt-5.2_cadquery-script's own
submission (docs/baseline_models_scores.json / edits collection) using the
same local scorer as score_groundedcad_baseline.py (voxel divisor 64).

Speed-only deviations from the official run config (do not affect Chamfer/
VolF1/Diff F1, only packaging): use_llm_critic=False.
Cheap local tool first; escalate to grounded CadQuery + render + edit-delta
(visual_iters=5, sandbox timeout 45s). Row wall-clock 420s.

Resumable: rows with an existing pipeline_result.json in the output dir are
skipped, so an interruption doesn't lose progress.

    python scripts/run_full48.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from groundedcad.ingest.request_parser import parse_request_from_row
from groundedcad.llm.base import auto_client_from_env
from groundedcad.pipeline import GroundedCADPipeline
from groundedcad.runtime.sandbox import Sandbox
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
OUT_DIR = ROOT / "data" / "full48_gpt52"
REPORT = ROOT / "docs" / "full48_gpt52_scores.json"
GPT_USER = "gpt-5.2_cadquery-script"
ROW_TIMEOUT_S = 420


def brep_stl(brep_id):
    if not brep_id:
        return None
    p = DATA / "breps" / f"{brep_id}.stl"
    return p if p.exists() else None


def _jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        # .tolist() recurses through ragged/nested object arrays (e.g. a
        # brep_start_path column holding separate .f3d/.step candidate
        # sub-arrays) into plain nested lists; falling through to str(obj)
        # instead silently turned that into an unparseable repr, so
        # parse_request_from_row could never resolve step_path downstream.
        return _jsonable(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _latest_step(out: Path) -> Path | None:
    candidates = [out / "tmp.step"]
    iters = out / "iterations"
    if iters.exists():
        for d in sorted(iters.iterdir(), reverse=True):
            candidates.append(d / "tmp.step")
            candidates.extend(d.glob("llm_attempt_*/tmp.step"))
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _make_pipeline():
    client = auto_client_from_env()
    visual_iters = int(os.getenv("GROUNDEDCAD_VISUAL_ITERS", "5"))
    print(f"LLM {type(client).__name__} model={client.model} visual_iters={visual_iters}", flush=True)
    return GroundedCADPipeline(
        grounding_client=client,
        planning_client=client,
        critic_client=client,
        max_iters=visual_iters,
        visual_iters=visual_iters,
        sandbox=Sandbox(render=True, timeout_s=45.0),
        inprocess=False,
        render=True,
        use_llm_critic=False,
        use_llm_cadquery=True,
        user_id="groundedcad_full48",
    )


def run_one_row(payload_path: Path, out: Path) -> None:
    row = json.loads(payload_path.read_text(encoding="utf-8"))
    req = parse_request_from_row(row, DATA)
    pipe = _make_pipeline()
    pipe.run(req, out, inplace=True)


def run_row_isolated(row: dict, out: Path) -> Path | None:
    """Run one edit in a child process so a hung OCC call cannot freeze the batch."""
    out.mkdir(parents=True, exist_ok=True)
    payload_path = OUT_DIR / f"_payload_{out.name}.json"
    payload_path.write_text(json.dumps(_jsonable(row)), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--row", str(payload_path), str(out)],
        cwd=str(ROOT),
        env=env,
    )
    try:
        proc.wait(timeout=ROW_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"ROW TIMEOUT {out.name} after {ROW_TIMEOUT_S}s", flush=True)
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        payload_path.unlink(missing_ok=True)
    pred = _latest_step(out)
    dest = out / "tmp.step"
    if pred and pred.resolve() != dest.resolve():
        shutil.copy2(pred, dest)
        return dest
    return pred


def main():
    parser = argparse.ArgumentParser(description="Score GroundedCAD vs gpt-5.2 CadQuery")
    parser.add_argument("--n-rows", type=int, default=None, help="Stop after N parquet rows (after --ids filter)")
    parser.add_argument("--ids", nargs="*", default=None, help="Only these request ids")
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default data/full48_gpt52)")
    parser.add_argument("--report", type=Path, default=None, help="Scores JSON path")
    parser.add_argument("--visual-iters", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="Do not resume from an existing report")
    args = parser.parse_args()

    global OUT_DIR, REPORT
    if args.out:
        OUT_DIR = args.out
    if args.report:
        REPORT = args.report
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.visual_iters is not None:
        os.environ["GROUNDEDCAD_VISUAL_ITERS"] = str(args.visual_iters)

    df = pd.read_parquet(PARQUET)
    requests = load_collection("requests")
    edits = load_collection("edits")
    by_req_user = {(str(e.get("request")), str(e.get("user"))): e for e in edits.values()}

    visual_iters = int(os.getenv("GROUNDEDCAD_VISUAL_ITERS", "5"))
    print(
        f"batch runner: {ROW_TIMEOUT_S}s/row, visual_iters={visual_iters}, out={OUT_DIR}, report={REPORT}",
        flush=True,
    )

    rows = []
    if REPORT.exists() and not args.fresh:
        rows = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    done_ids = {r["id"] for r in rows}

    want = set(args.ids) if args.ids else None
    scored_this_run = 0
    total_cap = args.n_rows

    for _, rec in df.iterrows():
        rid = str(rec["request"])
        if want is not None and rid not in want:
            continue
        if rid in done_ids:
            continue
        if total_cap is not None and scored_this_run >= total_cap:
            break
        text = str(rec.get("request_text") or "")
        print(f"START {rid[:24]}  {(' '.join(text.split()))[:80]}", flush=True)
        row = rec.to_dict()
        out = OUT_DIR / rid

        result_path = out / "pipeline_result.json"
        if result_path.exists():
            pred = out / "tmp.step"
        else:
            try:
                pred = run_row_isolated(row, out)
            except Exception as exc:  # noqa: BLE001
                print(f"ROW FAIL {rid}: {exc}", flush=True)
                pred = _latest_step(out)

        db_req = requests.get(rid)
        gt_user = db_req.get("user") if db_req else None
        start_id = db_req.get("brep_start") if db_req else None
        start_p = brep_stl(str(start_id) if start_id else None)
        gt_edit = next(
            (e for e in edits.values() if e.get("request") == rid and e.get("user") == gt_user),
            None,
        )
        gt_p = brep_stl(str(gt_edit.get("brep_end")) if gt_edit else None)

        ours = {"chamfer": 0.0, "volume_f1": 0.0, "diff_f1": 0.0, "valid": False}
        if pred and Path(pred).exists() and start_p and gt_p:
            try:
                start_m = load_geom(start_p)
                gt_m = load_geom(gt_p)
                pred_m = load_geom(pred)
                valid = len(pred_m[0]) > 0
                chamfer = chamfer_sim(sample_points(*gt_m), sample_points(*pred_m))
                vol_f1, diff = voxel_metrics(start_m, gt_m, pred_m)
                ours = {"chamfer": round(float(chamfer), 4), "volume_f1": round(float(vol_f1), 4), "diff_f1": round(float(diff), 4), "valid": bool(valid)}
            except Exception as exc:  # noqa: BLE001
                ours = {"chamfer": 0.0, "volume_f1": 0.0, "diff_f1": 0.0, "valid": False, "note": f"error:{exc}"}

        gpt_edit = by_req_user.get((rid, GPT_USER))
        gpt_p = brep_stl(str(gpt_edit.get("brep_end"))) if gpt_edit else None
        gpt = {"chamfer": 0.0, "volume_f1": 0.0, "diff_f1": 0.0, "valid": False}
        if gpt_p and start_p and gt_p:
            try:
                start_m = load_geom(start_p)
                gt_m = load_geom(gt_p)
                gpt_m = load_geom(gpt_p)
                valid = len(gpt_m[0]) > 0
                chamfer = chamfer_sim(sample_points(*gt_m), sample_points(*gpt_m))
                vol_f1, diff = voxel_metrics(start_m, gt_m, gpt_m)
                gpt = {"chamfer": round(float(chamfer), 4), "volume_f1": round(float(vol_f1), 4), "diff_f1": round(float(diff), 4), "valid": bool(valid)}
            except Exception:
                pass

        rows.append({
            "id": rid,
            "type": classify(text),
            "instruction": " ".join(text.split())[:100],
            "ours": ours,
            "gpt52": gpt,
        })
        REPORT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        scored_this_run += 1
        done_ids.add(rid)

        n = len(rows)
        cap = total_cap or (len(want) if want is not None else 48)
        mean_ours = np.mean([r["ours"]["diff_f1"] for r in rows])
        mean_gpt = np.mean([r["gpt52"]["diff_f1"] for r in rows])
        print(
            f"{n:02d}/{cap} {rid[:24]:26s} ours D={ours['diff_f1']:.3f} gpt5.2 D={gpt['diff_f1']:.3f} "
            f"| running mean: ours={mean_ours:.4f} gpt5.2={mean_gpt:.4f}",
            flush=True,
        )

    if not rows:
        print("no rows scored", flush=True)
        return
    print(f"\n=== FINAL {len(rows)}-row means ===")
    print(f"ours:   chamfer={np.mean([r['ours']['chamfer'] for r in rows]):.4f} "
          f"volf1={np.mean([r['ours']['volume_f1'] for r in rows]):.4f} "
          f"diff_f1={np.mean([r['ours']['diff_f1'] for r in rows]):.4f} "
          f"valid={sum(r['ours']['valid'] for r in rows)}/{len(rows)}")
    print(f"gpt5.2: chamfer={np.mean([r['gpt52']['chamfer'] for r in rows]):.4f} "
          f"volf1={np.mean([r['gpt52']['volume_f1'] for r in rows]):.4f} "
          f"diff_f1={np.mean([r['gpt52']['diff_f1'] for r in rows]):.4f} "
          f"valid={sum(r['gpt52']['valid'] for r in rows)}/{len(rows)}")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--row":
        run_one_row(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
