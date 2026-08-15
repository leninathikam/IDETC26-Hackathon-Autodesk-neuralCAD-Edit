"""Score finished model_outputs_v2 GroundedCAD STEPs. Does not touch the running job."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.ab_compare import _score_one, classify
from scripts.score_groundedcad_baseline import DATA, PARQUET, ROOT, load_collection

V2 = ROOT / "data" / "model_outputs_v2" / "cadquery_script" / "groundedcad" / "val_edit_text"
A_JSON = ROOT / "docs" / "baseline_scores.json"


def main() -> None:
    a_all = {r["id"]: r for r in json.loads(A_JSON.read_text(encoding="utf-8"))}
    df = pd.read_parquet(PARQUET)
    by_req = {str(r["request"]): r for _, r in df.iterrows()}
    requests = load_collection("requests")
    edits = load_collection("edits")
    done = list(V2.glob("*/brep_end/*/settings.json"))
    inprog = list(V2.glob("*/brep_end/*/instruction_spec.json"))
    print(f"settings (finished-ish): {len(done)}   dirs with inspect: {len(inprog)}", flush=True)
    rows = []
    for s in sorted(done, key=lambda p: p.stat().st_mtime):
        meta = json.loads(s.read_text(encoding="utf-8"))
        rid = str(meta.get("edit_request_id") or "")
        step = s.parent / "tmp.step"
        rec = by_req.get(rid)
        req_key = rid
        instr = str(rec.get("request_text") or "") if rec is not None else ""
        if rec is None:
            prefix = rid.split("_")[0]
            try:
                t = float(rid.split("_")[-1])
            except ValueError:
                t = None
            for k, r in by_req.items():
                if not k.startswith(prefix):
                    continue
                try:
                    tk = float(k.split("_")[-1])
                except ValueError:
                    continue
                if t is not None and abs(tk - t) < 5:
                    rec = r
                    req_key = k
                    instr = str(r.get("request_text") or "")
                    break
        db_req = requests.get(req_key) or requests.get(rid) or {}
        gt_user = db_req.get("user")
        start_id = db_req.get("brep_start")
        gt_edit = next(
            (e for e in edits.values() if e.get("request") in {req_key, rid} and e.get("user") == gt_user),
            None,
        )
        start_p = DATA / "breps" / f"{start_id}.stl" if start_id else None
        gt_id = gt_edit.get("brep_end") if gt_edit else None
        gt_p = DATA / "breps" / f"{gt_id}.stl" if gt_id else None
        b = {"chamfer": None, "volume_f1": None, "diff_f1": None, "valid": step.exists()}
        if step.exists() and start_p and start_p.exists() and gt_p is not None and gt_p.exists():
            c, v, d, ok = _score_one(start_p, gt_p, step)
            b = {"chamfer": round(c, 4), "volume_f1": round(v, 4), "diff_f1": round(d, 4), "valid": ok}
        a = a_all.get(req_key) or a_all.get(rid) or {}
        ad = float(a.get("diff_f1") or 0)
        bd = float(b["diff_f1"] or 0)
        clip = " ".join(instr.split())[:56]
        print(
            f"acc={str(meta.get('accepted')):5s}  A D={ad:.3f}  B D={bd:.3f}  "
            f"{classify(instr):20s} {clip}",
            flush=True,
        )
        rows.append((b, a, meta.get("accepted")))
    ds = [x[0]["diff_f1"] for x in rows if x[0].get("diff_f1") is not None]
    cs = [x[0]["chamfer"] for x in rows if x[0].get("chamfer") is not None]
    vs = [x[0]["volume_f1"] for x in rows if x[0].get("volume_f1") is not None]
    ads = [float(x[1].get("diff_f1") or 0) for x in rows]
    print("---", flush=True)
    print(f"scored {len(ds)} / 48", flush=True)
    if ds:
        print(
            f"THIS RUN mean  Chamfer {sum(cs)/len(cs):.3f}  VolF1 {sum(vs)/len(vs):.3f}  Diff F1 {sum(ds)/len(ds):.3f}",
            flush=True,
        )
        print(f"FROZEN A same rows mean Diff F1 {sum(ads)/len(ads):.3f}", flush=True)
        acc = sum(1 for x in rows if x[2])
        print(f"critic accepted {acc}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
