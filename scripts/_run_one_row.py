"""One-off: run a single parquet row and score it. Does not touch ab_compare.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from groundedcad.ingest.request_parser import parse_request_from_row
from groundedcad.llm.base import auto_client_from_env
from groundedcad.pipeline import GroundedCADPipeline
from groundedcad.runtime.sandbox import Sandbox
from scripts.ab_compare import A_JSON, DATA, PARQUET, _score_one, classify
from scripts.score_groundedcad_baseline import load_collection


def main(idx: int = 1) -> None:
    df = pd.read_parquet(PARQUET)
    rec = df.iloc[idx]
    rid = rec["request"]
    text = str(rec.get("request_text") or "")
    print("ROW", idx, rid, flush=True)
    print("INSTR", " ".join(text.split())[:240], flush=True)
    print("CLASS", classify(text), flush=True)
    a_all = {r["id"]: r for r in json.loads(A_JSON.read_text(encoding="utf-8"))}
    a = a_all.get(rid, {})
    print("A frozen", {k: a.get(k) for k in ("chamfer", "volume_f1", "diff_f1", "valid")}, flush=True)

    client = auto_client_from_env()
    print("LLM", getattr(client, "model", type(client).__name__), flush=True)
    pipe = GroundedCADPipeline(
        grounding_client=client,
        planning_client=client,
        critic_client=client,
        max_iters=2,
        sandbox=Sandbox(render=False),
        inprocess=True,
        render=False,
        use_llm_critic=False,
        use_llm_cadquery=True,
        user_id="one_row",
    )
    req = parse_request_from_row(rec.to_dict(), DATA)
    out = ROOT / "data" / "one_row" / rid
    result = pipe.run(req, out, inplace=True)
    notes = result.iterations[-1].notes if result.iterations else None
    tokens = result.iterations[-1].token_counts if result.iterations else {}
    print("accepted", result.accepted, "fail", result.failure_category, flush=True)
    print("tool", notes, flush=True)
    print("step", result.step_path, flush=True)
    print("tokens", tokens, flush=True)

    requests = load_collection("requests")
    edits = load_collection("edits")
    db_req = requests.get(rid) or {}
    gt_user = db_req.get("user")
    start_id = db_req.get("brep_start")
    gt_edit = next(
        (e for e in edits.values() if e.get("request") == rid and e.get("user") == gt_user),
        None,
    )
    start_p = DATA / "breps" / f"{start_id}.stl"
    gt_id = gt_edit.get("brep_end") if gt_edit else None
    gt_p = DATA / "breps" / f"{gt_id}.stl" if gt_id else None
    pred = Path(result.step_path) if result.step_path else None
    b: dict = {}
    if pred and pred.exists() and start_p.exists() and gt_p is not None and gt_p.exists():
        c, v, d, ok = _score_one(start_p, gt_p, pred)
        b = {"chamfer": round(c, 4), "volume_f1": round(v, 4), "diff_f1": round(d, 4), "valid": ok}
    print("B", b, flush=True)
    payload = {
        "index": idx,
        "id": rid,
        "instruction": " ".join(text.split())[:240],
        "A": {k: a.get(k) for k in ("chamfer", "volume_f1", "diff_f1", "valid")},
        "B": b,
        "accepted": result.accepted,
        "failure_category": result.failure_category,
        "tool": notes,
        "model": getattr(client, "model", None),
        "tokens": tokens,
    }
    dest = ROOT / "docs" / "one_row_run.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(idx)
