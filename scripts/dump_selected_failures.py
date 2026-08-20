"""Dump inference facts for selected baseline row numbers."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.classify_failures import latest_log

ROOT = Path(__file__).resolve().parents[1]
WANT = {1, 2, 3, 7, 13, 15, 17, 21, 23, 24, 27, 28}


def main() -> None:
    scores = json.loads((ROOT / "docs" / "baseline_scores.json").read_text(encoding="utf-8"))
    by_id = {}
    for s in (ROOT / "data" / "model_outputs").rglob("settings.json"):
        d = json.loads(s.read_text(encoding="utf-8"))
        rid = d.get("edit_request_id")
        if rid:
            by_id[rid] = (d, s.parent / "tmp.step")
    for i, row in enumerate(scores, 1):
        if i not in WANT:
            continue
        rid = row["id"]
        settings, step = by_id.get(rid, ({}, None))
        log = latest_log(step) if step else None
        tools = []
        notes = ""
        vol = n = bbox = size = exec_ok = op = None
        acc = settings.get("accepted")
        fail = settings.get("failure_category")
        if log:
            notes = (log.get("notes") or "")[:160]
            spec = log.get("edit_spec") or {}
            op = spec.get("operation")
            calls = spec.get("tool_calls") or (log.get("execution") or {}).get("tool_calls") or []
            for c in calls:
                args = {k: v for k, v in (c.get("arguments") or {}).items() if k not in ("step_path", "script")}
                tools.append((c.get("tool_name"), args))
            ex = log.get("execution") or {}
            exec_ok = ex.get("success")
            g = ex.get("geometry_summary") or {}
            vol, n, bbox, size = g.get("volume"), g.get("n_solids"), g.get("bbox"), g.get("size")
        print("=" * 72)
        print(f"{i:02d} C={row['chamfer']} V={row['volume_f1']} D={row['diff_f1']} valid={row['valid']}")
        print("id", rid)
        print("instr", row["instruction"])
        print("accept", acc, "fail", fail, "exec", exec_ok, "op", op)
        print("tools", tools)
        print("notes", notes.encode("ascii", "replace").decode())
        print("pred n_solids", n, "vol", vol, "size", size)
        print("bbox", bbox)


if __name__ == "__main__":
    main()
