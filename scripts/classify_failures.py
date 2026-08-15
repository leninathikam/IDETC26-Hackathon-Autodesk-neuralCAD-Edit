"""Dump tool + verifier facts for each baseline example (no scoring)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "docs" / "baseline_scores.json"
OUTPUTS = ROOT / "data" / "model_outputs"


def latest_log(step: Path) -> dict | None:
    iters = sorted((step.parent / "iterations").glob("*/log.json"))
    if not iters:
        return None
    return json.loads(iters[-1].read_text(encoding="utf-8"))


def main() -> None:
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    by_id = {}
    for s in OUTPUTS.rglob("settings.json"):
        d = json.loads(s.read_text(encoding="utf-8"))
        rid = d.get("edit_request_id")
        if rid:
            by_id[rid] = (d, s.parent / "tmp.step")
    for i, row in enumerate(scores, 1):
        rid = row["id"]
        meta = by_id.get(rid)
        tools = []
        pattern = ""
        op = ""
        notes = ""
        ok = None
        ver = {}
        exec_ok = None
        vol = {}
        fail = ""
        accepted = None
        if meta:
            settings, step = meta
            accepted = settings.get("accepted")
            fail = settings.get("failure_category")
            log = latest_log(step)
            if log:
                spec = log.get("edit_spec") or {}
                pattern = spec.get("pattern") or ""
                op = spec.get("operation") or ""
                notes = (log.get("notes") or "")[:120]
                src_calls = spec.get("tool_calls") or []
                if not src_calls:
                    src_calls = (log.get("execution") or {}).get("tool_calls") or []
                for c in src_calls:
                    args = {k: v for k, v in (c.get("arguments") or {}).items() if k not in ("step_path", "script")}
                    tools.append((c.get("tool_name"), args))
                if log.get("execution"):
                    exec_ok = log["execution"].get("success")
                    vol = (log["execution"].get("geometry_summary") or {})
                cr = log.get("critique") or {}
                ver = (cr.get("verification") or {})
                ok = cr.get("accept")
        print(
            f"{i:02d} D={row['diff_f1']:.3f} V={row['volume_f1']:.3f} C={row['chamfer']:.3f} "
            f"{row['type']:22s} accept={accepted} exec={exec_ok} crit={ok} fail={fail}"
        )
        print(f"    instr: {row['instruction']}")
        print(f"    spec: {pattern}/{op} tools={tools}")
        print(
            f"    ver: unint={ver.get('unintended_changes')} dim={ver.get('dimension_error')} "
            f"pos={ver.get('position_error')} topo={ver.get('topology_error')} "
            f"sat={ver.get('instruction_satisfied')} diag={(ver.get('diagnosis') or '')[:120]}"
        )
        if vol:
            print(f"    geom: n_solids={vol.get('n_solids')} vol={vol.get('volume')} bbox={vol.get('bbox')}")
        if notes:
            print("    notes:", notes.encode("ascii", "replace").decode("ascii"))
        print()


if __name__ == "__main__":
    main()
