"""Ablation runner on synthetic or parquet task subsets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groundedcad.ingest.request_parser import load_request_json
from groundedcad.llm.base import MockLLMClient, build_llm_client
from groundedcad.pipeline import GroundedCADPipeline
from groundedcad.runtime.sandbox import Sandbox
from scripts.make_synthetic_examples import ensure_examples


ABLATIONS = {
    "baseline_heuristic": {
        "description": "Heuristic ground+plan, deterministic verify (no LLM)",
        "provider": "mock",
    },
    "inspect_only": {
        "description": "Same as heuristic but logs geometry census emphasis",
        "provider": "mock",
    },
    "verify_strict": {
        "description": "Heuristic with stricter volume/body checks via mock planner seed",
        "provider": "mock",
    },
    "full_mock": {
        "description": "Full loop with mock LLM responses",
        "provider": "mock",
    },
}


def run_ablations(output_root: Path, no_render: bool = True) -> dict:
    examples = ensure_examples(Path("examples/synthetic"))
    output_root.mkdir(parents=True, exist_ok=True)
    report = {"created_at": time.time(), "ablations": {}}

    for abl_name, meta in ABLATIONS.items():
        client = build_llm_client(meta["provider"], "mock")
        # For verify_strict, script planner toward smaller max volume ratio via mock
        if abl_name == "verify_strict" and isinstance(client, MockLLMClient):
            client.scripted["planner"] = {
                "operation": "fillet",
                "tool_name": "fillet_edges_by_length",
                "parameters": {"radius": 2.0, "min_length": 0.0, "max_length": 1e9, "max_edges": 8},
                "invariants": ["valid_brep"],
                "success_checks": [
                    {"name": "valid_brep", "description": "valid", "check_type": "valid_brep"},
                    {
                        "name": "volume_delta",
                        "description": "strict volume",
                        "check_type": "volume_delta",
                        "params": {"max_ratio": 0.35},
                    },
                    {"name": "connectivity", "description": "conn", "check_type": "connectivity"},
                ],
                "notes": "strict verify ablation",
            }

        pipe = GroundedCADPipeline(
            grounding_client=client,
            planning_client=client,
            critic_client=client,
            max_iters=3,
            sandbox=Sandbox(render=not no_render),
            inprocess=True,
            render=not no_render,
            user_id=f"groundedcad_{abl_name}",
            use_llm_critic=False,
        )

        rows = []
        for ex_name, ex_dir in examples.items():
            req = load_request_json(Path(ex_dir) / "request.json")
            out = output_root / abl_name / ex_name
            t0 = time.time()
            result = pipe.run(req, out)
            rows.append(
                {
                    "example": ex_name,
                    "request_id": req.request_id,
                    "accepted": result.accepted,
                    "failure_category": result.failure_category,
                    "best_iteration": result.best_iteration,
                    "has_step": bool(result.step_path and Path(result.step_path).exists()),
                    "duration_s": time.time() - t0,
                    "n_iters": len(result.iterations),
                }
            )
        n = len(rows) or 1
        report["ablations"][abl_name] = {
            "meta": meta,
            "n": len(rows),
            "valid_rate": sum(1 for r in rows if r["has_step"]) / n,
            "accept_rate": sum(1 for r in rows if r["accepted"]) / n,
            "mean_iters": sum(r["n_iters"] for r in rows) / n,
            "mean_duration_s": sum(r["duration_s"] for r in rows) / n,
            "rows": rows,
        }

    (output_root / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/ablations")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    report = run_ablations(Path(args.output), no_render=not args.render)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in report["ablations"].items()}, indent=2))


if __name__ == "__main__":
    main()
