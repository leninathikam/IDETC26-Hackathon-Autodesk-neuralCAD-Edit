"""CLI entrypoints for GroundedCAD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groundedcad.ingest.request_parser import load_request_json, load_requests_parquet
from groundedcad.llm.base import build_llm_client
from groundedcad.pipeline import GroundedCADPipeline
from groundedcad.runtime.sandbox import Sandbox


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GroundedCAD editing agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run on a single JSON request")
    run.add_argument("--request", required=True, help="Path to request JSON")
    run.add_argument("--output", required=True, help="Output directory")
    run.add_argument("--provider", default="mock")
    run.add_argument("--model", default="mock")
    run.add_argument("--max-iters", type=int, default=5)
    run.add_argument("--inprocess", action="store_true")
    run.add_argument("--no-render", action="store_true")
    run.add_argument("--llm-critic", action="store_true")

    batch = sub.add_parser("batch", help="Run on a parquet of requests")
    batch.add_argument("--parquet", required=True)
    batch.add_argument("--root", default=None, help="Dataset root for relative paths")
    batch.add_argument("--output", required=True)
    batch.add_argument("--n-rows", type=int, default=10)
    batch.add_argument("--provider", default="mock")
    batch.add_argument("--model", default="mock")
    batch.add_argument("--max-iters", type=int, default=5)
    batch.add_argument("--inprocess", action="store_true")
    batch.add_argument("--no-render", action="store_true")

    demo = sub.add_parser("demo", help="Run built-in synthetic demo")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--inprocess", action="store_true", default=True)
    demo.add_argument("--no-render", action="store_true")

    inspect_p = sub.add_parser("inspect", help="Inspect a STEP file")
    inspect_p.add_argument("--step", required=True)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "inspect":
        from groundedcad.geometry.inspect import census_to_prompt, inspect_step

        census = inspect_step(args.step)
        print(census_to_prompt(census))
        print(
            json.dumps(
                {k: census[k] for k in ("valid", "volume", "n_solids", "n_faces", "n_edges")},
                indent=2,
            )
        )
        return 0

    if args.cmd == "demo":
        from scripts.make_synthetic_examples import ensure_examples

        examples_dir = Path("examples/synthetic")
        ensure_examples(examples_dir)
        request = load_request_json(examples_dir / "fillet_box" / "request.json")
        client = build_llm_client("mock", "mock")
        pipe = GroundedCADPipeline(
            grounding_client=client,
            planning_client=client,
            critic_client=client,
            max_iters=3,
            sandbox=Sandbox(render=not args.no_render),
            inprocess=True,
            render=not args.no_render,
        )
        result = pipe.run(request, args.output)
        print(result.model_dump_json(indent=2))
        return 0 if result.step_path else 1

    client = build_llm_client(args.provider, args.model)
    pipe = GroundedCADPipeline(
        grounding_client=client,
        planning_client=client,
        critic_client=client,
        max_iters=args.max_iters,
        sandbox=Sandbox(render=not args.no_render),
        inprocess=getattr(args, "inprocess", False),
        use_llm_critic=getattr(args, "llm_critic", False),
        render=not args.no_render,
    )

    if args.cmd == "run":
        request = load_request_json(args.request)
        result = pipe.run(request, args.output)
        print(
            json.dumps(
                {
                    "accepted": result.accepted,
                    "step": result.step_path,
                    "failure": result.failure_category,
                },
                indent=2,
            )
        )
        return 0 if result.step_path else 1

    if args.cmd == "batch":
        requests = load_requests_parquet(args.parquet, root_dir=args.root, n_rows=args.n_rows)
        summary = []
        for req in requests:
            out = Path(args.output) / req.request_id
            result = pipe.run(req, out)
            summary.append(
                {
                    "request_id": req.request_id,
                    "accepted": result.accepted,
                    "failure": result.failure_category,
                    "step": result.step_path,
                }
            )
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "batch_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
