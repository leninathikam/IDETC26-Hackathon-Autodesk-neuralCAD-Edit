"""Package synthetic demo outputs into neuralCAD-Edit-compatible folders."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groundedcad.ingest.request_parser import load_request_json
from groundedcad.llm.base import build_llm_client
from groundedcad.pipeline import GroundedCADPipeline
from groundedcad.runtime.sandbox import Sandbox
from scripts.make_synthetic_examples import ensure_examples


def package(output_root: Path) -> list[dict]:
    examples = ensure_examples(Path("examples/synthetic"))
    client = build_llm_client("mock", "mock")
    pipe = GroundedCADPipeline(
        grounding_client=client,
        planning_client=client,
        critic_client=client,
        max_iters=3,
        sandbox=Sandbox(render=False),
        inprocess=True,
        render=False,
        user_id="groundedcad_demo",
    )
    packaged = []
    for name, d in examples.items():
        req = load_request_json(Path(d) / "request.json")
        out = output_root / name
        result = pipe.run(req, out)
        packaged.append(
            {
                "example": name,
                "request_id": req.request_id,
                "step": result.step_path,
                "stl": result.stl_path,
                "accepted": result.accepted,
                "output_dir": str(out),
            }
        )
    (output_root / "package_manifest.json").write_text(json.dumps(packaged, indent=2), encoding="utf-8")
    return packaged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/submission_demo")
    args = ap.parse_args()
    rows = package(Path(args.output))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
