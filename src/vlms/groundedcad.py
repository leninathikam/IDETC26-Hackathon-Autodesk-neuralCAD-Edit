"""Official-harness adapter: GroundedCAD pipeline as a VLM function."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.vlms.base_vlm import BaseVLM, GenerateResponseResult

from groundedcad.ingest.request_parser import parse_request_from_row
from groundedcad.llm.base import build_llm_client
from groundedcad.pipeline import GroundedCADPipeline
from groundedcad.runtime.sandbox import Sandbox


class VLM(BaseVLM):
    """Plugs GroundedCAD into src/scripts_benchmark_inference/run_harness.py."""

    def __init__(self, config: dict, cache: bool = True):
        super().__init__(config=config, cache=cache)
        provider = os.getenv("GROUNDEDCAD_PROVIDER") or config.get("llm_provider") or "mock"
        model = os.getenv("GROUNDEDCAD_MODEL") or config.get("llm_model") or config.get("model", "mock")
        self.client = build_llm_client(provider, model)
        self.pipeline = GroundedCADPipeline(
            grounding_client=self.client,
            planning_client=self.client,
            critic_client=self.client,
            max_iters=int(config.get("max_iters", 5)),
            sandbox=Sandbox(render=bool(config.get("render", True)), timeout_s=60.0),
            user_id=config.get("userId", "groundedcad"),
            inprocess=bool(config.get("inprocess", False)),
            use_llm_critic=bool(config.get("use_llm_critic", False)),
            render=bool(config.get("render", True)),
            visual_iters=int(config.get("visual_iters", config.get("max_iters", 5))),
        )

    def create_messages(self, inputs: list, sys=None) -> list:
        return list(inputs)

    def generate_response(self, messages: list, output_path=None, return_token_counts=False) -> GenerateResponseResult:
        return GenerateResponseResult(response_json={}, response_text="", token_counts={})

    def groundedcad_edit(
        self,
        instruction_text: str,
        task_info_dict: dict[str, Any],
        harness_script_file: str,
        output_dir: str,
        **_: Any,
    ) -> dict[str, Any]:
        """Entry point used by run_harness.py via config['function']."""
        row = dict(task_info_dict)
        if instruction_text and not row.get("request_text") and not row.get("instruction"):
            row["instruction"] = instruction_text
        root = row.get("root_dir") or os.getenv("GROUNDEDCAD_DATA_ROOT", ".")
        request = parse_request_from_row(row, root)
        result = self.pipeline.run(request, Path(output_dir), inplace=True)

        # Copy artifacts into the harness output_dir if pipeline wrote elsewhere
        step = result.step_path
        return_dict: dict[str, Any] = {
            "token_counts": (result.iterations[-1].token_counts if result.iterations else self.client.total_tokens),
            "method": "groundedcad",
            "accepted": result.accepted,
            "failure_category": result.failure_category,
        }
        if step:
            return_dict["filename"] = step
        return return_dict
