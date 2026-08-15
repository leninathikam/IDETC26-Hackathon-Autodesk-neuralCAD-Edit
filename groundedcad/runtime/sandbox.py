"""Isolated CadQuery execution with timeout, artifacts, and rollback."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from groundedcad.agents.schemas import ExecutionResult, ToolCall
from groundedcad.tools.cadquery_tools import TOOL_REGISTRY, execute_tool


WORKER_SCRIPT = textwrap.dedent(
    r'''
    import json
    import sys
    import traceback
    from pathlib import Path

    def main():
        payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        out_dir = Path(payload["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = payload.get("mode", "tool")
        result_meta = {"success": False}
        try:
            sys.path.insert(0, payload.get("project_root", "."))
            from groundedcad.geometry.render import export_all_artifacts
            from groundedcad.geometry.inspect import inspect_shape, shape_from_workplane
            from groundedcad.tools.cadquery_tools import execute_tool

            if mode == "tool":
                result = execute_tool(payload["tool_name"], payload.get("arguments", {}))
            elif mode == "script":
                import cadquery as cq
                from cadquery import exporters
                import os
                input_file = payload.get("input_file")
                if input_file:
                    input_file = str(Path(input_file).expanduser().resolve())
                ns = {
                    "cq": cq,
                    "cadquery": cq,
                    "exporters": exporters,
                    "os": os,
                    "Path": Path,
                    "import_step": lambda p=None: cq.importers.importStep(p or input_file),
                }
                exec(payload["script"], ns, ns)
                if "my_cad_function" not in ns:
                    raise ValueError("my_cad_function not found")
                orig = ns["my_cad_function"]
                def my_cad_function(args):
                    if not isinstance(args, dict):
                        args = {"input_file": input_file, "output_dir": str(out_dir)}
                    else:
                        args = dict(args)
                        path = args.get("input_file")
                        if not isinstance(path, str) or not path:
                            args["input_file"] = input_file
                        else:
                            args["input_file"] = str(Path(path).expanduser().resolve())
                    return orig(args)
                result = my_cad_function({
                    "input_file": input_file,
                    "output_dir": str(out_dir),
                })
            else:
                raise ValueError(f"Unknown mode {mode}")

            artifacts = export_all_artifacts(result, out_dir, render=payload.get("render", True))
            shape = shape_from_workplane(result)
            census = inspect_shape(shape, source=artifacts["step"])
            result_meta.update({
                "success": True,
                "step_path": artifacts["step"],
                "stl_path": artifacts["stl"],
                "image_paths": artifacts["views"],
                "geometry_summary": census,
            })
        except Exception as e:
            result_meta["error"] = f"{e}\n{traceback.format_exc()}"
        Path(payload["result_file"]).write_text(json.dumps(result_meta), encoding="utf-8")

    if __name__ == "__main__":
        main()
    '''
).strip()


class Sandbox:
    def __init__(
        self,
        project_root: str | Path | None = None,
        timeout_s: float = 90.0,
        render: bool = True,
    ):
        self.project_root = str(Path(project_root or Path(__file__).resolve().parents[2]))
        self.timeout_s = timeout_s
        self.render = render

    def _run_worker(self, payload: dict[str, Any], output_dir: Path) -> ExecutionResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with tempfile.TemporaryDirectory(prefix="groundedcad_") as tmp:
            tmp_path = Path(tmp)
            payload_path = tmp_path / "payload.json"
            result_path = tmp_path / "result.json"
            worker_path = tmp_path / "worker.py"
            payload = {
                **payload,
                "output_dir": str(output_dir),
                "result_file": str(result_path),
                "project_root": self.project_root,
                "render": self.render,
            }
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            worker_path.write_text(WORKER_SCRIPT, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(worker_path), str(payload_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    cwd=self.project_root,
                )
                stdout = proc.stdout
                stderr = proc.stderr
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    success=False,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    error=f"Sandbox timeout after {self.timeout_s}s",
                    duration_s=time.time() - start,
                )

            if result_path.exists():
                meta = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                meta = {
                    "success": False,
                    "error": f"Worker produced no result. stderr={stderr}",
                }
            return ExecutionResult(
                success=bool(meta.get("success")),
                step_path=meta.get("step_path"),
                stl_path=meta.get("stl_path"),
                image_paths=meta.get("image_paths") or {},
                stdout=stdout,
                stderr=stderr,
                geometry_summary=meta.get("geometry_summary") or {},
                error=meta.get("error"),
                duration_s=time.time() - start,
                script=payload.get("script"),
                tool_calls=[
                    ToolCall(
                        tool_name=payload.get("tool_name", "raw_cadquery"),
                        arguments=payload.get("arguments") or {},
                    )
                ]
                if payload.get("mode") == "tool"
                else [
                    ToolCall(tool_name="raw_cadquery", arguments={}, rationale="script fallback")
                ],
            )

    def run_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        output_dir: str | Path,
    ) -> ExecutionResult:
        # Snapshot for rollback if previous step exists
        out = Path(output_dir)
        backup = None
        existing = out / "tmp.step"
        if existing.exists():
            backup = out / "tmp.step.bak"
            shutil.copy2(existing, backup)
        result = self._run_worker(
            {"mode": "tool", "tool_name": tool_name, "arguments": arguments},
            out,
        )
        if not result.success and backup and backup.exists():
            shutil.copy2(backup, existing)
        return result

    def run_script(
        self,
        script: str,
        input_file: str,
        output_dir: str | Path,
    ) -> ExecutionResult:
        from groundedcad.geometry.fallback import cadquery_available

        out = Path(output_dir)
        # Without CadQuery, raw scripts cannot execute; copy input as identity edit.
        if not cadquery_available():
            start = time.time()
            out.mkdir(parents=True, exist_ok=True)
            from groundedcad.geometry.fallback import export_simple_stl, export_simple_step, inspect_simple, load_simple

            solid = load_simple(input_file)
            step = export_simple_step(solid, out / "tmp.step")
            stl = export_simple_stl(solid, out / "tmp.stl")
            return ExecutionResult(
                success=False,
                step_path=None,
                stl_path=None,
                geometry_summary=inspect_simple(solid),
                duration_s=time.time() - start,
                script=script,
                error="IDENTITY_OUTPUT: CadQuery unavailable; refusing silent start-copy as a successful edit.",
                tool_calls=[ToolCall(tool_name="raw_cadquery", arguments={}, rationale="fallback identity")],
                stdout="CadQuery unavailable; identity copy is not a successful edit.",
            )

        backup = None
        existing = out / "tmp.step"
        if existing.exists():
            backup = out / "tmp.step.bak"
            shutil.copy2(existing, backup)
        result = self._run_worker(
            {
                "mode": "script",
                "script": script,
                "input_file": input_file,
            },
            out,
        )
        if not result.success and backup and backup.exists():
            shutil.copy2(backup, existing)
        return result

    def run_inprocess_tool(self, tool_name: str, arguments: dict[str, Any], output_dir: str | Path) -> ExecutionResult:
        """Faster path for tests (no subprocess)."""
        from groundedcad.geometry.inspect import inspect_shape, shape_from_workplane
        from groundedcad.geometry.render import export_all_artifacts

        start = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            result = execute_tool(tool_name, arguments)
            artifacts = export_all_artifacts(result, out, render=self.render)
            census = inspect_shape(shape_from_workplane(result), source=artifacts["step"])
            return ExecutionResult(
                success=True,
                step_path=artifacts["step"],
                stl_path=artifacts["stl"],
                image_paths=artifacts["views"],
                geometry_summary=census,
                duration_s=time.time() - start,
                tool_calls=[ToolCall(tool_name=tool_name, arguments=arguments)],
            )
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                success=False,
                error=f"{exc}\n{traceback.format_exc()}",
                duration_s=time.time() - start,
                tool_calls=[ToolCall(tool_name=tool_name, arguments=arguments)],
            )
