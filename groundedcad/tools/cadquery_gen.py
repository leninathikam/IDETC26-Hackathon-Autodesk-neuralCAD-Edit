"""CadQuery Generator stage: turn a minimal edit plan into executable CadQuery."""

from __future__ import annotations

from groundedcad.agents.schemas import ToolCall
from groundedcad.tools.cadquery_tools import RAW_FUNCTION_TEMPLATE


def generate_cadquery(tool: ToolCall, input_step: str) -> str:
    """Produce a my_cad_function script. Constrained tools stay preferred at runtime."""
    if tool.tool_name == "raw_cadquery":
        return str(tool.arguments.get("script") or RAW_FUNCTION_TEMPLATE)

    args = {k: v for k, v in tool.arguments.items() if k != "step_path"}
    follow = [
        {"tool_name": f.tool_name, "arguments": {k: v for k, v in f.arguments.items() if k != "step_path"}}
        for f in (tool.followups or [])
    ]
    return f'''
def my_cad_function(args):
    import os
    from pathlib import Path
    from groundedcad.tools.cadquery_tools import execute_tool
    input_file = os.path.expanduser(args["input_file"])
    result = execute_tool({tool.tool_name!r}, {{"step_path": input_file, **{args!r}}})
    path = str(Path(input_file))
    for follow in {follow!r}:
        step = path
        if hasattr(result, "val"):
            from groundedcad.geometry.render import export_step
            step = str(export_step(result, args.get("output_dir") or "."))
        result = execute_tool(follow["tool_name"], {{"step_path": step, **follow["arguments"]}})
    return result
'''.strip()
