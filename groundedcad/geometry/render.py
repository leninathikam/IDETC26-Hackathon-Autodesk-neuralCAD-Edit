"""Canonical view rendering and STEP/STL export helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

VIEW_PROJECTIONS = {
    "toprightiso": (1, -1, 1),
    "isometric": (1, -1, 1),
    "front": (0, 0, 1),
    "back": (0, 0, -1),
    "left": (-1, 0, 0),
    "right": (1, 0, 0),
    "top": (0, 1, 0),
    "bottom": (0, -1, 0),
}

CANONICAL_VIEWS = ["toprightiso", "front", "back", "left", "right", "top", "bottom"]


def _get_shape(result):
    from groundedcad.geometry.fallback import SimpleSolid

    if isinstance(result, SimpleSolid):
        return result
    if hasattr(result, "val"):
        return result.val()
    return result


def export_step(result, output_dir: str | Path, filename: str = "tmp.step") -> Path:
    from groundedcad.geometry.fallback import SimpleSolid, cadquery_available, export_simple_step

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    if isinstance(result, SimpleSolid) or not cadquery_available():
        solid = result if isinstance(result, SimpleSolid) else result
        return export_simple_step(solid, path)

    import cadquery as cq
    from cadquery import exporters

    if isinstance(result, cq.Assembly):
        result.save(str(path))
    else:
        exporters.export(_get_shape(result), str(path), exportType="STEP")
    return path


def export_stl(result, output_dir: str | Path, filename: str = "tmp.stl") -> Path:
    from groundedcad.geometry.fallback import SimpleSolid, cadquery_available, export_simple_stl

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    if isinstance(result, SimpleSolid) or not cadquery_available():
        solid = result if isinstance(result, SimpleSolid) else result
        return export_simple_stl(solid, path)

    from cadquery import exporters

    exporters.export(_get_shape(result), str(path), exportType="STL")
    return path


def render_png(
    shape,
    png_path: str | Path,
    proj: tuple[float, float, float] = (1, -1, 1),
    width: int = 1024,
    height: int = 1024,
) -> Optional[Path]:
    """Best-effort offscreen PNG render; returns None if rendering unavailable."""
    from groundedcad.geometry.fallback import SimpleSolid

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(shape, SimpleSolid):
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(img)
            bb = shape.bbox()
            draw.rectangle([40, 40, width - 40, height - 40], outline="black", width=3)
            draw.text((50, 50), f"fallback solid bodies={len(shape.bodies)}", fill="black")
            draw.text((50, 80), f"vol={shape.volume():.2f} size={bb.size()}", fill="black")
            draw.text((50, 110), f"proj={proj}", fill="black")
            img.save(png_path)
            return png_path
        except Exception:
            return None

    # Strategy 1: CadQuery vis.show screenshot
    try:
        from cadquery.vis import show

        show(shape, screenshot=str(png_path), width=width, height=height, interact=False)
        if png_path.exists():
            return png_path
    except Exception:
        pass

    # Strategy 2: OCP V3d viewer (Linux/macOS primarily)
    try:
        from OCP.AIS import AIS_InteractiveContext, AIS_Shape, AIS_Shaded
        from OCP.Aspect import Aspect_DisplayConnection, Aspect_TypeOfLine
        from OCP.OpenGl import OpenGl_GraphicDriver
        from OCP.Prs3d import Prs3d_LineAspect
        from OCP.Quantity import Quantity_Color, Quantity_NOC_BLACK, Quantity_TOC_RGB
        from OCP.V3d import V3d_Viewer
        import sys

        wrapped = shape.wrapped if hasattr(shape, "wrapped") else shape
        display_connection = Aspect_DisplayConnection()
        driver = OpenGl_GraphicDriver(display_connection)
        viewer = V3d_Viewer(driver)
        viewer.SetDefaultLights()
        viewer.SetLightOn()
        context = AIS_InteractiveContext(viewer)
        view = viewer.CreateView()

        if sys.platform == "linux":
            from OCP.Xw import Xw_Window

            window = Xw_Window(display_connection, "offscreen", 0, 0, width, height)
        elif sys.platform == "darwin":
            import AppKit
            from OCP.Cocoa import Cocoa_Window

            AppKit.NSApplication.sharedApplication()
            window = Cocoa_Window("offscreen", 0, 0, width, height)
        else:
            # Windows: try WNT if available, else fail soft
            try:
                from OCP.WNT import WNT_Window

                window = WNT_Window(display_connection, "offscreen", 0, 0, width, height)
            except Exception:
                raise RuntimeError("No offscreen window backend on this platform")

        view.SetWindow(window)
        if not window.IsMapped():
            window.Map()
        view.SetBackgroundColor(Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB))
        ais_shape = AIS_Shape(wrapped)
        ais_shape.SetColor(Quantity_Color(0.6, 0.6, 0.6, Quantity_TOC_RGB))
        drawer = ais_shape.Attributes()
        edge_aspect = Prs3d_LineAspect(
            Quantity_Color(Quantity_NOC_BLACK), Aspect_TypeOfLine.Aspect_TOL_SOLID, 2.0
        )
        drawer.SetFaceBoundaryAspect(edge_aspect)
        drawer.SetFaceBoundaryDraw(True)
        context.Display(ais_shape, AIS_Shaded, 0, True)
        view.SetProj(float(proj[0]), float(proj[1]), float(proj[2]))
        view.FitAll()
        view.Dump(str(png_path))
        if png_path.exists():
            return png_path
    except Exception:
        pass

    # Strategy 3: SVG fallback then convert with Pillow if possible
    try:
        from cadquery import exporters
        from PIL import Image, ImageDraw

        svg_path = png_path.with_suffix(".svg")
        exporters.export(shape, str(svg_path), exportType="SVG")
        # Create a simple placeholder PNG noting SVG was written
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"SVG fallback: {svg_path.name}", fill="black")
        img.save(png_path)
        return png_path if png_path.exists() else None
    except Exception:
        return None


def render_canonical_views(
    result,
    output_dir: str | Path,
    views: Optional[list[str]] = None,
    width: int = 1024,
    height: int = 1024,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shape = _get_shape(result)
    views = views or CANONICAL_VIEWS
    paths: dict[str, str] = {}
    for name in views:
        proj = VIEW_PROJECTIONS.get(name, (1, -1, 1))
        # neuralCAD-Edit uses topright isometric naming
        fname = "iso.png" if name in {"toprightiso", "isometric"} else f"{name}.png"
        path = out / fname
        rendered = render_png(shape, path, proj=proj, width=width, height=height)
        if rendered is not None:
            paths[name] = str(rendered)
    return paths


def export_all_artifacts(
    result,
    output_dir: str | Path,
    *,
    step_name: str = "tmp.step",
    stl_name: str = "tmp.stl",
    render: bool = True,
) -> dict[str, Any]:
    out = Path(output_dir)
    step = export_step(result, out, step_name)
    stl = export_stl(result, out, stl_name)
    views = render_canonical_views(result, out) if render else {}
    # Also write tmp.png iso for harness compatibility
    if "toprightiso" in views:
        iso = Path(views["toprightiso"])
        tmp = out / "tmp.png"
        if iso.exists() and not tmp.exists():
            tmp.write_bytes(iso.read_bytes())
        views["tmp"] = str(tmp if tmp.exists() else iso)
    return {"step": str(step), "stl": str(stl), "views": views}
