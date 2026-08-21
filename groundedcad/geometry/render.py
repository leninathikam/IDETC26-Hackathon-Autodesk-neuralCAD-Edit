"""Canonical view rendering and STEP/STL export helpers."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Optional

VIEW_PROJECTIONS = {
    "toprightiso": (1, -1, 1),
    "isometric": (1, -1, 1),
    "front": (0, -1, 0),
    "back": (0, 1, 0),
    "left": (-1, 0, 0),
    "right": (1, 0, 0),
    "top": (0, 0, 1),
    "bottom": (0, 0, -1),
}

# Keep the image roll stable for every cardinal camera.  The direction points
# from the model toward the camera; ``viewup`` lies in the image plane.
ORTHOGRAPHIC_VIEW_UP = {
    "front": (0, 0, 1),
    "back": (0, 0, 1),
    "left": (0, 0, 1),
    "right": (0, 0, 1),
    "top": (0, 1, 0),
    "bottom": (0, 1, 0),
}

# STEP has no universal semantic "front" marker.  State the world-axis
# convention in the contact sheet so a user and the model can tell an actual
# front/back reversal from a merely symmetric part.
VIEW_LABELS = {
    "toprightiso": "TOPRIGHTISO (+X,-Y,+Z)",
    "front": "FRONT (-Y)",
    "back": "BACK (+Y)",
    "left": "LEFT (-X)",
    "right": "RIGHT (+X)",
    "top": "TOP (+Z)",
    "bottom": "BOTTOM (-Z)",
}

CANONICAL_VIEWS = ["toprightiso", "front", "back", "left", "right", "top", "bottom"]


def _get_shape(result):
    from groundedcad.geometry.inspect import shape_from_workplane

    return shape_from_workplane(result)


def _camera_for_shape(
    shape, proj: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Frame the model bbox instead of assuming it is near world origin."""
    try:
        bb = shape.BoundingBox()
        focus = (
            0.5 * (float(bb.xmin) + float(bb.xmax)),
            0.5 * (float(bb.ymin) + float(bb.ymax)),
            0.5 * (float(bb.zmin) + float(bb.zmax)),
        )
        diagonal = math.sqrt(
            (float(bb.xmax) - float(bb.xmin)) ** 2
            + (float(bb.ymax) - float(bb.ymin)) ** 2
            + (float(bb.zmax) - float(bb.zmin)) ** 2
        )
        # Extra margin is needed for the oblique canonical view.
        distance = max(10.0, 3.0 * diagonal)
    except Exception:
        focus = (0.0, 0.0, 0.0)
        distance = 10.0
    return (
        tuple(focus[i] + float(proj[i]) * distance for i in range(3)),
        focus,
    )


def _orthographic_zoom(shape) -> float:
    """Fit a parallel camera whose default scale is otherwise only 1 mm."""
    try:
        bb = shape.BoundingBox()
        span = max(float(bb.xlen), float(bb.ylen), float(bb.zlen), 1.0)
        # VTK's Camera.Zoom divides parallel scale by this factor.  A factor
        # below one therefore expands the 1 mm default scale to the model.
        return 1.0 / (1.35 * span)
    except Exception:
        return 0.1


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
    *,
    orthographic: bool = False,
    viewup: tuple[float, float, float] | None = None,
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

    # Strategy 1: CadQuery's VTK renderer (the reliable Windows path).
    # ``show`` otherwise uses its default isometric camera and silently
    # produces the same PNG for every named projection.
    try:
        from cadquery.vis import show

        wrapped = shape.wrapped if hasattr(shape, "wrapped") else shape
        position, focus = _camera_for_shape(shape, proj)
        show(
            wrapped,
            screenshot=str(png_path),
            width=width,
            height=height,
            interact=False,
            position=position,
            focus=focus,
            viewup=viewup,
            orthographic=orthographic,
            # ``show`` applies these relative rotations even after an
            # absolute position/focus has been supplied.  Leaving its
            # isometric defaults here was the source of tilted named views.
            roll=0,
            elevation=0,
            azimuth=0,
            zoom=_orthographic_zoom(shape) if orthographic else 1.0,
            trihedron=False,
            gradient=False,
            bgcolor=(1.0, 1.0, 1.0),
        )
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

    # Do not manufacture a text-placeholder PNG as if it were visual model
    # evidence.  A missing view is safer than sending the LLM a misleading
    # image labelled as a front/back/top projection.
    try:
        from cadquery import exporters

        svg_path = png_path.with_suffix(".svg")
        exporters.export(shape, str(svg_path), exportType="SVG")
        return None
    except Exception:
        return None


def _has_visible_geometry(path: Path) -> bool:
    """Reject missing/near-white VTK screenshots before they reach the LLM."""
    try:
        from PIL import Image

        with Image.open(path).convert("RGB") as image:
            # The renderer background is white.  A few antialiased pixels are
            # not evidence of a usable view, so require a modest ink count.
            return sum(
                1
                for r, g, b in image.getdata()
                if min(r, g, b) < 235
            ) >= 500
    except Exception:
        return False


def stitch_orthographic_views(
    views: dict[str, str],
    output_path: str | Path,
    tile_size: int = 512,
) -> Optional[Path]:
    """Create one labelled CAD contact sheet for visual-language prompts.

    Individual renders remain useful evidence files, but a contact sheet makes
    their spatial relationship explicit and avoids relying on attachment order.
    """
    try:
        from PIL import Image, ImageDraw, ImageOps

        output_path = Path(output_path)
        layout = (
            ("toprightiso", "top", "back"),
            ("left", "front", "right"),
            ("", "bottom", ""),
        )
        label_height = 28
        canvas = Image.new("RGB", (tile_size * 3, (tile_size + label_height) * 3), "white")
        draw = ImageDraw.Draw(canvas)
        rendered = 0

        for row, names in enumerate(layout):
            for col, name in enumerate(names):
                x = col * tile_size
                y = row * (tile_size + label_height)
                if not name:
                    continue
                path = Path(views.get(name, ""))
                if path.exists() and _has_visible_geometry(path):
                    with Image.open(path) as source:
                        source = source.convert("RGB")
                        tile = ImageOps.contain(source, (tile_size, tile_size))
                        canvas.paste(
                            tile,
                            (x + (tile_size - tile.width) // 2, y + (tile_size - tile.height) // 2),
                        )
                    rendered += 1
                else:
                    draw.text((x + 8, y + 8), "MISSING VIEW", fill="red")
                draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), outline="black")
                draw.text(
                    (x + 8, y + tile_size + 6),
                    VIEW_LABELS.get(name, name.upper()),
                    fill="black",
                )
        if not rendered:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        return output_path
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
        is_orthographic = name not in {"toprightiso", "isometric"}
        # neuralCAD-Edit uses topright isometric naming
        fname = "iso.png" if name in {"toprightiso", "isometric"} else f"{name}.png"
        path = out / fname
        rendered = render_png(
            shape,
            path,
            proj=proj,
            width=width,
            height=height,
            orthographic=is_orthographic,
            viewup=ORTHOGRAPHIC_VIEW_UP.get(name),
        )
        if rendered is None or not _has_visible_geometry(path):
            path.unlink(missing_ok=True)
            if not is_orthographic:
                # ISO is intentionally perspective; a small tilt is harmless
                # there and is preferable to losing its overview tile.
                tilt = (0.13, -0.11, 0.17)
                fallback_proj = tuple(float(proj[i]) + tilt[i] for i in range(3))
                rendered = render_png(
                    shape, path, proj=fallback_proj, width=width, height=height
                )
        if rendered is not None:
            paths[name] = str(rendered)
    sheet = stitch_orthographic_views(paths, out / "orthographic_sheet.png")
    if sheet is not None:
        paths["orthographic_sheet"] = str(sheet)
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
