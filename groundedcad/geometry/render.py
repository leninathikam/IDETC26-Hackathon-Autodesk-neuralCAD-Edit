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

VIEW_UP = {
    # Straight-up/down cameras need a viewup that isn't parallel to their
    # look direction, or the camera math is degenerate (blank render).
    "top": (0, 0, 1),
    "bottom": (0, 0, 1),
}

CANONICAL_VIEWS = ["toprightiso", "front", "back", "left", "right", "top", "bottom"]


def _get_shape(result):
    from groundedcad.geometry.inspect import shape_from_workplane

    return shape_from_workplane(result)


def _bbox_largest_dim(shape) -> float:
    """Largest bounding-box dimension of a shape, in model units. Falls
    back to a sane default if the shape has no usable bbox (e.g. empty)."""
    try:
        bb = shape.BoundingBox() if hasattr(shape, "BoundingBox") else None
        if bb is None:
            return 20.0
        size = (
            float(bb.xmax) - float(bb.xmin),
            float(bb.ymax) - float(bb.ymin),
            float(bb.zmax) - float(bb.zmin),
        )
        largest = max(size)
        return largest if largest > 1e-6 else 20.0
    except Exception:
        return 20.0


def _bbox_center(shape) -> tuple[float, float, float]:
    """Center of a shape's bounding box, in model units. Cameras must focus
    here, not the world origin — a part offset from (0,0,0) will render
    blank/off-frame views (especially top/bottom) if the camera always
    looks at the origin instead of where the part actually sits."""
    try:
        bb = shape.BoundingBox() if hasattr(shape, "BoundingBox") else None
        if bb is None:
            return (0.0, 0.0, 0.0)
        return (
            (float(bb.xmax) + float(bb.xmin)) / 2.0,
            (float(bb.ymax) + float(bb.ymin)) / 2.0,
            (float(bb.zmax) + float(bb.zmin)) / 2.0,
        )
    except Exception:
        return (0.0, 0.0, 0.0)


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
    focus: Optional[tuple[float, float, float]] = None,
    distance: Optional[float] = None,
    viewup: Optional[tuple[float, float, float]] = None,
    orthographic: bool = False,
) -> Optional[Path]:
    """Best-effort offscreen PNG render; returns None if rendering unavailable.

    ``distance`` is the camera pull-back distance in model units, and
    ``focus`` is the point the camera looks at — both scaled/positioned
    from the part's actual bbox if not given explicitly. ``viewup`` fixes
    the degenerate-camera bug when looking straight up/down. ``orthographic``
    toggles parallel (True) vs perspective (False) projection.
    """
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

    if distance is None:
        distance = max(_bbox_largest_dim(shape) * 1.8, 5.0)
    if focus is None:
        focus = _bbox_center(shape)

    # Strategy 1: CadQuery's VTK renderer (the reliable Windows path).
    try:
        from cadquery.vis import show

        wrapped = shape.wrapped if hasattr(shape, "wrapped") else shape
        position = tuple(float(focus[i]) + float(proj[i]) * distance for i in range(3))
        # cadquery.vis.show() only calls renderer.ResetCamera() — which is
        # what would normally auto-fit the parallel-projection zoom to the
        # object's bbox — when neither `position` nor `focus` is given. We
        # always pass both (for correct perspective framing), so in
        # orthographic mode VTK's camera is left at its raw default
        # parallel_scale=1.0: a couple of model units tall, versus a part
        # that's typically tens of mm — the camera ends up jammed against
        # the surface. show()'s `zoom` kwarg calls camera.Zoom(factor),
        # which *divides* parallel_scale by factor, so pick factor =
        # 1/desired_half_height to land on a half-height proportional to
        # the part instead of that stale default. Perspective mode ignores
        # parallel_scale entirely, so this has no effect when orthographic
        # is False.
        zoom = (1.0 / max(distance / 3.0, 1.0)) if orthographic else 1.0
        show(
            wrapped,
            screenshot=str(png_path),
            width=width,
            height=height,
            interact=False,
            position=position,
            focus=tuple(float(f) for f in focus),
            viewup=tuple(float(v) for v in viewup) if viewup else None,
            orthographic=orthographic,
            zoom=zoom,
            trihedron=False,
            gradient=False,
            bgcolor=(1.0, 1.0, 1.0),
        )
        if png_path.exists():
            return png_path
    except Exception:
        pass

    # Strategy 2: OCP V3d viewer (Linux/macOS primarily) — already uses
    # FitAll(); does not support the orthographic toggle, used as fallback only.
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

    try:
        from cadquery import exporters

        svg_path = png_path.with_suffix(".svg")
        exporters.export(shape, str(svg_path), exportType="SVG")
        return None
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
    distance = max(_bbox_largest_dim(shape) * 1.8, 5.0)
    center = _bbox_center(shape)
    paths: dict[str, str] = {}
    for name in views:
        proj = VIEW_PROJECTIONS.get(name, (1, -1, 1))
        fname = "iso.png" if name in {"toprightiso", "isometric"} else f"{name}.png"
        path = out / fname
        rendered = render_png(
            shape, path, proj=proj, width=width, height=height,
            focus=center, distance=distance, viewup=VIEW_UP.get(name),
        )
        if rendered is not None:
            paths[name] = str(rendered)
    return paths


def render_orthographic_views(
    result,
    output_dir: str | Path,
    views: Optional[list[str]] = None,
    width: int = 1024,
    height: int = 1024,
) -> dict[str, str]:
    """Same canonical views, rendered orthographic (no perspective
    distortion) instead of the default perspective camera — saved with an
    '_ortho' suffix, separate files from the perspective renders."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shape = _get_shape(result)
    views = views or CANONICAL_VIEWS
    distance = max(_bbox_largest_dim(shape) * 1.8, 5.0)
    center = _bbox_center(shape)
    paths: dict[str, str] = {}
    for name in views:
        proj = VIEW_PROJECTIONS.get(name, (1, -1, 1))
        base = "iso" if name in {"toprightiso", "isometric"} else name
        path = out / f"{base}_ortho.png"
        rendered = render_png(
            shape, path, proj=proj, width=width, height=height,
            focus=center, distance=distance, viewup=VIEW_UP.get(name),
            orthographic=True,
        )
        if rendered is not None:
            paths[name] = str(rendered)
    return paths


def render_focused_view(
    result,
    anchor: tuple[float, float, float],
    output_dir: str | Path,
    context_ratio: float = 0.35,
    proj: tuple[float, float, float] = (1, -1, 1),
    width: int = 1024,
    height: int = 1024,
    filename: str = "focus.png",
) -> Optional[str]:
    """Zoomed render centered on ``anchor`` — shows the edit plus enough
    surrounding structure for orientation, sized as a fraction of the
    part's own bbox instead of a fixed close-up radius."""
    shape = _get_shape(result)
    largest_dim = _bbox_largest_dim(shape)
    window = max(largest_dim * context_ratio, 8.0)
    distance = window * 1.8
    out = Path(output_dir)
    path = out / filename
    rendered = render_png(
        shape, path, proj=proj, width=width, height=height, focus=anchor, distance=distance
    )
    return str(rendered) if rendered else None


def render_composite_grid(
    panels: list[tuple[str, str]],
    output_dir: str | Path,
    filename: str = "composite.png",
    cols: int = 3,
    cell_size: int = 350,
    label_height: int = 22,
    padding: int = 14,
) -> Optional[str]:
    """Stitch multiple already-rendered views into one grid image — fewer
    images sent to the LLM, more context per image. ``panels`` is an
    explicit ordered list of (label, image_path) pairs rather than a
    name-keyed dict, since perspective and orthographic renders of the
    same view (e.g. "front") share the same view name but are different
    images — a dict keyed by name can't hold both.

    Each cell has its label in its own strip above the image (not overlaid
    on top of it), the image is padded/centered rather than filling the
    cell edge-to-edge, and cells are separated by a thin light-grey divider
    rather than a heavy border. A short last row (fewer panels than `cols`)
    is center-justified instead of left-aligned."""
    from PIL import Image, ImageDraw

    available = [(label, path) for label, path in panels if path and Path(path).exists()]
    if not available:
        return None

    cols = max(1, min(cols, len(available)))
    rows = (len(available) + cols - 1) // cols
    cell_h = cell_size + label_height
    grid = Image.new("RGB", (cell_size * cols, cell_h * rows), "white")
    draw = ImageDraw.Draw(grid)

    last_row_count = len(available) - (rows - 1) * cols
    last_row_offset = (cols - last_row_count) * cell_size // 2 if last_row_count < cols else 0

    for i, (label, path) in enumerate(available):
        row, col = divmod(i, cols)
        x = col * cell_size + (last_row_offset if row == rows - 1 else 0)
        y = row * cell_h

        draw.text((x + padding // 2, y + 4), label, fill="black")

        img = Image.open(path)
        img.thumbnail((cell_size - 2 * padding, cell_size - 2 * padding))
        ix = x + (cell_size - img.width) // 2
        iy = y + label_height + (cell_size - img.height) // 2
        grid.paste(img, (ix, iy))

        draw.rectangle([x, y, x + cell_size - 1, y + cell_h - 1], outline="#999999", width=1)

    out_path = Path(output_dir) / filename
    grid.save(out_path)
    return str(out_path)


def render_full_composite(
    result,
    output_dir: str | Path,
    perspective_views: Optional[dict[str, str]] = None,
) -> tuple[dict[str, str], dict[str, str], Optional[str], Optional[str]]:
    """Render (or reuse) the 7 canonical perspective views plus the 4
    orthographic views, and stitch each set into its OWN bordered composite
    — "composite_ortho.png" (4 panels, unchanged from before) and
    "composite_perspective.png" (7 panels, new) — kept separate rather than
    merged into one grid, since they're different projections of the same
    views and mixing them in one image makes it harder to read either.
    Pass ``perspective_views`` when the caller already rendered them, to
    avoid rendering the same 7 views twice. Returns (perspective_views,
    orthographic_views, perspective_composite_path, ortho_composite_path)."""
    out = Path(output_dir)
    persp = perspective_views if perspective_views is not None else render_canonical_views(result, out)
    ortho = render_orthographic_views(result, out, views=["toprightiso", "front", "top", "right"])

    persp_panels = [
        (name, persp[name])
        for name in ("toprightiso", "top", "back", "left", "front", "right", "bottom")
        if persp.get(name)
    ]
    persp_composite = render_composite_grid(
        persp_panels, out, filename="composite_perspective.png", cols=3
    )

    ortho_panels = [
        (name, ortho[name])
        for name in ("toprightiso", "front", "top", "right")
        if ortho.get(name)
    ]
    ortho_composite = render_composite_grid(
        ortho_panels, out, filename="composite_ortho.png", cols=2
    )

    return persp, ortho, persp_composite, ortho_composite


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
    if "toprightiso" in views:
        iso = Path(views["toprightiso"])
        tmp = out / "tmp.png"
        if iso.exists() and not tmp.exists():
            tmp.write_bytes(iso.read_bytes())
        views["tmp"] = str(tmp if tmp.exists() else iso)

    if render:
        # Two extra composites added ALONGSIDE the 7 individual perspective
        # views above (those 7 stay exactly as they were, untouched, for
        # any caller still expecting them individually): one bordered grid
        # of the 7 perspective views, one bordered grid of the 4
        # orthographic views. Kept as two separate images rather than one
        # merged grid — different projections of the same views are easier
        # to read apart than stitched together.
        _, _, persp_composite, ortho_composite = render_full_composite(
            result, out, perspective_views=views
        )
        if persp_composite:
            views["composite_perspective"] = persp_composite
        if ortho_composite:
            views["composite_ortho"] = ortho_composite

    return {"step": str(step), "stl": str(stl), "views": views}