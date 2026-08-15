"""Volumetric / diff-based geometric edit metrics.

Drop-in metrics that score how well a model *edit* matches a ground-truth edit,
relative to the pre-edit (start) model:

    volumetric_f1        - F1 of voxel occupancy between two breps
    diff_f1              - F1 between the *edit diffs* (start -> gt vs start -> pred)
    surface_diff_chamfer - chamfer distance between the *surface changes*

All three share the repo-wide metric signature used by
``src.utils.evals_feature_geometric`` (``efg.chamfer_similarity_norm``,
``efg.compute_chamfer_distance``, ``efg.align_meshes`` etc.)::

    metric(gt_rel, edit_rel, db, **kwargs) -> float

``gt_rel`` / ``edit_rel`` are brep geometry relative paths (joined with
``db.root_dir`` to reach the file on disk); ``db`` is a ``DatabaseManager``.
So you can swap these in anywhere an old ``(source_brep, target_brep, db)``
metric is used.

The two *diff* metrics additionally need the pre-edit (start) model. By default
they resolve it automatically from ``db`` (start brep of the request the edit
belongs to), so the 3-argument call still works exactly like the older metrics.
Pass ``start_rel=`` explicitly to skip the lookup (faster in tight loops).

Every metric returns ``np.nan`` when it cannot be computed.

This module is intentionally self-contained (only depends on numpy, open3d,
scipy, and ``efg``) so it can be shared as a single file.
"""

import os.path as osp

import numpy as np
import open3d as o3d
from scipy import ndimage  # cavity-only interior fill (robust to open meshes)

from src.utils import evals_feature_geometric as efg


# Occupancy can be SURFACE-only (voxels the mesh surface passes through) or
# SOLID (interior filled). SOLID gives a true *volumetric* measure and is the
# default; it needs (approximately) watertight meshes, which B-rep STL exports
# normally are.
FILL_SOLID = True

# Safety cap: a pathologically oversized prediction is scored worst instead of
# allocating a huge voxel grid.
_MAX_GRID_VOXELS = 30_000_000

# Brep-doc keys that hold a geometry relative path, in preference order.
GEOM_KEYS = ["stl", "obj"]


def brep_geom_rel_path(brep, keys=GEOM_KEYS):
    """First available geometry relative path for a brep doc, or None."""
    if brep is None:
        return None
    for k in keys:
        val = brep.get(k)
        if val:
            return val[0] if isinstance(val, list) else val
    return None


# --- start-model resolution (edit brep -> its request's start brep) ---------
# The diff metrics need the pre-edit model. To keep the 3-arg swap-in interface
# we look it up from the DB: build (once per db, then cache) a map from each
# edit's end-brep geometry path -> its request's start-brep geometry path.
_START_LUT_CACHE = {}


def _start_rel_lut(db):
    key = getattr(db, "root_dir", id(db))
    lut = _START_LUT_CACHE.get(key)
    if lut is not None:
        return lut
    request_start = {}
    for req in db.requests.find({}):
        start_bid = req.get("brep_start")
        start_brep = db.breps.find_one({"_id": start_bid}) if start_bid else None
        request_start[req["_id"]] = brep_geom_rel_path(start_brep)
    lut = {}
    for edit in db.edits.find({}):
        end_bid = edit.get("brep_end")
        end_brep = db.breps.find_one({"_id": end_bid}) if end_bid else None
        rel = brep_geom_rel_path(end_brep)
        if rel:
            lut[rel] = request_start.get(edit.get("request"))
    _START_LUT_CACHE[key] = lut
    return lut


def resolve_start_rel(edit_rel, db):
    """Start-model geometry rel path for the edit whose end-brep is ``edit_rel``.

    Returns None if it can't be resolved. Result is cached per ``db``; call
    ``clear_start_cache()`` if the database changes within a session.
    """
    if isinstance(edit_rel, list):
        edit_rel = edit_rel[0] if edit_rel else None
    if not edit_rel:
        return None
    return _start_rel_lut(db).get(edit_rel)


def clear_start_cache():
    """Drop the cached edit->start lookup (e.g. after the DB is modified)."""
    _START_LUT_CACHE.clear()


# --- mesh loading -----------------------------------------------------------
def _resolve_mesh_path(rel, dbm):
    """Relative brep path (or list) -> existing abs path, else None."""
    if isinstance(rel, list):
        rel = rel[0]
    if not rel:
        return None
    p = osp.join(dbm.root_dir, rel)
    return p if osp.exists(p) else None


def _load_mesh(rel, dbm):
    p = _resolve_mesh_path(rel, dbm)
    if p is None:
        return None
    mesh = o3d.io.read_triangle_mesh(p)
    return mesh if len(mesh.vertices) > 0 else None


# --- shared voxelization helpers --------------------------------------------
def _shared_voxel_frame(meshes, voxel_divisor=128, size_meshes=None):
    """One voxel_size + bounds shared by all meshes so grid indices line up.

    voxel_size = (smallest bbox diagonal of ``size_meshes``) / voxel_divisor
    (repo IoU convention). ``size_meshes`` defaults to ``meshes``, but callers
    pass only the *reference* meshes (start / start+gt) so the resolution is
    FIXED per request and does NOT depend on the prediction: a degenerate/tiny
    prediction can no longer collapse the voxel size, blow up the grid, or
    inflate the diff. The bounds still span *all* ``meshes``, so the grid is only
    *extended* (never rescaled) to cover the prediction. A common origin/bounds
    means voxel grid indices are comparable across meshes (intersection is
    valid).
    """
    size_meshes = meshes if size_meshes is None else size_meshes
    size_bboxes = [m.get_axis_aligned_bounding_box() for m in size_meshes]
    diags = [np.linalg.norm(b.get_max_bound() - b.get_min_bound()) for b in size_bboxes]
    voxel_size = min(diags) / float(voxel_divisor)
    bboxes = [m.get_axis_aligned_bounding_box() for m in meshes]
    min_bound = np.minimum.reduce([np.asarray(b.get_min_bound()) for b in bboxes])
    max_bound = np.maximum.reduce([np.asarray(b.get_max_bound()) for b in bboxes])
    return voxel_size, min_bound, max_bound


def _grid_dims(voxel_size, min_bound, max_bound):
    """Integer (nx, ny, nz) of the shared voxel grid."""
    span = np.asarray(max_bound) - np.asarray(min_bound)
    return tuple(int(np.ceil(span[i] / voxel_size)) + 1 for i in range(3))


def _surface_mask(mesh, voxel_size, min_bound, max_bound, dims):
    """Boolean occupancy of the voxels the mesh *surface* passes through."""
    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=voxel_size, min_bound=min_bound, max_bound=max_bound)
    mask = np.zeros(dims, dtype=bool)
    for v in grid.get_voxels():
        i, j, k = v.grid_index
        if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
            mask[i, j, k] = True
    return mask


def _solid_mask(mesh, voxel_size, min_bound, max_bound, dims):
    """Boolean *filled* occupancy: surface voxels + any voxel-enclosed cavity.

    We do NOT use a ray/signed-distance inside test here: B-rep STL exports are
    frequently NON-watertight (open tube ends, cracks), and on such meshes
    Open3D's compute_occupancy / compute_signed_distance both "cap" the opening
    and fill a spurious solid beam extending along the open feature (e.g. a
    diagonal cylinder down a hollow shaft). Instead we voxelize the surface and
    fill only cavities that are fully enclosed by that surface shell
    (scipy.ndimage.binary_fill_holes): an open feature stays hollow because its
    interior connects to the grid border through the opening, while genuinely
    closed solids are filled. This is robust to non-watertight meshes and
    removes the beam artifact.
    """
    surface = _surface_mask(mesh, voxel_size, min_bound, max_bound, dims)
    return ndimage.binary_fill_holes(surface)


def _occupancy_mask(mesh, voxel_size, min_bound, max_bound, dims, fill=None):
    fill = FILL_SOLID if fill is None else fill
    if fill:
        return _solid_mask(mesh, voxel_size, min_bound, max_bound, dims)
    return _surface_mask(mesh, voxel_size, min_bound, max_bound, dims)


def _f1_from_masks(mask_a, mask_b):
    """F1 of two boolean occupancy grids = 2|A&B| / (|A|+|B|). In [0, 1]."""
    sa, sb = int(mask_a.sum()), int(mask_b.sum())
    if sa == 0 or sb == 0:
        return 0.0
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0:
        return 0.0
    return float(2.0 * inter / (sa + sb))


# --- metrics ----------------------------------------------------------------
def volumetric_f1(gt_rel, edit_rel, db, voxel_divisor=128, pre_align=False, fill=None):
    """Volumetric F1 between two breps via voxel occupancy (higher = better, [0, 1]).

    Both meshes are voxelized on a *shared* grid (common origin + voxel size);
    F1 = 2|gt n pred| / (|gt| + |pred|), i.e. the harmonic mean of the
    precision (|gt n pred| / |pred|) and recall (|gt n pred| / |gt|).
    voxel_size follows the repo's IoU convention: min(bbox diag) / voxel_divisor.
    Occupancy is solid-filled by default (fill=None -> FILL_SOLID).
    """
    gt_mesh = _load_mesh(gt_rel, db)
    pred_mesh = _load_mesh(edit_rel, db)
    if gt_mesh is None or pred_mesh is None:
        return np.nan
    if pre_align:
        pred_mesh = efg.align_meshes(pred_mesh, gt_mesh)
    # resolution fixed by GT (reference) only; bounds still cover the prediction
    voxel_size, min_bound, max_bound = _shared_voxel_frame(
        [gt_mesh, pred_mesh], voxel_divisor, size_meshes=[gt_mesh])
    if voxel_size <= 0:
        return np.nan
    dims = _grid_dims(voxel_size, min_bound, max_bound)
    if np.prod(dims, dtype=float) > _MAX_GRID_VOXELS:
        return 0.0  # prediction pathologically oversized vs reference -> worst
    gt_mask = _occupancy_mask(gt_mesh, voxel_size, min_bound, max_bound, dims, fill)
    pred_mask = _occupancy_mask(pred_mesh, voxel_size, min_bound, max_bound, dims, fill)
    return _f1_from_masks(gt_mask, pred_mask)


def diff_f1(gt_rel, edit_rel, db, voxel_divisor=128, start_rel=None, pre_align=False, fill=None):
    """F1 between the *edit diffs* relative to the pre-edit (start) model.

    A "diff" is the set of voxels that change (symmetric difference) between the
    start model and an end model:
        gt_diff    = voxels changed by  start -> ground-truth edit
        model_diff = voxels changed by  start -> model edit
    The returned F1 = 2|gt_diff n model_diff| / (|gt_diff| + |model_diff|) measures
    how well the model changed the *same regions* the ground-truth edit changed
    (higher = better, in [0, 1]).

    Occupancy is solid-filled by default (fill=None -> FILL_SOLID), so the diff
    is a true change in *volume* (material added/removed), not just surface.
    start / gt / model are all voxelized on ONE shared grid, so the diff uses
    exactly the same voxel size (and frame) as the F1 comparison itself.
    ``start_rel`` defaults to the start model resolved from ``dbm`` for the edit
    ``edit_rel`` (see ``resolve_start_rel``), so the 3-argument call works like
    the older ``efg`` metrics.

    Returns ``np.nan`` when meshes cannot be loaded; downstream aggregation treats
    NaN the same as a failed edit (0.0) so models are penalized for failures.
    """
    start = start_rel if start_rel is not None else resolve_start_rel(edit_rel, db)
    start_mesh = _load_mesh(start, db)
    gt_mesh = _load_mesh(gt_rel, db)
    pred_mesh = _load_mesh(edit_rel, db)
    if start_mesh is None or gt_mesh is None or pred_mesh is None:
        return np.nan
    if pre_align:
        gt_mesh = efg.align_meshes(gt_mesh, start_mesh)
        pred_mesh = efg.align_meshes(pred_mesh, start_mesh)
    # resolution fixed by start+GT (reference) only so it is constant per request;
    # bounds still cover the prediction, which is captured but can't rescale.
    voxel_size, min_bound, max_bound = _shared_voxel_frame(
        [start_mesh, gt_mesh, pred_mesh], voxel_divisor, size_meshes=[start_mesh, gt_mesh])
    if voxel_size <= 0:
        return np.nan
    dims = _grid_dims(voxel_size, min_bound, max_bound)
    if np.prod(dims, dtype=float) > _MAX_GRID_VOXELS:
        return 0.0  # prediction pathologically oversized vs reference -> worst
    start_mask = _occupancy_mask(start_mesh, voxel_size, min_bound, max_bound, dims, fill)
    gt_mask = _occupancy_mask(gt_mesh, voxel_size, min_bound, max_bound, dims, fill)
    pred_mask = _occupancy_mask(pred_mesh, voxel_size, min_bound, max_bound, dims, fill)
    gt_diff = np.logical_xor(start_mask, gt_mask)       # added + removed volume
    pred_diff = np.logical_xor(start_mask, pred_mask)
    # empty-diff policy: GT changed no voxels AND model changed none -> a correct
    # no-op -> perfect match (1.0). Exactly one empty -> total disagreement
    # (_f1_from_masks already returns 0.0). Never dropped.
    if gt_diff.sum() == 0 and pred_diff.sum() == 0:
        return 1.0
    return _f1_from_masks(gt_diff, pred_diff)


def _dist_to_mesh(query_points, mesh):
    """Unsigned distance from each query point to the continuous mesh surface.

    Using distance to the *mesh* (not nearest sampled point) makes change
    detection robust to sampling density, avoiding spurious "changed" points in
    unchanged regions.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(
        o3d.core.Tensor(query_points.astype(np.float32))).numpy()


def surface_diff_chamfer(gt_rel, edit_rel, dbm, start_rel=None, num_samples=20000,
                         thresh_mult=1.5, max_pts=4000, empty_penalty=2.0):
    """Chamfer distance between the *surface changes* of two edits (lower = better).

    For each edit we build a "surface diff" point cloud: points that were added
    or removed relative to the start surface (points on one mesh that lie
    farther than a threshold ``t`` from the other mesh surface). We then take the
    chamfer distance between the ground-truth surface diff and the model surface
    diff, so the metric measures whether the model changed the *same surface
    regions* as the ground-truth edit.

    Returns log1p(chamfer / start_diagonal): normalized by the start's
    characteristic length (constant per request -> scale-invariant, comparable
    across requests) with log1p to tame chamfer's heavy right tail.

    ``start_rel`` defaults to the start model resolved from ``dbm`` for the edit
    ``edit_rel`` (see ``resolve_start_rel``), so the 3-argument call works like
    the older ``efg`` metrics.
    """
    start_rel = start_rel if start_rel is not None else resolve_start_rel(edit_rel, dbm)
    sm = _load_mesh(start_rel, dbm)
    gm = _load_mesh(gt_rel, dbm)
    em = _load_mesh(edit_rel, dbm)
    if sm is None or gm is None or em is None:
        return np.nan

    def _sample(mesh):
        return np.asarray(mesh.sample_points_uniformly(number_of_points=num_samples).points)

    def _diag(mesh):
        return float(np.linalg.norm(np.ptp(np.asarray(mesh.vertices), 0)))

    sp_s, sp_g, sp_e = _sample(sm), _sample(gm), _sample(em)

    # threshold fixed by start+GT (reference) only, not the prediction, so a
    # shrunken prediction can't lower t and mark everything "changed".
    t = thresh_mult * (min(_diag(sm), _diag(gm)) / 128.0)

    def _changed(src, mesh):
        """Points in ``src`` that are farther than ``t`` from ``mesh`` (added/removed)."""
        return src[_dist_to_mesh(src, mesh) > t] if len(src) else src

    def _stack(*arrays):
        arrays = [a for a in arrays if len(a)]
        return np.vstack(arrays) if arrays else np.empty((0, 3))

    gt_diff = _stack(_changed(sp_g, sm), _changed(sp_s, gm))
    edit_diff = _stack(_changed(sp_e, sm), _changed(sp_s, em))

    # empty-diff policy: both empty => model correctly made no change (best = 0);
    # exactly one empty => total disagreement (worst). Never NaN, so no-ops and
    # failures stay in the analysis instead of being silently dropped.
    empty_worst = float(np.log1p(empty_penalty))
    if len(gt_diff) == 0 and len(edit_diff) == 0:
        return 0.0
    if len(gt_diff) == 0 or len(edit_diff) == 0:
        return empty_worst

    def _subsample(q, n=max_pts):
        if len(q) <= n:
            return q
        idx = np.random.default_rng(0).choice(len(q), n, replace=False)
        return q[idx]

    raw = float(efg.compute_chamfer_distance(
        _subsample(gt_diff), _subsample(edit_diff), normalize=False)[0])

    scale = _diag(sm)
    if scale < 1e-9:
        return np.nan
    return float(np.log1p(raw / scale))


# --- registry ---------------------------------------------------------------
# All callables use the swappable ``metric(gt_rel, edit_rel, db, **kwargs)``
# interface. ``LOWER_IS_BETTER`` records each metric's orientation (True =
# smaller is a better match) so callers can align it with a similarity scale.
METRICS = {
    "volumetric_f1": volumetric_f1,
    "diff_f1": diff_f1,
    "surface_diff_chamfer": surface_diff_chamfer,
}

LOWER_IS_BETTER = {
    "volumetric_f1": False,
    "diff_f1": False,
    "surface_diff_chamfer": True,
}
