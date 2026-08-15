"""Score GroundedCAD outputs vs GT using start/pred/GT meshes.

Does not require mongita package: reads mongita BSON collection files.
Chamfer / Volume F1 / Diff F1 follow the official formulas (numpy/scipy).
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "edit_192_external"
MONGO = DATA / "mongita_db"
OUTPUTS = ROOT / "data" / "model_outputs"
PARQUET = DATA / "parquets" / "val_edit_text.parquet"
REPORT = ROOT / "docs" / "baseline_scores.json"

VOXEL_DIVISOR = 64  # coarser than official 128 so the baseline finishes


def _read_cstring(buf: bytes, i: int) -> tuple[str, int]:
    j = buf.index(b"\x00", i)
    return buf[i:j].decode("utf-8", "replace"), j + 1


def _decode_value(buf: bytes, i: int, t: int):
    if t == 0x01:
        return struct.unpack_from("<d", buf, i)[0], i + 8
    if t == 0x02:
        n = struct.unpack_from("<i", buf, i)[0]
        i += 4
        return buf[i : i + n - 1].decode("utf-8", "replace"), i + n
    if t == 0x03:
        size = struct.unpack_from("<i", buf, i)[0]
        return _decode_doc(buf[i : i + size]), i + size
    if t == 0x04:
        size = struct.unpack_from("<i", buf, i)[0]
        doc = _decode_doc(buf[i : i + size])
        arr = [doc[str(k)] for k in range(len(doc))]
        return arr, i + size
    if t == 0x08:
        return bool(buf[i]), i + 1
    if t == 0x0A:
        return None, i
    if t == 0x10:
        return struct.unpack_from("<i", buf, i)[0], i + 4
    if t == 0x12:
        return struct.unpack_from("<q", buf, i)[0], i + 8
    if t == 0x07:
        return buf[i : i + 12].hex(), i + 12
    if t == 0x05:
        n = struct.unpack_from("<i", buf, i)[0]
        return None, i + 5 + n
    raise ValueError(f"unsupported bson type {t:#x}")


def _decode_doc(buf: bytes) -> dict:
    size = struct.unpack_from("<i", buf, 0)[0]
    i = 4
    out = {}
    while i < size - 1:
        t = buf[i]
        i += 1
        if t == 0:
            break
        name, i = _read_cstring(buf, i)
        val, i = _decode_value(buf, i, t)
        out[name] = val
    return out


def load_collection(name: str) -> dict[str, dict]:
    raw = (MONGO / f"v2_db.{name}" / "$.data").read_bytes()
    docs = {}
    i = 0
    while i + 4 <= len(raw):
        size = struct.unpack_from("<i", raw, i)[0]
        if size < 5 or i + size > len(raw):
            break
        doc = _decode_doc(raw[i : i + size])
        docs[str(doc.get("_id"))] = doc
        i += size
    return docs


def tessellate_step(path: Path, tol: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    import cadquery as cq

    cache = path.with_suffix(".baseline.stl")
    if not cache.exists() or cache.stat().st_mtime < path.stat().st_mtime:
        wp = cq.importers.importStep(str(path))
        cq.exporters.export(wp, str(cache))
    return load_stl_points(cache)


def sample_points(pts: np.ndarray, tris: np.ndarray, n: int = 4000) -> np.ndarray:
    if len(pts) == 0 or len(tris) == 0:
        return np.zeros((0, 3))
    tri_pts = pts[tris]
    areas = 0.5 * np.linalg.norm(np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]), axis=1)
    areas = np.clip(areas, 1e-12, None)
    prob = areas / areas.sum()
    idx = np.random.default_rng(0).choice(len(tris), size=n, p=prob)
    uvw = np.random.default_rng(1).random((n, 3))
    uvw = uvw / uvw.sum(axis=1, keepdims=True)
    return (uvw[:, 0:1] * tri_pts[idx, 0] + uvw[:, 1:2] * tri_pts[idx, 1] + uvw[:, 2:3] * tri_pts[idx, 2])


def chamfer_sim(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 0.0
    diag = np.linalg.norm(pts_a.max(0) - pts_a.min(0))
    if diag < 1e-8:
        return 0.0
    d1 = cKDTree(pts_a).query(pts_b, k=1)[0].mean()
    d2 = cKDTree(pts_b).query(pts_a, k=1)[0].mean()
    return float(1.0 - min((d1 + d2) / diag, 1.0))


def load_stl_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, triangles) from binary STL without a Python triangle loop."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 84:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    n = int(np.frombuffer(raw[80:84].tobytes(), dtype="<u4")[0])
    rec = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("attr", "<u2")])
    tri = np.frombuffer(raw[84 : 84 + n * 50], dtype=rec)
    verts = tri["v"].reshape(-1, 3).astype(np.float64)
    faces = np.arange(len(verts), dtype=np.int32).reshape(-1, 3)
    return verts, faces


def load_geom(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".stl":
        return load_stl_points(path)
    return tessellate_step(path, tol=0.35)


def occupancy(pts: np.ndarray, tris: np.ndarray, voxel_size, origin, dims):
    mask = np.zeros(dims, dtype=bool)
    if len(pts) == 0:
        return mask
    if len(pts) > 80_000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 80_000, replace=False)]
    idx = np.floor((pts - origin) / voxel_size).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, dims[0] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, dims[1] - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, dims[2] - 1)
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return ndimage.binary_fill_holes(mask)


def f1(a, b) -> float:
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 or sb == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    return float(2 * inter / (sa + sb))


def voxel_metrics(start, gt, pred):
    all_pts = np.vstack([start[0], gt[0], pred[0]])
    mn, mx = all_pts.min(0), all_pts.max(0)
    diag = float(np.linalg.norm(mx - mn))
    voxel = max(diag / VOXEL_DIVISOR, 1e-4)
    dims = tuple(int(np.ceil((mx[i] - mn[i]) / voxel)) + 2 for i in range(3))
    if np.prod(dims) > 8_000_000:
        voxel *= 2
        dims = tuple(int(np.ceil((mx[i] - mn[i]) / voxel)) + 2 for i in range(3))
    origin = mn - voxel
    ms = occupancy(start[0], start[1], voxel, origin, dims)
    mg = occupancy(gt[0], gt[1], voxel, origin, dims)
    mp = occupancy(pred[0], pred[1], voxel, origin, dims)
    vol = f1(mg, mp)
    gdiff = np.logical_xor(ms, mg)
    pdiff = np.logical_xor(ms, mp)
    if gdiff.sum() == 0 and pdiff.sum() == 0:
        d = 1.0
    else:
        d = f1(gdiff, pdiff)
    return vol, d


def classify(text: str) -> str:
    from groundedcad.agents.patterns import classify_edit

    return classify_edit(text)[0].value


def find_outputs() -> dict[str, Path]:
    found = {}
    for settings in OUTPUTS.rglob("settings.json"):
        data = json.loads(settings.read_text(encoding="utf-8"))
        rid = data.get("edit_request_id")
        step = settings.parent / "tmp.step"
        if rid and step.exists():
            found[rid] = step
    return found


def brep_stl(brep_id: str | None) -> Path | None:
    if not brep_id:
        return None
    p = DATA / "breps" / f"{brep_id}.stl"
    return p if p.exists() else None


def main():
    print("loading requests/edits (not the 722MB breps BSON)", flush=True)
    requests = load_collection("requests")
    edits = load_collection("edits")
    print(f"requests={len(requests)} edits={len(edits)}", flush=True)
    df = pd.read_parquet(PARQUET)
    preds = find_outputs()
    print(f"parquet={len(df)} pred_steps={len(preds)}", flush=True)
    rows = []
    for _, rec in df.iterrows():
        rid = rec["request"]
        text = str(rec.get("request_text") or "")
        etype = classify(text)
        pred_step = preds.get(rid)
        req = requests.get(rid)
        valid = False
        chamfer = vol_f1 = diff = 0.0
        note = ""
        if pred_step is None:
            note = "no_pred"
        elif req is None:
            note = "request_not_in_db"
        else:
            gt_user = req.get("user")
            start_id = req.get("brep_start")
            gt_edit = next(
                (e for e in edits.values() if e.get("request") == rid and e.get("user") == gt_user),
                None,
            )
            start_p = brep_stl(str(start_id) if start_id else None)
            gt_p = brep_stl(str(gt_edit.get("brep_end")) if gt_edit else None)
            if not start_p or not gt_p:
                note = f"missing_gt_or_start start={start_p} gt={gt_p}"
            else:
                try:
                    start_m = load_geom(start_p)
                    gt_m = load_geom(gt_p)
                    pred_m = load_geom(pred_step)
                    valid = len(pred_m[0]) > 0
                    chamfer = chamfer_sim(sample_points(*gt_m), sample_points(*pred_m))
                    vol_f1, diff = voxel_metrics(start_m, gt_m, pred_m)
                except Exception as exc:  # noqa: BLE001
                    note = f"metric_error:{exc}"
                    valid = False
        rows.append(
            {
                "id": rid,
                "type": etype,
                "chamfer": round(chamfer, 4),
                "volume_f1": round(vol_f1, 4),
                "diff_f1": round(diff, 4),
                "valid": valid,
                "instruction": " ".join(text.split())[:120],
                "note": note,
            }
        )
        print(
            f"{len(rows):02d} {etype:22s} C={chamfer:.3f} V={vol_f1:.3f} D={diff:.3f} valid={valid} {note}",
            flush=True,
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    print("\n=== mean by type (missing pred scored 0) ===")
    print(f"{'type':22s} n  Chamfer  VolF1  DiffF1  valid")
    for t, rs in sorted(by.items(), key=lambda kv: -np.mean([x["diff_f1"] for x in kv[1]])):
        n = len(rs)
        print(
            f"{t:22s} {n:2d}  {np.mean([x['chamfer'] for x in rs]):.3f}   "
            f"{np.mean([x['volume_f1'] for x in rs]):.3f}  {np.mean([x['diff_f1'] for x in rs]):.3f}  "
            f"{sum(x['valid'] for x in rs)}/{n}"
        )


if __name__ == "__main__":
    main()
