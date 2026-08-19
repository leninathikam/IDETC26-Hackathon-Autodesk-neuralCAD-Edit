"""Tag Diff-F1 failure modes from scores + pipeline_result (debug only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def tag_failure_bucket(
    *,
    ours: dict[str, Any],
    pipeline_result: dict[str, Any] | None = None,
    instruction: str = "",
) -> str:
    """Bucket a scored row for debugging: identity / wrong_edge_or_scope / over_edit / occ_fail / misclassify / ok."""
    diff = float(ours.get("diff_f1") or 0.0)
    chamfer = float(ours.get("chamfer") or 0.0)
    vol = float(ours.get("volume_f1") or 0.0)
    hist = []
    if pipeline_result:
        hist = list(pipeline_result.get("retry_history") or pipeline_result.get("settings", {}).get("retry_history") or [])
        if not hist and isinstance(pipeline_result.get("settings"), dict):
            hist = list(pipeline_result["settings"].get("retry_history") or [])
    # Candidate enumeration is intentionally allowed to decline an unsafe or
    # under-specified edit.  Its ``INCOMPLETE_PLAN`` record must not mask the
    # subsequent raw-CadQuery outcome (and previously labelled high-D rows as
    # misclassified merely because an early candidate was discarded).
    terminal_hist = [
        str(h) for h in hist
        if not str(h).lower().lstrip().startswith("candidate")
    ]
    joined = " | ".join(terminal_hist or [str(h) for h in hist]).upper()

    if diff >= 0.45 and chamfer >= 0.7:
        return "ok"
    if "INCOMPLETE_PLAN" in joined:
        return "misclassify"
    if "OCC" in joined or "NO CIRCULAR" in joined or "CHAMFER REQUIRES" in joined or "NO SUITABLE" in joined:
        return "occ_fail"
    if diff < 0.08 and chamfer >= 0.85:
        # Looks like start / wrong region
        if "IDENTITY" in joined or "266→266" in joined or "FACES" in joined and "→" in joined:
            return "identity"
        return "identity"
    if diff < 0.25 and vol < 0.7 and ("OVERSIZED" in joined or "EXTRA_BODIES" in joined):
        return "over_edit"
    if 0.08 <= diff < 0.45 and chamfer >= 0.7:
        return "wrong_edge_or_scope"
    if "SLOT_MISMATCH" in joined or "LOCALITY" in joined:
        return "wrong_edge_or_scope"
    if "OVERSIZED" in joined or "EXTRA_BODIES" in joined:
        return "over_edit"
    if diff < 0.08:
        return "identity"
    return "ok" if diff >= 0.35 else "wrong_edge_or_scope"


def load_pipeline_result(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / "pipeline_result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_buckets(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        b = r.get("failure_bucket") or "unknown"
        counts[b] = counts.get(b, 0) + 1
    return counts
