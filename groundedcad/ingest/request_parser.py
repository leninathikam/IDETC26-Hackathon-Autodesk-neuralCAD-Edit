"""Request ingestion and timestamp alignment for neuralCAD-Edit tasks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from groundedcad.agents.schemas import TemporalCue


MEDIA_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".f3d",
    ".step",
    ".smt",
    ".stp",
    ".stl",
    ".mp4",
    ".webm",
)


@dataclass
class EditRequest:
    """Normalized edit request consumed by the GroundedCAD pipeline."""

    request_id: str
    instruction: str
    step_path: Optional[str] = None
    video_path: Optional[str] = None
    transcript: str = ""
    transcript_words: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    views: dict[str, str] = field(default_factory=dict)
    difficulty: str = "unknown"
    modality: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)

    def temporal_cues(self) -> list[TemporalCue]:
        return align_transcript_and_events(self.transcript_words, self.events)


def _abspath_if_media(value: Any, root: Path) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (bytes, str)):
        try:
            return _abspath_if_media(value.tolist(), root)
        except Exception:
            pass
    if isinstance(value, str) and value.lower().endswith(MEDIA_EXTENSIONS):
        path = Path(value)
        if not path.is_absolute():
            return str((root / path).resolve())
        return value
    if isinstance(value, dict):
        return {k: _abspath_if_media(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_abspath_if_media(v, root) for v in value]
    return value


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    try:
        if pd.isna(value):
            return []
    except (ValueError, TypeError):
        pass
    return [value]


def format_task_row(row: dict[str, Any], root_dir: str | Path) -> dict[str, Any]:
    """Mirror neuralCAD-Edit path rewriting + view flattening."""
    root = Path(root_dir)
    task = {k: _abspath_if_media(v, root) for k, v in row.items()}

    views = _as_list(task.pop("views", []))
    for view in views:
        if view is None or (hasattr(view, "__len__") and not isinstance(view, str) and len(view) == 0):
            continue
        if isinstance(view, list):
            view = view[-1]
        name = Path(str(view)).stem.split("_")[-1]
        task[f"view_{name}"] = view

    brep_start = _as_list(task.pop("brep_start_path", []))
    flat: list[str] = []
    for item in brep_start:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    for filename in flat:
        ext = Path(filename).suffix.lstrip(".").lower()
        task[f"brep_start_path_{ext}"] = filename

    empty_strings = {"", "[]", "null"}
    def _keep(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and value in empty_strings:
            return False
        if isinstance(value, (list, dict)) and len(value) == 0:
            return False
        return True

    return {k: v for k, v in task.items() if _keep(v)}


def align_transcript_and_events(
    words: list[dict[str, Any]],
    events: list[dict[str, Any]],
    window_s: float = 0.75,
) -> list[TemporalCue]:
    """Align WhisperX-style words with pointer/drawing events."""
    cues: list[TemporalCue] = []

    if not words and not events:
        return cues

    if words and not events:
        # Collapse consecutive words into short phrases.
        chunk: list[dict[str, Any]] = []
        for w in words:
            chunk.append(w)
            if len(chunk) >= 8 or str(w.get("word", "")).endswith((".", "!", "?")):
                cues.append(
                    TemporalCue(
                        t_start=float(chunk[0].get("start", 0.0)),
                        t_end=float(chunk[-1].get("end", chunk[0].get("start", 0.0))),
                        text=" ".join(str(x.get("word", "")) for x in chunk).strip(),
                    )
                )
                chunk = []
        if chunk:
            cues.append(
                TemporalCue(
                    t_start=float(chunk[0].get("start", 0.0)),
                    t_end=float(chunk[-1].get("end", chunk[0].get("start", 0.0))),
                    text=" ".join(str(x.get("word", "")) for x in chunk).strip(),
                )
            )
        return cues

    for event in events:
        t = float(event.get("t", event.get("time", event.get("timestamp", 0.0))))
        nearby = [
            w
            for w in words
            if abs(float(w.get("start", t)) - t) <= window_s
            or abs(float(w.get("end", t)) - t) <= window_s
        ]
        text = " ".join(str(w.get("word", "")) for w in nearby).strip()
        cursor = None
        if "x" in event and "y" in event:
            cursor = (float(event["x"]), float(event["y"]))
        elif "cursor" in event and isinstance(event["cursor"], (list, tuple)):
            cursor = (float(event["cursor"][0]), float(event["cursor"][1]))
        cues.append(
            TemporalCue(
                t_start=t,
                t_end=t + float(event.get("duration", 0.2)),
                text=text or str(event.get("label", "")),
                cursor_xy=cursor,
                drawing=bool(event.get("drawing", event.get("is_drawing", False))),
                frame_path=event.get("frame_path"),
            )
        )
    return cues


def parse_request_from_row(row: dict[str, Any], root_dir: str | Path = ".") -> EditRequest:
    task = format_task_row(row, root_dir)

    instruction = (
        task.get("request_text")
        or task.get("instruction")
        or task.get("prompt")
        or task.get("transcript_oracle")
        or task.get("transcript")
        or task.get("text")
        or ""
    )
    if isinstance(instruction, list):
        instruction = " ".join(str(x) for x in instruction)

    step = (
        task.get("brep_start_path_step")
        or task.get("brep_start_path_stp")
        or task.get("step_path")
        or task.get("input_step")
    )

    if step:
        step = os.path.normpath(str(step))
        if Path(step).exists():
            step = str(Path(step).resolve())

    views = {
        k.replace("view_", ""): v
        for k, v in task.items()
        if k.startswith("view_") and isinstance(v, str)
    }

    words = task.get("transcript_words") or task.get("words") or []
    if isinstance(words, str):
        try:
            words = json.loads(words)
        except json.JSONDecodeError:
            words = []

    events = task.get("events") or task.get("pointer_events") or []
    if isinstance(events, str):
        try:
            events = json.loads(events)
        except json.JSONDecodeError:
            events = []

    return EditRequest(
        request_id=str(task.get("request") or task.get("request_id") or task.get("id") or "unknown"),
        instruction=str(instruction),
        step_path=step,
        video_path=task.get("video") or task.get("video_path"),
        transcript=str(task.get("transcript_oracle") or task.get("transcript") or instruction),
        transcript_words=list(words) if isinstance(words, list) else [],
        events=list(events) if isinstance(events, list) else [],
        views=views,
        difficulty=str(task.get("difficulty", "unknown")),
        modality=str(task.get("modality") or task.get("request_modality") or "text"),
        metadata=task,
    )


def load_requests_parquet(
    parquet_path: str | Path,
    root_dir: str | Path | None = None,
    n_rows: Optional[int] = None,
) -> list[EditRequest]:
    path = Path(parquet_path)
    root = Path(root_dir) if root_dir else path.parent
    df = pd.read_parquet(path)
    if n_rows is not None:
        df = df.head(n_rows)
    return [parse_request_from_row(row.to_dict(), root) for _, row in df.iterrows()]


def load_request_json(json_path: str | Path) -> EditRequest:
    path = Path(json_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    root = Path(data.get("root_dir", path.parent))
    return parse_request_from_row(data, root)


def write_benchmark_settings(
    output_dir: str | Path,
    *,
    request_id: str,
    edit_id: str,
    user_id: str,
    start_time: float,
    end_time: float,
    filename: Optional[str],
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write settings.json compatible with neuralCAD-Edit ingestion."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "edit_request_id": request_id,
        "edit_id": edit_id,
        "start_time": start_time,
        "end_time": end_time,
        "filename": filename,
        "isHuman": False,
        "userId": user_id,
    }
    if extra:
        settings.update(extra)
    settings_path = out / "settings.json"
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)
    return settings_path
