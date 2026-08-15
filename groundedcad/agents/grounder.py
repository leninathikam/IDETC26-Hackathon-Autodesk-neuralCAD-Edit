"""Temporal / multimodal reference grounding."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from groundedcad.agents.schemas import (
    EntityKind,
    EntityRef,
    GroundedIntent,
    OperationType,
    TemporalCue,
)
from groundedcad.geometry.inspect import census_to_prompt, find_entities_near_point
from groundedcad.ingest.request_parser import EditRequest
from groundedcad.agents.policy import AGENT_SYSTEM
from groundedcad.llm.base import LLMClient

GROUNDER_SYSTEM = AGENT_SYSTEM + """
Also return JSON with fields:
summary, operation_hints (list), dimensions (object), constraints (list),
ambiguities (list), confidence (0-1), target_text, reference_text, preserve (list),
targets (list of {entity_id, kind, description, confidence}).
"""

OP_KEYWORDS = {
    OperationType.FILLET: [r"\bfillet\b", r"\bround(ed)?\b", r"\brounds?\b"],
    OperationType.CHAMFER: [r"\bchamfer\b", r"\bbevel\b", r"\bgroove"],
    OperationType.HOLE: [r"\bholes?\b", r"\bdrill\b", r"\bbore\b", r"\btap\b"],
    OperationType.EXTRUDE_ADD: [r"\bextrude\b", r"\bthicken\b", r"\brib\b", r"\bflange\b"],
    OperationType.EXTRUDE_CUT: [r"\bcut\b", r"\bpocket\b", r"\bslot\b", r"\bcutout"],
    OperationType.TRANSFORM: [r"\bmove\b", r"\btranslate\b", r"\bshift\b"],
    OperationType.DUPLICATE: [r"\bduplicate\b", r"\bcopy\b", r"\bmirror\b"],
    OperationType.PATTERN: [r"\bpattern\b", r"\barray\b", r"\brepeat\b"],
    OperationType.SCALE: [r"\bscale\b", r"\bshrink\b", r"\benlarge\b"],
}


def _heuristic_ops(text: str) -> list[OperationType]:
    found = []
    lower = text.lower()
    for op, patterns in OP_KEYWORDS.items():
        if any(re.search(p, lower) for p in patterns):
            found.append(op)
    return found or [OperationType.CUSTOM]


_UNIT = r"(?:mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?)"

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a": 1, "an": 1, "another": 1, "both": 2, "couple": 2,
}


def _normalize_instruction(text: str) -> str:
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    return text


def _to_mm(value: float, unit: str | None) -> float:
    u = (unit or "mm").lower()
    if u.startswith("cm") or u.startswith("centimet"):
        return value * 10.0
    return value


_COUNT_NOUNS = r"holes?|buttons?|copies|instances|cutouts?|pins?|blades?|ports?|covers?|parts?"


def _extract_word_count(text: str) -> Optional[int]:
    """'add two click buttons' -> 2. Never guessed; only an explicit number word/digit."""
    word_alt = "|".join(_WORD_NUM)
    m = re.search(
        rf"(?:add|create|design|insert)\s+(\d+|{word_alt})\s+(?:\w+\s+){{0,2}}(?:{_COUNT_NOUNS})",
        text,
        flags=re.I,
    )
    if not m:
        return None
    tok = m.group(1).lower()
    return int(tok) if tok.isdigit() else _WORD_NUM.get(tok)


def _extract_dimensions(text: str) -> dict[str, float]:
    """Parse only explicit numbers. Never invent defaults."""
    dims: dict[str, float] = {}
    text = _normalize_instruction(text)
    count = _extract_word_count(text)
    if count is not None:
        dims["count"] = float(count)
    patterns = [
        ("diameter", rf"(?:diameter|dia|ø)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*({_UNIT})?"),
        ("diameter", rf"(\d+(?:\.\d+)?)\s*({_UNIT})?\s+(?:diameter|dia)"),
        ("radius", rf"(?:radius|fillet|\br\b)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*({_UNIT})?"),
        ("radius", rf"(\d+(?:\.\d+)?)\s*({_UNIT})?\s+(?:radius|fillet|rounds?)"),
        ("distance", rf"(?:chamfer|distance|offset)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*({_UNIT})?"),
        ("distance", rf"(\d+(?:\.\d+)?)\s*({_UNIT})?\s+chamfer"),
        ("groove", rf"(?:groove|grooves|knurl)\s*[:=]?\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*({_UNIT})?"),
        ("groove", rf"(\d+(?:\.\d+)?)\s*({_UNIT})?\s+grooves?"),
        ("depth", rf"(?:depth|deep)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*({_UNIT})?"),
        ("angle", r"(?:angle|deg(?:rees)?)\s*[:=]?\s*(\d+(?:\.\d+)?)"),
        ("count", rf"(?:add|create)\s+(\d+)\s+(?:{_COUNT_NOUNS})"),
        ("factor", r"scale(?:\s+the\s+part)?\s+(\d+(?:\.\d+)?)\s*x"),
        ("factor", r"\b(\d+(?:\.\d+)?)x\b"),
    ]
    for label, pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if not m or label in dims:
            continue
        raw = float(m.group(1))
        unit = m.group(2) if m.lastindex and m.lastindex >= 2 else None
        dims[label] = raw if label in {"angle", "count", "factor"} else _to_mm(raw, unit)
    if "value_mm" not in dims:
        leftover = re.search(rf"(\d+(?:\.\d+)?)\s*({_UNIT})", text, flags=re.I)
        if leftover and not dims:
            dims["value_mm"] = _to_mm(float(leftover.group(1)), leftover.group(2))
    return dims


def heuristic_ground(
    request: EditRequest,
    census: dict[str, Any],
) -> GroundedIntent:
    from groundedcad.agents.instruction_parser import parse_instruction

    cues = request.temporal_cues()
    parsed = parse_instruction(request.instruction or request.transcript)
    ops = parsed.operations
    dims = parsed.dimensions
    targets: list[EntityRef] = []

    # Cursor-normalized coords are often 0-1; map into bbox if present
    bbox = census.get("bbox") or {}
    for cue in cues:
        if not cue.cursor_xy:
            continue
        x_n, y_n = cue.cursor_xy
        # Assume normalized screen coords; project to XY of model bbox center plane
        if bbox:
            x = bbox["xmin"] + float(x_n) * (bbox["xmax"] - bbox["xmin"])
            y = bbox["ymin"] + float(y_n) * (bbox["ymax"] - bbox["ymin"])
            z = 0.5 * (bbox["zmin"] + bbox["zmax"])
        else:
            x, y, z = float(x_n), float(y_n), 0.0
        near = find_entities_near_point(census, (x, y, z), k=3)
        for ent in near:
            ent.confidence = min(0.9, ent.confidence + (0.2 if cue.drawing else 0.1))
            ent.description = f"{ent.description} | cue='{cue.text}' t={cue.t_start:.2f}"
            targets.append(ent)

    # Keyword entity hints — only when the edit *uses* existing holes/edges, not when adding a hole
    from groundedcad.agents.classifier import classify_instruction

    classified = classify_instruction(request.instruction or request.transcript)
    text = (request.instruction or request.transcript).lower()
    if classified.action != "add" and ("hole" in text or "circle" in text or "cylindrical" in text):
        for face in census.get("faces") or []:
            if face.get("radius"):
                targets.append(EntityRef.model_validate(face))
        for edge in census.get("edges") or []:
            geom = (edge.get("metadata") or {}).get("geom", "")
            if "CIRCLE" in str(geom).upper() or edge.get("radius"):
                targets.append(EntityRef.model_validate(edge))

    # Deduplicate by entity_id
    uniq: dict[str, EntityRef] = {}
    for t in targets:
        prev = uniq.get(t.entity_id)
        if prev is None or t.confidence > prev.confidence:
            uniq[t.entity_id] = t
    targets = list(uniq.values())[:12]

    ambiguities = []
    if len(targets) > 5:
        ambiguities.append("Many candidate targets; need disambiguation")
    if not targets:
        ambiguities.append("No grounded target; may need visual/label inspection")
    if len(ops) > 1:
        ambiguities.append("Multiple operations; apply each locally in order")
    raw = request.instruction or request.transcript
    return GroundedIntent(
        summary=raw[:240],
        targets=targets,
        operation_hints=ops,
        dimensions=dims,
        constraints=["smallest possible local edit", "do not invent dimensions", "preserve unrelated geometry"],
        ambiguities=ambiguities,
        temporal_cues=cues,
        confidence=0.75 if targets else 0.45,
        raw_instruction=raw,
        target_text=parsed.target,
        reference_text=parsed.location or "starting STEP census",
        preserve=parsed.preserve,
        location=parsed.location,
        pattern=parsed.pattern,
    )


def llm_ground(
    client: LLMClient,
    request: EditRequest,
    census: dict[str, Any],
    seed: Optional[GroundedIntent] = None,
) -> GroundedIntent:
    seed = seed or heuristic_ground(request, census)
    payload = {
        "instruction": request.instruction,
        "transcript": request.transcript,
        "modality": request.modality,
        "difficulty": request.difficulty,
        "temporal_cues": [c.model_dump() for c in seed.temporal_cues[:30]],
        "geometry_census": census_to_prompt(census),
        "heuristic_seed": seed.model_dump(),
    }
    images = list(request.views.values())[:6] or None
    try:
        data = client.complete_json(
            system=GROUNDER_SYSTEM,
            user=json.dumps(payload, indent=2),
            images=images,
        )
    except Exception:
        return seed

    ops = []
    for item in data.get("operation_hints") or []:
        try:
            ops.append(OperationType(str(item)))
        except Exception:
            continue
    targets = []
    for t in data.get("targets") or []:
        try:
            if "kind" in t:
                t = {**t, "kind": EntityKind(str(t["kind"]))}
            targets.append(EntityRef.model_validate(t))
        except Exception:
            continue
    if not targets:
        targets = seed.targets

    return GroundedIntent(
        summary=str(data.get("summary") or seed.summary),
        targets=targets,
        operation_hints=ops or seed.operation_hints,
        dimensions={**seed.dimensions, **(data.get("dimensions") or {})},
        constraints=list(data.get("constraints") or seed.constraints),
        ambiguities=list(data.get("ambiguities") or seed.ambiguities),
        temporal_cues=seed.temporal_cues,
        confidence=float(data.get("confidence", seed.confidence)),
        raw_instruction=seed.raw_instruction,
        target_text=str(data.get("target_text") or seed.target_text),
        reference_text=str(data.get("reference_text") or seed.reference_text),
        preserve=list(data.get("preserve") or seed.preserve),
        location=str(data.get("location") or seed.location),
    )


def ground_request(
    request: EditRequest,
    census: dict[str, Any],
    client: Optional[LLMClient] = None,
) -> GroundedIntent:
    seed = heuristic_ground(request, census)
    if client is None:
        return seed
    return llm_ground(client, request, census, seed=seed)
