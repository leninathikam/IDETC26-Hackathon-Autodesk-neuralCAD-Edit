"""Typed contracts for grounded CAD editing."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class EditPattern(str, Enum):
    HOLE_EDIT = "hole_edit"
    FEATURE_TRANSLATION = "feature_translation"
    FEATURE_ADDITION = "feature_addition"
    FEATURE_DELETION = "feature_deletion"
    DIMENSION_CHANGE = "dimension_change"
    PATTERN = "pattern"
    FILLET_CHAMFER = "fillet_chamfer"
    BOOLEAN_MODIFICATION = "boolean_modification"
    AMBIGUOUS = "ambiguous"


class OperationType(str, Enum):
    FILLET = "fillet"
    CHAMFER = "chamfer"
    HOLE = "hole"
    EXTRUDE_ADD = "extrude_add"
    EXTRUDE_CUT = "extrude_cut"
    BOOLEAN_UNION = "boolean_union"
    BOOLEAN_CUT = "boolean_cut"
    TRANSFORM = "transform"
    DUPLICATE = "duplicate"
    PATTERN = "pattern"
    SCALE = "scale"
    CUSTOM = "custom"


class EntityKind(str, Enum):
    BODY = "body"
    FACE = "face"
    EDGE = "edge"
    VERTEX = "vertex"
    COMPONENT = "component"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            0.5 * (self.xmin + self.xmax),
            0.5 * (self.ymin + self.ymax),
            0.5 * (self.zmin + self.zmax),
        )

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin)

    def volume_approx(self) -> float:
        sx, sy, sz = self.size
        return abs(sx * sy * sz)


class EntityRef(BaseModel):
    entity_id: str
    kind: EntityKind = EntityKind.UNKNOWN
    description: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    center: Optional[tuple[float, float, float]] = None
    radius: Optional[float] = None
    length: Optional[float] = None
    area: Optional[float] = None
    normal: Optional[tuple[float, float, float]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TemporalCue(BaseModel):
    t_start: float
    t_end: float
    text: str = ""
    cursor_xy: Optional[tuple[float, float]] = None
    drawing: bool = False
    frame_path: Optional[str] = None


class GroundedIntent(BaseModel):
    summary: str
    targets: list[EntityRef] = Field(default_factory=list)
    operation_hints: list[OperationType] = Field(default_factory=list)
    dimensions: dict[str, float] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    temporal_cues: list[TemporalCue] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    raw_instruction: str = ""
    target_text: str = ""
    reference_text: str = ""
    preserve: list[str] = Field(default_factory=list)
    location: str = ""
    pattern: str = ""


class SuccessCheck(BaseModel):
    name: str
    description: str
    check_type: Literal[
        "valid_brep",
        "body_count",
        "volume_delta",
        "bbox_delta",
        "dimension",
        "connectivity",
        "visual",
        "custom",
    ] = "custom"
    params: dict[str, Any] = Field(default_factory=dict)


class EditSpec(BaseModel):
    intent_summary: str
    operation: OperationType
    targets: list[EntityRef] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    invariants: list[str] = Field(default_factory=list)
    success_checks: list[SuccessCheck] = Field(default_factory=list)
    prefer_local: bool = True
    fallback_to_raw_script: bool = True
    notes: str = ""


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    followups: list["ToolCall"] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    success: bool
    step_path: Optional[str] = None
    stl_path: Optional[str] = None
    image_paths: dict[str, str] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    script: Optional[str] = None
    geometry_summary: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    duration_s: float = 0.0


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    measured: dict[str, Any] = Field(default_factory=dict)


class VerificationDecision(BaseModel):
    correct: bool = False
    instruction_satisfied: bool = False
    unintended_changes: bool = False
    dimension_error: bool = False
    position_error: bool = False
    topology_error: bool = False
    severity: Literal["none", "minor", "major"] = "major"
    diagnosis: str = ""
    specific_fix: str = ""


class Critique(BaseModel):
    accept: bool
    instruction_score: float = Field(default=0.0, ge=0.0, le=7.0)
    quality_score: float = Field(default=0.0, ge=0.0, le=7.0)
    failed_checks: list[CheckResult] = Field(default_factory=list)
    passed_checks: list[CheckResult] = Field(default_factory=list)
    revision_advice: str = ""
    evidence: list[str] = Field(default_factory=list)
    verification: Optional[VerificationDecision] = None


class IterationLog(BaseModel):
    iteration: int
    intent: Optional[GroundedIntent] = None
    edit_spec: Optional[EditSpec] = None
    execution: Optional[ExecutionResult] = None
    critique: Optional[Critique] = None
    token_counts: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class PipelineResult(BaseModel):
    request_id: str
    accepted: bool
    best_iteration: int = -1
    output_dir: str
    step_path: Optional[str] = None
    stl_path: Optional[str] = None
    views: dict[str, str] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    iterations: list[IterationLog] = Field(default_factory=list)
    failure_category: Optional[str] = None
