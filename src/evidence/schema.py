"""
ScoutEdge Evidence Output Schema v1.0
Authoritative Pydantic data model and pre-serialization schema linter.
"""

from typing import Literal, Optional, List, Any
import json
import os
import tempfile
from pydantic import BaseModel, Field, field_validator, model_validator

RecordType = Literal["event", "phase"]
VisibilityStatus = Literal["visible", "partially_visible", "occluded", "out_of_frame", "not_observable"]

EVENT_NAMES = {"RUS", "GTO", "BFC", "FFC", "BR", "FTE"}
# Phase names exactly as in Brief section 4B.
PHASES = [  # (name, from_event, to_event)
    ("Run-up", "RUS", "GTO"),
    ("Gather", "GTO", "BFC"),
    ("Delivery stride / release preparation", "BFC", "BR"),
    ("Ball release", "BR", None),
    ("Follow-through", "BR", "FTE"),
]
PHASE_NAMES = {p[0] for p in PHASES}
# accepted aliases (older files, reference annotations) -> canonical name
PHASE_ALIASES = {
    "run_up": "Run-up", "run-up": "Run-up", "gather": "Gather",
    "delivery_stride": "Delivery stride / release preparation", "delivery stride": "Delivery stride / release preparation",
    "ball_release": "Ball release", "follow_through": "Follow-through", "follow-through": "Follow-through",
}


def canonical_phase(name: str) -> str:
    if name in PHASE_NAMES:
        return name
    return PHASE_ALIASES.get(name.strip().lower(), name)


class DeliveryRecord(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    video_id: str
    video_sha256: str
    athlete_id: str
    delivery_id: str
    record_type: RecordType
    name: str

    # Phases only
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None

    # Events only
    event_frame: Optional[int] = None
    event_timestamp: Optional[float] = None
    event_window_start: Optional[int] = None
    event_window_end: Optional[int] = None

    visibility_status: VisibilityStatus
    confidence: Optional[float] = None
    evidence_source: str
    method_or_model: str
    pipeline_version: str
    human_verified: bool = False
    annotator: Optional[str] = None          # references only
    notes_or_uncertainty: str

    @field_validator("delivery_id")
    def validate_delivery_id(cls, v: str, info) -> str:
        video_id = info.data.get("video_id")
        if video_id and v != f"{video_id}-d1":
            raise ValueError(f"delivery_id must be '{video_id}-d1', got '{v}'")
        return v

    @field_validator("name")
    def validate_name(cls, v: str, info) -> str:
        rec_type = info.data.get("record_type")
        if rec_type == "event" and v not in EVENT_NAMES:
            raise ValueError(f"Invalid event name '{v}'. Allowed: {EVENT_NAMES}")
        if rec_type == "phase" and canonical_phase(v) not in PHASE_NAMES:
            raise ValueError(f"Invalid phase name '{v}'. Allowed: {PHASE_NAMES}")
        return v

    @model_validator(mode="after")
    def validate_record_consistency(self) -> "DeliveryRecord":
        if self.record_type == "event":
            if self.visibility_status == "not_observable":
                if self.event_frame is not None or self.event_timestamp is not None:
                    raise ValueError(
                        f"Event '{self.name}' has visibility_status='not_observable' "
                        f"but event_frame/timestamp is not null ({self.event_frame}, {self.event_timestamp})"
                    )
            elif self.visibility_status in ("visible", "partially_visible"):
                if self.event_frame is None and self.event_window_start is None:
                    raise ValueError(f"Observable event '{self.name}' must have event_frame or event_window")
            if (self.event_window_start is None) != (self.event_window_end is None):
                raise ValueError(f"Event '{self.name}': window start/end must both be set or both null")
            if self.event_window_start is not None and self.event_window_start > self.event_window_end:
                raise ValueError(f"Event '{self.name}': window start > end")
            if (self.event_frame is None) != (self.event_timestamp is None):
                raise ValueError(f"Event '{self.name}': event_frame and event_timestamp must both be set or both null")
            if any(v is not None for v in (self.start_frame, self.end_frame)):
                raise ValueError(f"Event '{self.name}' must not carry phase frames")

        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1] or null, got {self.confidence}")
        if self.record_type == "phase":
            if self.event_frame is not None:
                raise ValueError(f"Phase '{self.name}' must not carry event_frame")
            if canonical_phase(self.name) == "Ball release":
                if self.start_frame is not None and self.start_frame != self.end_frame:
                    raise ValueError("Ball release is an instantaneous phase: start_frame must equal end_frame")
            elif self.start_frame is not None and self.end_frame is not None:
                if self.start_frame >= self.end_frame:
                    raise ValueError(f"Phase '{self.name}' start_frame ({self.start_frame}) >= end_frame ({self.end_frame})")
        return self


def validate_records(records: List[dict]) -> List[DeliveryRecord]:
    """Validates raw dicts against DeliveryRecord schema."""
    validated = []
    for r in records:
        validated.append(DeliveryRecord(**r))
    return validated


def atomic_write_predictions(predictions: List[DeliveryRecord], target_path: str):
    """
    Atomically writes validated predictions to disk.
    Serializes to a temp file, validates, then renames to avoid partial writes.
    """
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)

    # Convert to pure JSON-serializable list
    data = [rec.model_dump(exclude_none=False) for rec in predictions]
    for d in data:
        if d.get("annotator") is None:
            d.pop("annotator", None)

    # Write to temporary file first
    temp_fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix=".tmp_pred_", suffix=".json")
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Validate by reading back
        with open(temp_path, "r", encoding="utf-8") as f:
            read_back = json.load(f)
            validate_records(read_back)

        # Atomic replacement
        os.replace(temp_path, target_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Atomic prediction write failed for {target_path}: {e}")
