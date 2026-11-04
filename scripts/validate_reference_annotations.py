#!/usr/bin/env python3
"""
ScoutEdge Reference Annotation Validator
Validates reference annotation JSON files in annotations/reference/ against Schema v1.0 logic:
- Record Counts: exactly 6 events and 4 phases
- Visibility Logic: visible events require integer event_frame; not_observable require null event_frame
- Window Logic: uncertainty windows must be valid integers with start < end
- Phase Logic: start_frame < end_frame, proper inheritance of event uncertainty windows without overlap
- Metadata: human_verified=True, schema_version="1.0", delivery_id consistency, and complete schema keys
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = BASE_DIR / "annotations" / "reference"

STANDARD_SCHEMA_KEYS = {
    "schema_version",
    "video_id",
    "video_sha256",
    "athlete_id",
    "delivery_id",
    "record_type",
    "name",
    "start_frame",
    "end_frame",
    "start_timestamp",
    "end_timestamp",
    "event_frame",
    "event_timestamp",
    "event_window_start",
    "event_window_end",
    "visibility_status",
    "confidence",
    "evidence_source",
    "method_or_model",
    "pipeline_version",
    "human_verified",
    "notes_or_uncertainty",
    "annotator",
}

EXPECTED_EVENT_NAMES = ["RUS", "GTO", "BFC", "FFC", "BR", "FTE"]
EXPECTED_PHASE_NAMES = ["Run-up", "Gather", "Delivery stride", "Follow-through"]

PHASE_BOUNDARY_MAPPING = [
    ("Run-up", "RUS", "GTO"),
    ("Gather", "GTO", "BFC"),
    ("Delivery stride", "BFC", "BR"),
    ("Follow-through", "BR", "FTE"),
]


def is_valid_int(val: Any) -> bool:
    """Check if value is an integer (and not a boolean)."""
    return isinstance(val, int) and not isinstance(val, bool)


def is_valid_number(val: Any) -> bool:
    """Check if value is a float or int (and not a boolean)."""
    return (isinstance(val, (int, float))) and not isinstance(val, bool)


def validate_annotation_file(json_path: Path) -> List[str]:
    """
    Validate a single reference annotation JSON file.
    Returns a list of error strings (empty if valid).
    """
    errors = []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        return [f"Failed to read or parse JSON: {e}"]

    if not isinstance(records, list):
        return ["Root JSON structure must be an array of records."]

    video_id = json_path.stem

    # -------------------------------------------------------------------------
    # 1. Record Count & Type Checks
    # -------------------------------------------------------------------------
    event_records = [r for r in records if r.get("record_type") == "event"]
    phase_records = [r for r in records if r.get("record_type") == "phase"]
    other_records = [r for r in records if r.get("record_type") not in ("event", "phase")]

    if len(records) != 10:
        errors.append(f"Expected exactly 10 records, found {len(records)}.")

    if len(event_records) != 6:
        errors.append(f"Expected exactly 6 event records, found {len(event_records)}.")

    if len(phase_records) != 4:
        errors.append(f"Expected exactly 4 phase records, found {len(phase_records)}.")

    if other_records:
        errors.append(f"Found {len(other_records)} records with invalid record_type.")

    event_names = [r.get("name") for r in event_records]
    if event_names != EXPECTED_EVENT_NAMES:
        errors.append(f"Event names/order mismatch. Expected {EXPECTED_EVENT_NAMES}, got {event_names}.")

    phase_names = [r.get("name") for r in phase_records]
    if phase_names != EXPECTED_PHASE_NAMES:
        errors.append(f"Phase names/order mismatch. Expected {EXPECTED_PHASE_NAMES}, got {phase_names}.")

    events_by_name: Dict[str, Dict[str, Any]] = {r.get("name"): r for r in event_records}
    phases_by_name: Dict[str, Dict[str, Any]] = {r.get("name"): r for r in phase_records}

    # -------------------------------------------------------------------------
    # 2. Metadata & Standard Keys
    # -------------------------------------------------------------------------
    for idx, rec in enumerate(records):
        rec_name = rec.get("name", f"Record[{idx}]")
        rec_type = rec.get("record_type", "unknown")

        missing_keys = STANDARD_SCHEMA_KEYS - set(rec.keys())
        if missing_keys:
            errors.append(f"{rec_type} '{rec_name}': Missing required schema keys: {sorted(missing_keys)}.")

        if rec.get("schema_version") != "1.0":
            errors.append(f"{rec_type} '{rec_name}': schema_version must be '1.0', got '{rec.get('schema_version')}'.")

        if rec.get("human_verified") is not True:
            errors.append(f"{rec_type} '{rec_name}': human_verified must be True, got {rec.get('human_verified')}.")

        if rec.get("method_or_model") != "manual_inspection":
            errors.append(f"{rec_type} '{rec_name}': method_or_model must be 'manual_inspection', got '{rec.get('method_or_model')}'.")

        expected_delivery_id = f"{video_id}-d1"
        if rec.get("delivery_id") != expected_delivery_id:
            errors.append(f"{rec_type} '{rec_name}': delivery_id must be '{expected_delivery_id}', got '{rec.get('delivery_id')}'.")

        if rec.get("video_id") != video_id:
            errors.append(f"{rec_type} '{rec_name}': video_id mismatch. Expected '{video_id}', got '{rec.get('video_id')}'.")

    # -------------------------------------------------------------------------
    # 3. Visibility & Window Logic for Events
    # -------------------------------------------------------------------------
    for ev in event_records:
        name = ev.get("name", "UNKNOWN")
        vs = ev.get("visibility_status")
        ef = ev.get("event_frame")
        et = ev.get("event_timestamp")
        ws = ev.get("event_window_start")
        we = ev.get("event_window_end")

        # Phase fields must be null on events
        for pf in ("start_frame", "end_frame", "start_timestamp", "end_timestamp"):
            if ev.get(pf) is not None:
                errors.append(f"Event '{name}': Phase field '{pf}' must be null, got {ev.get(pf)}.")

        if vs == "visible":
            if not is_valid_int(ef):
                errors.append(f"Event '{name}': visibility_status is 'visible' but event_frame={ef} is not an integer.")
            elif ef < 0:
                errors.append(f"Event '{name}': event_frame must be >= 0, got {ef}.")

            if not is_valid_number(et):
                errors.append(f"Event '{name}': visibility_status is 'visible' but event_timestamp={et} is not a valid number.")

        elif vs == "not_observable":
            if ef is not None:
                errors.append(f"Event '{name}': visibility_status is 'not_observable' but event_frame must be null, got {ef}.")
            if et is not None:
                errors.append(f"Event '{name}': visibility_status is 'not_observable' but event_timestamp must be null, got {et}.")

        else:
            errors.append(f"Event '{name}': Invalid visibility_status '{vs}'.")

        # Window logic
        if ws is not None or we is not None:
            if not is_valid_int(ws) or not is_valid_int(we):
                errors.append(f"Event '{name}': Uncertainty window bounds must both be valid integers, got ({ws}, {we}).")
            elif ws < 0:
                errors.append(f"Event '{name}': event_window_start must be >= 0, got {ws}.")
            elif ws >= we:
                errors.append(f"Event '{name}': Uncertainty window start ({ws}) must be < end ({we}).")

    # -------------------------------------------------------------------------
    # 4. Phase Logic & Boundary Inheritance
    # -------------------------------------------------------------------------
    prev_phase_end: Any = None
    for phase_name, start_ev_name, end_ev_name in PHASE_BOUNDARY_MAPPING:
        if phase_name not in phases_by_name:
            continue

        ph = phases_by_name[phase_name]
        sf = ph.get("start_frame")
        ef = ph.get("end_frame")
        st = ph.get("start_timestamp")
        et = ph.get("end_timestamp")
        vs = ph.get("visibility_status")

        # Event fields must be null on phases
        for ev_f in ("event_frame", "event_timestamp", "event_window_start", "event_window_end"):
            if ph.get(ev_f) is not None:
                errors.append(f"Phase '{phase_name}': Event field '{ev_f}' must be null, got {ph.get(ev_f)}.")

        start_ev = events_by_name.get(start_ev_name, {})
        end_ev = events_by_name.get(end_ev_name, {})

        # Determine expected boundaries from events
        expected_sf = None
        if start_ev.get("event_frame") is not None:
            expected_sf = start_ev["event_frame"]
        elif start_ev.get("event_window_end") is not None:
            expected_sf = start_ev["event_window_end"]

        expected_ef = None
        if end_ev.get("event_frame") is not None:
            expected_ef = end_ev["event_frame"]
        elif end_ev.get("event_window_start") is not None:
            expected_ef = end_ev["event_window_start"]

        can_be_valid = (expected_sf is not None and expected_ef is not None and expected_ef > expected_sf)

        if vs == "visible":
            if not is_valid_int(sf) or not is_valid_int(ef):
                errors.append(f"Phase '{phase_name}': start_frame and end_frame must be integers, got ({sf}, {ef}).")
            elif sf >= ef:
                errors.append(f"Phase '{phase_name}': start_frame ({sf}) must be < end_frame ({ef}).")

            if not is_valid_number(st) or not is_valid_number(et):
                errors.append(f"Phase '{phase_name}': start_timestamp and end_timestamp must be numbers, got ({st}, {et}).")
            elif st is not None and et is not None and st > et:
                errors.append(f"Phase '{phase_name}': start_timestamp ({st}) must be <= end_timestamp ({et}).")

            # Check boundary inheritance from events
            if expected_sf is not None and sf != expected_sf:
                errors.append(f"Phase '{phase_name}': start_frame ({sf}) does not match boundary event {start_ev_name} ({expected_sf}).")
            if expected_ef is not None and ef != expected_ef:
                errors.append(f"Phase '{phase_name}': end_frame ({ef}) does not match boundary event {end_ev_name} ({expected_ef}).")

            # Check non-overlap with previous phase
            if prev_phase_end is not None and sf is not None:
                if sf < prev_phase_end:
                    errors.append(f"Phase '{phase_name}' overlaps previous phase: start_frame ({sf}) < previous end_frame ({prev_phase_end}).")

            prev_phase_end = ef

        elif vs == "not_observable":
            if sf is not None or ef is not None or st is not None or et is not None:
                errors.append(f"Phase '{phase_name}': not_observable phase must have null frames/timestamps, got ({sf}, {ef}, {st}, {et}).")
            prev_phase_end = None
        else:
            errors.append(f"Phase '{phase_name}': Invalid visibility_status '{vs}'.")

    return errors


def validate_all_reference_annotations():
    """Validate reference annotation JSON files (default: annotations/reference/*.json)."""
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ANNOTATIONS_DIR

    if target.is_file():
        json_files = [target]
    elif target.is_dir():
        json_files = sorted(target.glob("*.json"))
    else:
        print(f"[!] Path does not exist: {target}")
        sys.exit(1)

    if not json_files:
        print(f"[!] No JSON files found in {target}")
        sys.exit(1)

    print(f"[*] Validating {len(json_files)} reference annotation file(s) from {target}...\n")

    total_files = len(json_files)
    passed_files = 0
    failed_files = 0

    for json_path in json_files:
        file_errors = validate_annotation_file(json_path)

        if not file_errors:
            passed_files += 1
            print(f" [PASS] {json_path.name}")
        else:
            failed_files += 1
            print(f" [FAIL] {json_path.name}:")
            for err in file_errors:
                print(f"        - {err}")

    print("\n" + "=" * 60)
    print(f"Validation Summary: {passed_files}/{total_files} file(s) PASSED, {failed_files} FAILED")
    print("=" * 60)

    if failed_files > 0:
        sys.exit(1)


if __name__ == "__main__":
    validate_all_reference_annotations()
