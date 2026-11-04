"""Schema v1.0 contract (Brief section 5)."""
import glob
import json
import os

import pytest
from pydantic import ValidationError

from src.evidence.schema import DeliveryRecord, PHASE_NAMES

BASE = dict(schema_version="1.0", video_id="dev_01", video_sha256="ab" * 32, athlete_id="ATH-PLACEHOLDER-01",
            delivery_id="dev_01-d1", evidence_source="pose:right_wrist", method_or_model="m", pipeline_version="abc123",
            notes_or_uncertainty="")


def ev(**kw):
    return DeliveryRecord(**{**BASE, "record_type": "event", "name": "BR", "visibility_status": "visible",
                             "event_frame": 10, "event_timestamp": 0.3333, **kw})


def test_valid_event_and_not_observable():
    ev()
    ev(visibility_status="not_observable", event_frame=None, event_timestamp=None, event_window_start=8,
       event_window_end=12, confidence=None)


def test_not_observable_must_have_null_frame():
    with pytest.raises(ValidationError):
        ev(visibility_status="not_observable")


def test_delivery_id_format():
    with pytest.raises(ValidationError):
        ev(delivery_id="dev_01-d2")


def test_bad_names_and_window():
    with pytest.raises(ValidationError):
        ev(name="XYZ")
    with pytest.raises(ValidationError):
        ev(event_window_start=12, event_window_end=8)
    with pytest.raises(ValidationError):
        ev(event_window_start=8)
    with pytest.raises(ValidationError):
        ev(confidence=1.5)


def test_phase_rules():
    DeliveryRecord(**{**BASE, "record_type": "phase", "name": "Gather", "visibility_status": "visible",
                      "start_frame": 5, "end_frame": 9, "start_timestamp": 0.1667, "end_timestamp": 0.3})
    with pytest.raises(ValidationError):
        DeliveryRecord(**{**BASE, "record_type": "phase", "name": "Gather", "visibility_status": "visible",
                          "start_frame": 9, "end_frame": 5})
    assert {"Run-up", "Gather", "Delivery stride / release preparation", "Ball release", "Follow-through"} == PHASE_NAMES


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions", "*.json"))))
def test_written_predictions_are_schema_valid(path):
    recs = json.load(open(path))
    names = [r["name"] for r in recs]
    assert sorted(n for n, r in zip(names, recs) if r["record_type"] == "event") == sorted(["RUS", "GTO", "BFC", "FFC", "BR", "FTE"])
    assert len([r for r in recs if r["record_type"] == "phase"]) == 5
    for r in recs:
        DeliveryRecord(**r)
        assert r["human_verified"] is False
