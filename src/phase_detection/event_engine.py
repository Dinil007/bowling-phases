"""
Stage 5: turn event estimates into schema v1.0 records (6 events + 5 phases).

Phase boundaries come only from observed event frames; if either boundary event is not observable
the phase frames are null and the record says which boundary is missing.
"""
from typing import List
from src.evidence.schema import DeliveryRecord, PHASES


def build_records(estimates, meta, entry, sha, pipeline_version, extra_notes: str = "") -> List[DeliveryRecord]:
    base = dict(schema_version="1.0", video_id=entry.clip_id, video_sha256=sha, athlete_id=entry.athlete_id,
                delivery_id=f"{entry.clip_id}-d1", pipeline_version=pipeline_version, human_verified=False)
    recs, by = [], {e.name: e for e in estimates}
    for e in estimates:
        notes = e.notes + (" " + extra_notes if extra_notes else "")
        recs.append(DeliveryRecord(
            **base, record_type="event", name=e.name,
            event_frame=e.frame, event_timestamp=meta.ts(e.frame) if e.frame is not None else None,
            event_window_start=e.window[0] if e.window else None, event_window_end=e.window[1] if e.window else None,
            visibility_status=e.visibility if e.frame is not None or e.visibility != "visible" else "not_observable",
            confidence=e.confidence if e.frame is not None else None,
            evidence_source=e.evidence, method_or_model=e.method, notes_or_uncertainty=notes.strip()))
    for name, a, b in PHASES:
        ea = by[a]
        if b is None:  # Ball release: instantaneous
            ok = ea.frame is not None
            sf = ef = ea.frame if ok else None
            note = "Instantaneous event phase (no duration)." if ok else "Ball release not observable."
        else:
            eb = by[b]
            ok = ea.frame is not None and eb.frame is not None and eb.frame > ea.frame
            sf, ef = (ea.frame, eb.frame) if ok else (None, None)
            if ok:
                note = f"From {a} (frame {ea.frame}) to {b} (frame {eb.frame}, exclusive)."
            else:
                miss = [x for x, e in ((a, ea), (b, eb)) if e.frame is None]
                note = (f"Boundary event(s) {', '.join(miss)} not observable; phase extent unknown." if miss
                        else f"Boundary order violated ({a}={ea.frame}, {b}={eb.frame}); phase withheld.")
        conf = None
        if ok and ea.confidence is not None and (b is None or by[b].confidence is not None):
            conf = round(min(ea.confidence, by[b].confidence) if b else ea.confidence, 3)
        recs.append(DeliveryRecord(
            **base, record_type="phase", name=name, start_frame=sf, end_frame=ef,
            start_timestamp=meta.ts(sf) if sf is not None else None,
            end_timestamp=meta.ts(ef) if ef is not None else None,
            visibility_status="visible" if ok else "not_observable", confidence=conf,
            evidence_source=f"events:{a}" + (f",{b}" if b else ""), method_or_model="phase-from-events-v1",
            notes_or_uncertainty=note))
    return recs


def failure_records(entry, sha, pipeline_version, err) -> List[DeliveryRecord]:
    """Schema-valid, fully not_observable output for a clip that failed explicitly (no person / ambiguous)."""
    base = dict(schema_version="1.0", video_id=entry.clip_id, video_sha256=sha, athlete_id=entry.athlete_id,
                delivery_id=f"{entry.clip_id}-d1", pipeline_version=pipeline_version, human_verified=False)
    msg = f"{err.tag}: {err}"
    recs = [DeliveryRecord(**base, record_type="event", name=n, visibility_status="not_observable",
                           evidence_source="none", method_or_model="pipeline-failure", notes_or_uncertainty=msg)
            for n in ("RUS", "GTO", "BFC", "FFC", "BR", "FTE")]
    recs += [DeliveryRecord(**base, record_type="phase", name=p[0], visibility_status="not_observable",
                            evidence_source="none", method_or_model="pipeline-failure", notes_or_uncertainty=msg)
             for p in PHASES]
    return recs
