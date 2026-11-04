#!/usr/bin/env python3
"""
ScoutEdge Manual Annotation GUI
Standalone OpenCV-based annotation tool for bowling delivery events.

Usage:
    python scripts/annotate_gui.py [video_id]

    If video_id is omitted, all 8 development clips are annotated sequentially.

Keyboard Controls:
    Space       - Pause / Play
    Left Arrow  - Step back exactly 1 frame
    Right Arrow - Step forward exactly 1 frame
    r           - Log RUS  (Run-Up Start) at current frame
    g           - Log GTO  (Gather Take-Off) at current frame
    b           - Log BFC  (Back-Foot Contact) at current frame
    f           - Log FFC  (Front-Foot Contact) at current frame
    v           - Log BR   (Ball Release) at current frame
    e           - Log FTE  (Follow-Through End) at current frame
    n           - Mark current event as not_observable (clears frame/timestamp)
    [           - Set event_window_start for current event
    ]           - Set event_window_end for current event
    q / ESC     - Export annotations to JSON and quit (or move to next clip)

Output:
    annotations/reference/<video_id>.json  (schema v1.0, human_verified=true)
"""

import csv
import json
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "data" / "input_manifest.csv"
ANNOTATIONS_DIR = BASE_DIR / "annotations" / "reference"

ANNOTATOR = "Dinil Raj"
PIPELINE_VERSION = "ref-v1.0"
CONFIDENCE = 1.0
METHOD_OR_MODEL = "manual_inspection"
SCHEMA_VERSION = "1.0"

EVENT_NAMES = ["RUS", "GTO", "BFC", "FFC", "BR", "FTE"]

EVENT_KEYS = {
    ord("r"): "RUS",
    ord("g"): "GTO",
    ord("b"): "BFC",
    ord("f"): "FFC",
    ord("v"): "BR",
    ord("e"): "FTE",
}

EVENT_KEY_CHARS = {
    "RUS": "r",
    "GTO": "g",
    "BFC": "b",
    "FFC": "f",
    "BR": "v",
    "FTE": "e",
}

EVIDENCE_SOURCE = {
    "RUS": "human_annotation:run_up_onset",
    "GTO": "human_annotation:gather_bound_takeoff",
    "BFC": "human_annotation:back_foot_landing",
    "FFC": "human_annotation:front_foot_strike",
    "BR": "human_annotation:ball_release",
    "FTE": "human_annotation:follow_through_end",
}

DEFAULT_NOTES = {
    "RUS": "First observable forward motion or frame 0.",
    "GTO": "Last foot leaves turf for bound.",
    "BFC": "Back foot touchdown post-bound.",
    "FFC": "Front foot landing in delivery stride.",
    "BR": "Ball release from bowling hand.",
    "FTE": "Follow-through end / deceleration complete.",
}

# 4 delivery phases derived from boundary events
PHASE_DEFS = [
    {
        "name": "Run-up",
        "start_event": "RUS",
        "end_event": "GTO",
        "notes": "Run-up approach.",
        "evidence_source": "human_annotation:phase_run_up",
    },
    {
        "name": "Gather",
        "start_event": "GTO",
        "end_event": "BFC",
        "notes": "Aerial bound.",
        "evidence_source": "human_annotation:phase_gather",
    },
    {
        "name": "Delivery stride",
        "start_event": "BFC",
        "end_event": "BR",
        "notes": "Stride from back foot landing to release.",
        "evidence_source": "human_annotation:phase_delivery_stride",
    },
    {
        "name": "Follow-through",
        "start_event": "BR",
        "end_event": "FTE",
        "notes": "Post-release deceleration.",
        "evidence_source": "human_annotation:phase_follow_through",
    },
]

KEY_SPACE = 32
KEY_Q = ord("q")
KEY_ESC = 27
KEY_LEFT_CODES = {81, 2424832}
KEY_RIGHT_CODES = {83, 2555904}
KEY_UP_CODES = {82, 2490368}
KEY_DOWN_CODES = {84, 2621440}


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def load_manifest():
    """Load input_manifest.csv and return a list of clip dicts."""
    clips = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append(row)
    return clips


def get_clip_by_id(clips, video_id):
    """Find a clip dict by clip_id."""
    for clip in clips:
        if clip["clip_id"] == video_id:
            return clip
    return None


# ---------------------------------------------------------------------------
# Record factories
# ---------------------------------------------------------------------------
def make_event_record(video_id, video_sha256, athlete_id, event_name):
    """Create a blank event record compliant with schema v1.0."""
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "video_sha256": video_sha256,
        "athlete_id": athlete_id,
        "delivery_id": f"{video_id}-d1",
        "record_type": "event",
        "name": event_name,
        "start_frame": None,
        "end_frame": None,
        "start_timestamp": None,
        "end_timestamp": None,
        "event_frame": None,
        "event_timestamp": None,
        "event_window_start": None,
        "event_window_end": None,
        "visibility_status": "visible",
        "confidence": CONFIDENCE,
        "evidence_source": EVIDENCE_SOURCE[event_name],
        "method_or_model": METHOD_OR_MODEL,
        "pipeline_version": PIPELINE_VERSION,
        "human_verified": True,
        "annotator": ANNOTATOR,
        "notes_or_uncertainty": DEFAULT_NOTES[event_name],
    }


def get_event_boundary_frame(rec, role: str):
    """
    Get the frame number for an event when bounding a phase.

    If an exact event_frame was logged, use it.
    If the event was marked not_observable and has an uncertainty window:
      - role == "end": use event_window_start as phase end_frame
      - role == "start": use event_window_end as next phase start_frame
    """
    if not rec:
        return None

    if rec.get("event_frame") is not None:
        return rec["event_frame"]

    if role == "end":
        return rec.get("event_window_start")
    elif role == "start":
        return rec.get("event_window_end")

    return None


def make_phase_record(video_id, video_sha256, athlete_id, phase_def, start_rec, end_rec, fps):
    """Create a phase record calculated from boundary events compliant with schema rules."""
    phase_name = phase_def["name"]
    start_frame = get_event_boundary_frame(start_rec, role="start")
    end_frame = get_event_boundary_frame(end_rec, role="end")

    is_valid = (start_frame is not None and end_frame is not None and end_frame > start_frame)

    start_timestamp = round(start_frame / fps, 4) if (is_valid and fps > 0) else None
    end_timestamp = round(end_frame / fps, 4) if (is_valid and fps > 0) else None

    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "video_sha256": video_sha256,
        "athlete_id": athlete_id,
        "delivery_id": f"{video_id}-d1",
        "record_type": "phase",
        "name": phase_name,
        "start_frame": start_frame if is_valid else None,
        "end_frame": end_frame if is_valid else None,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "event_frame": None,
        "event_timestamp": None,
        "event_window_start": None,
        "event_window_end": None,
        "visibility_status": "visible" if is_valid else "not_observable",
        "confidence": CONFIDENCE if is_valid else None,
        "evidence_source": phase_def["evidence_source"],
        "method_or_model": METHOD_OR_MODEL,
        "pipeline_version": PIPELINE_VERSION,
        "human_verified": True,
        "annotator": ANNOTATOR,
        "notes_or_uncertainty": phase_def["notes"] if is_valid else f"Phase '{phase_name}' not observable in this clip interval.",
    }


def put_outlined_text(
    img,
    text,
    org,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    color=(0, 255, 0),
    thickness=1,
    outline_color=(0, 0, 0),
    outline_thickness=3,
):
    """Draw text with a high-contrast dark outline behind it so it pops on any background."""
    cv2.putText(
        img,
        text,
        org,
        font_face,
        font_scale,
        outline_color,
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Annotation loop
# ---------------------------------------------------------------------------
def annotate_video(clip):
    """Run the interactive OpenCV annotation loop for a single clip."""
    video_id = clip["clip_id"]
    video_path = BASE_DIR / clip["file_name"]
    video_sha256 = clip["sha256"]
    athlete_id = clip["athlete_id"]
    fps = float(clip["frame_rate"])

    if not video_path.exists():
        print(f"[!] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Dynamic display resizing: scale to max height of 800px while maintaining aspect ratio
    MAX_DISPLAY_HEIGHT = 800
    if orig_height > MAX_DISPLAY_HEIGHT:
        scale = MAX_DISPLAY_HEIGHT / float(orig_height)
        disp_width = int(round(orig_width * scale))
        disp_height = MAX_DISPLAY_HEIGHT
    else:
        disp_width = orig_width
        disp_height = orig_height

    # width and height used for all UI overlay calculations and display rendering
    width, height = disp_width, disp_height

    print(f"\n{'='*60}")
    print(f"Annotating: {video_id}")
    print(f"  File:      {video_path}")
    print(f"  Resolution: {orig_width}x{orig_height} (Display: {width}x{height}), FPS: {fps}, Total frames: {total_frames}")
    print(f"  SHA-256:   {video_sha256}")
    print(f"  Athlete:   {athlete_id}")
    print(f"{'='*60}")
    print("Controls: Space=Pause/Play | Left/Right=Step 1 frame | Up/Down=Select event |")
    print("          r/g/b/f/v/e=Log event | n=Not observable |")
    print("          [ = window start | ] = window end | q=Export & quit")
    print()

    records = {}
    for name in EVENT_NAMES:
        records[name] = make_event_record(video_id, video_sha256, athlete_id, name)

    json_path = ANNOTATIONS_DIR / f"{video_id}.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            for r in existing_data:
                if r.get("record_type") == "event" and r.get("name") in records:
                    records[r["name"]] = r
            print(f"  [i] Pre-loaded existing annotations from {json_path}")
        except Exception as e:
            print(f"  [!] Could not load existing annotations: {e}")

    current_event = EVENT_NAMES[0]
    frame_index = 0
    paused = False
    step_requested = None

    use_waitkey_ex = hasattr(cv2, "waitKeyEx")

    while True:
        if step_requested is not None:
            frame_index += step_requested
            frame_index = max(0, min(frame_index, total_frames - 1))
            step_requested = None
            paused = True

        if not paused:
            frame_index += 1
            if frame_index >= total_frames:
                frame_index = total_frames - 1
                paused = True

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"[!] Failed to read frame {frame_index}")
            break

        timestamp = frame_index / fps if fps > 0 else 0.0

        # Resize frame to display resolution before drawing any UI elements
        if (orig_width, orig_height) != (width, height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        top_band_h = 280
        bottom_band_h = 58
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, top_band_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Solid black rectangle at the bottom for hint lines
        cv2.rectangle(frame, (0, height - bottom_band_h), (width, height), (0, 0, 0), -1)

        # Responsive scaling & layout for mobile/vertical videos
        is_narrow = width < 700
        f_scale_title = 0.65 if is_narrow else 0.8
        f_scale_info = 0.55 if is_narrow else 0.7
        f_scale_list = 0.48 if is_narrow else 0.52

        # 1. Main stats
        put_outlined_text(
            frame,
            f"Frame: {frame_index} / {total_frames - 1}",
            (15, 28),
            font_scale=f_scale_title,
            color=(0, 255, 0),
        )
        put_outlined_text(
            frame,
            f"Timestamp: {timestamp:.4f}s  @ {fps:.2f} fps",
            (15, 54),
            font_scale=f_scale_info,
            color=(0, 255, 0),
        )

        state_text = "PAUSED" if paused else "PLAYING"
        state_color = (0, 255, 255) if paused else (0, 255, 128)
        put_outlined_text(
            frame,
            state_text,
            (15, 80),
            font_scale=f_scale_info,
            color=state_color,
        )

        # 2. Prominent, larger & brighter 'Current Event' display
        ce_text = f"Current Event: {current_event if current_event else 'None'}"
        f_scale_ce = 0.72 if is_narrow else 0.85
        if is_narrow:
            ce_org = (15, 110)
            status_y = 138
        else:
            ce_org = (width - 390, 38)
            status_y = 115

        (tw, th), baseline = cv2.getTextSize(ce_text, cv2.FONT_HERSHEY_SIMPLEX, f_scale_ce, 2)
        pad = 6
        bx1 = ce_org[0] - pad
        by1 = ce_org[1] - th - pad
        bx2 = ce_org[0] + tw + pad
        by2 = ce_org[1] + baseline + pad
        badge_border = (0, 215, 255) if current_event else (70, 70, 70)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_border, 2)

        put_outlined_text(
            frame,
            ce_text,
            ce_org,
            font_scale=f_scale_ce,
            color=(0, 255, 255) if current_event else (160, 160, 160),
            thickness=2,
            outline_thickness=4,
        )

        # 3. Event list with hotkeys explicitly mapped in brackets
        status_x = 15
        line_spacing = 22 if is_narrow else 24
        for i, name in enumerate(EVENT_NAMES):
            rec = records[name]
            ef = rec["event_frame"]
            vs = rec["visibility_status"]
            win_start = rec.get("event_window_start")
            win_end = rec.get("event_window_end")

            color = (0, 255, 0) if (vs == "visible" and ef is not None) else (0, 165, 255)
            tag = ">" if name == current_event else " "
            key_char = EVENT_KEY_CHARS.get(name, "?")

            if ef is not None:
                frame_desc = f"frame={ef}"
            elif win_start is not None or win_end is not None:
                frame_desc = f"win=[{win_start if win_start is not None else '--'}, {win_end if win_end is not None else '--'}]"
            else:
                frame_desc = "frame=--"

            status_text = f"{tag} [{key_char}] {name}: {frame_desc} [{vs[:5]}]"
            put_outlined_text(
                frame,
                status_text,
                (status_x, status_y + i * line_spacing),
                font_scale=f_scale_list,
                color=color,
            )

        # 4. Split two-line bottom hint over solid black background
        line1 = "PRESS: r=RUS | g=GTO | b=BFC | f=FFC | v=BR | e=FTE | q=Save & Next"
        line2 = "IF UNCLEAR: Press 'n', then mark window with '[' (start) and ']' (end)"

        hint_scale_1 = 0.46
        hint_scale_2 = 0.42
        t1 = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, hint_scale_1, 1)[0]
        if t1[0] > width - 20:
            hint_scale_1 = hint_scale_1 * (width - 20) / t1[0]

        t2 = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, hint_scale_2, 1)[0]
        if t2[0] > width - 20:
            hint_scale_2 = hint_scale_2 * (width - 20) / t2[0]

        put_outlined_text(
            frame,
            line1,
            (12, height - 32),
            font_scale=hint_scale_1,
            color=(255, 255, 255),
        )
        put_outlined_text(
            frame,
            line2,
            (12, height - 10),
            font_scale=hint_scale_2,
            color=(0, 215, 255),
        )

        cv2.imshow("ScoutEdge Annotation GUI", frame)

        delay = 1 if paused else 30
        if use_waitkey_ex:
            key = cv2.waitKeyEx(delay)
        else:
            key = cv2.waitKey(delay)

        if key == -1:
            continue

        if key == KEY_Q or key == KEY_ESC:
            break
        elif key == KEY_SPACE:
            paused = not paused
        elif key in KEY_LEFT_CODES:
            step_requested = -1
        elif key in KEY_RIGHT_CODES:
            step_requested = 1
        elif key in KEY_UP_CODES:
            current_idx = EVENT_NAMES.index(current_event) if current_event in EVENT_NAMES else 0
            current_event = EVENT_NAMES[(current_idx - 1) % len(EVENT_NAMES)]
        elif key in KEY_DOWN_CODES:
            current_idx = EVENT_NAMES.index(current_event) if current_event in EVENT_NAMES else 0
            current_event = EVENT_NAMES[(current_idx + 1) % len(EVENT_NAMES)]
        elif key in EVENT_KEYS:
            event_name = EVENT_KEYS[key]
            current_event = event_name
            rec = records[event_name]
            rec["event_frame"] = frame_index
            rec["event_timestamp"] = round(timestamp, 4)
            rec["visibility_status"] = "visible"
            print(f"[+] {event_name} logged at frame {frame_index} (t={timestamp:.4f}s)")
        elif key == ord("n"):
            if current_event is None:
                print("[!] No event selected. Press an event key (r/g/b/f/v/e) first.")
                continue
            rec = records[current_event]
            rec["visibility_status"] = "not_observable"
            rec["event_frame"] = None
            rec["event_timestamp"] = None
            print(f"[!] {current_event} marked as not_observable")
        elif key == ord("["):
            if current_event is None:
                print("[!] No event selected. Press an event key (r/g/b/f/v/e) first.")
                continue
            rec = records[current_event]
            rec["event_window_start"] = frame_index
            print(f"[-] {current_event} event_window_start = {frame_index}")
        elif key == ord("]"):
            if current_event is None:
                print("[!] No event selected. Press an event key (r/g/b/f/v/e) first.")
                continue
            rec = records[current_event]
            rec["event_window_end"] = frame_index
            print(f"[-] {current_event} event_window_end = {frame_index}")

    cap.release()
    cv2.destroyAllWindows()
    export_annotations(video_id, records, fps=fps)


# ---------------------------------------------------------------------------
# JSON Export
# ---------------------------------------------------------------------------
def export_annotations(video_id, records, fps=None):
    """Write all event records and calculated phase records to annotations/reference/<video_id>.json."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ANNOTATIONS_DIR / f"{video_id}.json"

    # Base event records
    event_output = [records[name] for name in EVENT_NAMES]

    # Extract video-level metadata from logged records
    sample_rec = records.get("RUS") or next(iter(records.values()))
    video_sha256 = sample_rec["video_sha256"]
    athlete_id = sample_rec["athlete_id"]

    # Resolve fps if not provided
    if fps is None or fps <= 0:
        try:
            clips = load_manifest()
            clip = get_clip_by_id(clips, video_id)
            if clip and "frame_rate" in clip:
                fps = float(clip["frame_rate"])
        except Exception:
            pass
    if fps is None or fps <= 0:
        fps = 30.0

    # Calculate and append the 4 delivery phase records
    phase_output = []
    for p_def in PHASE_DEFS:
        start_rec = records.get(p_def["start_event"])
        end_rec = records.get(p_def["end_event"])
        phase_rec = make_phase_record(
            video_id=video_id,
            video_sha256=video_sha256,
            athlete_id=athlete_id,
            phase_def=p_def,
            start_rec=start_rec,
            end_rec=end_rec,
            fps=fps,
        )
        phase_output.append(phase_rec)

    output = event_output + phase_output

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(
        f"\n[+] Exported {len(output)} records ({len(event_output)} events, "
        f"{len(phase_output)} phases) to {json_path}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    clips = load_manifest()
    available_ids = [c["clip_id"] for c in clips]

    if len(sys.argv) > 1:
        video_id = sys.argv[1]
        clip = get_clip_by_id(clips, video_id)
        if clip is None:
            print(f"[!] Unknown video_id: {video_id}")
            print(f"    Available: {', '.join(available_ids)}")
            sys.exit(1)
        annotate_video(clip)
    else:
        print("ScoutEdge Manual Annotation GUI")
        print(f"Available clips: {', '.join(available_ids)}")
        print("Usage: python scripts/annotate_gui.py <video_id>\n")
        print("Annotating all clips sequentially...\n")
        for clip in clips:
            annotate_video(clip)
        print("\n[+] All clips annotated.")


if __name__ == "__main__":
    main()
