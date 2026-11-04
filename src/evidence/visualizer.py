"""
Annotated verification video (Brief 4C): every frame shows frame number, timestamp, current phase,
event markers (exact frame and uncertainty window) and the visibility status of every event.
The bowler box and skeleton are drawn inside the dense analysis window.
"""
import os
import cv2
import numpy as np
from src.ingest.decoder import SequentialReader
from src.phase_detection.rtmpose import SKELETON

COL = {"visible": (80, 200, 80), "partially_visible": (0, 200, 255), "not_observable": (90, 90, 255),
       "occluded": (90, 90, 255), "out_of_frame": (90, 90, 255)}


def phase_at(f, recs):
    for r in recs:
        if r.record_type != "phase" or r.start_frame is None:
            continue
        if r.name == "Ball release" and f == r.start_frame:
            return r.name
        if r.name != "Ball release" and r.start_frame <= f < r.end_frame:
            return r.name
    return "-"


def draw_overlay(img, f, meta, recs, scale, W=None):
    events = [r for r in recs if r.record_type == "event"]
    h, w = img.shape[:2]
    if W is not None and f in W.boxes and W.boxes[f] is not None:
        b = (W.boxes[f] * scale).astype(int)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 200, 0) if W.status[f] == "tracked" else (0, 140, 255), 2)
        if f in W.kp:
            k, s = W.kp[f] * scale, W.sc[f]
            for a, c in SKELETON:
                if s[a] > 0.3 and s[c] > 0.3:
                    cv2.line(img, tuple(k[a].astype(int)), tuple(k[c].astype(int)), (0, 255, 255), 1, cv2.LINE_AA)
    band = 22 * (3 + len(events))
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (min(w, 430), band), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(ov, 0.6, img, 0.4, 0)
    put = lambda t, y, c=(255, 255, 255), s=0.5: cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)
    put(f"{meta_id(recs)}  frame {f}  t={meta.ts(f):.3f}s", 18)
    put(f"phase: {phase_at(f, recs)}", 40, (255, 230, 120))
    y = 62
    for r in events:
        tag = f"{r.name}: {r.visibility_status}"
        tag += f" @ {r.event_frame}" if r.event_frame is not None else ""
        tag += f" win[{r.event_window_start},{r.event_window_end}]" if r.event_window_start is not None else ""
        here = r.event_frame == f
        inwin = r.event_window_start is not None and r.event_window_start <= f <= r.event_window_end
        put(("> " if here else "  ") + tag, y, COL.get(r.visibility_status, (255, 255, 255)) if not here else (255, 255, 255), 0.45)
        y += 22
        if here:
            cv2.putText(img, r.name, (w // 2 - 40, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
        elif inwin:
            cv2.putText(img, f"({r.name} window)", (w // 2 - 80, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
    return img


def meta_id(recs):
    return recs[0].video_id if recs else ""


def render_video(meta, recs, W, out_path, max_width=960, pad_s=0.5):
    frames = [r.event_frame for r in recs if r.record_type == "event" and r.event_frame is not None]
    frames += [v for r in recs if r.record_type == "event" for v in (r.event_window_start, r.event_window_end) if v is not None]
    frames += [W.start, W.end] if W is not None else []
    s0 = max(0, min(frames) - int(pad_s * meta.fps)) if frames else 0
    s1 = min(meta.frame_count - 1, max(frames) + int(pad_s * meta.fps)) if frames else meta.frame_count - 1
    scale = min(1.0, max_width / float(meta.width))
    size = (int(meta.width * scale) // 2 * 2, int(meta.height * scale) // 2 * 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, size)
    with SequentialReader(meta.path) as rd:
        for f, fr in rd.frames(start=s0, stop=s1 + 1):
            img = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
            vw.write(draw_overlay(img, f, meta, recs, size[0] / meta.width, W))
    vw.release()
    return out_path, (s0, s1)
