"""
One contact sheet per event (Brief 4C): frames -3..+3 around the prediction.
For a not-observable event the sheet is centred on the middle of its uncertainty window and is
labelled as such, so a reviewer can still inspect where the system looked.
Tiles are crops around the bowler (when tracked) so feet/hand are inspectable.
"""
import os
import cv2
import numpy as np
from src.ingest.decoder import read_frames


def make_sheets(meta, recs, W, out_dir, radius=3, tile_h=300):
    os.makedirs(out_dir, exist_ok=True)
    events = [r for r in recs if r.record_type == "event"]
    centres = {}
    for r in events:
        if r.event_frame is not None:
            centres[r.name] = (r.event_frame, "prediction")
        elif r.event_window_start is not None:
            centres[r.name] = ((r.event_window_start + r.event_window_end) // 2, "window centre (NOT OBSERVABLE)")
    need = [f for c, _ in centres.values() for f in range(c - radius, c + radius + 1) if 0 <= f < meta.frame_count]
    have = {f: W.thumbs[f] for f in need if W is not None and f in W.thumbs}
    scale_have = W.thumb_scale if W is not None else 1.0
    missing = [f for f in need if f not in have]
    raw = read_frames(meta.path, missing) if missing else {}
    paths = []
    for r in events:
        p = os.path.join(out_dir, f"{r.video_id}_{r.name}_contact_sheet.png")
        if r.name not in centres:
            img = np.zeros((tile_h + 60, 900, 3), np.uint8)
            cv2.putText(img, f"{r.video_id} {r.name}: {r.visibility_status} - no frame or window to show", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imwrite(p, img)
            paths.append(p)
            continue
        c, kind = centres[r.name]
        # crop region: bowler box at the centre frame, expanded
        crop = None
        if W is not None and W.boxes.get(c) is not None:
            b = W.boxes[c]
            cx, cy, bh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[3] - b[1])
            half = max(bh * 0.9, 60)
            crop = [cx - half * 0.8, cy - half, cx + half * 0.8, cy + half]
        tiles = []
        for f in range(c - radius, c + radius + 1):
            if f in have:
                img, s = have[f], scale_have
            elif f in raw:
                img, s = raw[f], 1.0
            else:
                tiles.append(np.zeros((tile_h, int(tile_h * 0.8), 3), np.uint8))
                continue
            if crop is not None:
                x1, y1, x2, y2 = [int(v * s) for v in crop]
                H, Wd = img.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(Wd, x2), min(H, y2)
                img = img[y1:y2, x1:x2] if x2 > x1 + 10 and y2 > y1 + 10 else img
            t = cv2.resize(img, (int(tile_h * img.shape[1] / img.shape[0]), tile_h))
            t = cv2.copyMakeBorder(t, 28, 4, 2, 2, cv2.BORDER_CONSTANT,
                                   value=(0, 220, 255) if f == c else (40, 40, 40))
            cv2.putText(t, f"{f - c:+d}  f{f}  {meta.ts(f):.3f}s", (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0) if f == c else (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        hmax = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        strip = np.hstack(tiles)
        head = np.zeros((56, strip.shape[1], 3), np.uint8)
        title = f"{r.video_id}  {r.name}  [{r.visibility_status}]  centre f{c} = {kind}"
        if r.event_window_start is not None:
            title += f"  window [{r.event_window_start},{r.event_window_end}]"
        cv2.putText(head, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(head, f"{r.method_or_model} | {r.evidence_source}"[:140], (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imwrite(p, np.vstack([head, strip]))
        paths.append(p)
    return paths
