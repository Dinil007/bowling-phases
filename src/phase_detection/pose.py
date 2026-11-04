"""
Stage 3: dense native-frame refinement around the delivery anchor.

Every frame in [anchor - pre_s, anchor + post_s] (clipped to the anchor's shot) is decoded in order,
the bowler is tracked forward and backward from the anchor, and on each frame we store:
bowler box, RTMPose Halpe-26 keypoints + scores, camera motion, sports-ball detections and a
downscaled copy of the frame for the visual evidence.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import cv2
import numpy as np

from src.ingest.decoder import SequentialReader, VideoMeta
from src.processing.detector import COCO_PERSON, COCO_SPORTS_BALL
from src.processing.fallback import DenseBowlerTracker
from src.processing.motion import MotionEstimator, compose, warp_box, IDENTITY


@dataclass
class DenseWindow:
    start: int
    end: int                      # inclusive
    anchor: int
    fps: float
    frames: List[int]
    boxes: Dict[int, np.ndarray] = field(default_factory=dict)
    status: Dict[int, str] = field(default_factory=dict)
    kp: Dict[int, np.ndarray] = field(default_factory=dict)
    sc: Dict[int, np.ndarray] = field(default_factory=dict)
    A: Dict[int, np.ndarray] = field(default_factory=dict)        # f-1 -> f
    balls: Dict[int, np.ndarray] = field(default_factory=dict)
    thumbs: Dict[int, np.ndarray] = field(default_factory=dict)
    thumb_scale: float = 1.0
    img_hw: tuple = (0, 0)


def shot_bounds(cuts: List[bool], f: int):
    s = f
    while s > 0 and not cuts[s]:
        s -= 1
    e = f
    while e + 1 < len(cuts) and not cuts[e + 1]:
        e += 1
    return s, e


def dense_refine(meta: VideoMeta, scan, anchor, detector, pose_model, cfg: dict) -> DenseWindow:
    fps = meta.fps
    s0, e0 = shot_bounds(scan.cuts, anchor.frame)
    start = max(s0, anchor.frame - int(round(cfg["pre_s"] * fps)))
    end = min(e0, anchor.frame + int(round(cfg["post_s"] * fps)), meta.frame_count - 1)
    W = DenseWindow(start, end, anchor.frame, fps, list(range(start, end + 1)),
                    img_hw=(meta.height, meta.width))
    work_w = cfg.get("work_max_width", 1280)
    sc_work = min(1.0, work_w / float(meta.width))
    thumb_w = cfg.get("thumb_max_width", 960)
    W.thumb_scale = min(1.0, thumb_w / float(meta.width))

    # 1) decode window once; keep working-resolution copies in memory
    work, mot = {}, MotionEstimator(cfg.get("gmc_width", 320), 0.0)  # no cut test inside a shot
    trk_box = scan.tracker.box_at(anchor.track_id, anchor.frame)
    with SequentialReader(meta.path) as rd:
        for idx, fr in rd.frames(start=start, stop=end + 1):
            mask = [trk_box] if trk_box is not None else []
            a, _, _ = mot.step(fr, mask)
            W.A[idx] = a if idx > start else IDENTITY.copy()
            work[idx] = cv2.resize(fr, None, fx=sc_work, fy=sc_work, interpolation=cv2.INTER_AREA) if sc_work < 1 else fr
            W.thumbs[idx] = cv2.resize(fr, None, fx=W.thumb_scale, fy=W.thumb_scale, interpolation=cv2.INTER_AREA) \
                if W.thumb_scale < 1 else fr.copy()
    frames = sorted(work)
    if not frames:
        return W
    W.frames = frames
    W.end = frames[-1]

    # 2) detections (person + sports ball) on every window frame
    dets = {}
    for f in frames:
        p, b = detector.detect(work[f], cls_id=COCO_PERSON, conf=cfg.get("person_conf", 0.3),
                               extra_cls=COCO_SPORTS_BALL, extra_conf=cfg.get("ball_conf", 0.25))
        p[:, :4] /= sc_work
        b[:, :4] /= sc_work
        dets[f], W.balls[f] = p, b

    # 3) track bowler forward and backward from the anchor
    a0 = min(max(anchor.frame, frames[0]), frames[-1])
    init = scan.tracker.box_at(anchor.track_id, a0)
    if init is None:
        init = np.array(scan.tracks[anchor.track_id]["boxes"][0])
    md = cfg.get("max_degraded_frames", 10)
    fwd = DenseBowlerTracker(init, md)
    for f in [x for x in frames if x >= a0]:
        A = W.A[f] if f > a0 else IDENTITY
        box, st = fwd.step(A, dets[f])
        W.boxes[f], W.status[f] = box, st
    bwd = DenseBowlerTracker(W.boxes.get(a0, init), md)
    for f in [x for x in reversed(frames) if x < a0]:
        A_inv = np.linalg.inv(W.A[f + 1])  # map frame f+1 -> f
        box, st = bwd.step(A_inv, dets[f])
        W.boxes[f], W.status[f] = box, st

    # 4) pose on every tracked frame (native-pixel keypoints)
    for f in frames:
        b = W.boxes.get(f)
        if b is None:
            continue
        kp, s = pose_model(work[f], b * sc_work)
        W.kp[f], W.sc[f] = kp / sc_work, s
    return W
