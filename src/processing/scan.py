"""
Stage 1: whole-video low-cost scan (Bible: "whole-video low-cost scan and native-frame refinement").

Pass A (every frame, cheap): camera motion + shot cuts at 320 px.
Pass A (sparse, scan.detect_fps): YOLOX person detection + motion-compensated tracking.
Pass B (scan.pose_fps): RTMPose on candidate tracks only, used to find the delivery.
The result is only used to *locate* the bowler and the delivery; events are measured later on
native frames in the dense window.
"""
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from src.ingest.decoder import SequentialReader, VideoMeta
from src.processing.motion import MotionEstimator
from src.processing.tracker import SparseTracker


@dataclass
class ScanResult:
    A: List[np.ndarray]
    cuts: List[bool]
    det_frames: List[int]
    dets: Dict[int, np.ndarray]
    tracks: Dict[int, dict]
    tracker: SparseTracker
    pose: Dict[int, Dict[int, tuple]] = field(default_factory=dict)  # tid -> frame -> (kp, sc)
    candidates: List[int] = field(default_factory=list)


def scan_video(meta: VideoMeta, detector, pose_model, cfg: dict) -> ScanResult:
    det_step = max(1, int(round(meta.fps / cfg["detect_fps"])))
    pose_step = max(1, int(round(meta.fps / cfg["pose_fps"])))
    mot = MotionEstimator(cfg.get("gmc_width", 320), cfg.get("cut_hist_corr", 0.5))
    trk = SparseTracker(cfg.get("track_iou", 0.25), cfg.get("track_high", 0.5), cfg.get("track_max_gap", 3))
    A, cuts, det_frames, dets = [], [], [], {}
    last_boxes, cut_since = [], False
    with SequentialReader(meta.path) as rd:
        for idx, fr in rd.frames():
            a, is_cut, _ = mot.step(fr, last_boxes)
            A.append(a)
            cuts.append(bool(is_cut) and idx > 0)
            cut_since = cut_since or (is_cut and idx > 0)
            if idx % det_step == 0:
                d = detector.detect(fr, conf=cfg.get("person_conf", 0.35))
                d = d[(d[:, 3] - d[:, 1]) >= cfg.get("min_box_frac", 0.06) * meta.height]
                dets[idx] = d
                det_frames.append(idx)
                trk.update(idx, d, A, cut_since)
                cut_since = False
                last_boxes = d.tolist()
    if not any(len(v) for v in dets.values()):
        return ScanResult(A, cuts, det_frames, dets, trk.tracks, trk)

    # candidate tracks: long and large enough, ranked by ground-relative motion (bowlers run)
    from src.processing.isolate import track_speed
    min_len = max(2, int(cfg.get("min_track_s", 0.6) * cfg["detect_fps"]))
    hmax = max((b[3] - b[1]) for t in trk.tracks.values() for b in t["boxes"])
    res = ScanResult(A, cuts, det_frames, dets, trk.tracks, trk)
    scored = []
    for tid, t in trk.tracks.items():
        if len(t["frames"]) < min_len:
            continue
        hs = np.array([b[3] - b[1] for b in t["boxes"]])
        if np.median(hs) < cfg.get("cand_min_rel_height", 0.25) * hmax:
            continue
        sp = list(track_speed(res, tid, meta.fps).values())
        motion = float(np.percentile(sp, 90)) if sp else 0.0
        scored.append((motion, tid))
    cands = [tid for _, tid in sorted(scored, reverse=True)[: cfg.get("max_candidates", 4)]]

    pose = {tid: {} for tid in cands}

    def run_pose(todo):
        if not todo:
            return
        with SequentialReader(meta.path) as rd:
            keys = sorted(todo)
            for idx, fr in rd.frames(start=keys[0], stop=keys[-1] + 1):
                for tid in todo.get(idx, []):
                    b = trk.box_at(tid, idx)
                    if b is not None and idx not in pose[tid]:
                        pose[tid][idx] = pose_model(fr, b)

    # tier 1: pose at every detection sample of each candidate
    tier1 = {}
    for tid in cands:
        for f in trk.tracks[tid]["frames"]:
            tier1.setdefault(f, []).append(tid)
    run_pose(tier1)
    # tier 2: denser pose only around samples where a wrist is at/above shoulder height
    from src.processing.isolate import arm_raise
    rad = int(round(cfg.get("tier2_radius_s", 0.4) * meta.fps))
    tier2 = {}
    for tid in cands:
        t = trk.tracks[tid]
        for f in list(pose[tid]):
            if arm_raise(*pose[tid][f]) > cfg.get("tier2_raise", 0.0):
                for g in range(max(t["frames"][0], f - rad), min(t["frames"][-1], f + rad) + 1, pose_step):
                    if g not in pose[tid]:
                        tier2.setdefault(g, []).append(tid)
    run_pose(tier2)
    return ScanResult(A, cuts, det_frames, dets, trk.tracks, trk, pose, cands)
