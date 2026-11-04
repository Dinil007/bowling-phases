"""
Stage 2: bowler isolation and delivery anchor.

Cue (view-agnostic, works on partial clips): the bowling delivery is the only moment in a clip where
a large, moving person swings a hand well above the shoulder (gather / arm swing / release).
For each candidate track we find "arm-overhead" pose samples and score clusters by
  score = overhead_hits + w_motion * pre_motion + w_size * relative_height
The FIRST high-scoring cluster is the delivery (Brief: process only the first delivery).
Ambiguity is raised only when two different (non-overlapping) people score within the margin.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.ingest.manifest import AmbiguousBowlerError, NoPersonDetectedError
from src.phase_detection.rtmpose import KP
from src.processing.motion import compose, warp_box
from src.processing.tracker import iou


@dataclass
class Anchor:
    track_id: int
    frame: int              # first overhead sample of the delivery cluster
    score: float
    method: str
    notes: str


def arm_raise(kp, sc, thr=0.3) -> float:
    """Max over both arms of (shoulder_y - wrist_y) / torso_length (image y grows downward)."""
    torso = np.linalg.norm(kp[KP["neck"]] - kp[KP["hip"]])
    if torso < 1 or sc[KP["neck"]] < thr or sc[KP["hip"]] < thr:
        return -9.0
    best = -9.0
    for s, w in (("left_shoulder", "left_wrist"), ("right_shoulder", "right_wrist")):
        if sc[KP[s]] >= thr and sc[KP[w]] >= thr:
            best = max(best, (kp[KP[s], 1] - kp[KP[w], 1]) / torso)
    return best


def track_speed(scan, tid, fps) -> dict:
    """Ground-relative speed of box bottom-centre (body heights / s) at each sparse frame."""
    t = scan.tracks[tid]
    out = {}
    for k in range(1, len(t["frames"])):
        f0, f1 = t["frames"][k - 1], t["frames"][k]
        if any(scan.cuts[f0 + 1: f1 + 1]):
            continue
        b0 = warp_box(compose(scan.A, f0, f1), t["boxes"][k - 1])
        b1 = np.array(t["boxes"][k])
        c0 = np.array([(b0[0] + b0[2]) / 2, b0[3]])
        c1 = np.array([(b1[0] + b1[2]) / 2, b1[3]])
        h = max(1.0, b1[3] - b1[1])
        grow = abs((b1[3] - b1[1]) - (b0[3] - b0[1])) / h      # looming (front-on / behind)
        out[f1] = (np.linalg.norm(c1 - c0) / h + 2.0 * grow) * fps / (f1 - f0)
    return out


def arm_swing(scan, tid, fps, max_gap):
    """Bowling-arm swing speed: change of (wrist - shoulder) between pose samples, torso-lengths / s."""
    fs = sorted(scan.pose[tid])
    out = {}
    for a, b in zip(fs[:-1], fs[1:]):
        if b - a > max_gap:
            continue
        k0, s0 = scan.pose[tid][a]
        k1, s1 = scan.pose[tid][b]
        tor = np.linalg.norm(k1[KP["neck"]] - k1[KP["hip"]])
        if tor < 1 or s1[KP["neck"]] < 0.3 or s1[KP["hip"]] < 0.3:
            continue
        best = 0.0
        for w, sh in (("left_wrist", "left_shoulder"), ("right_wrist", "right_shoulder")):
            w, sh = KP[w], KP[sh]
            if min(s0[w], s1[w], s0[sh], s1[sh]) > 0.3:
                d = np.linalg.norm((k1[w] - k1[sh]) - (k0[w] - k0[sh])) / tor / ((b - a) / fps)
                best = max(best, d)
        out[b] = best
    return out


def find_anchor(scan, fps: float, cfg: dict) -> Anchor:
    if not scan.tracks or not any(len(v) for v in scan.dets.values()):
        raise NoPersonDetectedError("NO_PERSON: no person detected anywhere in the clip")
    if not scan.candidates:
        raise NoPersonDetectedError("NO_PERSON: no person track long/large enough to be a bowler")
    raise_thr = cfg.get("overhead_raise", 0.25)
    swing_thr = cfg.get("swing_speed", 6.0)
    max_gap = int(round(fps / cfg.get("pose_fps", 10.0))) * 2 + 1
    hmax = max(np.median([b[3] - b[1] for b in scan.tracks[t]["boxes"]]) for t in scan.candidates)
    clusters = []
    for tid in scan.candidates:
        sp = track_speed(scan, tid, fps)
        sw = arm_swing(scan, tid, fps, max_gap)
        frames = sorted(scan.pose[tid])
        rz = {f: arm_raise(*scan.pose[tid][f]) for f in frames}
        hits = [f for f in frames if rz[f] > raise_thr or sw.get(f, 0) > swing_thr]
        if not hits:
            continue
        groups, cur = [], [hits[0]]
        for f in hits[1:]:
            if f - cur[-1] <= cfg.get("cluster_gap_s", 0.5) * fps:
                cur.append(f)
            else:
                groups.append(cur)
                cur = [f]
        groups.append(cur)
        rel_h = np.median([b[3] - b[1] for b in scan.tracks[tid]["boxes"]]) / hmax
        for g in groups:
            pre = [v for f, v in sp.items() if g[0] - 1.5 * fps <= f <= g[-1]]
            pre_motion = float(np.percentile(pre, 75)) if pre else 0.0
            max_sw = max(sw.get(f, 0.0) for f in g)
            max_rz = max(rz[f] for f in g)
            peak = max(g, key=lambda f: (sw.get(f, 0.0), rz[f]))
            score = (min(max_sw, 25.0) / 10.0 + max(0.0, min(max_rz, 1.5))
                     + cfg.get("w_motion", 1.0) * min(pre_motion, 3.0) + cfg.get("w_size", 1.0) * rel_h)
            clusters.append({"tid": tid, "frame": g[0], "peak": peak, "hits": len(g), "pre_motion": pre_motion,
                             "swing": max_sw, "raise": max_rz, "rel_h": rel_h, "score": score})
    if not clusters:
        raise AmbiguousBowlerError("AMBIGUOUS_BOWLER: no candidate shows a bowling arm action")
    best = max(clusters, key=lambda c: c["score"])
    good = sorted([c for c in clusters if c["score"] >= cfg.get("accept_frac", 0.75) * best["score"]],
                  key=lambda c: c["frame"])
    first = good[0]
    for c in good[1:]:
        # truly simultaneous actions only: a batter's stroke or a fielder's throw follows the release
        if c["tid"] == first["tid"] or abs(c["peak"] - first["peak"]) > cfg.get("ambiguity_window_s", 0.2) * fps:
            continue
        b1 = scan.tracker.box_at(first["tid"], first["peak"])
        b2 = scan.tracker.box_at(c["tid"], c["peak"])
        if b1 is not None and b2 is not None and iou(b1, b2) > 0.3:
            continue  # same person, split track
        if abs(c["score"] - first["score"]) < cfg.get("ambiguity_margin", 0.1) * max(c["score"], first["score"]):
            raise AmbiguousBowlerError(
                f"AMBIGUOUS_BOWLER: tracks {first['tid']} and {c['tid']} both show a delivery action "
                f"near frame {first['peak']} with similar scores ({first['score']:.2f} vs {c['score']:.2f})")
    later = [c for c in good if c["frame"] > first["frame"] + 1.5 * fps]
    note = (f"Delivery anchor at frame {first['peak']} (track {first['tid']}; arm swing {first['swing']:.1f} torso/s, "
            f"wrist raise {first['raise']:.2f} torso, pre-motion {first['pre_motion']:.2f} body-heights/s, "
            f"score {first['score']:.2f} vs best {best['score']:.2f}).")
    if later:
        note += f" {len(later)} later arm-action cluster(s) ignored (Brief: first delivery only)."
    return Anchor(first["tid"], first["peak"], float(first["score"]), "arm-swing-cluster-v2", note)
