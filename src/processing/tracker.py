"""
Lightweight tracking-by-detection with camera-motion compensation.

Association: previous box warped by camera motion (GMC) plus constant-velocity prediction, then
greedy IoU matching with a two-stage high/low score scheme (ByteTrack idea). Tracks never cross a
shot cut. Built here; the ideas come from ByteTrack (MIT) and BoT-SORT (MIT), no code reused.
"""
from typing import Dict, List
import numpy as np
from src.processing.motion import compose, warp_box


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class SparseTracker:
    def __init__(self, iou_thresh: float = 0.25, high: float = 0.5, max_gap_samples: int = 3,
                 dist_gate: float = 0.6):
        self.iou_thresh = iou_thresh
        self.dist_gate = dist_gate
        self.high = high
        self.max_gap = max_gap_samples
        self.tracks: Dict[int, dict] = {}
        self._next = 0

    def _new(self, f, box):
        self.tracks[self._next] = {"id": self._next, "frames": [f], "boxes": [box[:4].tolist()],
                                   "scores": [float(box[4])], "alive": True, "miss": 0}
        self._next += 1

    def update(self, f: int, dets: np.ndarray, A_list: List[np.ndarray], cut_since_last: bool):
        if cut_since_last:
            for t in self.tracks.values():
                t["alive"] = False
        alive = [t for t in self.tracks.values() if t["alive"]]
        preds = []
        for t in alive:
            lf, lb = t["frames"][-1], np.array(t["boxes"][-1])
            pb = warp_box(compose(A_list, lf, f), lb)
            if len(t["frames"]) >= 2:  # constant velocity in compensated coords (damped)
                pf, pbx = t["frames"][-2], np.array(t["boxes"][-2])
                prev_in_last = warp_box(compose(A_list, pf, lf), pbx)
                v = (lb - prev_in_last) / max(1, lf - pf)
                pb = pb + 0.5 * v * (f - lf)
            preds.append(pb)
        unmatched_t = list(range(len(alive)))
        unmatched_d = list(range(len(dets)))
        for stage in ("high", "low"):
            pairs = []
            for di in unmatched_d:
                if (dets[di, 4] >= self.high) != (stage == "high"):
                    continue
                for ti in unmatched_t:
                    s = iou(preds[ti], dets[di])
                    if s >= self.iou_thresh:
                        pairs.append((1.0 + s, ti, di))
                    else:  # fast movers at low sample rate: centre-distance gate, similar size
                        pb, db = preds[ti], dets[di]
                        ph, dh = pb[3] - pb[1], db[3] - db[1]
                        dist = np.hypot((pb[0] + pb[2] - db[0] - db[2]) / 2, (pb[1] + pb[3] - db[1] - db[3]) / 2)
                        if ph > 0 and 0.7 < dh / ph < 1.43 and dist < self.dist_gate * ph:
                            pairs.append((1.0 - dist / (self.dist_gate * ph), ti, di))
            for s, ti, di in sorted(pairs, reverse=True):
                if ti in unmatched_t and di in unmatched_d:
                    t = alive[ti]
                    t["frames"].append(f)
                    t["boxes"].append(dets[di, :4].tolist())
                    t["scores"].append(float(dets[di, 4]))
                    t["miss"] = 0
                    unmatched_t.remove(ti)
                    unmatched_d.remove(di)
        for ti in unmatched_t:
            alive[ti]["miss"] += 1
            if alive[ti]["miss"] > self.max_gap:
                alive[ti]["alive"] = False
        for di in unmatched_d:
            if dets[di, 4] >= self.high:
                self._new(f, dets[di])

    def box_at(self, tid: int, f: int):
        """Linear interpolation of a track's box at frame f (None outside its span)."""
        t = self.tracks[tid]
        fr = t["frames"]
        if f < fr[0] or f > fr[-1]:
            return None
        j = int(np.searchsorted(fr, f))
        if fr[j] == f:
            return np.array(t["boxes"][j])
        a, b = fr[j - 1], fr[j]
        w = (f - a) / (b - a)
        return (1 - w) * np.array(t["boxes"][j - 1]) + w * np.array(t["boxes"][j])
