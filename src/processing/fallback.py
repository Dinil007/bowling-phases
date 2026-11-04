"""
Bounded degraded-state handling for dense bowler tracking (Bible §3).

If no detection matches the bowler in a frame, the last box is propagated by camera motion plus
damped velocity for at most `max_degraded` frames ("degraded"); after that the bowler is "lost"
and all evidence in those frames is treated as not observable.
"""
import numpy as np
from src.processing.motion import apply
from src.processing.tracker import iou


class DenseBowlerTracker:
    def __init__(self, init_box, max_degraded: int = 10, iou_thresh: float = 0.2, dist_gate: float = 0.5):
        self.box = np.asarray(init_box, float)[:4]
        self.vel = np.zeros(4)
        self.degraded = 0
        self.max_degraded = max_degraded
        self.iou_thresh = iou_thresh
        self.dist_gate = dist_gate
        self.lost = False

    def predict(self, A):
        p = apply(A, [[self.box[0], self.box[1]], [self.box[2], self.box[3]]])
        return np.array([p[0, 0], p[0, 1], p[1, 0], p[1, 1]]) + 0.5 * self.vel

    def step(self, A, dets):
        """Returns (box, status) with status in {'tracked','degraded','lost'}."""
        if self.lost:
            return None, "lost"
        pred = self.predict(A)
        best, best_s = None, 0.0
        h = max(1.0, pred[3] - pred[1])
        for d in dets:
            s = iou(pred, d)
            if s < self.iou_thresh:
                dist = np.hypot((pred[0] + pred[2] - d[0] - d[2]) / 2, (pred[1] + pred[3] - d[1] - d[3]) / 2)
                dh = d[3] - d[1]
                s = 0.5 * (1 - dist / (self.dist_gate * h)) if (dist < self.dist_gate * h and 0.7 < dh / h < 1.43) else 0
            if s > best_s:
                best, best_s = d, s
        if best is not None:
            nb = np.asarray(best[:4], float)
            self.vel = 0.6 * self.vel + 0.4 * (nb - pred + 0.5 * self.vel)
            self.box, self.degraded = nb, 0
            return self.box.copy(), "tracked"
        self.degraded += 1
        if self.degraded > self.max_degraded:
            self.lost = True
            return None, "lost"
        self.box = pred
        self.vel *= 0.8
        return self.box.copy(), "degraded"
