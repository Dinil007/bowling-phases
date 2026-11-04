"""
Camera ego-motion (global motion compensation) and shot-cut detection.

GMC follows the BoT-SORT idea: sparse LK feature tracks on background pixels (person boxes masked),
robust similarity transform via RANSAC. A[t] maps pixel coords in frame t-1 to frame t (native px).
Ground-relative motion of a body point is then |p_t - A[t] p_{t-1}|, which removes camera pan/shake.
"""
from typing import List, Optional
import cv2
import numpy as np

IDENTITY = np.eye(3, dtype=np.float64)


class MotionEstimator:
    def __init__(self, work_width: int = 320, cut_hist_corr: float = 0.5):
        self.work_width = work_width
        self.cut_corr = cut_hist_corr
        self.prev_gray = None
        self.prev_hist = None
        self.scale = 1.0

    def _hist(self, small_bgr):
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()

    def step(self, frame: np.ndarray, mask_boxes: Optional[List] = None):
        """Returns (A 3x3 prev->curr in native px, is_cut, inlier_count)."""
        H, W = frame.shape[:2]
        self.scale = self.work_width / float(W)
        small = cv2.resize(frame, (self.work_width, max(8, int(round(H * self.scale)))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = self._hist(small)
        A, is_cut, inl = IDENTITY.copy(), False, 0
        if self.prev_gray is not None:
            corr = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
            is_cut = corr < self.cut_corr
            if not is_cut:
                mask = np.full(gray.shape, 255, np.uint8)
                for b in (mask_boxes or []):
                    x1, y1, x2, y2 = [int(v * self.scale) for v in b[:4]]
                    pad = int(0.15 * (y2 - y1))
                    cv2.rectangle(mask, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 0, -1)
                p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=300, qualityLevel=0.01,
                                             minDistance=7, mask=mask)
                if p0 is not None and len(p0) >= 12:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None,
                                                        winSize=(21, 21), maxLevel=3)
                    g0, g1 = p0[st == 1], p1[st == 1]
                    if len(g0) >= 10:
                        M, inliers = cv2.estimateAffinePartial2D(g0, g1, method=cv2.RANSAC,
                                                                 ransacReprojThreshold=1.5)
                        if M is not None:
                            inl = int(inliers.sum()) if inliers is not None else 0
                            M = M.astype(np.float64)
                            M[:, 2] /= self.scale
                            A = np.vstack([M, [0, 0, 1]])
        self.prev_gray, self.prev_hist = gray, hist
        return A, is_cut, inl


def compose(A_list: List[np.ndarray], i: int, j: int) -> np.ndarray:
    """Transform mapping frame i coords to frame j coords (j >= i), using per-frame A[t] (t-1 -> t)."""
    T = IDENTITY.copy()
    for t in range(i + 1, j + 1):
        T = A_list[t] @ T
    return T


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ T.T
    return h[:, :2]


def warp_box(T: np.ndarray, box) -> np.ndarray:
    p = apply(T, [[box[0], box[1]], [box[2], box[3]]])
    return np.array([p[0, 0], p[0, 1], p[1, 0], p[1, 1]])


def shot_id_per_frame(cuts: List[bool]) -> np.ndarray:
    return np.cumsum(np.asarray(cuts, int))
