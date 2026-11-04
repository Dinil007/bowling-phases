"""
Scale-normalised, camera-compensated kinematic signals from the dense window.

All speeds are ground-relative (camera motion removed with the per-frame GMC transform) and in
body-heights per second, so they are independent of resolution, zoom and distance to camera.
"""
import numpy as np
from src.phase_detection.rtmpose import KP
from src.processing.motion import apply

FOOT = {"left": [KP["left_ankle"], KP["left_heel"], KP["left_big_toe"], KP["left_small_toe"]],
        "right": [KP["right_ankle"], KP["right_heel"], KP["right_big_toe"], KP["right_small_toe"]]}
ARM = {"left": (KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"]),
       "right": (KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"])}


def point_track(W, idxs, min_score=0.3):
    """Score-weighted mean of keypoints `idxs` per frame -> (pts[N,2] with NaN, score[N])."""
    pts = np.full((len(W.frames), 2), np.nan)
    scs = np.zeros(len(W.frames))
    for i, f in enumerate(W.frames):
        if f not in W.kp:
            continue
        s = W.sc[f][idxs]
        k = W.kp[f][idxs]
        good = s >= min_score
        scs[i] = float(s.mean())
        if good.any():
            pts[i] = (k[good] * s[good, None]).sum(0) / s[good].sum()
    return pts, scs


def body_height(W) -> float:
    hs = [b[3] - b[1] for f, b in W.boxes.items() if b is not None and W.status.get(f) == "tracked"]
    return float(np.median(hs)) if hs else 1.0


def ground_speed(W, pts, H, centered: bool = True) -> np.ndarray:
    """Ground-relative speed in body-heights per second (NaN where undefined).

    centered=True uses the zero-lag central difference (p[t+1] - T p[t-1]) / 2, where T maps
    frame t-1 coordinates into frame t+1 (camera motion removed); falls back to one-sided ends.
    """
    n = len(W.frames)
    v = np.full(n, np.nan)
    for i in range(n):
        f = W.frames[i]
        if centered and 0 < i < n - 1 and not (np.isnan(pts[i - 1]).any() or np.isnan(pts[i + 1]).any()):
            T = W.A[f + 1] @ W.A[f]
            v[i] = np.linalg.norm(pts[i + 1] - apply(T, pts[i - 1:i])[0]) / 2 / H * W.fps
        elif i > 0 and not (np.isnan(pts[i]).any() or np.isnan(pts[i - 1]).any()):
            v[i] = np.linalg.norm(pts[i] - apply(W.A[f], pts[i - 1:i])[0]) / H * W.fps
    return v


def nanmedfilt(x, k=3):
    out = x.copy()
    r = k // 2
    for i in range(len(x)):
        seg = x[max(0, i - r): i + r + 1]
        seg = seg[~np.isnan(seg)]
        out[i] = np.median(seg) if len(seg) else np.nan
    return out


def runs(mask):
    """Start/end (inclusive) indices of True runs."""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out
