"""
Video probing and deterministic sequential frame access.

- True metadata: frame rate from the container (ffprobe r_frame_rate when available, else OpenCV),
  frame count by counting decoded frames (header counts can be wrong), resolution, duration.
- Variable frame rate is flagged by inspecting packet PTS deltas (ffprobe) or decoder timestamps.
- Timestamps follow the Brief: t = frame_index / true_fps. If VFR is detected, the PTS of each
  frame is used instead and the clip is flagged in the run log and record notes.
- Original files are opened read-only and never modified.
"""
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from src.ingest.manifest import UnreadableFileError


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    is_vfr: bool
    pts: Optional[List[float]] = None
    probe_source: str = "opencv"
    notes: List[str] = field(default_factory=list)

    def ts(self, idx: int) -> float:
        """Brief 4B/5: timestamps are derived from the frame index and the true frame rate."""
        return round(idx / self.fps, 4)

    def pts_ts(self, idx: int):
        """Container presentation time (only for diagnostics on VFR clips)."""
        if self.pts and 0 <= idx < len(self.pts):
            return round(self.pts[idx] - self.pts[0], 4)
        return None


def _ffprobe(path: str) -> Optional[dict]:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate,avg_frame_rate,width,height:packet=pts_time",
             "-of", "json", path], capture_output=True, text=True, timeout=120)
        return json.loads(out.stdout) if out.returncode == 0 else None
    except Exception:
        return None


def _rate(s: str) -> Optional[float]:
    try:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else None
    except Exception:
        return None


def probe(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise UnreadableFileError(f"OpenCV cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cv = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    # Exact frame count by decoding (grab is cheap; no pixel conversion).
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    if n == 0 or w == 0 or h == 0:
        raise UnreadableFileError(f"Video has no decodable frames: {path}")

    meta = VideoMeta(path=path, width=w, height=h, fps=fps_cv, frame_count=n,
                     duration_s=0.0, is_vfr=False)
    pr = _ffprobe(path)
    if pr and pr.get("streams"):
        st = pr["streams"][0]
        r = _rate(st.get("r_frame_rate", ""))
        if r:
            meta.fps = r
        pts = sorted(float(p["pts_time"]) for p in pr.get("packets", []) if p.get("pts_time") not in (None, "N/A"))
        meta.probe_source = "ffprobe"
        if len(pts) >= 3:
            d = np.diff(pts)
            nominal = 1.0 / meta.fps if meta.fps else np.median(d)
            dev = np.abs(d - nominal)
            if np.max(dev) > 0.25 * nominal:
                meta.is_vfr = True
                bad = int((dev > 0.25 * nominal).sum())
                meta.notes.append(
                    f"VFR flag: {bad} of {len(d)} frame intervals deviate >25% from 1/{meta.fps:g} s "
                    f"(max {np.max(d) * 1000:.1f} ms). Timestamps are index/fps as the Brief specifies; "
                    f"near those intervals true capture time can differ by up to {np.max(dev) * 1000:.0f} ms.")
            meta.pts = pts if len(pts) == n else None
    if not meta.fps or meta.fps <= 0:
        raise UnreadableFileError(f"Cannot determine frame rate: {path}")
    meta.duration_s = round(n / meta.fps, 4)
    return meta


class SequentialReader:
    """Reads frames strictly in order (frame-accurate; avoids codec seek inaccuracy)."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise UnreadableFileError(f"OpenCV cannot open video: {path}")
        self.idx = -1

    def frames(self, start: int = 0, stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        while self.idx + 1 < start:
            if not self.cap.grab():
                return
            self.idx += 1
        while stop is None or self.idx + 1 < stop:
            ok, frame = self.cap.read()
            if not ok:
                return
            self.idx += 1
            yield self.idx, frame

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_frames(path: str, indices: List[int]) -> dict:
    """Deterministically fetch a sorted set of frame indices by sequential decoding."""
    want = sorted(set(i for i in indices if i >= 0))
    out = {}
    if not want:
        return out
    with SequentialReader(path) as r:
        it = iter(want)
        nxt = next(it)
        for idx, fr in r.frames(start=want[0], stop=want[-1] + 1):
            if idx == nxt:
                out[idx] = fr
                nxt = next(it, None)
                if nxt is None:
                    break
    return out
