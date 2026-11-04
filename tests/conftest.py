import os
import sys

import cv2
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def write_video(path, n=30, fps=30.0, size=(320, 240), color=0):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        vw.write(np.full((size[1], size[0], 3), color, np.uint8))
    vw.release()
    return path


@pytest.fixture
def blank_clip(tmp_path):
    from src.ingest.manifest import compute_sha256
    p = write_video(str(tmp_path / "blank.mp4"))
    return p, compute_sha256(p)


def manifest(tmp_path, rows):
    p = tmp_path / "manifest.csv"
    lines = ["clip_id,file_name,sha256,frame_rate,resolution,duration,camera_view,athlete_id"]
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    p.write_text("\n".join(lines) + "\n")
    return str(p)


WEIGHTS = all(os.path.isfile(os.path.join(REPO, "weights", f)) for f in
              ("yolox_s.onnx", "rtmpose-m_halpe26_256x192.onnx"))
