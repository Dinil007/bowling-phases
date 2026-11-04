"""
Top-down 2D pose: RTMPose-m, Halpe-26 keypoints (OpenMMLab, Apache-2.0) via onnxruntime.

Reused: RTMPose ONNX weights (body7 training, 256x192), SimCC decoding scheme from MMPose.
Built here: affine crop, normalisation, decode to native-frame pixels.
Halpe-26 adds heels and toes, which are the direct evidence for foot-ground contact.
"""
from typing import Tuple
import cv2
import numpy as np
from src.processing.detector import make_session

HALPE26 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
           "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
           "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
           "right_knee", "left_ankle", "right_ankle", "head", "neck", "hip",
           "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
           "left_heel", "right_heel"]
KP = {n: i for i, n in enumerate(HALPE26)}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (15, 24), (15, 20), (16, 25), (16, 21),
            (17, 18), (18, 19)]

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


class RTMPose:
    def __init__(self, model_path: str, input_wh: Tuple[int, int] = (192, 256),
                 bbox_scale: float = 1.25, threads: int = 2, simcc_split: float = 2.0):
        self.sess = make_session(model_path, threads)
        self.inp = self.sess.get_inputs()[0].name
        self.w, self.h = input_wh
        self.bbox_scale = bbox_scale
        self.split = simcc_split

    def _affine(self, box):
        x1, y1, x2, y2 = box[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * self.bbox_scale, (y2 - y1) * self.bbox_scale
        aspect = self.w / self.h
        if bw > bh * aspect:
            bh = bw / aspect
        else:
            bw = bh * aspect
        src = np.float32([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2], [cx - bw / 2, cy + bh / 2]])
        dst = np.float32([[0, 0], [self.w, 0], [0, self.h]])
        return cv2.getAffineTransform(src, dst), (bw, bh)

    def __call__(self, img: np.ndarray, box) -> Tuple[np.ndarray, np.ndarray]:
        """Returns keypoints [26,2] in native pixels and scores [26] (SimCC peak, clipped to 0..1)."""
        M, _ = self._affine(box)
        crop = cv2.warpAffine(img, M, (self.w, self.h), flags=cv2.INTER_LINEAR)
        x = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) - MEAN) / STD
        sx, sy = self.sess.run(None, {self.inp: x.transpose(2, 0, 1)[None]})
        sx, sy = sx[0], sy[0]
        px = sx.argmax(1) / self.split
        py = sy.argmax(1) / self.split
        score = np.minimum(sx.max(1), sy.max(1)).clip(0, 1)
        Minv = cv2.invertAffineTransform(M)
        pts = np.stack([px, py, np.ones_like(px)], 1) @ Minv.T
        return pts.astype(np.float32), score.astype(np.float32)
