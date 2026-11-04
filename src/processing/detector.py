"""
Person detector: YOLOX-s (Megvii, Apache-2.0) via onnxruntime.

Reused: YOLOX ONNX weights and the standard YOLOX grid decode (Megvii demo_postprocess).
Built here: letterbox pre-processing, decode, class filtering and NMS wrapper.
"""
from typing import List, Tuple
import cv2
import numpy as np
import onnxruntime as ort

COCO_PERSON = 0
COCO_SPORTS_BALL = 32


def make_session(path: str, threads: int = 2) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


class YOLOXDetector:
    def __init__(self, model_path: str, input_size: int = 640, threads: int = 2):
        self.sess = make_session(model_path, threads)
        self.inp = self.sess.get_inputs()[0].name
        self.size = input_size
        grids, strides = [], []
        for s in (8, 16, 32):
            n = input_size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids, 0).astype(np.float32)
        self.strides = np.concatenate(strides, 0).astype(np.float32)

    def _raw(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        resized = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32)  # BGR, 0-255 (YOLOX legacy=False)
        out = self.sess.run(None, {self.inp: x})[0][0]
        out[:, :2] = (out[:, :2] + self.grids) * self.strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * self.strides
        return out, r

    def detect(self, img: np.ndarray, cls_id: int = COCO_PERSON, conf: float = 0.35,
               nms: float = 0.45, extra_cls: int = None, extra_conf: float = 0.3):
        """Returns boxes [N,5] (x1,y1,x2,y2,score) for cls_id, plus optional boxes for extra_cls."""
        out, r = self._raw(img)

        def pick(cid, thr):
            scores = out[:, 4] * out[:, 5 + cid]
            keep = scores > thr
            if not keep.any():
                return np.zeros((0, 5), np.float32)
            b = out[keep, :4]
            s = scores[keep]
            xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                             b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / r
            idx = cv2.dnn.NMSBoxes(
                [[float(a[0]), float(a[1]), float(a[2] - a[0]), float(a[3] - a[1])] for a in xyxy],
                [float(v) for v in s], thr, nms)
            idx = np.array(idx).reshape(-1)
            res = np.concatenate([xyxy[idx], s[idx, None]], 1).astype(np.float32)
            H, W = img.shape[:2]
            res[:, [0, 2]] = res[:, [0, 2]].clip(0, W - 1)
            res[:, [1, 3]] = res[:, [1, 3]].clip(0, H - 1)
            return res[np.argsort(-res[:, 4])]

        persons = pick(cls_id, conf)
        if extra_cls is None:
            return persons
        return persons, pick(extra_cls, extra_conf)
