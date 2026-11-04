#!/usr/bin/env python3
"""
Download model weights with SHA-256 verification (weights are never committed).

    python scripts/fetch_weights.py

Each model lists one or more sources; the first one whose file matches the pinned SHA-256 wins.
Both models are Apache-2.0. See reports/technical_design.md for provenance and licences.
"""
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(REPO, "weights")

MODELS = [
    {
        "name": "yolox_s.onnx",  # YOLOX-s COCO detector, Megvii (Apache-2.0), official release asset
        "sha256": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        "sources": [("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx", None)],
    },
    {
        "name": "rtmpose-m_halpe26_256x192.onnx",  # RTMPose-m body7 Halpe-26, OpenMMLab (Apache-2.0)
        "sha256": "26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf",
        "sources": [
            # GitHub mirror of the OpenMMLab mmdeploy export (ossia/score-packages, verified SHA above)
            ("https://github.com/ossia/score-packages/releases/download/v1.0.0/end2end.onnx.zip", "end2end.onnx"),
            # official OpenMMLab model zoo zip (accepted only if the contained ONNX has the same SHA)
            ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
             "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip", "end2end.onnx"),
        ],
    },
]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "scoutedge-eval-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> int:
    os.makedirs(WEIGHTS, exist_ok=True)
    ok = True
    for m in MODELS:
        dst = os.path.join(WEIGHTS, m["name"])
        if os.path.isfile(dst) and sha256(open(dst, "rb").read()) == m["sha256"]:
            print(f"[ok] {m['name']} already present (sha256 verified)")
            continue
        done = False
        for url, member in m["sources"]:
            try:
                print(f"[..] {m['name']} <- {url}")
                data = fetch(url)
                if member:
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        name = next(n for n in z.namelist() if n.endswith(member))
                        data = z.read(name)
                got = sha256(data)
                if got != m["sha256"]:
                    print(f"[!!] checksum mismatch from {url}: {got}")
                    continue
                with open(dst + ".part", "wb") as f:
                    f.write(data)
                os.replace(dst + ".part", dst)
                print(f"[ok] {m['name']} saved and verified")
                done = True
                break
            except Exception as e:
                print(f"[!!] {url}: {e}")
        ok &= done
        if not done:
            print(f"[FAIL] could not obtain {m['name']} with the pinned checksum", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
