"""Brief 4A explicit failures: missing file, unreadable file, checksum mismatch, no person,
unresolvable multiple bowlers. Each must give a clear message and a non-zero exit code."""
import json
import os

import numpy as np
import pytest

from src.ingest.manifest import (AmbiguousBowlerError, ChecksumMismatchError, MissingFileError,
                                 NoPersonDetectedError, UnreadableFileError, read_manifest, verify_entry)
from tests.conftest import WEIGHTS, manifest


def test_missing_file(tmp_path):
    m = manifest(tmp_path, [("c1", "nope.mp4", "0" * 64, 30, "1x1", 1, "other", "ATH-1")])
    with pytest.raises(MissingFileError) as e:
        verify_entry(read_manifest(m)[0])
    assert e.value.code == 10


def test_checksum_mismatch(tmp_path, blank_clip):
    p, sha = blank_clip
    m = manifest(tmp_path, [("c1", os.path.basename(p), "f" * 64, 30, "320x240", 1, "other", "ATH-1")])
    with pytest.raises(ChecksumMismatchError) as e:
        verify_entry(read_manifest(m)[0])
    assert e.value.code == 12


def test_unreadable_file(tmp_path):
    from src.ingest.decoder import probe
    from src.ingest.manifest import compute_sha256
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    m = manifest(tmp_path, [("c1", "bad.mp4", compute_sha256(str(bad)), 30, "1x1", 1, "other", "ATH-1")])
    e0 = read_manifest(m)[0]
    verify_entry(e0)
    with pytest.raises(UnreadableFileError) as e:
        probe(e0.path)
    assert e.value.code == 11


def test_manifest_relative_paths_resolve_from_manifest_dir(tmp_path, blank_clip):
    p, sha = blank_clip
    m = manifest(tmp_path, [("c1", os.path.basename(p), sha, 30, "320x240", 1, "other", "ATH-1")])
    assert read_manifest(m)[0].path == os.path.abspath(p)


def test_ambiguous_bowler_raises():
    """Two different, non-overlapping people with the same arm action at the same moment."""
    from src.processing.isolate import find_anchor
    from src.processing.tracker import SparseTracker
    from src.phase_detection.rtmpose import KP

    def pose(raise_wrist):
        kp = np.zeros((26, 2), np.float32)
        sc = np.ones(26, np.float32)
        kp[KP["neck"]] = (0, 0); kp[KP["hip"]] = (0, 100)
        kp[KP["left_shoulder"]] = (-10, 0); kp[KP["right_shoulder"]] = (10, 0)
        kp[KP["left_wrist"]] = (-10, 50); kp[KP["right_wrist"]] = (10, -raise_wrist)
        return kp, sc

    class S:
        pass
    s = S()
    tr = SparseTracker()
    tr.tracks = {0: {"frames": [0, 6, 12], "boxes": [[0, 0, 50, 200]] * 3, "scores": [0.9] * 3},
                 1: {"frames": [0, 6, 12], "boxes": [[400, 0, 450, 200]] * 3, "scores": [0.9] * 3}}
    s.tracker, s.tracks = tr, tr.tracks
    s.dets = {0: np.zeros((2, 5))}
    s.candidates = [0, 1]
    s.cuts = [False] * 20
    s.A = [np.eye(3)] * 20
    s.pose = {t: {0: pose(-50), 6: pose(80), 12: pose(-50)} for t in (0, 1)}
    with pytest.raises(AmbiguousBowlerError) as e:
        find_anchor(s, 30.0, {"pose_fps": 5.0})
    assert e.value.code == 14


@pytest.mark.skipif(not WEIGHTS, reason="model weights not fetched")
def test_no_person_end_to_end(tmp_path, blank_clip):
    from src.run import main
    p, sha = blank_clip
    m = manifest(tmp_path, [("c1", os.path.basename(p), sha, 30, "320x240", 1, "other", "ATH-1")])
    out = tmp_path / "out"
    rc = main(["--manifest", m, "--out", str(out)])
    assert rc == 3
    log = json.load(open(out / "run_log.json"))
    assert log["failures"][0]["code"] == NoPersonDetectedError.code
    recs = json.load(open(out / "predictions" / "c1.json"))   # schema-valid, all not_observable
    assert all(r["visibility_status"] == "not_observable" for r in recs)


@pytest.mark.skipif(not WEIGHTS, reason="model weights not fetched")
def test_missing_file_end_to_end_exit_code(tmp_path):
    from src.run import main
    m = manifest(tmp_path, [("c1", "missing.mp4", "0" * 64, 30, "1x1", 1, "other", "ATH-1")])
    assert main(["--manifest", m, "--out", str(tmp_path / "o")]) == 3
    assert main(["--manifest", str(tmp_path / "no_such.csv"), "--out", str(tmp_path / "o")]) == 1
