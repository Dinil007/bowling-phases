"""Timestamp-from-frame conversion uses the true (container) frame rate (Brief 4B/5)."""
from src.ingest.decoder import VideoMeta, probe
from tests.conftest import write_video


def meta(fps):
    return VideoMeta(path="x", width=1, height=1, fps=fps, frame_count=1000, duration_s=0, is_vfr=False)


def test_ts_from_index_and_fps():
    assert meta(30.0).ts(0) == 0.0
    assert meta(30.0).ts(141) == 4.7
    assert meta(25.0).ts(50) == 2.0
    assert abs(meta(23.0).ts(107) - 107 / 23.0) < 1e-4
    assert abs(meta(30000 / 1001).ts(300) - 300 * 1001 / 30000) < 1e-4


def test_probe_reads_true_metadata(tmp_path):
    p = write_video(str(tmp_path / "c.mp4"), n=45, fps=25.0, size=(160, 120))
    m = probe(p)
    assert (m.width, m.height, m.frame_count) == (160, 120, 45)
    assert abs(m.fps - 25.0) < 1e-6
    assert abs(m.duration_s - 1.8) < 1e-3
