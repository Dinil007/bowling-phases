"""Unit tests for tracking, camera-motion maths and event logic on synthetic signals."""
import numpy as np

from src.processing.motion import apply, compose
from src.processing.tracker import iou
from src.processing.fallback import DenseBowlerTracker
from src.phase_detection.kinematics import runs, nanmedfilt


def test_iou():
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-6


def test_compose_translation():
    A = [np.eye(3)] + [np.array([[1, 0, 2], [0, 1, 0], [0, 0, 1]], float)] * 3
    T = compose(A, 0, 3)
    assert np.allclose(apply(T, [[0, 0]]), [[6, 0]])


def test_degraded_budget_then_lost():
    t = DenseBowlerTracker([0, 0, 10, 20], max_degraded=2)
    I = np.eye(3)
    assert t.step(I, np.array([[0, 0, 10, 20, 0.9]]))[1] == "tracked"
    assert t.step(I, np.zeros((0, 5)))[1] == "degraded"
    assert t.step(I, np.zeros((0, 5)))[1] == "degraded"
    assert t.step(I, np.zeros((0, 5)))[1] == "lost"


def test_runs_and_filter():
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]
    x = np.array([1, 100, 1, np.nan, 2.0])
    assert nanmedfilt(x, 3)[1] == 1
