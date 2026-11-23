# -*- coding: utf-8 -*-
"""Regression tests for resource cleanup, validation, and drawing recovery."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import actions
import camera
import cli
import draw
from face_guard import FaceGuard
from gestures import StepAccumulator, analyze_two_hands
from hand_tracker import Hand, HandTracker


class _FakeCapture:
    instances = []
    behavior = "success"

    def __init__(self):
        self.released = False
        self.read_count = 0
        self.__class__.instances.append(self)

    def set(self, *_args):
        return True

    def open(self, *_args):
        return True

    def isOpened(self):
        return True

    def read(self):
        self.read_count += 1
        if self.behavior == "fail_after_first" and self.read_count > 1:
            return False, None
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def get(self, prop):
        if prop == camera.cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop == camera.cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        return 30

    def release(self):
        self.released = True


class CameraTests(unittest.TestCase):
    def setUp(self):
        _FakeCapture.instances = []

    def test_failed_camera_candidates_are_always_released(self):
        _FakeCapture.behavior = "fail_after_first"
        with patch.object(camera.cv2, "VideoCapture", _FakeCapture):
            self.assertIsNone(camera.open_camera(0))
        self.assertEqual(len(_FakeCapture.instances), 4)
        self.assertTrue(all(cap.released for cap in _FakeCapture.instances))

    def test_successful_camera_remains_open(self):
        _FakeCapture.behavior = "success"
        with patch.object(camera.cv2, "VideoCapture", _FakeCapture):
            cap = camera.open_camera(0)
        self.assertIsNotNone(cap)
        self.assertFalse(cap.released)
        self.assertEqual(len(_FakeCapture.instances), 1)


class ResourceTests(unittest.TestCase):
    def test_hand_tracker_close_is_idempotent(self):
        class Closable:
            def __init__(self):
                self.calls = 0

            def close(self):
                self.calls += 1

        tracker = HandTracker.__new__(HandTracker)
        tracker.gesture_recognizer = None
        tracker.landmarker = Closable()
        tracker._close_lock = threading.Lock()
        tracker._closed = False
        tracker.close()
        tracker.close()
        self.assertEqual(tracker.landmarker.calls, 1)

    def test_face_guard_close_is_idempotent(self):
        class Closable:
            def __init__(self):
                self.calls = 0

            def close(self):
                self.calls += 1

        guard = FaceGuard.__new__(FaceGuard)
        guard.detector = Closable()
        guard.enabled = True
        detector = guard.detector
        guard.close()
        guard.close()
        self.assertEqual(detector.calls, 1)
        self.assertFalse(guard.enabled)
        self.assertIsNone(guard.detector)

    def test_combo_releases_modifier_after_failure(self):
        with (
            patch.object(actions, "_down"),
            patch.object(actions, "_up") as release,
            patch.object(actions, "tap", side_effect=RuntimeError("injection failed")),
            patch.object(actions.time, "sleep"),
        ):
            with self.assertRaises(RuntimeError):
                actions.combo(actions.VK["alt"], actions.VK["tab"])
        release.assert_called_once_with(actions.VK["alt"])


class DrawingRecoveryTests(unittest.TestCase):
    def test_point_stroke_survives_undo_redraw(self):
        canvas = draw.InkCanvas(200, 200, 5)
        canvas.begin((50, 50))
        canvas.end()
        canvas.begin((20, 100))
        canvas.extend((100, 100))
        canvas.end()
        self.assertTrue(canvas.undo())
        self.assertGreater(int(canvas.mask[50, 50]), 0)
        self.assertEqual(canvas.n_strokes, 1)

    def test_image_encoding_error_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.unsupported")
            with patch.object(draw.cv2, "imencode", side_effect=ValueError("bad extension")):
                self.assertFalse(draw.save_png(np.zeros((10, 10, 3), np.uint8), path))


class GeometryTests(unittest.TestCase):
    @staticmethod
    def _hand(x, aspect):
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[:, 0] = x
        landmarks[:, 1] = 0.65
        landmarks[0, 0] = x - 0.05
        landmarks[0, 1] = 0.75
        landmarks[9, 0] = x + 0.05
        landmarks[9, 1] = 0.60
        return Hand(landmarks=landmarks, score=1.0, aspect=aspect)

    def test_two_hand_distance_uses_metric_coordinates(self):
        distances = []
        expected = []
        for aspect in (16 / 9, 1.0):
            left = self._hand(0.2, aspect)
            left.side = "left"
            right = self._hand(0.8, aspect)
            right.side = "right"
            distances.append(analyze_two_hands([left, right]).distance)
            expected.append(np.hypot(0.6 * aspect, 0) / np.hypot(0.1 * aspect, 0.15))
        self.assertAlmostEqual(distances[0], expected[0], places=5)
        self.assertAlmostEqual(distances[1], expected[1], places=5)


class ValidationTests(unittest.TestCase):
    def test_step_accumulator_rejects_zero_step(self):
        with self.assertRaises(ValueError):
            StepAccumulator(step_size=0)

    def test_cli_rejects_invalid_values(self):
        for validator, value in (
            (cli.positive_int, "0"),
            (cli.positive_float, "-0.1"),
            (cli.non_negative_int, "-1"),
            (cli.probability, "1.1"),
        ):
            with self.subTest(validator=validator.__name__, value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    validator(value)


if __name__ == "__main__":
    unittest.main()
