import numpy as np
import pytest

from tennis_tracker.yolo_ball import COCO_SPORTS_BALL_CLASS, YoloBallDetector


class _FakeBox:
    def __init__(self, xyxy, conf):
        self.xyxy = [xyxy]
        self.conf = [conf]


class _FakeResults:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, boxes):
        self._boxes = boxes
        self.last_call_kwargs = None

    def predict(self, frame, classes, conf, device, verbose):
        self.last_call_kwargs = dict(classes=classes, conf=conf, device=device, verbose=verbose)
        return [_FakeResults(self._boxes)]


def _make_detector(model, confidence=0.25, device="cpu"):
    # Bypasses __init__ (which lazily imports ultralytics) so these tests
    # don't need the real package installed just to exercise .detect()'s logic.
    detector = YoloBallDetector.__new__(YoloBallDetector)
    detector.model = model
    detector.confidence = confidence
    detector.device = device
    return detector


def test_detect_returns_none_when_no_ball_found():
    detector = _make_detector(_FakeModel([]))

    position, radius = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert position is None
    assert radius is None


def test_detect_picks_highest_confidence_box():
    boxes = [_FakeBox((0.0, 0.0, 10.0, 10.0), 0.3), _FakeBox((20.0, 20.0, 40.0, 40.0), 0.9)]
    detector = _make_detector(_FakeModel(boxes))

    position, radius = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert position == pytest.approx((30.0, 30.0))
    assert radius == pytest.approx(10.0)


def test_detect_filters_to_sports_ball_class():
    model = _FakeModel([_FakeBox((0.0, 0.0, 10.0, 10.0), 0.9)])
    detector = _make_detector(model, confidence=0.4, device="cpu")

    detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert model.last_call_kwargs == {
        "classes": [COCO_SPORTS_BALL_CLASS], "conf": 0.4, "device": "cpu", "verbose": False,
    }
