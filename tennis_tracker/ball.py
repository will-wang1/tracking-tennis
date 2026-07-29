"""Phase 3: ball detection.

The plan calls out TrackNet as the standard purpose-built approach for this
task, but its pretrained weights are distributed as ad-hoc pickled PyTorch
checkpoints outside of pip (typically via Google Drive links), which is both
fragile to depend on and a supply-chain risk (unpickling arbitrary weights
from a third party can execute code). This module instead implements the
plan's documented fallback: background subtraction + color/shape filtering +
temporal continuity, which needs no external model weights. It's the
documented trade-off (easier to stand up, less robust to camera motion) —
swap in a real TrackNet checkpoint here later if you have one you trust.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Default HSV range for a standard optic-yellow tennis ball. Real footage
# will need this tuned per-lighting-condition (see BallDetector args).
DEFAULT_HSV_LOWER = (24, 60, 90)
DEFAULT_HSV_UPPER = (55, 255, 255)

DEFAULT_MIN_AREA = 8
DEFAULT_MAX_AREA = 900
DEFAULT_MIN_CIRCULARITY = 0.55


@dataclass
class BallDetection:
    frame_index: int
    timestamp: float
    position: tuple[float, float] | None  # (x, y) in pixels, None if not found
    radius: float | None = None


def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter * perimeter))


class BallDetector:
    """Per-frame ball candidate detection via motion + color + shape filtering."""

    def __init__(
        self,
        hsv_lower: tuple[int, int, int] = DEFAULT_HSV_LOWER,
        hsv_upper: tuple[int, int, int] = DEFAULT_HSV_UPPER,
        min_area: float = DEFAULT_MIN_AREA,
        max_area: float = DEFAULT_MAX_AREA,
        min_circularity: float = DEFAULT_MIN_CIRCULARITY,
        max_jump_px: float = 150.0,
    ):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.max_jump_px = max_jump_px
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=32, detectShadows=False
        )
        self._last_position: tuple[float, float] | None = None

    def _candidates(self, frame: np.ndarray) -> list[tuple[tuple[float, float], float, float]]:
        """Returns list of (center, radius, circularity) for plausible ball blobs."""
        fg_mask = self._bg_subtractor.apply(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        combined = cv2.bitwise_and(fg_mask, color_mask)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        combined = cv2.dilate(combined, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            circularity = _circularity(contour)
            if circularity < self.min_circularity:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            candidates.append(((float(x), float(y)), float(radius), circularity))
        return candidates

    def detect(self, frame: np.ndarray) -> tuple[tuple[float, float] | None, float | None]:
        candidates = self._candidates(frame)
        if not candidates:
            return None, None

        if self._last_position is not None:
            def dist(c):
                (x, y), _, _ = c
                lx, ly = self._last_position
                return ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5

            candidates = [c for c in candidates if dist(c) <= self.max_jump_px] or candidates

        # Prefer the most circular candidate; ties broken by proximity to last position.
        candidates.sort(key=lambda c: (-c[2],))
        position, radius, _ = candidates[0]
        self._last_position = position
        return position, radius


def detect_video(video_path: str | Path, **detector_kwargs):
    """Yield a BallDetection per frame of ``video_path``."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = BallDetector(**detector_kwargs)

    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            position, radius = detector.detect(frame)
            yield BallDetection(
                frame_index=frame_index,
                timestamp=frame_index / fps,
                position=position,
                radius=radius,
            )
            frame_index += 1
    finally:
        cap.release()


def visualize(video_path: str | Path, output_path: str | Path, trail_length: int = 15, **detector_kwargs) -> int:
    """Write an annotated copy of ``video_path`` with the detected ball + trail overlaid."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    trail: deque = deque(maxlen=trail_length)
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    try:
        for detection in detect_video(video_path, **detector_kwargs):
            ok, frame = cap.read()
            if not ok:
                break
            if detection.position is not None:
                trail.append(detection.position)
                x, y = detection.position
                radius = max(3, int(detection.radius or 4))
                cv2.circle(frame, (int(x), int(y)), radius, (0, 0, 255), 2)

            for i in range(1, len(trail)):
                p1 = tuple(int(v) for v in trail[i - 1])
                p2 = tuple(int(v) for v in trail[i])
                cv2.line(frame, p1, p2, (0, 165, 255), 2)

            writer.write(frame)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    return frame_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3: ball detection.")
    sub = parser.add_subparsers(dest="command", required=True)

    viz_p = sub.add_parser("visualize", help="Write an annotated video with ball position + trail.")
    viz_p.add_argument("video", help="Path to the source video file.")
    viz_p.add_argument("output_video", help="Path to write the annotated video to.")

    args = parser.parse_args(argv)

    if args.command == "visualize":
        count = visualize(args.video, args.output_video)
        print(f"Wrote {count} annotated frames to {args.output_video}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
