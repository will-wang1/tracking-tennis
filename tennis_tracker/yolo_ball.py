"""YOLO-based ball detector: an alternative to ball.py's classical detector
and tracknet.py's trained model.

https://github.com/ultralytics/yolov5's checkpoints are loadable straight
through the `ultralytics` package (already a dependency here) -- that's the
actively maintained way to run them now, no need to clone the yolov5 repo or
go through torch.hub. There's no tennis-specific class in a stock YOLO
model, so this uses COCO's "sports ball" class (32): a reasonable
general-purpose ball detector with no training required, but not
specialized for a small, fast, motion-blurred tennis ball the way a
purpose-trained TrackNet model is.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2

from tennis_tracker.ball import BallDetection

COCO_SPORTS_BALL_CLASS = 32
DEFAULT_YOLO_MODEL = "yolov5su.pt"
DEFAULT_CONFIDENCE = 0.25


class YoloBallDetector:
    """Wraps an ultralytics YOLO model, keeping the highest-confidence "sports ball" box per frame."""

    def __init__(
        self,
        model_path: str = DEFAULT_YOLO_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
        device: str = "cpu",
    ):
        # Imported lazily so the classical/tracknet detectors never pull in ultralytics/torch.
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def detect(self, frame) -> tuple[tuple[float, float] | None, float | None]:
        results = self.model.predict(
            frame, classes=[COCO_SPORTS_BALL_CLASS], conf=self.confidence, device=self.device, verbose=False,
        )[0]
        if len(results.boxes) == 0:
            return None, None

        best = max(results.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = (float(v) for v in best.xyxy[0])
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        radius = max(x2 - x1, y2 - y1) / 2
        return center, radius


def detect_video(video_path: str | Path, verbose: bool = False, **detector_kwargs):
    """Yield a BallDetection per frame of ``video_path``, same contract as ball.detect_video."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector = YoloBallDetector(**detector_kwargs)

    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            position, radius = detector.detect(frame)
            yield BallDetection(
                frame_index=frame_index, timestamp=frame_index / fps, position=position, radius=radius,
            )
            frame_index += 1
            if verbose and (frame_index % 30 == 0 or frame_index == total_frames):
                print(f"\rball detection: frame {frame_index}/{total_frames or '?'}", end="", flush=True)
    finally:
        cap.release()
        if verbose and frame_index:
            print()  # newline after the live-updating progress line


def visualize(video_path: str | Path, output_path: str | Path, trail_length: int = 15, **detector_kwargs) -> int:
    """Write an annotated copy of ``video_path`` with the YOLO-detected ball + trail overlaid."""
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
        for detection in detect_video(video_path, verbose=True, **detector_kwargs):
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
    parser = argparse.ArgumentParser(description="YOLO-based ball detection.")
    sub = parser.add_subparsers(dest="command", required=True)

    viz_p = sub.add_parser("visualize", help="Write an annotated video with ball position + trail.")
    viz_p.add_argument("video", help="Path to the source video file.")
    viz_p.add_argument("output_video", help="Path to write the annotated video to.")
    viz_p.add_argument("--model", default=DEFAULT_YOLO_MODEL, help="ultralytics model name or path (auto-downloads).")
    viz_p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    viz_p.add_argument("--device", default="cpu", help='"cpu" or "cuda".')

    args = parser.parse_args(argv)

    if args.command == "visualize":
        count = visualize(
            args.video, args.output_video,
            model_path=args.model, confidence=args.confidence, device=args.device,
        )
        print(f"Wrote {count} annotated frames to {args.output_video}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
