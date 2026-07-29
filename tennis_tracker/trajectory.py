"""Phase 4: Kalman-filter the raw ball detections and detect hit events.

A constant-velocity Kalman filter predicts where the ball "should" be each
frame. When the ball gets struck, the actual measurement diverges sharply
from that prediction — the residual (predicted vs. observed position) spikes.
That spike, not the raw ball position, is what flags a hit event.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter

from tennis_tracker.ball import BallDetection, detect_video

# Process/measurement noise: tuned for pixel-scale ball motion at ~30fps.
# Larger PROCESS_NOISE lets the filter adapt faster to real velocity changes
# (fewer false "hits"); smaller MEASUREMENT_NOISE trusts detections more,
# which sharpens genuine hit residual spikes.
DEFAULT_PROCESS_NOISE = 5.0
DEFAULT_MEASUREMENT_NOISE = 4.0


@dataclass
class TrackedPoint:
    frame_index: int
    timestamp: float
    position: tuple[float, float]  # Kalman-smoothed (post-update) position
    velocity: tuple[float, float]
    residual: float | None  # predicted-vs-observed distance; None if no detection this frame


@dataclass
class HitEvent:
    frame_index: int
    timestamp: float
    position: tuple[float, float]
    residual: float


def _build_kalman_filter(initial_position: tuple[float, float], dt: float) -> KalmanFilter:
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.x = np.array([initial_position[0], initial_position[1], 0.0, 0.0])
    kf.P *= 500.0
    kf.R *= DEFAULT_MEASUREMENT_NOISE
    kf.Q *= DEFAULT_PROCESS_NOISE
    return kf


def smooth_trajectory(detections: list[BallDetection], fps: float) -> list[TrackedPoint]:
    """Run a constant-velocity Kalman filter over ball detections.

    Frames with no detection are predict-only (no measurement update), so
    short gaps in ball detection are bridged smoothly.
    """
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    kf: KalmanFilter | None = None
    tracked: list[TrackedPoint] = []

    for detection in detections:
        if kf is None:
            if detection.position is None:
                continue
            kf = _build_kalman_filter(detection.position, dt)
            tracked.append(
                TrackedPoint(
                    frame_index=detection.frame_index,
                    timestamp=detection.timestamp,
                    position=detection.position,
                    velocity=(0.0, 0.0),
                    residual=None,
                )
            )
            continue

        kf.predict()
        residual = None
        if detection.position is not None:
            predicted = (float(kf.x[0]), float(kf.x[1]))
            residual = float(np.hypot(detection.position[0] - predicted[0], detection.position[1] - predicted[1]))
            kf.update(np.array(detection.position))

        tracked.append(
            TrackedPoint(
                frame_index=detection.frame_index,
                timestamp=detection.timestamp,
                position=(float(kf.x[0]), float(kf.x[1])),
                velocity=(float(kf.x[2]), float(kf.x[3])),
                residual=residual,
            )
        )

    return tracked


def detect_hits(
    tracked_points: list[TrackedPoint],
    threshold_std_multiplier: float = 3.0,
    min_absolute_threshold: float = 5.0,
    min_separation_sec: float = 0.15,
) -> list[HitEvent]:
    """Flag hit events from residual spikes in the smoothed trajectory.

    After a real hit, the constant-velocity filter's residual doesn't spike
    and immediately recover — it jumps once, then decays gradually over many
    frames as the filter's velocity estimate catches up. That decay tail
    means the raw residual magnitude stays elevated long after the hit, so
    thresholding on magnitude directly (mean + k*std over the whole clip)
    gets swamped by its own tail and misses genuine spikes. Instead this
    thresholds on the *increase* in residual from one frame to the next
    (the onset of a spike) — decay is monotonically decreasing so it never
    triggers a false positive, while a hit's sharp jump does.
    """
    residual_points = [p for p in tracked_points if p.residual is not None]
    if len(residual_points) < 2:
        return []

    residuals = np.array([p.residual for p in residual_points])
    deltas = np.diff(residuals)
    threshold = max(min_absolute_threshold, float(deltas.mean() + threshold_std_multiplier * deltas.std()))

    candidates = [
        residual_points[i + 1] for i, delta in enumerate(deltas) if delta >= threshold
    ]
    candidates.sort(key=lambda p: p.residual, reverse=True)

    accepted: list[TrackedPoint] = []
    for candidate in candidates:
        if any(abs(candidate.timestamp - a.timestamp) < min_separation_sec for a in accepted):
            continue
        accepted.append(candidate)

    accepted.sort(key=lambda p: p.frame_index)
    return [
        HitEvent(frame_index=p.frame_index, timestamp=p.timestamp, position=p.position, residual=p.residual)
        for p in accepted
    ]


def process_video(video_path: str, **detector_kwargs) -> tuple[list[TrackedPoint], list[HitEvent]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    detections = list(detect_video(video_path, **detector_kwargs))
    tracked = smooth_trajectory(detections, fps=fps)
    hits = detect_hits(tracked)
    return tracked, hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4: trajectory smoothing & hit detection.")
    parser.add_argument("video", help="Path to the source video file.")
    args = parser.parse_args(argv)

    _, hits = process_video(args.video)
    print(f"Detected {len(hits)} hit event(s):")
    for hit in hits:
        print(f"  frame {hit.frame_index:5d}  t={hit.timestamp:6.2f}s  pos={hit.position}  residual={hit.residual:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
