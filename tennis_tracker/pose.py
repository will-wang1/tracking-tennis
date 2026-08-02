"""Phase 2: player detection + pose estimation via MediaPipe Pose Landmarker.

MediaPipe's PoseLandmarker natively supports detecting multiple people
per frame (num_poses=2 covers both players), so a separate YOLOv8 person
detector isn't needed. What it doesn't give us is a stable identity across
frames, so this module adds a small nearest-centroid tracker on top to keep
"player 0" / "player 1" consistent over time.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Must be set before the first mediapipe import below: silences the
# "Created TensorFlow Lite XNNPACK delegate", "Feedback manager requires...",
# and "Using NORM_RECT without IMAGE_DIMENSIONS" lines mediapipe/TFLite/absl
# print on every run — all benign and non-actionable, not actual errors.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "pose_landmarker_lite.task"

# (landmark_index, landmark_index) pairs to draw as skeleton edges.
SKELETON_CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]

# Named indices used later for forehand/backhand classification (Phase 5).
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

# Max centroid displacement (as a fraction of frame width) allowed between
# consecutive frames for a detection to be matched to an existing track.
MAX_MATCH_DISTANCE_FRAC = 0.15


@dataclass
class PlayerPose:
    player_id: int
    landmarks_px: np.ndarray  # shape (33, 2), pixel coordinates
    visibility: np.ndarray  # shape (33,)
    bbox: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max


@dataclass
class FramePoses:
    frame_index: int
    timestamp: float
    players: list[PlayerPose] = field(default_factory=list)


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the pose landmarker model on first use; reuse the cached copy after."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def _centroid(landmarks_px: np.ndarray) -> np.ndarray:
    return landmarks_px.mean(axis=0)


class PlayerTracker:
    """Greedy nearest-centroid tracker that keeps player IDs stable across frames."""

    def __init__(self, max_players: int = 2, max_match_distance: float = 100.0):
        self.max_players = max_players
        self.max_match_distance = max_match_distance
        self._next_id = 0
        self._track_centroids: dict[int, np.ndarray] = {}

    def update(self, detections: list[tuple[np.ndarray, np.ndarray, tuple]]) -> list[PlayerPose]:
        """``detections`` is a list of (landmarks_px, visibility, bbox). Returns assigned PlayerPoses."""
        centroids = [_centroid(landmarks) for landmarks, _, _ in detections]

        unmatched_detections = set(range(len(detections)))
        assignments: dict[int, int] = {}  # detection_index -> track_id

        # Greedy matching: repeatedly pick the closest (track, detection) pair.
        candidate_pairs = []
        for track_id, track_centroid in self._track_centroids.items():
            for det_idx, det_centroid in enumerate(centroids):
                dist = float(np.linalg.norm(track_centroid - det_centroid))
                if dist <= self.max_match_distance:
                    candidate_pairs.append((dist, track_id, det_idx))
        candidate_pairs.sort(key=lambda p: p[0])

        used_tracks: set[int] = set()
        for dist, track_id, det_idx in candidate_pairs:
            if track_id in used_tracks or det_idx not in unmatched_detections:
                continue
            assignments[det_idx] = track_id
            used_tracks.add(track_id)
            unmatched_detections.discard(det_idx)

        # Assign new track IDs to leftover detections, up to max_players total.
        for det_idx in sorted(unmatched_detections):
            if len(assignments) >= self.max_players:
                break
            new_id = self._next_id
            self._next_id += 1
            assignments[det_idx] = new_id

        players: list[PlayerPose] = []
        for det_idx, track_id in assignments.items():
            landmarks_px, visibility, bbox = detections[det_idx]
            self._track_centroids[track_id] = centroids[det_idx]
            players.append(PlayerPose(track_id, landmarks_px, visibility, bbox))

        players.sort(key=lambda p: p.player_id)
        return players


def _landmarks_to_pixels(pose_landmarks, width: int, height: int) -> tuple[np.ndarray, np.ndarray, tuple]:
    pts = np.array([[lm.x * width, lm.y * height] for lm in pose_landmarks], dtype=np.float32)
    visibility = np.array([lm.visibility for lm in pose_landmarks], dtype=np.float32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
    return pts, visibility, bbox


def track_poses(video_path: str | Path, max_players: int = 2, model_path: Path = DEFAULT_MODEL_PATH):
    """Yield a FramePoses per video frame, with player IDs stable across frames."""
    path = Path(video_path)
    resolved_model_path = ensure_model(model_path)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(resolved_model_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=max_players,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = PlayerTracker(max_players=max_players, max_match_distance=cap.get(cv2.CAP_PROP_FRAME_WIDTH) * MAX_MATCH_DISTANCE_FRAC)

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((frame_index / fps) * 1000)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                detections = [
                    _landmarks_to_pixels(pose_landmarks, width, height)
                    for pose_landmarks in result.pose_landmarks
                ]
                players = tracker.update(detections)

                yield FramePoses(frame_index=frame_index, timestamp=frame_index / fps, players=players)
                frame_index += 1
    finally:
        cap.release()


_PLAYER_COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 255), (255, 255, 0)]


def draw_pose_overlay(frame: np.ndarray, frame_poses: FramePoses) -> np.ndarray:
    annotated = frame.copy()
    for player in frame_poses.players:
        color = _PLAYER_COLORS[player.player_id % len(_PLAYER_COLORS)]
        pts = player.landmarks_px

        for start, end in SKELETON_CONNECTIONS:
            p1, p2 = pts[start], pts[end]
            cv2.line(annotated, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
        for x, y in pts:
            cv2.circle(annotated, (int(x), int(y)), 3, color, -1)

        x_min, y_min, x_max, y_max = player.bbox
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 1)
        cv2.putText(
            annotated,
            f"player {player.player_id}",
            (x_min, max(0, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return annotated


def visualize(video_path: str | Path, output_path: str | Path, max_players: int = 2) -> int:
    """Write an annotated copy of ``video_path`` with skeleton overlays to ``output_path``."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    try:
        for frame_poses in track_poses(video_path, max_players=max_players):
            ok, frame = cap.read()
            if not ok:
                break
            annotated = draw_pose_overlay(frame, frame_poses)
            writer.write(annotated)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    return frame_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2: player pose tracking.")
    sub = parser.add_subparsers(dest="command", required=True)

    viz_p = sub.add_parser("visualize", help="Write an annotated video with skeleton overlays.")
    viz_p.add_argument("video", help="Path to the source video file.")
    viz_p.add_argument("output_video", help="Path to write the annotated video to.")
    viz_p.add_argument("--max-players", type=int, default=2)

    args = parser.parse_args(argv)

    if args.command == "visualize":
        count = visualize(args.video, args.output_video, max_players=args.max_players)
        print(f"Wrote {count} annotated frames to {args.output_video}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
