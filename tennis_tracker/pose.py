"""Phase 2: player detection + pose estimation via MediaPipe Pose Landmarker.

MediaPipe's PoseLandmarker natively supports detecting multiple people
per frame (num_poses=2 covers both players), so a separate YOLOv8 person
detector isn't needed. What it doesn't give us is a stable identity across
frames, so this module adds a small nearest-centroid tracker on top to keep
"player 0" / "player 1" consistent over time.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

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

@contextlib.contextmanager
def _suppress_native_stderr():
    """Redirect the OS-level stderr file descriptor to devnull for the block.

    mediapipe/TFLite write some benign startup/per-inference notices
    ("Created TensorFlow Lite XNNPACK delegate", "Feedback manager
    requires...", "Using NORM_RECT without IMAGE_DIMENSIONS...") straight to
    the native stderr fd from C++, bypassing Python's logging entirely —
    TF_CPP_MIN_LOG_LEVEL/GLOG_minloglevel don't reliably suppress them in
    this build. Redirecting the fd itself works regardless of which native
    logging library wrote it. Scoped tightly around one call at a time
    (never held open across a generator's yield) so a real error elsewhere
    is never at risk of being silently swallowed.
    """
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# "lite" is fastest but least accurate; players who are small/distant in a
# typical broadcast-angle tennis shot are exactly the case it struggles
# with. Offline batch processing isn't latency-sensitive, so "full" is a
# meaningfully more accurate default for little extra cost; "heavy" is the
# most accurate still if "full" isn't enough.
POSE_MODEL_VARIANTS = ("lite", "full", "heavy")
DEFAULT_POSE_MODEL_VARIANT = "full"


def _model_url(variant: str) -> str:
    name = f"pose_landmarker_{variant}"
    return f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/{name}/float16/latest/{name}.task"


def _default_model_path(variant: str) -> Path:
    return DEFAULT_MODEL_DIR / f"pose_landmarker_{variant}.task"

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


def ensure_model(variant: str = DEFAULT_POSE_MODEL_VARIANT, model_path: Path | None = None) -> Path:
    """Download the requested pose landmarker model variant on first use; reuse the cached copy after."""
    if variant not in POSE_MODEL_VARIANTS:
        raise ValueError(f"Unknown pose model variant {variant!r}; choose from {POSE_MODEL_VARIANTS}")
    path = model_path or _default_model_path(variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(_model_url(variant), path)
    return path


def _centroid(landmarks_px: np.ndarray) -> np.ndarray:
    return landmarks_px.mean(axis=0)


class PlayerTracker:
    """Greedy nearest-centroid tracker that keeps player IDs stable across frames.

    A player briefly missing a detection — motion blur or self-occlusion
    during a fast swing, exactly the moment a shot happens — used to risk an
    ID swap: the track's last-known position stayed frozen while they kept
    moving, so by the time they reappeared they could be too far from that
    frozen spot to match. This predicts each missed track's position forward
    using its last observed velocity instead of assuming it stood still, and
    expires a track only after ``max_missed_frames`` consecutive misses, so
    a stale track can't linger forever and steal a match from someone new.
    """

    def __init__(self, max_players: int = 2, max_match_distance: float = 100.0, max_missed_frames: int = 15):
        self.max_players = max_players
        self.max_match_distance = max_match_distance
        self.max_missed_frames = max_missed_frames
        self._next_id = 0
        self._track_centroids: dict[int, np.ndarray] = {}
        self._track_velocities: dict[int, np.ndarray] = {}
        self._track_missed_count: dict[int, int] = {}

    def _predicted_centroid(self, track_id: int) -> np.ndarray:
        missed = self._track_missed_count.get(track_id, 0)
        velocity = self._track_velocities.get(track_id, np.zeros(2))
        return self._track_centroids[track_id] + velocity * (missed + 1)

    def update(self, detections: list[tuple[np.ndarray, np.ndarray, tuple]]) -> list[PlayerPose]:
        """``detections`` is a list of (landmarks_px, visibility, bbox). Returns assigned PlayerPoses."""
        centroids = [_centroid(landmarks) for landmarks, _, _ in detections]

        unmatched_detections = set(range(len(detections)))
        assignments: dict[int, int] = {}  # detection_index -> track_id

        # Greedy matching: repeatedly pick the closest (predicted track, detection) pair.
        candidate_pairs = []
        for track_id in self._track_centroids:
            predicted = self._predicted_centroid(track_id)
            for det_idx, det_centroid in enumerate(centroids):
                dist = float(np.linalg.norm(predicted - det_centroid))
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
            new_centroid = centroids[det_idx]
            if track_id in self._track_centroids:
                self._track_velocities[track_id] = new_centroid - self._track_centroids[track_id]
            self._track_centroids[track_id] = new_centroid
            self._track_missed_count[track_id] = 0
            players.append(PlayerPose(track_id, landmarks_px, visibility, bbox))

        matched_track_ids = set(assignments.values())
        for track_id in list(self._track_centroids):
            if track_id in matched_track_ids:
                continue
            self._track_missed_count[track_id] = self._track_missed_count.get(track_id, 0) + 1
            if self._track_missed_count[track_id] > self.max_missed_frames:
                del self._track_centroids[track_id]
                self._track_velocities.pop(track_id, None)  # never set if only matched once
                del self._track_missed_count[track_id]

        players.sort(key=lambda p: p.player_id)
        return players


def _landmarks_to_pixels(pose_landmarks, width: int, height: int) -> tuple[np.ndarray, np.ndarray, tuple]:
    pts = np.array([[lm.x * width, lm.y * height] for lm in pose_landmarks], dtype=np.float32)
    visibility = np.array([lm.visibility for lm in pose_landmarks], dtype=np.float32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
    return pts, visibility, bbox


def _filter_to_most_active_tracks(frame_poses_list: list[FramePoses], max_players: int) -> list[FramePoses]:
    """Keep only the ``max_players`` tracks with the most total movement, remapped to sequential IDs.

    Real footage usually has more humans in frame than just the players —
    ball kids, umpire, linespeople — who a detector capped at exactly
    ``max_players`` has no way to distinguish from the players themselves;
    it just keeps whichever candidates it's most confident about each
    frame, which can inconsistently jump between different people frame to
    frame. Over-detecting more candidates than needed and then keeping only
    the ones that moved the most across the whole clip is a much more
    reliable signal: real players move substantially during play, while
    officials and ball kids mostly don't.
    """
    positions_by_track: dict[int, list[np.ndarray]] = {}
    for frame_poses in frame_poses_list:
        for player in frame_poses.players:
            positions_by_track.setdefault(player.player_id, []).append(_centroid(player.landmarks_px))

    def total_movement(track_id: int) -> float:
        positions = np.array(positions_by_track[track_id])
        if len(positions) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))

    selected_track_ids = sorted(positions_by_track, key=total_movement, reverse=True)[:max_players]
    id_remap = {track_id: new_id for new_id, track_id in enumerate(sorted(selected_track_ids))}

    filtered: list[FramePoses] = []
    for frame_poses in frame_poses_list:
        players = [
            PlayerPose(id_remap[p.player_id], p.landmarks_px, p.visibility, p.bbox)
            for p in frame_poses.players
            if p.player_id in id_remap
        ]
        players.sort(key=lambda p: p.player_id)
        filtered.append(FramePoses(frame_index=frame_poses.frame_index, timestamp=frame_poses.timestamp, players=players))
    return filtered


def track_poses(
    video_path: str | Path,
    max_players: int = 2,
    model_variant: str = DEFAULT_POSE_MODEL_VARIANT,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_path: Path | None = None,
    over_detect_poses: int | None = None,
    verbose: bool = False,
):
    """Yield a FramePoses per video frame, with player IDs stable across frames.

    If players are going undetected during real play (not just briefly
    mid-swing, which PlayerTracker already tolerates, but genuinely missing
    for long stretches), the two things worth trying first are a larger
    ``model_variant`` ("full" or "heavy" — "lite" is fastest but least
    accurate, and struggles most with players who are small/distant in
    frame, which is normal for a broadcast-angle tennis shot) and lowering
    the confidence thresholds below their conservative 0.5 defaults.

    If instead detection is finding people but the *wrong* ones (ball kids,
    umpire, linespeople), raise ``over_detect_poses`` (default:
    ``max(6, max_players * 3)``) — more candidates are tracked internally
    per frame, and only the ``max_players`` tracks that moved the most
    across the whole clip are kept and returned as the players.
    """
    over_detect_poses = over_detect_poses or max(6, max_players * 3)

    path = Path(video_path)
    resolved_model_path = ensure_model(model_variant, model_path)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(resolved_model_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=over_detect_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = PlayerTracker(
        max_players=over_detect_poses, max_match_distance=cap.get(cv2.CAP_PROP_FRAME_WIDTH) * MAX_MATCH_DISTANCE_FRAC
    )

    raw_frames: list[FramePoses] = []
    frame_index = 0
    try:
        with _suppress_native_stderr():
            landmarker = PoseLandmarker.create_from_options(options)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((frame_index / fps) * 1000)

                with _suppress_native_stderr():
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                detections = [
                    _landmarks_to_pixels(pose_landmarks, width, height)
                    for pose_landmarks in result.pose_landmarks
                ]
                players = tracker.update(detections)

                raw_frames.append(FramePoses(frame_index=frame_index, timestamp=frame_index / fps, players=players))
                frame_index += 1
                if verbose and (frame_index % 30 == 0 or frame_index == total_frames):
                    print(f"\rpose tracking: frame {frame_index}/{total_frames or '?'}", end="", flush=True)
        finally:
            landmarker.close()
    finally:
        cap.release()
        if verbose and frame_index:
            print()  # newline after the live-updating progress line

    yield from _filter_to_most_active_tracks(raw_frames, max_players)


_PLAYER_COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 255), (255, 255, 0)]

# MediaPipe still returns a landmark for body parts it can't actually see —
# occluded by the body's own pose, out of frame, motion-blurred mid-swing —
# it just marks them with low visibility. Drawing those anyway is what a
# "wild"/broken-looking skeleton usually is: real joints connected to
# guessed ones. Skipping points and connections below this threshold trims
# the guesses and leaves only the parts mediapipe actually saw.
DEFAULT_MIN_LANDMARK_VISIBILITY = 0.5


def draw_pose_overlay(
    frame: np.ndarray,
    frame_poses: FramePoses,
    player_labels: dict[int, str] | None = None,
    min_landmark_visibility: float = DEFAULT_MIN_LANDMARK_VISIBILITY,
) -> np.ndarray:
    """Draw skeletons + a "player N" label per tracked player.

    ``player_labels``, if given, maps player_id -> extra text appended to
    that player's own label (e.g. "player 0 - FOREHAND") instead of a
    separate floating label elsewhere in the frame. Landmarks/connections
    below ``min_landmark_visibility`` are skipped rather than drawn as
    guesses (see DEFAULT_MIN_LANDMARK_VISIBILITY).
    """
    annotated = frame.copy()
    for player in frame_poses.players:
        color = _PLAYER_COLORS[player.player_id % len(_PLAYER_COLORS)]
        pts = player.landmarks_px
        visible = player.visibility >= min_landmark_visibility

        for start, end in SKELETON_CONNECTIONS:
            if not (visible[start] and visible[end]):
                continue
            p1, p2 = pts[start], pts[end]
            cv2.line(annotated, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
        for (x, y), ok in zip(pts, visible):
            if not ok:
                continue
            cv2.circle(annotated, (int(x), int(y)), 3, color, -1)

        x_min, y_min, x_max, y_max = player.bbox
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 1)
        label = f"player {player.player_id}"
        extra = (player_labels or {}).get(player.player_id)
        if extra:
            label = f"{label} - {extra}"
        cv2.putText(
            annotated,
            label,
            (x_min, max(0, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return annotated


def visualize(
    video_path: str | Path,
    output_path: str | Path,
    max_players: int = 2,
    model_variant: str = DEFAULT_POSE_MODEL_VARIANT,
    pose_confidence: float = 0.5,
    over_detect_poses: int | None = None,
    min_landmark_visibility: float = DEFAULT_MIN_LANDMARK_VISIBILITY,
) -> int:
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
        for frame_poses in track_poses(
            video_path, max_players=max_players, model_variant=model_variant,
            min_pose_detection_confidence=pose_confidence, min_pose_presence_confidence=pose_confidence,
            min_tracking_confidence=pose_confidence, over_detect_poses=over_detect_poses,
        ):
            ok, frame = cap.read()
            if not ok:
                break
            annotated = draw_pose_overlay(frame, frame_poses, min_landmark_visibility=min_landmark_visibility)
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
    viz_p.add_argument(
        "--model-variant", choices=POSE_MODEL_VARIANTS, default=DEFAULT_POSE_MODEL_VARIANT,
        help='"lite" is fastest but least accurate; try "full" or "heavy" if players go undetected.',
    )
    viz_p.add_argument(
        "--pose-confidence", type=float, default=0.5,
        help="Lower (e.g. 0.3) if players are going undetected; raises false positives as a tradeoff.",
    )
    viz_p.add_argument(
        "--over-detect-poses", type=int, default=None,
        help="Candidates tracked per frame before filtering to --max-players (default: max(6, max_players*3)). "
        "Raise this if the wrong people (ball kids, umpire) are being picked as players.",
    )
    viz_p.add_argument(
        "--min-landmark-visibility", type=float, default=DEFAULT_MIN_LANDMARK_VISIBILITY,
        help="Skip drawing skeleton points/lines mediapipe is less than this confident it actually saw.",
    )

    args = parser.parse_args(argv)

    if args.command == "visualize":
        count = visualize(
            args.video, args.output_video, max_players=args.max_players,
            model_variant=args.model_variant, pose_confidence=args.pose_confidence,
            over_detect_poses=args.over_detect_poses, min_landmark_visibility=args.min_landmark_visibility,
        )
        print(f"Wrote {count} annotated frames to {args.output_video}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
