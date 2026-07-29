"""Phase 5: forehand/backhand classification via a geometric heuristic.

At each ball-hit event, find the nearest player's pose and check which side
of their body the racket-side wrist is extended on. A forehand swings the
racket arm out on the same side as the dominant hand; a backhand swings it
across the body to the opposite side. This needs each player's handedness
(there's no way to infer it from pose alone), and per the plan this is the
"fast, interpretable, but brittle to camera angle/occlusion" option — a
two-handed backhand or a slice will confuse it, since it only looks at one
wrist's side of the body, not full swing shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tennis_tracker.pose import (
    FramePoses,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    PlayerPose,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from tennis_tracker.trajectory import HitEvent

# How far (in shoulder-widths) the racket wrist must sit past the torso
# centerline before it counts as a confident forehand/backhand, rather than
# "unclear" (e.g. racket near the body, as at a volley or serve).
DEFAULT_SIDE_THRESHOLD = 0.15

Handedness = str  # "right" or "left"


@dataclass
class ShotClassification:
    hit: HitEvent
    player_id: int
    shot_type: str  # "forehand", "backhand", or "unclear"
    confidence: float  # signed projection normalized by shoulder width; magnitude ~ decisiveness


def nearest_player(position: tuple[float, float], frame_poses: FramePoses) -> PlayerPose | None:
    """Return whichever tracked player's bounding-box center is closest to ``position``."""
    if not frame_poses.players:
        return None

    def centroid(p: PlayerPose) -> np.ndarray:
        x_min, y_min, x_max, y_max = p.bbox
        return np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

    target = np.array(position)
    return min(frame_poses.players, key=lambda p: float(np.linalg.norm(centroid(p) - target)))


def classify_pose(player_pose: PlayerPose, handedness: Handedness) -> tuple[str, float]:
    """Classify a single pose as forehand/backhand/unclear given the player's handedness."""
    pts = player_pose.landmarks_px
    left_shoulder, right_shoulder = pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER]
    left_hip, right_hip = pts[LEFT_HIP], pts[RIGHT_HIP]

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    torso_center = (shoulder_center + hip_center) / 2

    shoulder_vector = right_shoulder - left_shoulder
    shoulder_width = float(np.linalg.norm(shoulder_vector))
    if shoulder_width < 1e-6:
        return "unclear", 0.0

    racket_wrist_idx = RIGHT_WRIST if handedness == "right" else LEFT_WRIST
    wrist = pts[racket_wrist_idx]

    # Signed projection of the wrist's offset from torso center onto the
    # shoulder-line axis, normalized by shoulder width: positive = wrist
    # displaced toward the right-shoulder side, negative = left-shoulder side.
    offset = wrist - torso_center
    projection = float(np.dot(offset, shoulder_vector) / shoulder_width)
    normalized = projection / shoulder_width

    # Racket-hand side matches dominant hand for a forehand; crossing to the
    # opposite side of the body is a backhand.
    same_side_is_forehand = normalized if handedness == "right" else -normalized

    if abs(normalized) <= DEFAULT_SIDE_THRESHOLD:
        return "unclear", same_side_is_forehand

    return ("forehand" if same_side_is_forehand > 0 else "backhand"), same_side_is_forehand


def classify_hits(
    hits: list[HitEvent],
    poses_by_frame: dict[int, FramePoses],
    handedness: dict[int, Handedness],
    max_frame_offset: int = 5,
) -> list[ShotClassification]:
    """Classify each hit event using the nearest player's pose at (or near) that frame."""
    results: list[ShotClassification] = []

    for hit in hits:
        frame_poses = _find_nearby_frame_poses(poses_by_frame, hit.frame_index, max_frame_offset)
        if frame_poses is None:
            continue

        player = nearest_player(hit.position, frame_poses)
        if player is None:
            continue

        player_handedness = handedness.get(player.player_id, "right")
        shot_type, confidence = classify_pose(player, player_handedness)
        results.append(
            ShotClassification(hit=hit, player_id=player.player_id, shot_type=shot_type, confidence=confidence)
        )

    return results


def _find_nearby_frame_poses(
    poses_by_frame: dict[int, FramePoses], frame_index: int, max_offset: int
) -> FramePoses | None:
    if frame_index in poses_by_frame:
        return poses_by_frame[frame_index]
    for offset in range(1, max_offset + 1):
        for candidate in (frame_index - offset, frame_index + offset):
            if candidate in poses_by_frame:
                return poses_by_frame[candidate]
    return None
