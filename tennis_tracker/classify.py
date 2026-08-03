"""Phase 5: forehand/backhand classification via a geometric heuristic.

At each ball-hit event, find the nearest player's pose and check which side
of their body the racket-side wrist is extended on. A forehand swings the
racket arm out on the same side as the dominant hand; a backhand swings it
across the body to the opposite side. This needs each player's handedness
(there's no way to infer it from pose alone), and per the plan this is the
"fast, interpretable, but brittle to camera angle/occlusion" option — a
two-handed backhand or a slice will confuse it, since it only looks at one
wrist's side of the body, not full swing shape.

classify_hits() can optionally use a trained tennis_tracker.shot_classifier
model instead (see that module) — a THETIS-dataset-trained classifier that
looks at a whole window of swing frames rather than one wrist's side, and
can distinguish more shot types than just forehand/backhand/unclear. This
heuristic remains the always-available default and fallback.
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


NO_POSE_DATA = "no_pose_data"  # distinct from "unclear": no player/pose was found near the hit at all

# Frames each side of a hit to gather for the THETIS-trained classifier
# (tennis_tracker.shot_classifier) -- roughly a full swing's worth at 30fps.
# Below MIN_SHOT_CLASSIFIER_FRAMES available frames (occlusion, a hit right
# at the start/end of the clip), fall back to the geometric heuristic rather
# than feed the classifier a near-empty sequence.
DEFAULT_SHOT_CLASSIFIER_WINDOW = 15
MIN_SHOT_CLASSIFIER_FRAMES = 4


@dataclass
class ShotClassification:
    hit: HitEvent
    player_id: int | None  # None when shot_type == NO_POSE_DATA
    shot_type: str  # "forehand", "backhand", "unclear", or NO_POSE_DATA
    confidence: float  # signed projection normalized by shoulder width; 0.0 when shot_type == NO_POSE_DATA


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


def player_landmark_window(
    poses_by_frame: dict[int, FramePoses], player_id: int, center_frame: int, window: int
) -> list[np.ndarray]:
    """Gather one player's landmarks across frames [center_frame - window, center_frame + window].

    Frames where that player wasn't tracked are skipped rather than padded
    here — extract_clip_features (tennis_tracker.shot_classifier) handles a
    sequence shorter than its expected sample count.
    """
    sequence = []
    for frame_index in range(center_frame - window, center_frame + window + 1):
        frame_poses = poses_by_frame.get(frame_index)
        if frame_poses is None:
            continue
        player = next((p for p in frame_poses.players if p.player_id == player_id), None)
        if player is not None:
            sequence.append(player.landmarks_px)
    return sequence


def classify_hits(
    hits: list[HitEvent],
    poses_by_frame: dict[int, FramePoses],
    handedness: dict[int, Handedness],
    max_frame_offset: int = 5,
    shot_classifier_model=None,
    shot_classifier_window: int = DEFAULT_SHOT_CLASSIFIER_WINDOW,
) -> list[ShotClassification]:
    """Classify each hit event using the nearest player's pose at (or near) that frame.

    Always returns exactly one ShotClassification per hit — a hit with no
    player/pose found nearby (occlusion, briefly out of frame, often right
    at the moment of a fast swing) is reported as NO_POSE_DATA rather than
    silently dropped, so "N hits detected" and the classified-shot counts
    stay reconcilable instead of quietly disagreeing.

    ``shot_classifier_model``, if given (a tennis_tracker.shot_classifier
    ShotClassifierNet, e.g. from load_model()), is used in place of the
    geometric heuristic whenever enough frames of that player's pose are
    available around the hit — it can distinguish shot types the heuristic
    can't (slices, two-handed backhands, volleys, serve types), depending
    on what it was trained on. Falls back to the heuristic when no model is
    given, or when too few frames are available (occlusion, a hit right at
    a clip boundary).
    """
    results: list[ShotClassification] = []

    for hit in hits:
        frame_poses = _find_nearby_frame_poses(poses_by_frame, hit.frame_index, max_frame_offset)
        player = nearest_player(hit.position, frame_poses) if frame_poses is not None else None

        if player is None:
            results.append(ShotClassification(hit=hit, player_id=None, shot_type=NO_POSE_DATA, confidence=0.0))
            continue

        player_handedness = handedness.get(player.player_id, "right")
        shot_type, confidence = None, None
        if shot_classifier_model is not None:
            sequence = player_landmark_window(poses_by_frame, player.player_id, hit.frame_index, shot_classifier_window)
            if len(sequence) >= MIN_SHOT_CLASSIFIER_FRAMES:
                from tennis_tracker.shot_classifier import classify_landmarks_sequence

                shot_type, confidence = classify_landmarks_sequence(shot_classifier_model, sequence)
        if shot_type is None:
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
