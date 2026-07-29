import numpy as np

from tennis_tracker.classify import classify_hits, classify_pose, nearest_player
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


def _make_pose(player_id, left_shoulder, right_shoulder, left_hip, right_hip, left_wrist, right_wrist):
    pts = np.zeros((33, 2), dtype=np.float32)
    pts[LEFT_SHOULDER] = left_shoulder
    pts[RIGHT_SHOULDER] = right_shoulder
    pts[LEFT_HIP] = left_hip
    pts[RIGHT_HIP] = right_hip
    pts[LEFT_WRIST] = left_wrist
    pts[RIGHT_WRIST] = right_wrist
    visibility = np.ones(33, dtype=np.float32)
    bbox = (0, 0, 200, 200)
    return PlayerPose(player_id=player_id, landmarks_px=pts, visibility=visibility, bbox=bbox)


# Body facing the camera: left shoulder/hip on image-left, right on image-right,
# torso centered around x=100. Shoulders are 60px apart.
LEFT_SHOULDER_PX = (70, 100)
RIGHT_SHOULDER_PX = (130, 100)
LEFT_HIP_PX = (75, 180)
RIGHT_HIP_PX = (125, 180)


def test_right_handed_forehand_wrist_extended_to_dominant_side():
    # Right-handed forehand: racket (right) wrist extended well past the right shoulder.
    pose = _make_pose(
        0, LEFT_SHOULDER_PX, RIGHT_SHOULDER_PX, LEFT_HIP_PX, RIGHT_HIP_PX,
        left_wrist=(60, 110), right_wrist=(190, 90),
    )

    shot_type, confidence = classify_pose(pose, "right")

    assert shot_type == "forehand"
    assert confidence > 0


def test_right_handed_backhand_wrist_crosses_body():
    # Right-handed backhand: racket (right) wrist has crossed to the left side of the body.
    pose = _make_pose(
        0, LEFT_SHOULDER_PX, RIGHT_SHOULDER_PX, LEFT_HIP_PX, RIGHT_HIP_PX,
        left_wrist=(60, 110), right_wrist=(20, 90),
    )

    shot_type, confidence = classify_pose(pose, "right")

    assert shot_type == "backhand"
    assert confidence < 0


def test_left_handed_mirrors_right_handed_logic():
    # Left-handed forehand: racket (left) wrist extended past the left shoulder.
    pose = _make_pose(
        0, LEFT_SHOULDER_PX, RIGHT_SHOULDER_PX, LEFT_HIP_PX, RIGHT_HIP_PX,
        left_wrist=(10, 90), right_wrist=(140, 110),
    )

    shot_type, _ = classify_pose(pose, "left")

    assert shot_type == "forehand"


def test_wrist_near_center_is_unclear():
    pose = _make_pose(
        0, LEFT_SHOULDER_PX, RIGHT_SHOULDER_PX, LEFT_HIP_PX, RIGHT_HIP_PX,
        left_wrist=(90, 110), right_wrist=(105, 90),
    )

    shot_type, _ = classify_pose(pose, "right")

    assert shot_type == "unclear"


def test_nearest_player_picks_closest_bbox_center():
    near = PlayerPose(0, np.zeros((33, 2)), np.ones(33), bbox=(0, 0, 20, 20))
    far = PlayerPose(1, np.zeros((33, 2)), np.ones(33), bbox=(500, 500, 520, 520))
    frame_poses = FramePoses(frame_index=0, timestamp=0.0, players=[near, far])

    chosen = nearest_player((10, 10), frame_poses)

    assert chosen.player_id == 0


def test_classify_hits_end_to_end():
    forehand_pose = _make_pose(
        0, LEFT_SHOULDER_PX, RIGHT_SHOULDER_PX, LEFT_HIP_PX, RIGHT_HIP_PX,
        left_wrist=(60, 110), right_wrist=(190, 90),
    )
    frame_poses = FramePoses(frame_index=10, timestamp=10 / 30, players=[forehand_pose])
    hit = HitEvent(frame_index=10, timestamp=10 / 30, position=(100.0, 100.0), residual=20.0)

    results = classify_hits([hit], {10: frame_poses}, handedness={0: "right"})

    assert len(results) == 1
    assert results[0].shot_type == "forehand"
    assert results[0].player_id == 0
