"""Phase 6: end-to-end pipeline — wires ball tracking + pose + classification
together and produces an annotated video plus a CSV/JSON shot log.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tennis_tracker.ball import BallDetection, detect_video
from tennis_tracker.classify import NO_POSE_DATA, Handedness, ShotClassification, classify_hits
from tennis_tracker.pose import (
    DEFAULT_POSE_MODEL_VARIANT,
    POSE_MODEL_VARIANTS,
    FramePoses,
    draw_pose_overlay,
    track_poses,
)
from tennis_tracker.trajectory import HitEvent, TrackedPoint, detect_hits, smooth_trajectory

# How many frames after a hit its shot-type label stays drawn on screen,
# so it's readable rather than flashing for a single frame.
LABEL_PERSIST_FRAMES = 15

# A gap in ball detections longer than this (seconds) marks the point as
# over: the ball left play and didn't come back soon, so anything flagged
# as a "hit" after that is more likely a false positive (players resetting,
# picking up balls) than a real shot.
DEFAULT_MAX_BALL_GAP_SEC = 1.5


def parse_handedness_arg(value: str | None) -> dict[int, Handedness]:
    """Parse "0:right,1:left" into {0: "right", 1: "left"}."""
    if not value:
        return {}
    result: dict[int, Handedness] = {}
    for pair in value.split(","):
        player_id_str, hand = pair.split(":")
        result[int(player_id_str.strip())] = hand.strip().lower()
    return result


def get_ball_detections(
    video_path: str | Path,
    detector: str = "classical",
    ball_detector_kwargs: dict | None = None,
    tracknet_model_path: str | Path | None = None,
    tracknet_device: str = "cpu",
) -> list[BallDetection]:
    """Run ball detection with either the classical CV detector or a trained TrackNet model.

    Both produce the same BallDetection stream (one per frame, position=None
    when not found), so everything downstream — Kalman smoothing, hit
    detection, contact fitting — works unchanged regardless of which
    detector produced it.
    """
    if detector == "tracknet":
        if tracknet_model_path is None:
            raise ValueError('tracknet_model_path is required when detector="tracknet"')
        # Imported lazily so using the classical detector never pulls in torch.
        from tennis_tracker.tracknet import load_model, run_tracknet_on_video

        model = load_model(tracknet_model_path, device=tracknet_device)
        return list(run_tracknet_on_video(video_path, model, device=tracknet_device))

    if detector != "classical":
        raise ValueError(f'detector must be "classical" or "tracknet", got {detector!r}')

    return list(detect_video(video_path, **(ball_detector_kwargs or {})))


def find_point_end_frame(detections: list[BallDetection], fps: float, max_gap_sec: float = DEFAULT_MAX_BALL_GAP_SEC) -> int:
    """Find the last frame of the point: the frame right before the first too-long gap in ball detections.

    If no gap that long ever occurs, the whole clip counts as the point.
    Returns the last detection's frame index if ``detections`` is empty.
    """
    if not detections:
        return 0

    max_gap_frames = max(1, int(max_gap_sec * fps))
    gap_start_frame: int | None = None
    for d in detections:
        if d.position is None:
            if gap_start_frame is None:
                gap_start_frame = d.frame_index
            elif d.frame_index - gap_start_frame + 1 > max_gap_frames:
                return gap_start_frame - 1
        else:
            gap_start_frame = None

    return detections[-1].frame_index


@dataclass
class PipelineResult:
    frame_poses: list[FramePoses]
    detections: list[BallDetection]
    tracked: list[TrackedPoint]
    hits: list[HitEvent]
    classifications: list[ShotClassification]
    point_end_frame: int


def run_pipeline(
    video_path: str | Path,
    handedness: dict[int, Handedness],
    max_players: int = 2,
    ball_detector_kwargs: dict | None = None,
    detector: str = "classical",
    tracknet_model_path: str | Path | None = None,
    tracknet_device: str = "cpu",
    max_ball_gap_sec: float = DEFAULT_MAX_BALL_GAP_SEC,
    pose_model_variant: str = DEFAULT_POSE_MODEL_VARIANT,
    pose_confidence: float = 0.5,
) -> PipelineResult:
    """Run pose tracking, ball tracking, hit detection, and classification over a video."""
    frame_poses_list = list(track_poses(
        video_path, max_players=max_players, model_variant=pose_model_variant,
        min_pose_detection_confidence=pose_confidence, min_pose_presence_confidence=pose_confidence,
        min_tracking_confidence=pose_confidence,
    ))
    poses_by_frame = {fp.frame_index: fp for fp in frame_poses_list}

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    detections = get_ball_detections(
        video_path, detector=detector, ball_detector_kwargs=ball_detector_kwargs,
        tracknet_model_path=tracknet_model_path, tracknet_device=tracknet_device,
    )
    point_end_frame = find_point_end_frame(detections, fps, max_gap_sec=max_ball_gap_sec)
    tracked = smooth_trajectory(detections, fps=fps)
    hits = [h for h in detect_hits(tracked) if h.frame_index <= point_end_frame]
    classifications = classify_hits(hits, poses_by_frame, handedness)

    return PipelineResult(frame_poses_list, detections, tracked, hits, classifications, point_end_frame)


def write_shot_log(classifications: list[ShotClassification], output_path: str | Path) -> None:
    path = Path(output_path)
    rows = [
        {
            "frame_index": c.hit.frame_index,
            "timestamp": round(c.hit.timestamp, 3),
            "player_id": c.player_id,
            "shot_type": c.shot_type,
            "confidence": round(c.confidence, 4),
        }
        for c in classifications
    ]
    counts = {"forehand": 0, "backhand": 0, "unclear": 0, NO_POSE_DATA: 0}
    for row in rows:
        counts[row["shot_type"]] += 1

    if path.suffix.lower() == ".csv":
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_index", "timestamp", "player_id", "shot_type", "confidence"])
            writer.writeheader()
            writer.writerows(rows)
    else:
        with path.open("w") as f:
            json.dump({"summary": counts, "shots": rows}, f, indent=2)


def render_annotated_video(
    video_path: str | Path,
    output_path: str | Path,
    frame_poses_list: list[FramePoses],
    detections: list[BallDetection],
    tracked: list[TrackedPoint],
    classifications: list[ShotClassification],
    end_frame: int | None = None,
) -> int:
    """Draws pose skeletons, the ball trail + speed, and shot labels onto a copy of the video.

    ``detections``/``tracked`` are reused from whatever already ran ball
    detection and Kalman smoothing (run_pipeline) rather than re-run here —
    with TrackNet in particular, re-running detection a second time just for
    rendering would double a real GPU inference cost, not just a cheap
    classical-CV pass. The shot-type label is drawn on the hitting player's
    own "player N" tag (e.g. "player 0 - FOREHAND") rather than as a
    separate floating label, so it's unambiguous who hit it. ``end_frame``,
    if given, stops the output there (e.g. where the point ended) instead
    of continuing to render dead time afterward.
    """
    poses_by_frame = {fp.frame_index: fp for fp in frame_poses_list}
    detections_by_frame = {d.frame_index: d for d in detections}
    velocity_by_frame = {t.frame_index: t.velocity for t in tracked}

    player_labels_by_frame: dict[int, dict[int, str]] = {}
    for c in classifications:
        if c.player_id is None:  # NO_POSE_DATA: no player to attach a label to
            continue
        for offset in range(LABEL_PERSIST_FRAMES):
            player_labels_by_frame.setdefault(c.hit.frame_index + offset, {})[c.player_id] = c.shot_type.upper()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    trail: deque = deque(maxlen=15)

    frame_index = 0
    try:
        while True:
            if end_frame is not None and frame_index > end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break

            detection = detections_by_frame.get(frame_index)
            if detection is not None and detection.position is not None:
                trail.append(detection.position)
                x, y = detection.position
                cv2.circle(frame, (int(x), int(y)), max(3, int(detection.radius or 4)), (0, 0, 255), 2)

                velocity = velocity_by_frame.get(frame_index)
                if velocity is not None:
                    speed_px_s = float(np.hypot(velocity[0], velocity[1]))
                    cv2.putText(
                        frame, f"{speed_px_s:.0f} px/s", (int(x) - 35, int(y) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1,
                    )
            for i in range(1, len(trail)):
                p1 = tuple(int(v) for v in trail[i - 1])
                p2 = tuple(int(v) for v in trail[i])
                cv2.line(frame, p1, p2, (0, 165, 255), 2)

            if frame_index in poses_by_frame:
                labels = player_labels_by_frame.get(frame_index, {})
                frame = draw_pose_overlay(frame, poses_by_frame[frame_index], player_labels=labels)

            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    return frame_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6: full tennis shot tracking pipeline.")
    parser.add_argument("video", help="Path to the source video file.")
    parser.add_argument("--output-video", required=True, help="Path to write the annotated video to.")
    parser.add_argument("--output-log", required=True, help="Path to write the shot log to (.json or .csv).")
    parser.add_argument(
        "--handedness",
        default="",
        help='Per-player handedness, e.g. "0:right,1:left". Defaults to right for unlisted players.',
    )
    parser.add_argument("--max-players", type=int, default=2)
    parser.add_argument(
        "--detector", choices=["classical", "tracknet"], default="classical",
        help='Ball detector to use. "tracknet" requires --tracknet-model.',
    )
    parser.add_argument("--tracknet-model", default=None, help="Trained TrackNet checkpoint (from tracknet.py train).")
    parser.add_argument("--tracknet-device", default="cpu", help='"cpu" or "cuda", for the tracknet detector.')
    parser.add_argument(
        "--ball-hsv-lower", default=None, help='Override ball color lower HSV bound, e.g. "24,60,90".'
    )
    parser.add_argument(
        "--ball-hsv-upper", default=None, help='Override ball color upper HSV bound, e.g. "55,255,255".'
    )
    parser.add_argument(
        "--max-ball-gap-sec", type=float, default=DEFAULT_MAX_BALL_GAP_SEC,
        help="Gap in ball detections (seconds) that marks the point as over; tracking/hits stop there.",
    )
    parser.add_argument(
        "--pose-model-variant", choices=POSE_MODEL_VARIANTS, default=DEFAULT_POSE_MODEL_VARIANT,
        help='"lite" is fastest but least accurate; try "full" (default) or "heavy" if players go undetected.',
    )
    parser.add_argument(
        "--pose-confidence", type=float, default=0.5,
        help="Lower (e.g. 0.3) if players are going undetected during real play; raises false positives as a tradeoff.",
    )
    args = parser.parse_args(argv)

    handedness = parse_handedness_arg(args.handedness)
    ball_detector_kwargs = {}
    if args.ball_hsv_lower:
        ball_detector_kwargs["hsv_lower"] = tuple(int(v) for v in args.ball_hsv_lower.split(","))
    if args.ball_hsv_upper:
        ball_detector_kwargs["hsv_upper"] = tuple(int(v) for v in args.ball_hsv_upper.split(","))

    result = run_pipeline(
        args.video, handedness, max_players=args.max_players, ball_detector_kwargs=ball_detector_kwargs,
        detector=args.detector, tracknet_model_path=args.tracknet_model, tracknet_device=args.tracknet_device,
        max_ball_gap_sec=args.max_ball_gap_sec, pose_model_variant=args.pose_model_variant,
        pose_confidence=args.pose_confidence,
    )

    write_shot_log(result.classifications, args.output_log)
    frame_count = render_annotated_video(
        args.video, args.output_video, result.frame_poses, result.detections, result.tracked,
        result.classifications, end_frame=result.point_end_frame,
    )

    counts = {"forehand": 0, "backhand": 0, "unclear": 0, NO_POSE_DATA: 0}
    for c in result.classifications:
        counts[c.shot_type] += 1

    print(f"Processed {frame_count} frames (point ended at frame {result.point_end_frame}), detected {len(result.hits)} hit(s).")
    print(f"Shots: {counts['forehand']} forehand, {counts['backhand']} backhand, {counts['unclear']} unclear.")
    if counts[NO_POSE_DATA]:
        print(
            f"({counts[NO_POSE_DATA]} hit(s) could not be classified — no player pose found nearby, "
            "e.g. occluded or briefly out of frame during the swing)"
        )
    print(f"Annotated video: {args.output_video}")
    print(f"Shot log: {args.output_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
