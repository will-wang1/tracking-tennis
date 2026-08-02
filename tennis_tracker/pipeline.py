"""Phase 6: end-to-end pipeline — wires ball tracking + pose + classification
together and produces an annotated video plus a CSV/JSON shot log.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import cv2

from tennis_tracker.ball import BallDetection, detect_video
from tennis_tracker.classify import NO_POSE_DATA, Handedness, ShotClassification, classify_hits
from tennis_tracker.pose import FramePoses, draw_pose_overlay, track_poses
from tennis_tracker.trajectory import HitEvent, detect_hits, smooth_trajectory

# How many frames after a hit its shot-type label stays drawn on screen,
# so it's readable rather than flashing for a single frame.
LABEL_PERSIST_FRAMES = 15

_LABEL_COLORS = {
    "forehand": (0, 255, 0), "backhand": (0, 0, 255), "unclear": (0, 255, 255), NO_POSE_DATA: (128, 128, 128),
}


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


def run_pipeline(
    video_path: str | Path,
    handedness: dict[int, Handedness],
    max_players: int = 2,
    ball_detector_kwargs: dict | None = None,
    detector: str = "classical",
    tracknet_model_path: str | Path | None = None,
    tracknet_device: str = "cpu",
) -> tuple[list[FramePoses], list[BallDetection], list[HitEvent], list[ShotClassification]]:
    """Run pose tracking, ball tracking, hit detection, and classification over a video."""
    frame_poses_list = list(track_poses(video_path, max_players=max_players))
    poses_by_frame = {fp.frame_index: fp for fp in frame_poses_list}

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    detections = get_ball_detections(
        video_path, detector=detector, ball_detector_kwargs=ball_detector_kwargs,
        tracknet_model_path=tracknet_model_path, tracknet_device=tracknet_device,
    )
    tracked = smooth_trajectory(detections, fps=fps)
    hits = detect_hits(tracked)
    classifications = classify_hits(hits, poses_by_frame, handedness)

    return frame_poses_list, detections, hits, classifications


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
    classifications: list[ShotClassification],
) -> int:
    """Draws pose skeletons, the ball trail, and shot labels onto a copy of the video.

    ``detections`` are reused from whatever already ran ball detection
    (run_pipeline) rather than re-run here — with TrackNet in particular,
    re-running detection a second time just for rendering would double a
    real GPU inference cost, not just a cheap classical-CV pass.
    """
    poses_by_frame = {fp.frame_index: fp for fp in frame_poses_list}
    detections_by_frame = {d.frame_index: d for d in detections}
    labels_by_frame: dict[int, tuple[str, tuple[int, int]]] = {}
    for c in classifications:
        x, y = c.hit.position
        for offset in range(LABEL_PERSIST_FRAMES):
            labels_by_frame[c.hit.frame_index + offset] = (c.shot_type, (int(x), int(y)))

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
            ok, frame = cap.read()
            if not ok:
                break

            detection = detections_by_frame.get(frame_index)
            if detection is not None and detection.position is not None:
                trail.append(detection.position)
                cv2.circle(
                    frame, (int(detection.position[0]), int(detection.position[1])),
                    max(3, int(detection.radius or 4)), (0, 0, 255), 2,
                )
            for i in range(1, len(trail)):
                p1 = tuple(int(v) for v in trail[i - 1])
                p2 = tuple(int(v) for v in trail[i])
                cv2.line(frame, p1, p2, (0, 165, 255), 2)

            if frame_index in poses_by_frame:
                frame = draw_pose_overlay(frame, poses_by_frame[frame_index])

            if frame_index in labels_by_frame:
                shot_type, label_pos = labels_by_frame[frame_index]
                color = _LABEL_COLORS.get(shot_type, (255, 255, 255))
                cv2.putText(
                    frame, shot_type.upper(), (label_pos[0] - 40, label_pos[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                )

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
    args = parser.parse_args(argv)

    handedness = parse_handedness_arg(args.handedness)
    ball_detector_kwargs = {}
    if args.ball_hsv_lower:
        ball_detector_kwargs["hsv_lower"] = tuple(int(v) for v in args.ball_hsv_lower.split(","))
    if args.ball_hsv_upper:
        ball_detector_kwargs["hsv_upper"] = tuple(int(v) for v in args.ball_hsv_upper.split(","))

    frame_poses_list, detections, hits, classifications = run_pipeline(
        args.video, handedness, max_players=args.max_players, ball_detector_kwargs=ball_detector_kwargs,
        detector=args.detector, tracknet_model_path=args.tracknet_model, tracknet_device=args.tracknet_device,
    )

    write_shot_log(classifications, args.output_log)
    frame_count = render_annotated_video(
        args.video, args.output_video, frame_poses_list, detections, classifications
    )

    counts = {"forehand": 0, "backhand": 0, "unclear": 0, NO_POSE_DATA: 0}
    for c in classifications:
        counts[c.shot_type] += 1

    print(f"Processed {frame_count} frames, detected {len(hits)} hit(s).")
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
