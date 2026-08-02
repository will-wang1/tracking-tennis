"""Phase G: physics-based contact time/speed, usable today without TrackNet.

This combines pieces that already exist and are independently validated:
the classical ball detector (ball.py), the Kalman-based hit detector
(trajectory.py) to find *approximate* hit frames, the physics-constrained
contact fit (contact_fit.py) to refine those into precise contact time/
position/speed given a camera calibration (calibration.py), and the
pose-based forehand/backhand classifier (classify.py). TrackNet (Phase E)
isn't needed for any of this — it would eventually replace ball.py's
detections with more reliable ones, but the physics fitting downstream
works the same either way.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from tennis_tracker.ball import BallDetection
from tennis_tracker.calibration import CameraCalibration, calibrate_camera, load_correspondences_json
from tennis_tracker.classify import Handedness, classify_pose, nearest_player
from tennis_tracker.contact_fit import ContactEvent, fit_contact_event
from tennis_tracker.physics import TennisBallParams
from tennis_tracker.pipeline import get_ball_detections, parse_handedness_arg
from tennis_tracker.pose import track_poses
from tennis_tracker.trajectory import HitEvent, detect_hits, smooth_trajectory

MPS_TO_KMH = 3.6

DEFAULT_WINDOW_SIZE = 15  # frames of ball detections to fit on each side of a hit
DEFAULT_GAP = 2  # frames adjacent to the hit excluded (motion blur/occlusion at contact)
DEFAULT_MIN_OBSERVATIONS = 6  # fit_trajectory needs >=6 to fit 6 parameters
DEFAULT_GAP_WARNING_M = 2.0  # position_gap_m above this marks the fit as low-confidence


@dataclass
class PhysicsShotResult:
    hit_frame_index: int
    hit_timestamp: float  # from the approximate 2D Kalman-based hit detector
    player_id: int | None
    shot_type: str | None
    classification_confidence: float | None
    contact_event: ContactEvent | None  # None if the physics fit was skipped/failed
    fit_note: str  # "ok" | "low_confidence" | "insufficient_observations" | "fit_failed"

    @property
    def incoming_speed_kmh(self) -> float | None:
        return self.contact_event.incoming_speed * MPS_TO_KMH if self.contact_event else None

    @property
    def outgoing_speed_kmh(self) -> float | None:
        return self.contact_event.outgoing_speed * MPS_TO_KMH if self.contact_event else None


def _build_hit_window(
    detections: list[BallDetection],
    hit: HitEvent,
    window_size: int,
    gap: int,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    pre_obs = [
        (d.timestamp, d.position[0], d.position[1])
        for d in detections
        if d.position is not None and hit.frame_index - window_size <= d.frame_index < hit.frame_index - gap
    ]
    post_obs = [
        (d.timestamp, d.position[0], d.position[1])
        for d in detections
        if d.position is not None and hit.frame_index + gap < d.frame_index <= hit.frame_index + window_size
    ]
    return pre_obs, post_obs


def run_physics_pipeline(
    video_path: str | Path,
    calibration: CameraCalibration,
    handedness: dict[int, Handedness],
    max_players: int = 2,
    ball_detector_kwargs: dict | None = None,
    physics_params: TennisBallParams | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    gap: int = DEFAULT_GAP,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    gap_warning_m: float = DEFAULT_GAP_WARNING_M,
    detector: str = "classical",
    tracknet_model_path: str | Path | None = None,
    tracknet_device: str = "cpu",
) -> list[PhysicsShotResult]:
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

    results: list[PhysicsShotResult] = []
    for hit in hits:
        player_id = None
        shot_type = None
        confidence = None
        frame_poses = poses_by_frame.get(hit.frame_index)
        if frame_poses is not None:
            player = nearest_player(hit.position, frame_poses)
            if player is not None:
                player_id = player.player_id
                shot_type, confidence = classify_pose(player, handedness.get(player_id, "right"))

        pre_obs, post_obs = _build_hit_window(detections, hit, window_size, gap)
        if len(pre_obs) < min_observations or len(post_obs) < min_observations:
            results.append(
                PhysicsShotResult(
                    hit_frame_index=hit.frame_index, hit_timestamp=hit.timestamp, player_id=player_id,
                    shot_type=shot_type, classification_confidence=confidence,
                    contact_event=None, fit_note="insufficient_observations",
                )
            )
            continue

        try:
            event = fit_contact_event(pre_obs, post_obs, calibration, physics_params)
        except Exception:
            results.append(
                PhysicsShotResult(
                    hit_frame_index=hit.frame_index, hit_timestamp=hit.timestamp, player_id=player_id,
                    shot_type=shot_type, classification_confidence=confidence,
                    contact_event=None, fit_note="fit_failed",
                )
            )
            continue

        fit_note = "ok" if event.position_gap_m <= gap_warning_m else "low_confidence"
        results.append(
            PhysicsShotResult(
                hit_frame_index=hit.frame_index, hit_timestamp=hit.timestamp, player_id=player_id,
                shot_type=shot_type, classification_confidence=confidence,
                contact_event=event, fit_note=fit_note,
            )
        )

    return results


def write_physics_shot_log(results: list[PhysicsShotResult], output_path: str | Path) -> None:
    path = Path(output_path)
    rows = []
    for r in results:
        row = {
            "hit_frame_index": r.hit_frame_index,
            "hit_timestamp": round(r.hit_timestamp, 3),
            "player_id": r.player_id,
            "shot_type": r.shot_type,
            "classification_confidence": round(r.classification_confidence, 4) if r.classification_confidence is not None else None,
            "fit_note": r.fit_note,
            "contact_time": round(r.contact_event.contact_time, 4) if r.contact_event else None,
            "contact_position_m": list(r.contact_event.contact_position) if r.contact_event else None,
            "incoming_speed_kmh": round(r.incoming_speed_kmh, 2) if r.incoming_speed_kmh is not None else None,
            "incoming_speed_std_kmh": round(r.contact_event.incoming_speed_std * MPS_TO_KMH, 2) if r.contact_event else None,
            "outgoing_speed_kmh": round(r.outgoing_speed_kmh, 2) if r.outgoing_speed_kmh is not None else None,
            "outgoing_speed_std_kmh": round(r.contact_event.outgoing_speed_std * MPS_TO_KMH, 2) if r.contact_event else None,
            "position_gap_m": round(r.contact_event.position_gap_m, 4) if r.contact_event else None,
        }
        rows.append(row)

    if path.suffix.lower() == ".csv":
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    else:
        with path.open("w") as f:
            json.dump({"shots": rows}, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase G: physics-based contact time/speed pipeline.")
    parser.add_argument("video", help="Path to the source video file.")
    parser.add_argument("--calibration-json", required=True, help="Court landmark correspondences JSON.")
    parser.add_argument("--output-log", required=True, help="Path to write the shot log to (.json or .csv).")
    parser.add_argument("--handedness", default="", help='e.g. "0:right,1:left". Defaults to right.')
    parser.add_argument("--max-players", type=int, default=2)
    parser.add_argument(
        "--detector", choices=["classical", "tracknet"], default="classical",
        help='Ball detector to use. "tracknet" requires --tracknet-model.',
    )
    parser.add_argument("--tracknet-model", default=None, help="Trained TrackNet checkpoint (from tracknet.py train).")
    parser.add_argument("--tracknet-device", default="cpu", help='"cpu" or "cuda", for the tracknet detector.')
    args = parser.parse_args(argv)

    handedness = parse_handedness_arg(args.handedness)

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    correspondences = load_correspondences_json(args.calibration_json)
    calibration = calibrate_camera(correspondences, (width, height))
    print(f"Calibration reprojection error: {calibration.reprojection_error_px:.2f} px")

    results = run_physics_pipeline(
        args.video, calibration, handedness, max_players=args.max_players,
        detector=args.detector, tracknet_model_path=args.tracknet_model, tracknet_device=args.tracknet_device,
    )
    write_physics_shot_log(results, args.output_log)

    for r in results:
        speed_str = (
            f"in={r.incoming_speed_kmh:.1f}km/h out={r.outgoing_speed_kmh:.1f}km/h"
            if r.contact_event
            else "(no physics fit)"
        )
        shot_type_str = r.shot_type or "unknown"
        print(f"frame {r.hit_frame_index:5d}  player {r.player_id}  {shot_type_str:>9}  {speed_str}  [{r.fit_note}]")

    print(f"Shot log: {args.output_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
