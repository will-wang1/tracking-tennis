"""Phase 1: video loading, frame extraction, and source-quality validation.

Ball tracking is sensitive to frame rate, so this module's job is to get frames
out of a video file reliably and tell the caller up front whether the source
footage even meets the minimum bar (30fps, 50/60fps preferred) before any
downstream tracking work is attempted.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

MIN_RECOMMENDED_FPS = 30.0
PREFERRED_FPS = 50.0


@dataclass
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_sec(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def get_video_info(video_path: str | Path) -> VideoInfo:
    """Open a video file just long enough to read its metadata."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video (unsupported codec/container?): {path}")

    try:
        info = VideoInfo(
            path=path,
            fps=cap.get(cv2.CAP_PROP_FPS),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()

    return info


def validate_video(info: VideoInfo) -> list[str]:
    """Return human-readable warnings about footage quality. Empty list means clean."""
    warnings: list[str] = []

    if info.fps <= 0:
        warnings.append(
            "Could not read a valid FPS from this file (got "
            f"{info.fps}). The container may be missing metadata; re-encode before tracking."
        )
    elif info.fps < MIN_RECOMMENDED_FPS:
        warnings.append(
            f"Source FPS is {info.fps:.2f}, below the {MIN_RECOMMENDED_FPS:.0f}fps minimum. "
            "Ball tracking (especially on fast serves) will be unreliable at this frame rate."
        )
    elif info.fps < PREFERRED_FPS:
        warnings.append(
            f"Source FPS is {info.fps:.2f}, which clears the {MIN_RECOMMENDED_FPS:.0f}fps "
            f"minimum but is below the {PREFERRED_FPS:.0f}fps preferred for fast serves."
        )

    if info.width <= 0 or info.height <= 0:
        warnings.append(f"Could not read a valid resolution (got {info.width}x{info.height}).")

    if info.frame_count <= 0:
        warnings.append(
            "Frame count reported as 0 or unknown; some containers under-report this — "
            "frame extraction will still work by reading until EOF."
        )

    return warnings


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    target_fps: float | None = None,
    image_ext: str = "jpg",
) -> list[float]:
    """Extract frames from ``video_path`` into ``output_dir``.

    If ``target_fps`` is None (or >= the source FPS), every frame is saved at
    the source rate. Otherwise frames are subsampled to approximate
    ``target_fps`` by keeping every Nth source frame.

    Returns the timestamp (seconds, from video start) of each saved frame.
    """
    path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video (unsupported codec/container?): {path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if target_fps and source_fps > 0 and target_fps < source_fps:
        step = max(1, round(source_fps / target_fps))
    else:
        step = 1

    timestamps: list[float] = []
    try:
        source_index = 0
        saved_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if source_index % step == 0:
                out_path = out_dir / f"frame_{saved_index:06d}.{image_ext}"
                cv2.imwrite(str(out_path), frame)
                timestamp = source_index / source_fps if source_fps > 0 else 0.0
                timestamps.append(timestamp)
                saved_index += 1
            source_index += 1
    finally:
        cap.release()

    return timestamps


def _print_info(info: VideoInfo, warnings: list[str]) -> None:
    print(f"path:        {info.path}")
    print(f"resolution:  {info.width}x{info.height}")
    print(f"fps:         {info.fps:.3f}")
    print(f"frame_count: {info.frame_count}")
    print(f"duration:    {info.duration_sec:.2f}s")
    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nno warnings: footage meets frame-rate/resolution requirements.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1: video ingestion for the tennis shot tracker.")
    sub = parser.add_subparsers(dest="command", required=True)

    info_p = sub.add_parser("info", help="Print video metadata and quality warnings.")
    info_p.add_argument("video", help="Path to the source video file.")

    extract_p = sub.add_parser("extract", help="Extract frames from a video.")
    extract_p.add_argument("video", help="Path to the source video file.")
    extract_p.add_argument("output_dir", help="Directory to write extracted frames into.")
    extract_p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target extraction FPS. Omit to extract every source frame.",
    )
    extract_p.add_argument(
        "--ext",
        default="jpg",
        help="Image format for saved frames (default: jpg).",
    )

    args = parser.parse_args(argv)

    if args.command == "info":
        info = get_video_info(args.video)
        warnings = validate_video(info)
        _print_info(info, warnings)
        return 1 if warnings else 0

    if args.command == "extract":
        info = get_video_info(args.video)
        warnings = validate_video(info)
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
        timestamps = extract_frames(args.video, args.output_dir, target_fps=args.fps, image_ext=args.ext)
        print(f"Extracted {len(timestamps)} frames to {args.output_dir}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
