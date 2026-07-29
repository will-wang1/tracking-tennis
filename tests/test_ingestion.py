import cv2
import numpy as np
import pytest

from tennis_tracker.ingestion import extract_frames, get_video_info, validate_video


def _write_synthetic_video(path, fps, width=64, height=48, num_frames=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_get_video_info_reads_metadata(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _write_synthetic_video(video_path, fps=60.0, num_frames=30)

    info = get_video_info(video_path)

    assert info.width == 64
    assert info.height == 48
    assert info.frame_count == 30
    assert info.fps == pytest.approx(60.0, rel=0.05)


def test_get_video_info_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_video_info(tmp_path / "does_not_exist.mp4")


def test_validate_video_flags_low_fps(tmp_path):
    video_path = tmp_path / "low_fps.mp4"
    _write_synthetic_video(video_path, fps=15.0, num_frames=15)

    info = get_video_info(video_path)
    warnings = validate_video(info)

    assert any("below the 30fps minimum" in w for w in warnings)


def test_validate_video_clean_for_good_footage(tmp_path):
    video_path = tmp_path / "good_fps.mp4"
    _write_synthetic_video(video_path, fps=60.0, num_frames=30)

    info = get_video_info(video_path)
    warnings = validate_video(info)

    assert warnings == []


def test_extract_frames_every_frame(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _write_synthetic_video(video_path, fps=30.0, num_frames=30)
    out_dir = tmp_path / "frames"

    timestamps = extract_frames(video_path, out_dir)

    saved = sorted(out_dir.glob("frame_*.jpg"))
    assert len(saved) == 30
    assert len(timestamps) == 30
    assert timestamps[0] == 0.0
    assert timestamps[1] == pytest.approx(1 / 30, rel=0.05)


def test_extract_frames_subsampled_to_target_fps(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _write_synthetic_video(video_path, fps=30.0, num_frames=30)
    out_dir = tmp_path / "frames"

    timestamps = extract_frames(video_path, out_dir, target_fps=10.0)

    saved = sorted(out_dir.glob("frame_*.jpg"))
    # step = round(30/10) = 3 -> every 3rd frame -> 10 frames
    assert len(saved) == 10
    assert len(timestamps) == 10
