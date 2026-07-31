import csv

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis_tracker.tracknet import (  # noqa: E402
    ClipFrameDataset,
    TrackNet,
    build_dataset_from_root,
    find_clip_directories,
    generate_heatmap,
    heatmap_to_position,
    load_label_csv,
    load_model,
    run_tracknet_on_video,
    save_model,
    train,
)


def _write_clip(clip_dir, n_frames=8, width=128, height=96, header=None, invisible_frames=()):
    clip_dir.mkdir(parents=True, exist_ok=True)
    header = header or ["File Name", "Visibility Class", "X", "Y", "Trajectory Pattern"]
    rows = []
    for i in range(n_frames):
        file_name = f"{i:04d}.jpg"
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        x, y = 10 + i * 8, 20 + i * 4
        if i not in invisible_frames:
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        cv2.imwrite(str(clip_dir / file_name), frame)
        if i in invisible_frames:
            rows.append([file_name, 0, "", "", ""])
        else:
            rows.append([file_name, 1, x, y, 0])

    with (clip_dir / "Label.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_generate_heatmap_peaks_at_target_position():
    heatmap = generate_heatmap(x=30, y=10, height=20, width=64, sigma=3.0)

    assert heatmap.shape == (20, 64)
    peak_x, peak_y = heatmap_to_position(heatmap)
    assert peak_x == 30
    assert peak_y == 10
    assert heatmap.max() == pytest.approx(1.0, abs=1e-6)


def test_tracknet_forward_pass_shape():
    model = TrackNet(num_frames=3)
    model.eval()
    x = torch.rand(2, 9, 64, 96)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 3, 64, 96)
    assert output.min() >= 0.0 and output.max() <= 1.0


def test_load_label_csv_standard_headers(tmp_path):
    clip_dir = tmp_path / "Clip1"
    _write_clip(clip_dir, n_frames=5)

    labels = load_label_csv(clip_dir / "Label.csv")

    assert len(labels) == 5
    assert labels[0].file_name == "0000.jpg"
    assert labels[0].visibility_class == 1
    assert labels[0].x == 10.0 and labels[0].y == 20.0


def test_load_label_csv_alternate_headers(tmp_path):
    clip_dir = tmp_path / "Clip1"
    _write_clip(clip_dir, n_frames=3, header=["file name", "visibility", "x-coordinate", "y-coordinate", "status"])

    labels = load_label_csv(clip_dir / "Label.csv")

    assert len(labels) == 3
    assert labels[1].x == 18.0


def test_load_label_csv_zero_visibility_gives_none_position(tmp_path):
    clip_dir = tmp_path / "Clip1"
    _write_clip(clip_dir, n_frames=5, invisible_frames=(2,))

    labels = load_label_csv(clip_dir / "Label.csv")

    assert labels[2].visibility_class == 0
    assert labels[2].x is None and labels[2].y is None


def test_clip_frame_dataset_windows_and_zero_heatmap_for_invisible(tmp_path):
    clip_dir = tmp_path / "Clip1"
    _write_clip(clip_dir, n_frames=8, invisible_frames=(2,))

    dataset = ClipFrameDataset(clip_dir, num_frames=3, input_size=(64, 48))

    assert len(dataset) == 8 - 3 + 1
    frames, heatmaps = dataset[0]  # window covering frame indices 0,1,2 -> frame 2 is invisible
    assert frames.shape == (9, 48, 64)
    assert heatmaps.shape == (3, 48, 64)
    assert heatmaps[2].max() == 0.0  # invisible frame -> all-zero target
    assert heatmaps[0].max() > 0.0


def test_find_clip_directories_and_game_filter(tmp_path):
    for game in (1, 2):
        for clip in (1, 2):
            _write_clip(tmp_path / f"game{game}" / f"Clip{clip}", n_frames=6)

    all_dirs = find_clip_directories(tmp_path)
    assert len(all_dirs) == 4

    filtered = find_clip_directories(tmp_path, games=[1])
    assert len(filtered) == 2
    assert all("game1" in str(d) for d in filtered)

    capped = find_clip_directories(tmp_path, max_clips_per_game=1)
    assert len(capped) == 2


def test_build_dataset_from_root_spans_multiple_clips_without_crossing(tmp_path):
    _write_clip(tmp_path / "game1" / "Clip1", n_frames=6)
    _write_clip(tmp_path / "game1" / "Clip2", n_frames=6)

    dataset = build_dataset_from_root(tmp_path, num_frames=3, input_size=(64, 48))

    # Each 6-frame clip yields 6-3+1=4 windows; two clips -> 8, never 6+6-3+1=10
    # (which would happen if a window crossed the clip boundary).
    assert len(dataset) == 8


def test_training_loop_reduces_loss_on_toy_dataset(tmp_path):
    _write_clip(tmp_path / "game1" / "Clip1", n_frames=10)
    dataset = build_dataset_from_root(tmp_path, num_frames=3, input_size=(64, 48))

    model = TrackNet(num_frames=3)
    history = train(model, dataset, epochs=8, batch_size=2, lr=1e-2)

    assert len(history) == 8
    assert history[-1] < history[0]


def test_save_and_load_model_round_trip(tmp_path):
    model = TrackNet(num_frames=3)
    path = tmp_path / "model.pt"

    save_model(model, path)
    loaded = load_model(path)

    assert loaded.num_frames == 3
    x = torch.rand(1, 9, 64, 64)
    with torch.no_grad():
        out1 = model.eval()(x)
        out2 = loaded(x)
    torch.testing.assert_close(out1, out2)


def test_run_tracknet_on_video_yields_one_detection_per_frame(tmp_path):
    video_path = tmp_path / "clip.mp4"
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (96, 64))
    for i in range(6):
        out.write(np.full((64, 96, 3), i * 10, dtype=np.uint8))
    out.release()

    model = TrackNet(num_frames=3)
    detections = list(run_tracknet_on_video(video_path, model, input_size=(64, 48)))

    assert len(detections) == 6
    # First (num_frames - 1) frames lack enough context and must report "not found".
    assert detections[0].position is None
    assert detections[1].position is None
    for d in detections:
        assert d.frame_index in range(6)
