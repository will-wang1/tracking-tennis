import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis_tracker.tracknet import (  # noqa: E402
    BallHeatmapDataset,
    TrackNet,
    generate_heatmap,
    heatmap_to_position,
    train,
)


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
    x = torch.rand(2, 9, 64, 96)  # batch=2, 3 frames * 3 channels, small H/W for a fast test

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 3, 64, 96)
    assert output.min() >= 0.0 and output.max() <= 1.0


def _write_synthetic_labeled_video(path, n_frames=8, width=128, height=96):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    labels = {}
    for i in range(n_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        x, y = 10 + i * 10, 20 + i * 5
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        out.write(frame)
        labels[i] = (float(x), float(y))
    out.release()
    return labels


def test_dataset_produces_correctly_shaped_windows(tmp_path):
    video_path = tmp_path / "labeled.mp4"
    labels = _write_synthetic_labeled_video(video_path, n_frames=8)

    dataset = BallHeatmapDataset(video_path, labels, num_frames=3, input_size=(64, 48))

    assert len(dataset) == 8 - 3 + 1
    frames, heatmaps = dataset[0]
    assert frames.shape == (9, 48, 64)
    assert heatmaps.shape == (3, 48, 64)


def test_dataset_skips_windows_with_missing_labels(tmp_path):
    video_path = tmp_path / "labeled.mp4"
    labels = _write_synthetic_labeled_video(video_path, n_frames=8)
    del labels[4]  # simulate an unlabeled frame

    dataset = BallHeatmapDataset(video_path, labels, num_frames=3, input_size=(64, 48))

    # No 3-frame window may include frame 4.
    for start in dataset.window_starts:
        assert 4 not in (start, start + 1, start + 2)


def test_training_loop_reduces_loss_on_toy_dataset(tmp_path):
    video_path = tmp_path / "labeled.mp4"
    labels = _write_synthetic_labeled_video(video_path, n_frames=10)
    dataset = BallHeatmapDataset(video_path, labels, num_frames=3, input_size=(64, 48))

    model = TrackNet(num_frames=3)
    history = train(model, dataset, epochs=8, batch_size=2, lr=1e-2)

    assert len(history) == 8
    assert history[-1] < history[0]
