import csv

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis_tracker.tracknet import (  # noqa: E402
    ARCHITECTURE_VERSION,
    NUM_FRAMES,
    NUM_HEATMAP_CLASSES,
    ClipFrameDataset,
    TrackNet,
    build_dataset_from_root,
    find_clip_directories,
    generate_heatmap_classes,
    import_pretrained_checkpoint,
    load_label_csv,
    load_model,
    postprocess_heatmap,
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


def test_generate_heatmap_classes_peaks_at_target_position():
    heatmap = generate_heatmap_classes(
        x=30, y=10, orig_width=64, orig_height=20, target_width=64, target_height=20,
    )

    assert heatmap.shape == (20, 64)
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    assert x == 30
    assert y == 10
    assert heatmap.max() == 255


def test_generate_heatmap_classes_all_zero_when_no_position():
    heatmap = generate_heatmap_classes(
        x=None, y=None, orig_width=64, orig_height=20, target_width=64, target_height=20,
    )

    assert heatmap.max() == 0


def test_postprocess_heatmap_finds_circular_blob():
    # A filled circle of class-255 pixels on an otherwise-zero map, matching
    # what an argmax'd prediction confident about the ball would look like.
    class_map = np.zeros((60, 80), dtype=np.uint8)
    cv2.circle(class_map, (40, 30), 4, 255, -1)

    position, radius = postprocess_heatmap(class_map, scale_x=2.0, scale_y=2.0)

    assert position is not None
    x, y = position
    assert x == pytest.approx(80.0, abs=6.0)  # 40 * scale_x=2.0
    assert y == pytest.approx(60.0, abs=6.0)  # 30 * scale_y=2.0
    assert radius is not None


def test_postprocess_heatmap_returns_none_for_blank_map():
    class_map = np.zeros((60, 80), dtype=np.uint8)

    position, radius = postprocess_heatmap(class_map)

    assert position is None
    assert radius is None


def test_tracknet_forward_pass_shape():
    model = TrackNet(input_size=(96, 64))
    model.eval()
    x = torch.rand(2, NUM_FRAMES * 3, 64, 96)

    with torch.no_grad():
        logits = model(x)
        probs = model(x, testing=True)

    assert logits.shape == (2, NUM_HEATMAP_CLASSES, 64 * 96)
    assert probs.shape == logits.shape
    # testing=True applies softmax over the class dimension -> sums to 1 per pixel.
    assert torch.allclose(probs.sum(dim=1), torch.ones(2, 64 * 96), atol=1e-4)


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

    dataset = ClipFrameDataset(clip_dir, input_size=(64, 48))

    assert len(dataset) == 8 - (NUM_FRAMES - 1)
    frames, target = dataset[0]  # window covering frame indices 0,1,2 -> current (idx 2) is invisible
    assert frames.shape == (NUM_FRAMES * 3, 48, 64)
    assert target.shape == (48 * 64,)
    assert target.max() == 0  # invisible current frame -> all-zero (class 0) target

    frames, target = dataset[1]  # window covering 1,2,3 -> current (idx 3) is visible
    assert target.max() > 0


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

    dataset = build_dataset_from_root(tmp_path, input_size=(64, 48))

    # Each 6-frame clip yields 6-2=4 windows; two clips -> 8, never 6+6-2=10
    # (which would happen if a window crossed the clip boundary).
    assert len(dataset) == 8


def test_training_loop_reduces_loss_on_toy_dataset(tmp_path):
    _write_clip(tmp_path / "game1" / "Clip1", n_frames=10)
    dataset = build_dataset_from_root(tmp_path, input_size=(64, 48))

    model = TrackNet(input_size=(64, 48))
    history = train(model, dataset, epochs=8, batch_size=2, lr=1e-2)

    assert len(history) == 8
    assert history[-1] < history[0]


def test_train_checkpoints_after_every_epoch_not_just_at_the_end(tmp_path):
    _write_clip(tmp_path / "game1" / "Clip1", n_frames=10)
    dataset = build_dataset_from_root(tmp_path, input_size=(64, 48))
    model = TrackNet(input_size=(64, 48))
    checkpoint_path = tmp_path / "checkpoint.pt"

    seen_checkpoint_before_epoch_2 = []

    def progress_callback(epoch, batch_index, num_batches, batch_loss):
        if epoch == 1 and batch_index == 1:
            # Epoch 0 has fully finished by the time epoch 1 starts, so if
            # checkpointing happens every epoch (not only after train()
            # returns), the file must already exist here.
            seen_checkpoint_before_epoch_2.append(checkpoint_path.exists())

    train(
        model, dataset, epochs=2, batch_size=2, checkpoint_path=checkpoint_path,
        progress_callback=progress_callback,
    )

    assert seen_checkpoint_before_epoch_2 == [True]
    assert load_model(checkpoint_path).input_size == (64, 48)


def test_train_calls_progress_callback_once_per_batch(tmp_path):
    _write_clip(tmp_path / "game1" / "Clip1", n_frames=10)
    dataset = build_dataset_from_root(tmp_path, input_size=(64, 48))
    model = TrackNet(input_size=(64, 48))

    calls = []
    train(model, dataset, epochs=2, batch_size=2, progress_callback=lambda *args: calls.append(args))

    num_batches_per_epoch = -(-len(dataset) // 2)  # ceil division, matches DataLoader's last partial batch
    assert len(calls) == 2 * num_batches_per_epoch
    # 1-indexed batch numbers within each epoch, epoch itself 0-indexed.
    assert calls[0][0] == 0 and calls[0][1] == 1
    assert calls[-1][0] == 1 and calls[-1][1] == num_batches_per_epoch


def test_save_and_load_model_round_trip(tmp_path):
    model = TrackNet(input_size=(64, 64))
    path = tmp_path / "model.pt"

    save_model(model, path)
    loaded = load_model(path)

    assert loaded.input_size == (64, 64)
    x = torch.rand(1, NUM_FRAMES * 3, 64, 64)
    with torch.no_grad():
        out1 = model.eval()(x)
        out2 = loaded(x)
    torch.testing.assert_close(out1, out2)


def test_load_model_rejects_incompatible_checkpoint(tmp_path):
    path = tmp_path / "old_format.pt"
    torch.save({"num_frames": 3, "state_dict": TrackNet().state_dict()}, path)

    with pytest.raises(RuntimeError, match=ARCHITECTURE_VERSION):
        load_model(path)


def test_run_tracknet_on_video_yields_one_detection_per_frame(tmp_path):
    video_path = tmp_path / "clip.mp4"
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (96, 64))
    for i in range(6):
        out.write(np.full((64, 96, 3), i * 10, dtype=np.uint8))
    out.release()

    model = TrackNet(input_size=(64, 48))
    detections = list(run_tracknet_on_video(video_path, model))

    assert len(detections) == 6
    # First (NUM_FRAMES - 1) frames lack enough context and must report "not found".
    assert detections[0].position is None
    assert detections[1].position is None
    for d in detections:
        assert d.frame_index in range(6)


def test_import_pretrained_checkpoint_wraps_a_bare_state_dict(tmp_path):
    # Stand-in for yastrebksv/TrackNet's own released checkpoint, which is
    # just a bare state_dict (no architecture tag or input_size) -- since
    # this module's TrackNet mirrors their BallTrackerNet's layer names
    # exactly, our own model's state_dict works as a same-shape stand-in to
    # test the wrapping/loading logic without needing their actual weights.
    source_model = TrackNet(input_size=(64, 48))
    raw_path = tmp_path / "raw_state_dict.pt"
    torch.save(source_model.state_dict(), raw_path)

    output_path = tmp_path / "wrapped.pt"
    import_pretrained_checkpoint(raw_path, output_path, input_size=(64, 48))

    loaded = load_model(output_path)
    assert loaded.input_size == (64, 48)
    x = torch.rand(1, NUM_FRAMES * 3, 48, 64)
    with torch.no_grad():
        torch.testing.assert_close(source_model.eval()(x), loaded(x))


def test_import_pretrained_checkpoint_reports_key_mismatch_clearly(tmp_path):
    raw_path = tmp_path / "mismatched.pt"
    torch.save({"totally_unrelated_key": torch.zeros(3)}, raw_path)

    with pytest.raises(RuntimeError, match="doesn't match"):
        import_pretrained_checkpoint(raw_path, tmp_path / "out.pt")
