import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Dataset  # noqa: E402

from tennis_tracker.shot_classifier import (  # noqa: E402
    ARCHITECTURE_VERSION,
    NUM_LANDMARKS,
    ShotClassifierNet,
    build_feature_dataset,
    classify_landmarks_sequence,
    discover_action_clips,
    extract_clip_features,
    load_model,
    normalize_landmarks,
    sample_frame_indices,
    save_model,
    train,
)


def _fake_landmarks(offset=0.0):
    """A (33, 2) array where hip/shoulder indices form a known, simple torso."""
    pts = np.zeros((33, 2), dtype=np.float32)
    pts[23] = [90, 200]  # LEFT_HIP
    pts[24] = [110, 200]  # RIGHT_HIP
    pts[11] = [90, 100]  # LEFT_SHOULDER
    pts[12] = [110, 100]  # RIGHT_SHOULDER
    pts += offset
    return pts


def test_normalize_landmarks_centers_on_hip_and_scales_by_torso_length():
    pts = _fake_landmarks()

    normalized = normalize_landmarks(pts)

    hip_center = (pts[23] + pts[24]) / 2
    shoulder_center = (pts[11] + pts[12]) / 2
    expected_scale = np.linalg.norm(shoulder_center - hip_center)
    np.testing.assert_allclose(normalized[23], (pts[23] - hip_center) / expected_scale)
    # Hip center itself normalizes to the origin.
    np.testing.assert_allclose(normalized[23] + normalized[24], [0, 0], atol=1e-5)


def test_normalize_landmarks_is_translation_invariant():
    pts_a = _fake_landmarks(offset=0.0)
    pts_b = _fake_landmarks(offset=500.0)  # same pose, shifted far away in the frame

    np.testing.assert_allclose(normalize_landmarks(pts_a), normalize_landmarks(pts_b), atol=1e-5)


def test_sample_frame_indices_pads_short_sequences_by_repeating_last():
    indices = sample_frame_indices(num_available=3, num_samples=5)

    assert indices == [0, 1, 2, 2, 2]


def test_sample_frame_indices_evenly_spans_long_sequences():
    indices = sample_frame_indices(num_available=100, num_samples=5)

    assert indices[0] == 0
    assert indices[-1] == 99
    assert len(indices) == 5
    assert indices == sorted(indices)


def test_extract_clip_features_shape_matches_samples_times_landmarks():
    sequence = [_fake_landmarks(offset=float(i)) for i in range(10)]

    features = extract_clip_features(sequence, num_samples=8)

    assert features.shape == (8 * NUM_LANDMARKS * 2,)


def test_extract_clip_features_rejects_empty_sequence():
    with pytest.raises(ValueError):
        extract_clip_features([])


def test_shot_classifier_net_forward_shape():
    model = ShotClassifierNet(num_classes=4, window_frames=8, hidden_size=32)
    x = torch.rand(3, 8 * NUM_LANDMARKS * 2)

    out = model(x)

    assert out.shape == (3, 4)


def test_discover_action_clips_groups_by_folder(tmp_path):
    root = tmp_path / "thetis"
    for action, n in [("forehand_flat", 2), ("backhand", 3)]:
        action_dir = root / "VIDEO_RGB" / action
        action_dir.mkdir(parents=True)
        for i in range(n):
            (action_dir / f"p1_{action}_{i}.avi").write_bytes(b"")

    classes = discover_action_clips(root)

    assert set(classes) == {"forehand_flat", "backhand"}
    assert len(classes["forehand_flat"]) == 2
    assert len(classes["backhand"]) == 3


def test_discover_action_clips_raises_when_video_subdir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_action_clips(tmp_path / "nonexistent")


def test_build_feature_dataset_skips_clips_with_no_detected_person(tmp_path, monkeypatch):
    root = tmp_path / "thetis"
    for action in ("forehand_flat", "backhand"):
        action_dir = root / "VIDEO_RGB" / action
        action_dir.mkdir(parents=True)
        (action_dir / "p1_clip_0.avi").write_bytes(b"")

    def fake_extract(clip_path, **kwargs):
        # Only the backhand clip has a detectable person; forehand's is empty
        # (e.g. mediapipe found nobody in it) and must be skipped, not crash
        # or poison the dataset with junk.
        if "backhand" in str(clip_path):
            return [_fake_landmarks(offset=float(i)) for i in range(5)]
        return []

    monkeypatch.setattr("tennis_tracker.shot_classifier._extract_single_player_landmarks", fake_extract)

    output_path = tmp_path / "features.npz"
    build_feature_dataset(root, output_path, num_samples=4, verbose=False)

    data = np.load(output_path)
    assert list(data["class_names"]) == ["backhand", "forehand_flat"]
    assert len(data["labels"]) == 1
    assert data["labels"][0] == 0  # "backhand" is class index 0 (sorted before "forehand_flat")


def test_build_feature_dataset_raises_when_all_clips_undetectable(tmp_path, monkeypatch):
    root = tmp_path / "thetis"
    action_dir = root / "VIDEO_RGB" / "smash"
    action_dir.mkdir(parents=True)
    (action_dir / "p1_smash_0.avi").write_bytes(b"")

    monkeypatch.setattr("tennis_tracker.shot_classifier._extract_single_player_landmarks", lambda *a, **k: [])

    with pytest.raises(ValueError, match="No clips yielded"):
        build_feature_dataset(root, tmp_path / "features.npz", verbose=False)


class _ToyFeatureDataset(Dataset):
    """Random-but-learnable toy dataset: label = which half of the feature vector has larger mean."""

    def __init__(self, num_samples=40, feature_dim=16):
        rng = np.random.default_rng(0)
        self.features = np.zeros((num_samples, feature_dim), dtype=np.float32)
        self.labels = np.zeros(num_samples, dtype=np.int64)
        for i in range(num_samples):
            label = i % 2
            base = rng.normal(0, 0.1, feature_dim).astype(np.float32)
            if label == 1:
                base[feature_dim // 2 :] += 3.0
            else:
                base[: feature_dim // 2] += 3.0
            self.features[i] = base
            self.labels[i] = label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), int(self.labels[idx])


def test_training_loop_reduces_loss_on_toy_dataset():
    # A 1-frame window so the model's expected input size (NUM_LANDMARKS * 2)
    # matches the toy dataset's feature_dim.
    model = ShotClassifierNet(num_classes=2, window_frames=1, hidden_size=32)
    dataset = _ToyFeatureDataset(feature_dim=NUM_LANDMARKS * 2)

    history = train(model, dataset, epochs=15, batch_size=8, lr=1e-2)

    assert len(history) == 15
    assert history[-1] < history[0]


def test_save_and_load_model_round_trip(tmp_path):
    model = ShotClassifierNet(num_classes=3, window_frames=4, hidden_size=16)
    model.class_names = ["a", "b", "c"]
    path = tmp_path / "shot_classifier.pt"

    save_model(model, path)
    loaded = load_model(path)

    assert loaded.class_names == ["a", "b", "c"]
    assert loaded.window_frames == 4
    x = torch.rand(1, 4 * NUM_LANDMARKS * 2)
    with torch.no_grad():
        out1 = model.eval()(x)
        out2 = loaded(x)
    torch.testing.assert_close(out1, out2)


def test_load_model_rejects_incompatible_checkpoint(tmp_path):
    path = tmp_path / "old_format.pt"
    torch.save({"some_other_format": True}, path)

    with pytest.raises(RuntimeError, match=ARCHITECTURE_VERSION):
        load_model(path)


def test_classify_landmarks_sequence_returns_a_trained_class_name():
    model = ShotClassifierNet(num_classes=2, window_frames=3, hidden_size=16)
    model.class_names = ["forehand_flat", "backhand"]
    sequence = [_fake_landmarks(offset=float(i)) for i in range(5)]

    shot_type, confidence = classify_landmarks_sequence(model, sequence)

    assert shot_type in model.class_names
    assert 0.0 <= confidence <= 1.0
