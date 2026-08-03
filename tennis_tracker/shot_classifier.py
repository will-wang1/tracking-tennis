"""THETIS-trained shot classifier: a learned alternative to classify.py's geometric heuristic.

classify.py's default classifier is a hand-written rule (which side of the
body the racket wrist swings to) -- fast and needs no training data, but
brittle: a two-handed backhand or a slice confuses it, and it only ever
outputs "forehand"/"backhand"/"unclear". This module instead learns shot
type from labeled examples, using the THETIS dataset
(https://github.com/THETIS-dataset/dataset) -- 12 stroke classes (forehand
flat/slice/open-stands, backhand/two-handed/slice/volley, service
flat/kick/slice, smash, forehand volley) performed by 55 subjects.

THETIS's "skeleton" data is itself distributed as rendered .avi videos, not
as numeric joint-position files, so there's nothing structured to parse out
of it directly. Instead, this runs tennis_tracker.pose's own pose detector
over THETIS's VIDEO_RGB/<action>/*.avi clips (one subject per clip, so
max_players=1) to extract landmark sequences, using each clip's containing
folder name as its label -- the same detector used at real inference time,
so training and inference see the same landmark representation.

Each clip's landmark sequence is normalized (centered on the hip midpoint,
scaled by torso length) so the same pose produces the same features
regardless of the subject's distance from or position in frame -- important
since THETIS's close-up indoor footage looks nothing like a wide broadcast
shot, and a raw-pixel representation wouldn't transfer between them. A fixed
number of frames are then sampled evenly across the sequence (padding short
ones by repeating the last frame) to get a fixed-size feature vector, fed
into a small feedforward classifier.

Workflow:
    1. shot_classifier.py extract-features <thetis_root> --output features.npz
    2. shot_classifier.py train features.npz --output shot_classifier.pt
    3. Pass --shot-classifier-model shot_classifier.pt to pipeline.py /
       physics_pipeline.py to use it in place of the geometric heuristic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tennis_tracker.pose import (
    DEFAULT_POSE_MODEL_VARIANT,
    LEFT_HIP,
    LEFT_SHOULDER,
    POSE_MODEL_VARIANTS,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    track_poses,
)

ARCHITECTURE_VERSION = "thetis-mlp-v1"
NUM_LANDMARKS = 33
DEFAULT_WINDOW_FRAMES = 16  # frames sampled per clip/window -> fixed feature size
DEFAULT_HIDDEN_SIZE = 256


def normalize_landmarks(landmarks_px: np.ndarray) -> np.ndarray:
    """Center on the hip midpoint and scale by torso length.

    This makes the same pose produce the same normalized coordinates
    regardless of the player's distance from or position in the camera
    frame -- essential since THETIS's close-up training footage and a real
    broadcast-angle match shot put the same pose at very different pixel
    scales. Torso length (shoulder-center to hip-center) is used as the
    scale reference rather than shoulder width, since shoulder width shrinks
    toward zero for a side-on stance -- common in tennis -- which would
    blow up the normalization.
    """
    hip_center = (landmarks_px[LEFT_HIP] + landmarks_px[RIGHT_HIP]) / 2
    shoulder_center = (landmarks_px[LEFT_SHOULDER] + landmarks_px[RIGHT_SHOULDER]) / 2
    scale = float(np.linalg.norm(shoulder_center - hip_center))
    if scale < 1e-6:
        scale = 1.0
    return (landmarks_px - hip_center) / scale


def sample_frame_indices(num_available: int, num_samples: int) -> list[int]:
    """Evenly-spaced frame indices spanning ``num_available`` frames.

    If there are fewer available frames than requested, the last frame is
    repeated to pad out to ``num_samples`` rather than erroring -- a short
    clip (or a hit near the start/end of a video) still produces a
    fixed-size feature vector.
    """
    if num_available <= 0:
        raise ValueError("num_available must be positive")
    if num_available >= num_samples:
        return [int(round(i)) for i in np.linspace(0, num_available - 1, num_samples)]
    return list(range(num_available)) + [num_available - 1] * (num_samples - num_available)


def extract_clip_features(landmarks_sequence: list[np.ndarray], num_samples: int = DEFAULT_WINDOW_FRAMES) -> np.ndarray:
    """Sample ``num_samples`` frames evenly across the sequence, normalize each, and flatten/concatenate."""
    if not landmarks_sequence:
        raise ValueError("landmarks_sequence is empty")
    indices = sample_frame_indices(len(landmarks_sequence), num_samples)
    frames = [normalize_landmarks(landmarks_sequence[i]).flatten() for i in indices]
    return np.concatenate(frames).astype(np.float32)


class ShotClassifierNet(nn.Module):
    """Small feedforward classifier over a flattened, sampled, normalized landmark sequence."""

    def __init__(self, num_classes: int, window_frames: int = DEFAULT_WINDOW_FRAMES, hidden_size: int = DEFAULT_HIDDEN_SIZE):
        super().__init__()
        self.window_frames = window_frames
        self.hidden_size = hidden_size
        self.class_names: list[str] = []  # set by the caller (train CLI) before saving
        input_size = window_frames * NUM_LANDMARKS * 2
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def discover_action_clips(
    dataset_root: str | Path, video_subdir: str = "VIDEO_RGB", video_glob: str = "*.avi"
) -> dict[str, list[Path]]:
    """Map action name -> clip paths, from THETIS's <root>/VIDEO_RGB/<action>/*.avi layout.

    Class names come from whatever subdirectories actually exist rather
    than a hardcoded list of THETIS's 12 category names, so this tolerates
    naming variants or a trimmed-down subset of classes.
    """
    root = Path(dataset_root) / video_subdir
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found -- expected a {video_subdir!r} folder under {dataset_root}")

    classes: dict[str, list[Path]] = {}
    for action_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        clips = sorted(action_dir.glob(video_glob))
        if clips:
            classes[action_dir.name] = clips
    return classes


def _extract_single_player_landmarks(clip_path: str | Path, **pose_kwargs) -> list[np.ndarray]:
    """Run pose detection over one clip, returning the (single) detected player's landmarks per frame.

    THETIS clips have exactly one subject, so max_players=1.
    """
    frame_poses_list = track_poses(clip_path, max_players=1, **pose_kwargs)
    return [fp.players[0].landmarks_px for fp in frame_poses_list if fp.players]


def build_feature_dataset(
    dataset_root: str | Path,
    output_path: str | Path,
    video_subdir: str = "VIDEO_RGB",
    video_glob: str = "*.avi",
    num_samples: int = DEFAULT_WINDOW_FRAMES,
    model_variant: str = DEFAULT_POSE_MODEL_VARIANT,
    pose_confidence: float = 0.5,
    verbose: bool = True,
) -> None:
    """Run pose detection over every clip and cache the resulting features to ``output_path`` (.npz).

    This is a separate, one-time step from train() so that retrying
    different training hyperparameters doesn't mean re-running pose
    detection over the whole dataset every time.
    """
    class_clips = discover_action_clips(dataset_root, video_subdir=video_subdir, video_glob=video_glob)
    class_names = sorted(class_clips)
    if not class_names:
        raise FileNotFoundError(f"No action subdirectories with clips found under {Path(dataset_root) / video_subdir}")

    total_clips = sum(len(clips) for clips in class_clips.values())
    processed = 0
    features: list[np.ndarray] = []
    labels: list[int] = []

    for class_index, class_name in enumerate(class_names):
        for clip_path in class_clips[class_name]:
            sequence = _extract_single_player_landmarks(
                clip_path, model_variant=model_variant, min_pose_detection_confidence=pose_confidence,
                min_pose_presence_confidence=pose_confidence, min_tracking_confidence=pose_confidence,
            )
            processed += 1
            if verbose:
                print(f"\rextracting pose features: clip {processed}/{total_clips}", end="", flush=True)
            if len(sequence) < 2:
                continue  # no person detected at all in this clip -- skip rather than poison training with junk
            features.append(extract_clip_features(sequence, num_samples=num_samples))
            labels.append(class_index)

    if verbose and total_clips:
        print()

    if not features:
        raise ValueError(
            "No clips yielded usable pose sequences -- check the dataset path and try a lower --pose-confidence"
        )

    np.savez(
        output_path,
        features=np.stack(features),
        labels=np.array(labels, dtype=np.int64),
        class_names=np.array(class_names),
        num_samples=num_samples,
    )


class ShotClipDataset(Dataset):
    """Wraps a .npz produced by build_feature_dataset() for PyTorch training."""

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path, allow_pickle=False)
        self.features = data["features"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)
        self.class_names = [str(c) for c in data["class_names"]]
        self.num_samples = int(data["num_samples"])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.features[idx]), int(self.labels[idx])


def train(
    model: ShotClassifierNet,
    dataset: Dataset,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    progress_callback=None,
) -> list[float]:
    """Trains ``model`` in place; returns the per-epoch mean loss history.

    Small enough (an MLP over a few thousand short clips) that this skips
    the AMP/cudnn-tuning machinery tracknet.py's train() needs for its much
    larger CNN -- plain full-precision training is already fast here.
    """
    model.to(device)
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    num_batches = len(loader)

    history = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch_index, (features, labels) in enumerate(loader, start=1):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            if progress_callback is not None:
                progress_callback(epoch, batch_index, num_batches, loss.item())
        history.append(float(np.mean(epoch_losses)))
        if checkpoint_path is not None:
            save_model(model, checkpoint_path)

    return history


def print_training_progress(epoch: int, batch_index: int, num_batches: int, batch_loss: float) -> None:
    print(f"\repoch {epoch + 1}  batch {batch_index}/{num_batches}  loss {batch_loss:.4f}", end="", flush=True)
    if batch_index == num_batches:
        print()


def save_model(model: ShotClassifierNet, path: str | Path) -> None:
    torch.save(
        {
            "architecture": ARCHITECTURE_VERSION,
            "class_names": model.class_names,
            "window_frames": model.window_frames,
            "hidden_size": model.hidden_size,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(path: str | Path, device: str = "cpu") -> ShotClassifierNet:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != ARCHITECTURE_VERSION:
        raise RuntimeError(
            f"{path} isn't a checkpoint for this shot-classifier architecture ({ARCHITECTURE_VERSION!r})."
        )
    model = ShotClassifierNet(
        num_classes=len(checkpoint["class_names"]), window_frames=checkpoint["window_frames"],
        hidden_size=checkpoint["hidden_size"],
    )
    model.class_names = list(checkpoint["class_names"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def classify_landmarks_sequence(
    model: ShotClassifierNet, landmarks_sequence: list[np.ndarray], device: str = "cpu"
) -> tuple[str, float]:
    """Classify one player's landmark sequence (e.g. frames around a hit) into one of the model's trained classes."""
    model.eval()
    features = extract_clip_features(landmarks_sequence, num_samples=model.window_frames)
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    class_index = int(torch.argmax(probs).item())
    return model.class_names[class_index], float(probs[class_index].item())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="THETIS-trained shot classifier (an alternative to classify.py's heuristic).")
    sub = parser.add_subparsers(dest="command", required=True)

    extract_p = sub.add_parser(
        "extract-features", help="Run pose detection over a THETIS-layout dataset and cache landmark features."
    )
    extract_p.add_argument("dataset_root", help="Directory containing VIDEO_RGB/<action>/*.avi (THETIS layout).")
    extract_p.add_argument("--output", required=True, help="Path to save the extracted features to (.npz).")
    extract_p.add_argument("--video-subdir", default="VIDEO_RGB")
    extract_p.add_argument("--video-glob", default="*.avi")
    extract_p.add_argument("--num-samples", type=int, default=DEFAULT_WINDOW_FRAMES)
    extract_p.add_argument("--model-variant", choices=POSE_MODEL_VARIANTS, default=DEFAULT_POSE_MODEL_VARIANT)
    extract_p.add_argument("--pose-confidence", type=float, default=0.5)

    train_p = sub.add_parser("train", help="Train the shot classifier on features from 'extract-features'.")
    train_p.add_argument("features_npz", help="Path to the .npz produced by 'extract-features'.")
    train_p.add_argument("--output", required=True, help="Path to save the trained model checkpoint.")
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    train_p.add_argument("--device", default="cpu")

    args = parser.parse_args(argv)

    if args.command == "extract-features":
        build_feature_dataset(
            args.dataset_root, args.output, video_subdir=args.video_subdir, video_glob=args.video_glob,
            num_samples=args.num_samples, model_variant=args.model_variant, pose_confidence=args.pose_confidence,
        )
        print(f"Saved extracted features to {args.output}")
        return 0

    if args.command == "train":
        dataset = ShotClipDataset(args.features_npz)
        print(f"Training on {len(dataset)} clips across {len(dataset.class_names)} classes: {dataset.class_names}")
        model = ShotClassifierNet(
            num_classes=len(dataset.class_names), window_frames=dataset.num_samples, hidden_size=args.hidden_size,
        )
        model.class_names = dataset.class_names
        history = train(
            model, dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device,
            checkpoint_path=args.output, progress_callback=print_training_progress,
        )
        print(f"Saved model to {args.output} (checkpointed every epoch); final loss {history[-1]:.4f}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
