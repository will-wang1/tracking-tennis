"""Phase E: TrackNet-style temporal CNN for ball detection.

This is a from-scratch, simplified reimplementation of the TrackNetV2-style
architecture — a VGG-like encoder/decoder that takes several consecutive
frames stacked on the channel axis and outputs one heatmap per input frame.
Using multiple frames at once (rather than one frame in isolation) is what
lets it see through motion blur: a fast-moving ball leaves a trail of
appearance across frames that a single-frame detector can't use.

Training data is the standard TrackNet tennis dataset layout: a root
directory of game1/, game2/, ... folders, each containing Clip1/, Clip2/,
... folders, each holding sequential frame images (0000.jpg, 0001.jpg, ...)
plus a Label.csv with one row per frame: file name, visibility class (0=ball
not in frame, 1=clearly visible, 2=hard to see, 3=occluded), x/y pixel
coordinate, and trajectory pattern (0=flying, 1=hit, 2=bouncing — recorded
here but not consumed by this model; hit/bounce detection is handled
separately by tennis_tracker.trajectory and tennis_tracker.contact_fit).

Windows never cross a Clip boundary, since each clip is an independent
rally — frame 0000 of Clip2 has nothing to do with the last frame of Clip1.

There are no pretrained weights bundled here — see the project's earlier
discussion for why: third-party pretrained checkpoints for this exact task
are typically distributed as ad-hoc pickled files of uncertain provenance,
a real code-execution risk to load blindly. Once you've trained a model
with train() below, run_tracknet_on_video() analyzes a real video the same
way tennis_tracker.ball.detect_video() does — same per-frame generator
contract — so a trained model is a drop-in alternative to the classical
detector.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from tennis_tracker.ball import BallDetection

DEFAULT_NUM_FRAMES = 3
DEFAULT_INPUT_SIZE = (512, 288)  # (width, height), matches the published TrackNetV2 setup
DEFAULT_HEATMAP_SIGMA = 5.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

_FILE_NAME_ALIASES = ["file name", "filename", "file_name", "file"]
_VISIBILITY_ALIASES = ["visibility class", "visibility", "vc", "visibility_class"]
_X_ALIASES = ["x-coordinate", "x coordinate", "x"]
_Y_ALIASES = ["y-coordinate", "y coordinate", "y"]
_TRAJECTORY_ALIASES = ["trajectory pattern", "trajectory", "status", "trajectory_pattern"]


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TrackNet(nn.Module):
    """VGG-style encoder/decoder producing one heatmap per input frame."""

    def __init__(self, num_frames: int = DEFAULT_NUM_FRAMES, channels_per_frame: int = 3):
        super().__init__()
        self.num_frames = num_frames
        in_channels = num_frames * channels_per_frame

        self.enc1 = nn.Sequential(_ConvBlock(in_channels, 64), _ConvBlock(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(_ConvBlock(64, 128), _ConvBlock(128, 128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(_ConvBlock(128, 256), _ConvBlock(256, 256), _ConvBlock(256, 256))
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(_ConvBlock(256, 512), _ConvBlock(512, 512), _ConvBlock(512, 512))

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = nn.Sequential(_ConvBlock(512, 256), _ConvBlock(256, 256), _ConvBlock(256, 256))
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = nn.Sequential(_ConvBlock(256, 128), _ConvBlock(128, 128))
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = nn.Sequential(_ConvBlock(128, 64), _ConvBlock(64, 64))

        self.output_conv = nn.Conv2d(64, num_frames, kernel_size=1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw pre-sigmoid output. Training uses this + BCEWithLogitsLoss, which is
        numerically stable and autocast-safe, unlike computing sigmoid then BCELoss
        separately (fp16 can round a sigmoid output infinitesimally outside [0, 1],
        which CUDA's BCELoss kernel then rejects with a hard assertion failure)."""
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.bottleneck(x)
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return self.output_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


def generate_heatmap(x: float, y: float, height: int, width: int, sigma: float = DEFAULT_HEATMAP_SIGMA) -> np.ndarray:
    """A 2D Gaussian heatmap peaking at (x, y), used as the training target for one frame."""
    yy, xx = np.mgrid[0:height, 0:width]
    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return heatmap.astype(np.float32)


def heatmap_to_position(heatmap: np.ndarray) -> tuple[float, float]:
    """Argmax of a predicted heatmap, as an (x, y) pixel position in that heatmap's own resolution."""
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return float(x), float(y)


@dataclass
class FrameLabel:
    file_name: str
    visibility_class: int  # 0=not in frame, 1=clear, 2=hard to see, 3=occluded
    x: float | None  # None when visibility_class == 0 (no ball position to give)
    y: float | None
    trajectory_pattern: int | None  # 0=flying, 1=hit, 2=bouncing; not consumed by this model


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace("-", " ").replace("_", " ")


def _match_column(fieldnames: list[str], aliases: list[str], required: bool = True) -> str | None:
    normalized = {_normalize_header(f): f for f in fieldnames}
    for alias in aliases:
        match = normalized.get(_normalize_header(alias))
        if match is not None:
            return match
    if required:
        raise ValueError(f"Could not find a column matching any of {aliases} in header {fieldnames}")
    return None


def load_label_csv(path: str | Path) -> list[FrameLabel]:
    """Parse a Label.csv from the TrackNet dataset. Tolerant of minor header-naming variants."""
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        file_col = _match_column(fieldnames, _FILE_NAME_ALIASES)
        vis_col = _match_column(fieldnames, _VISIBILITY_ALIASES)
        x_col = _match_column(fieldnames, _X_ALIASES)
        y_col = _match_column(fieldnames, _Y_ALIASES)
        traj_col = _match_column(fieldnames, _TRAJECTORY_ALIASES, required=False)

        labels = []
        for row in reader:
            visibility = int(row[vis_col])
            x_raw, y_raw = (row[x_col] or "").strip(), (row[y_col] or "").strip()
            if visibility == 0 or not x_raw or not y_raw:
                x, y = None, None
            else:
                x, y = float(x_raw), float(y_raw)
            trajectory = None
            if traj_col is not None and row[traj_col] not in (None, ""):
                trajectory = int(row[traj_col])
            labels.append(
                FrameLabel(file_name=row[file_col], visibility_class=visibility, x=x, y=y, trajectory_pattern=trajectory)
            )
    return labels


class ClipFrameDataset(Dataset):
    """One Clip*/ folder: sequential frame images + Label.csv, from the TrackNet dataset.

    Windows of ``num_frames`` consecutive frames are built entirely within
    this clip. Frames labeled visibility_class=0 (ball not in frame) get an
    all-zero heatmap target rather than being skipped — training the model
    to predict "no ball here" is itself useful signal, not something to
    discard.
    """

    def __init__(
        self,
        clip_dir: str | Path,
        num_frames: int = DEFAULT_NUM_FRAMES,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        heatmap_sigma: float = DEFAULT_HEATMAP_SIGMA,
    ):
        self.clip_dir = Path(clip_dir)
        self.num_frames = num_frames
        self.input_size = input_size
        self.heatmap_sigma = heatmap_sigma

        label_path = self.clip_dir / "Label.csv"
        if not label_path.exists():
            candidates = list(self.clip_dir.glob("*.csv"))
            if not candidates:
                raise FileNotFoundError(f"No Label.csv found in {clip_dir}")
            label_path = candidates[0]
        self.labels = load_label_csv(label_path)
        if not self.labels:
            raise ValueError(f"No labeled frames found in {label_path}")

        first_frame_path = self.clip_dir / self.labels[0].file_name
        first_frame = cv2.imread(str(first_frame_path))
        if first_frame is None:
            raise RuntimeError(f"Could not read frame image: {first_frame_path}")
        self.orig_height, self.orig_width = first_frame.shape[:2]

    def __len__(self) -> int:
        return max(0, len(self.labels) - self.num_frames + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        width, height = self.input_size
        scale_x = width / self.orig_width
        scale_y = height / self.orig_height

        frame_tensors = []
        heatmaps = []
        for i in range(self.num_frames):
            label = self.labels[idx + i]
            frame_path = self.clip_dir / label.file_name
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read frame image: {frame_path}")
            frame = cv2.resize(frame, (width, height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frame_tensors.append(frame_rgb.transpose(2, 0, 1))  # HWC -> CHW

            if label.x is None or label.y is None:
                heatmaps.append(np.zeros((height, width), dtype=np.float32))
            else:
                heatmaps.append(
                    generate_heatmap(label.x * scale_x, label.y * scale_y, height, width, self.heatmap_sigma)
                )

        stacked_frames = np.concatenate(frame_tensors, axis=0)  # (num_frames*3, H, W)
        stacked_heatmaps = np.stack(heatmaps, axis=0)  # (num_frames, H, W)
        return torch.from_numpy(stacked_frames), torch.from_numpy(stacked_heatmaps)


def find_clip_directories(
    dataset_root: str | Path, games: list[int] | None = None, max_clips_per_game: int | None = None
) -> list[Path]:
    """Discover Clip*/ directories under dataset_root/game*/, per the TrackNet dataset layout."""
    root = Path(dataset_root)
    game_dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name.lower().startswith("game")),
        key=lambda d: d.name.lower(),
    )
    if games is not None:
        wanted = {f"game{g}" for g in games}
        game_dirs = [d for d in game_dirs if d.name.lower() in wanted]

    clip_dirs = []
    for game_dir in game_dirs:
        clips = sorted(
            (d for d in game_dir.iterdir() if d.is_dir() and d.name.lower().startswith("clip")),
            key=lambda d: d.name.lower(),
        )
        if max_clips_per_game is not None:
            clips = clips[:max_clips_per_game]
        clip_dirs.extend(clips)
    return clip_dirs


def build_dataset_from_root(
    dataset_root: str | Path,
    games: list[int] | None = None,
    max_clips_per_game: int | None = None,
    num_frames: int = DEFAULT_NUM_FRAMES,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    heatmap_sigma: float = DEFAULT_HEATMAP_SIGMA,
) -> ConcatDataset:
    """Builds one combined Dataset spanning every Clip*/ folder found under dataset_root/game*/.

    ``games`` restricts to specific game numbers (e.g. [1, 2]); omit for all
    10. ``max_clips_per_game`` caps clips per game — handy for a quick smoke
    run rather than loading all ~20k frames.
    """
    clip_dirs = find_clip_directories(dataset_root, games=games, max_clips_per_game=max_clips_per_game)
    if not clip_dirs:
        raise FileNotFoundError(f"No game*/Clip* directories found under {dataset_root}")

    clip_datasets = [
        ClipFrameDataset(d, num_frames=num_frames, input_size=input_size, heatmap_sigma=heatmap_sigma)
        for d in clip_dirs
    ]
    clip_datasets = [d for d in clip_datasets if len(d) > 0]
    if not clip_datasets:
        raise ValueError(f"Found clip directories under {dataset_root} but none had enough frames for a window")
    return ConcatDataset(clip_datasets)


def train(
    model: TrackNet,
    dataset: Dataset,
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 1e-3,
    device: str = "cpu",
    num_workers: int = 0,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    checkpoint_path: str | Path | None = None,
    progress_callback=None,
) -> list[float]:
    """Trains ``model`` in place; returns the per-epoch mean loss history.

    ``progress_callback``, if given, is called after every batch as
    ``progress_callback(epoch, batch_index, num_batches, batch_loss)`` (all
    1-indexed except epoch) — a full training run can have thousands of
    batches, so this is what lets a CLI show live progress instead of a
    blank terminal for however long the run takes.

    ``num_workers`` > 0 loads/decodes the next batch's images in background
    worker processes while the GPU computes on the current batch, instead of
    stalling on disk I/O between every batch. Every input in this dataset is
    the same fixed size, so cuDNN's autotuner (enabled below) can pick the
    fastest convolution algorithm for that shape once and reuse it — on CUDA
    this is normally a large speedup for a fixed-input-size CNN like this one.

    ``use_amp`` enables mixed-precision training on CUDA (most of the compute
    happens in float16 instead of float32) — a genuine compute speedup on
    GPUs with Tensor Cores, not just a data-pipeline fix like the two above.
    It's automatically a no-op on CPU regardless of this flag. Loss is
    BCEWithLogitsLoss on the model's raw logits rather than sigmoid output +
    BCELoss — the latter is both disallowed under autocast and can trip a
    hard CUDA assertion if fp16 rounding pushes a sigmoid output outside
    [0, 1]. ``grad_clip_norm`` bounds gradient norms as an extra guard
    against the instability that error was a symptom of.

    ``checkpoint_path``, if given, saves the model after every epoch (not
    just at the end) — a crash partway through a long run then loses at most
    one epoch's progress instead of all of it.
    """
    torch.backends.cudnn.benchmark = True
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = use_amp and device_type == "cuda" and torch.cuda.is_available()

    model.to(device)
    model.train()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_type == "cuda",
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)
    num_batches = len(loader)

    history = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch_index, (frames, targets) in enumerate(loader, start=1):
            frames = frames.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type, enabled=amp_enabled):
                logits = model.forward_logits(frames)
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            if progress_callback is not None:
                progress_callback(epoch, batch_index, num_batches, batch_loss)
        history.append(float(np.mean(epoch_losses)))
        if checkpoint_path is not None:
            save_model(model, checkpoint_path)

    return history


def print_training_progress(epoch: int, batch_index: int, num_batches: int, batch_loss: float) -> None:
    """A ready-made progress_callback for train() that prints a live-updating progress line."""
    print(f"\repoch {epoch + 1}  batch {batch_index}/{num_batches}  loss {batch_loss:.4f}", end="", flush=True)
    if batch_index == num_batches:
        print()


def save_model(model: TrackNet, path: str | Path) -> None:
    torch.save({"num_frames": model.num_frames, "state_dict": model.state_dict()}, path)


def load_model(path: str | Path, device: str = "cpu") -> TrackNet:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TrackNet(num_frames=checkpoint["num_frames"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def run_tracknet_on_video(
    video_path: str | Path,
    model: TrackNet,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str = "cpu",
):
    """Yield a BallDetection per frame of ``video_path`` using a trained TrackNet model.

    Same per-frame generator contract as tennis_tracker.ball.detect_video —
    this is meant as a drop-in alternative once a model is actually trained.
    The first (num_frames - 1) frames yield position=None (not enough
    temporal context yet), matching detect_video's "not found" convention.
    """
    model.eval()
    model.to(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = input_size
    scale_x = orig_width / width
    scale_y = orig_height / height

    buffer: deque = deque(maxlen=model.num_frames)
    frame_index = 0
    try:
        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                resized = cv2.resize(frame, (width, height))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                buffer.append(rgb.transpose(2, 0, 1))

                if len(buffer) < model.num_frames:
                    position = None
                else:
                    stacked = np.concatenate(list(buffer), axis=0)
                    tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)
                    heatmaps = model(tensor)[0].cpu().numpy()
                    last_heatmap = heatmaps[-1]  # freshest frame's prediction
                    if float(last_heatmap.max()) < confidence_threshold:
                        position = None
                    else:
                        hx, hy = heatmap_to_position(last_heatmap)
                        position = (hx * scale_x, hy * scale_y)

                yield BallDetection(frame_index=frame_index, timestamp=frame_index / fps, position=position)
                frame_index += 1
    finally:
        cap.release()


def visualize(
    video_path: str | Path,
    output_path: str | Path,
    model: TrackNet,
    trail_length: int = 15,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str = "cpu",
) -> int:
    """Write an annotated copy of ``video_path`` with the TrackNet-detected ball + trail overlaid."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    trail: deque = deque(maxlen=trail_length)
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    try:
        for detection in run_tracknet_on_video(
            video_path, model, confidence_threshold=confidence_threshold, device=device
        ):
            ok, frame = cap.read()
            if not ok:
                break
            if detection.position is not None:
                trail.append(detection.position)
                x, y = detection.position
                cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), 2)
            for i in range(1, len(trail)):
                p1 = tuple(int(v) for v in trail[i - 1])
                p2 = tuple(int(v) for v in trail[i])
                cv2.line(frame, p1, p2, (0, 165, 255), 2)
            writer.write(frame)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    return frame_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase E: TrackNet-style ball detector.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train on the TrackNet dataset (game*/Clip*/*.jpg + Label.csv).")
    train_p.add_argument("dataset_root", help="Dataset root directory (contains game1/, game2/, ...).")
    train_p.add_argument("--output", required=True, help="Path to save the trained model checkpoint.")
    train_p.add_argument("--games", default=None, help='Comma-separated game numbers, e.g. "1,2,3". Default: all.')
    train_p.add_argument("--max-clips-per-game", type=int, default=None, help="Cap clips per game (quick smoke run).")
    train_p.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--batch-size", type=int, default=2)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--device", default="cpu")
    train_p.add_argument(
        "--num-workers", type=int, default=4,
        help="Background processes for loading/decoding images in parallel with GPU compute. 0 disables.",
    )
    train_p.add_argument(
        "--input-width", type=int, default=DEFAULT_INPUT_SIZE[0],
        help="Resize frames to this width before feeding the network. Smaller = faster, less precise.",
    )
    train_p.add_argument(
        "--input-height", type=int, default=DEFAULT_INPUT_SIZE[1],
        help="Resize frames to this height before feeding the network. Smaller = faster, less precise.",
    )
    train_p.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed-precision (fp16) training. On by default on CUDA; harmless to leave on.",
    )

    viz_p = sub.add_parser("visualize", help="Run a trained model over a video, drawing the detected ball trail.")
    viz_p.add_argument("video")
    viz_p.add_argument("output_video")
    viz_p.add_argument("--model", required=True, help="Trained model checkpoint from 'train'.")
    viz_p.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    viz_p.add_argument("--device", default="cpu", help='"cpu" or "cuda" — GPU inference is much faster.')

    args = parser.parse_args(argv)

    if args.command == "train":
        games = [int(g) for g in args.games.split(",")] if args.games else None
        input_size = (args.input_width, args.input_height)
        dataset = build_dataset_from_root(
            args.dataset_root, games=games, max_clips_per_game=args.max_clips_per_game,
            num_frames=args.num_frames, input_size=input_size,
        )
        print(f"Training on {len(dataset)} windows from {args.dataset_root} at {input_size[0]}x{input_size[1]}")
        model = TrackNet(num_frames=args.num_frames)
        history = train(
            model, dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device,
            num_workers=args.num_workers, use_amp=not args.no_amp, checkpoint_path=args.output,
            progress_callback=print_training_progress,
        )
        print(f"Saved model to {args.output} (checkpointed every epoch); final loss {history[-1]:.4f}")
        return 0

    if args.command == "visualize":
        model = load_model(args.model, device=args.device)
        count = visualize(
            args.video, args.output_video, model,
            confidence_threshold=args.confidence_threshold, device=args.device,
        )
        print(f"Wrote {count} annotated frames to {args.output_video}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
