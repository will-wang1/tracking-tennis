"""Phase E: TrackNet-style temporal CNN for ball detection.

This ports yastrebksv/TrackNet (https://github.com/yastrebksv/TrackNet) -- a
proven, published PyTorch implementation of TrackNetV2 built for this exact
tennis dataset format -- in place of the from-scratch simplified
architecture this module used before. The two differ in a few consequential
ways:

- Input: the current frame plus its two predecessors stacked on the channel
  axis (9 channels), newest first. Multiple frames at once is what lets the
  model see through motion blur -- a fast ball leaves a trail of appearance
  across frames a single-frame detector can't use.
- Output/loss: rather than regressing a [0,1] heatmap per input frame
  (sigmoid + BCE), the model predicts, independently per pixel, which of 256
  grayscale intensity classes (0-255) that pixel's Gaussian heatmap value
  falls into (softmax classification + cross-entropy) -- a single heatmap,
  for only the *current* (most recent) of the stacked frames.
- Postprocessing: rather than taking the single brightest pixel, the
  predicted class map is thresholded to binary and searched for a circular
  blob via Hough circle detection. This rejects non-circular bright regions
  (a common false-positive shape for a small, fast, motion-blurred ball)
  that plain argmax can't distinguish from the real thing.

This is an architecture change, not a compatible extension: a checkpoint
trained with the previous version of this module cannot be loaded here and
must be retrained (see train() below) -- load_model() checks for this and
raises a clear error rather than failing cryptically.

Training data is still the standard TrackNet tennis dataset layout: a root
directory of game1/, game2/, ... folders, each containing Clip1/, Clip2/,
... folders, each holding sequential frame images (0000.jpg, 0001.jpg, ...)
plus a Label.csv with one row per frame: file name, visibility class (0=ball
not in frame, 1=clearly visible, 2=hard to see, 3=occluded), x/y pixel
coordinate, and trajectory pattern (0=flying, 1=hit, 2=bouncing -- recorded
here but not consumed by this model; hit/bounce detection is handled
separately by tennis_tracker.trajectory and tennis_tracker.contact_fit).

Windows never cross a Clip boundary, since each clip is an independent
rally -- frame 0000 of Clip2 has nothing to do with the last frame of Clip1.

Once you've trained a model with train() below, run_tracknet_on_video()
analyzes a real video the same way tennis_tracker.ball.detect_video() does
-- same per-frame generator contract -- so a trained model is a drop-in
alternative to the classical detector.
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

NUM_FRAMES = 3  # fixed by the architecture: 3 stacked RGB frames = 9 input channels
NUM_HEATMAP_CLASSES = 256  # per-pixel grayscale intensity classes (0-255) -- the classification target
ARCHITECTURE_VERSION = "yastrebksv-v1"

# (width, height); both divisible by 8 for clean pool/upsample. The reference
# implementation defaults to 640x360 -- pass that explicitly via
# --input-width/--input-height for literal fidelity. This module defaults
# smaller since the 256-class output head is already more VRAM-hungry than
# plain heatmap regression, and this repo's known training GPU (4GB) is
# tight on memory.
DEFAULT_INPUT_SIZE = (512, 288)
DEFAULT_GAUSSIAN_KERNEL_SIZE = 20  # ground-truth Gaussian half-width, in native-resolution pixels
DEFAULT_GAUSSIAN_VARIANCE = 10.0

DEFAULT_BINARY_THRESHOLD = 127  # argmax class map -> binary mask cutoff, before Hough circle detection
DEFAULT_HOUGH_PARAM1 = 50.0
DEFAULT_HOUGH_PARAM2 = 2.0  # very low accumulator threshold -- the ball is a weak, small circular blob
DEFAULT_MIN_RADIUS = 2
DEFAULT_MAX_RADIUS = 7

_FILE_NAME_ALIASES = ["file name", "filename", "file_name", "file"]
_VISIBILITY_ALIASES = ["visibility class", "visibility", "vc", "visibility_class"]
_X_ALIASES = ["x-coordinate", "x coordinate", "x"]
_Y_ALIASES = ["y-coordinate", "y coordinate", "y"]
_TRAJECTORY_ALIASES = ["trajectory pattern", "trajectory", "status", "trajectory_pattern"]


class _ConvBlock(nn.Module):
    """Conv2d -> ReLU -> BatchNorm2d, in that order.

    Matches yastrebksv/TrackNet's ConvBlock exactly -- an unusual order
    (ReLU before BatchNorm rather than after), not just a generic conv block,
    kept as-is for fidelity to the reference implementation.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class TrackNet(nn.Module):
    """VGG-style encoder/decoder, ported from yastrebksv/TrackNet's BallTrackerNet.

    Outputs, per pixel, a distribution over NUM_HEATMAP_CLASSES (0-255)
    grayscale intensity classes for a single heatmap (the current/most
    recent of the NUM_FRAMES stacked input frames) -- reshaped to
    (batch, 256, H*W) for nn.CrossEntropyLoss. Fully convolutional (pooled
    3x by 2x, then upsampled 3x by 2x -- net effect: input resolution is
    preserved), so it works at any input resolution divisible by 8, not just
    the reference's 640x360. ``input_size`` isn't used inside the network
    itself; it's just carried along so save_model/load_model can round-trip
    the resolution a checkpoint was trained at, so inference automatically
    matches without the caller needing to remember and repass it.
    """

    def __init__(self, input_size: tuple[int, int] = DEFAULT_INPUT_SIZE):
        super().__init__()
        self.input_size = input_size
        in_channels = NUM_FRAMES * 3

        self.conv1 = _ConvBlock(in_channels, 64)
        self.conv2 = _ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = _ConvBlock(64, 128)
        self.conv4 = _ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = _ConvBlock(128, 256)
        self.conv6 = _ConvBlock(256, 256)
        self.conv7 = _ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = _ConvBlock(256, 512)
        self.conv9 = _ConvBlock(512, 512)
        self.conv10 = _ConvBlock(512, 512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = _ConvBlock(512, 256)
        self.conv12 = _ConvBlock(256, 256)
        self.conv13 = _ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = _ConvBlock(256, 128)
        self.conv15 = _ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = _ConvBlock(128, 64)
        self.conv17 = _ConvBlock(64, 64)
        self.conv18 = _ConvBlock(64, NUM_HEATMAP_CLASSES)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x: torch.Tensor, testing: bool = False) -> torch.Tensor:
        """Returns raw per-pixel class logits reshaped to (batch, 256, H*W).

        Training feeds this straight into nn.CrossEntropyLoss (which applies
        log-softmax internally). ``testing=True`` additionally applies
        softmax, turning the output into per-class probabilities -- plain
        inference doesn't need this, since argmax of logits and argmax of
        softmax(logits) always agree.
        """
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        out = x.reshape(batch_size, NUM_HEATMAP_CLASSES, -1)
        if testing:
            out = self.softmax(out)
        return out

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def generate_heatmap_classes(
    x: float | None,
    y: float | None,
    orig_width: int,
    orig_height: int,
    target_width: int,
    target_height: int,
    kernel_size: int = DEFAULT_GAUSSIAN_KERNEL_SIZE,
    variance: float = DEFAULT_GAUSSIAN_VARIANCE,
) -> np.ndarray:
    """Render the ball's ground-truth Gaussian heatmap at native resolution, then
    downsample to the model's working resolution -- matches yastrebksv/TrackNet's
    gt_gen.py exactly. Returns per-pixel *class labels* (0-255 grayscale
    intensity), the nn.CrossEntropyLoss target -- not a [0, 1] regression target.
    """
    heatmap = np.zeros((orig_height, orig_width), dtype=np.uint8)
    if x is not None and y is not None:
        xi, yi = int(x), int(y)
        for i in range(-kernel_size, kernel_size + 1):
            px = xi + i
            if px < 0 or px >= orig_width:
                continue
            for j in range(-kernel_size, kernel_size + 1):
                py = yi + j
                if py < 0 or py >= orig_height:
                    continue
                value = int(255 * np.exp(-(i**2 + j**2) / (2 * variance)))
                if value > 0:
                    heatmap[py, px] = value
    return cv2.resize(heatmap, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def postprocess_heatmap(
    class_map: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    binary_threshold: int = DEFAULT_BINARY_THRESHOLD,
    hough_param1: float = DEFAULT_HOUGH_PARAM1,
    hough_param2: float = DEFAULT_HOUGH_PARAM2,
    min_radius: int = DEFAULT_MIN_RADIUS,
    max_radius: int = DEFAULT_MAX_RADIUS,
) -> tuple[tuple[float, float] | None, float | None]:
    """Threshold the predicted per-pixel class map to binary, then find a circular
    blob via Hough circle detection -- yastrebksv/TrackNet's postprocessing
    exactly, chosen over plain argmax-of-heatmap because it rejects
    non-circular bright regions (a common false-positive shape for a small,
    blurry, fast-moving ball) rather than just taking whichever pixel scored
    highest.
    """
    frame = class_map.astype(np.uint8)
    _, binary = cv2.threshold(frame, binary_threshold, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=hough_param1, param2=hough_param2, minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return None, None
    x, y, radius = circles[0][0]
    return (float(x) * scale_x, float(y) * scale_y), float(radius) * ((scale_x + scale_y) / 2)


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

    Each item is a window of NUM_FRAMES=3 consecutive frames (never crossing
    a clip boundary), stacked newest-first on the channel axis, with the
    *current* (most recent) frame's ball position as the target heatmap.
    Frames labeled visibility_class=0 (ball not in frame) get an all-zero
    (class 0 everywhere) heatmap target rather than being skipped --
    training the model to predict "no ball here" is itself useful signal,
    not something to discard.
    """

    def __init__(
        self,
        clip_dir: str | Path,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        gaussian_kernel_size: int = DEFAULT_GAUSSIAN_KERNEL_SIZE,
        gaussian_variance: float = DEFAULT_GAUSSIAN_VARIANCE,
    ):
        self.clip_dir = Path(clip_dir)
        self.input_size = input_size
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_variance = gaussian_variance

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
        return max(0, len(self.labels) - (NUM_FRAMES - 1))

    def _load_resized(self, file_name: str) -> np.ndarray:
        width, height = self.input_size
        frame = cv2.imread(str(self.clip_dir / file_name))
        if frame is None:
            raise RuntimeError(f"Could not read frame image: {self.clip_dir / file_name}")
        return cv2.resize(frame, (width, height))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.labels[idx + NUM_FRAMES - 1]
        prev = self.labels[idx + NUM_FRAMES - 2]
        preprev = self.labels[idx]

        # Newest-first channel order, matching yastrebksv/TrackNet's own
        # dataset -- the exact convention doesn't matter on its own, but
        # run_tracknet_on_video() must stack frames the same way at
        # inference as training does here.
        img = self._load_resized(current.file_name)
        img_prev = self._load_resized(prev.file_name)
        img_preprev = self._load_resized(preprev.file_name)
        stacked = np.concatenate((img, img_prev, img_preprev), axis=2).astype(np.float32) / 255.0
        stacked = np.rollaxis(stacked, 2, 0)  # HWC -> CHW

        width, height = self.input_size
        heatmap_classes = generate_heatmap_classes(
            current.x, current.y, self.orig_width, self.orig_height, width, height,
            kernel_size=self.gaussian_kernel_size, variance=self.gaussian_variance,
        )
        target = heatmap_classes.astype(np.int64).reshape(-1)

        return torch.from_numpy(stacked), torch.from_numpy(target)


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
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    gaussian_kernel_size: int = DEFAULT_GAUSSIAN_KERNEL_SIZE,
    gaussian_variance: float = DEFAULT_GAUSSIAN_VARIANCE,
) -> ConcatDataset:
    """Builds one combined Dataset spanning every Clip*/ folder found under dataset_root/game*/.

    ``games`` restricts to specific game numbers (e.g. [1, 2]); omit for all
    10. ``max_clips_per_game`` caps clips per game -- handy for a quick smoke
    run rather than loading all ~20k frames.
    """
    clip_dirs = find_clip_directories(dataset_root, games=games, max_clips_per_game=max_clips_per_game)
    if not clip_dirs:
        raise FileNotFoundError(f"No game*/Clip* directories found under {dataset_root}")

    clip_datasets = [
        ClipFrameDataset(
            d, input_size=input_size, gaussian_kernel_size=gaussian_kernel_size, gaussian_variance=gaussian_variance,
        )
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

    Loss is nn.CrossEntropyLoss over the 256 per-pixel grayscale-intensity
    classes (see TrackNet.forward's docstring) -- this pairs well with
    mixed-precision training (no special fp16/autocast workaround needed,
    unlike the sigmoid+BCE setup this module used before).

    ``progress_callback``, if given, is called after every batch as
    ``progress_callback(epoch, batch_index, num_batches, batch_loss)`` (all
    1-indexed except epoch) -- a full training run can have thousands of
    batches, so this is what lets a CLI show live progress instead of a
    blank terminal for however long the run takes.

    ``num_workers`` > 0 loads/decodes the next batch's images in background
    worker processes while the GPU computes on the current batch, instead of
    stalling on disk I/O between every batch. Every input in this dataset is
    the same fixed size, so cuDNN's autotuner (enabled below) can pick the
    fastest convolution algorithm for that shape once and reuse it -- on CUDA
    this is normally a large speedup for a fixed-input-size CNN like this one.

    ``use_amp`` enables mixed-precision training on CUDA (most of the compute
    happens in float16 instead of float32) -- a genuine compute speedup on
    GPUs with Tensor Cores, not just a data-pipeline fix like the two above.
    It's automatically a no-op on CPU regardless of this flag.
    ``grad_clip_norm`` bounds gradient norms as a general training-stability
    guard.

    ``checkpoint_path``, if given, saves the model after every epoch (not
    just at the end) -- a crash partway through a long run then loses at most
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
    loss_fn = nn.CrossEntropyLoss()
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
                logits = model(frames)  # (batch, 256, H*W); testing=False (default) -> raw logits
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
    torch.save(
        {"architecture": ARCHITECTURE_VERSION, "input_size": model.input_size, "state_dict": model.state_dict()},
        path,
    )


def load_model(path: str | Path, device: str = "cpu") -> TrackNet:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != ARCHITECTURE_VERSION:
        raise RuntimeError(
            f"{path} isn't a checkpoint for this TrackNet architecture ({ARCHITECTURE_VERSION!r}). This module "
            "was rewritten to port yastrebksv/TrackNet, which is not a compatible extension of the previous "
            "version -- checkpoints trained before this change must be retrained with the current "
            "'tracknet.py train' command."
        )
    model = TrackNet(input_size=tuple(checkpoint["input_size"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


# The resolution yastrebksv/TrackNet's own released weights were trained
# and tuned at (its README/scripts hardcode 640x360 throughout) -- import
# at anything else and the Hough-circle postprocessing parameters, tuned
# for the ball's apparent size at that scale, would be wrong for it.
PRETRAINED_INPUT_SIZE = (640, 360)


def import_pretrained_checkpoint(
    raw_checkpoint_path: str | Path,
    output_path: str | Path,
    input_size: tuple[int, int] = PRETRAINED_INPUT_SIZE,
    device: str = "cpu",
) -> None:
    """Wrap yastrebksv/TrackNet's own released checkpoint for use with this module.

    Their checkpoint (e.g. the "Pretrained model" linked from
    https://github.com/yastrebksv/TrackNet's README) is a bare
    ``model.state_dict()``, not wrapped with an architecture tag or the
    training resolution the way save_model() does here. This module's
    TrackNet uses the exact same layer names/structure as their
    BallTrackerNet (ConvBlock order, conv1..conv18, pool1..pool3,
    ups1..ups3), so the weights load directly -- this just adds the
    wrapping so it plugs into load_model()/get_ball_detections() like any
    checkpoint trained with this module's own train().

    ``weights_only=True`` (used here, as everywhere else this module loads
    a checkpoint) restricts unpickling to plain tensor data, blocking the
    classic arbitrary-code-execution vector in a malicious pickle -- the
    supply-chain risk ball.py's docstring flags with loading third-party
    model weights. It doesn't certify the weights are trustworthy or
    correct, just that loading them can't execute arbitrary code.

    Raises a RuntimeError with the mismatched key names if the checkpoint's
    state_dict doesn't line up with this module's TrackNet layer-for-layer
    (e.g. if it's actually a different release/architecture than expected).
    """
    raw_state_dict = torch.load(raw_checkpoint_path, map_location=device, weights_only=True)
    model = TrackNet(input_size=input_size)
    try:
        model.load_state_dict(raw_state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"{raw_checkpoint_path}'s state_dict doesn't match this module's TrackNet layer-for-layer "
            f"(see error below) -- is this really yastrebksv/TrackNet's released state_dict, not some other "
            f"checkpoint format?\n\n{e}"
        ) from e
    save_model(model, output_path)


def run_tracknet_on_video(
    video_path: str | Path,
    model: TrackNet,
    input_size: tuple[int, int] | None = None,
    device: str = "cpu",
    verbose: bool = False,
    binary_threshold: int = DEFAULT_BINARY_THRESHOLD,
    hough_param1: float = DEFAULT_HOUGH_PARAM1,
    hough_param2: float = DEFAULT_HOUGH_PARAM2,
    min_radius: int = DEFAULT_MIN_RADIUS,
    max_radius: int = DEFAULT_MAX_RADIUS,
):
    """Yield a BallDetection per frame of ``video_path`` using a trained TrackNet model.

    Same per-frame generator contract as tennis_tracker.ball.detect_video --
    this is meant as a drop-in alternative once a model is actually trained.
    The first (NUM_FRAMES - 1) frames yield position=None (not enough
    temporal context yet), matching detect_video's "not found" convention.

    ``input_size`` defaults to whatever resolution ``model`` was trained at
    (round-tripped through save_model/load_model) -- passing a different
    value here than training used would silently change how large the ball
    appears to the network relative to what it learned, so leave this unset
    unless you have a specific reason to override it.
    """
    model.eval()
    model.to(device)
    input_size = input_size or model.input_size

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = input_size
    scale_x = orig_width / width
    scale_y = orig_height / height

    buffer: deque = deque(maxlen=NUM_FRAMES)
    frame_index = 0
    try:
        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                buffer.append(cv2.resize(frame, (width, height)))

                if len(buffer) < NUM_FRAMES:
                    position, radius = None, None
                else:
                    newest_first = list(buffer)[::-1]  # [current, prev, preprev]
                    stacked = np.concatenate(newest_first, axis=2).astype(np.float32) / 255.0
                    stacked = np.rollaxis(stacked, 2, 0)
                    tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)
                    logits = model(tensor)[0].cpu().numpy()  # (256, H*W)
                    class_map = logits.argmax(axis=0).reshape(height, width)
                    position, radius = postprocess_heatmap(
                        class_map, scale_x, scale_y,
                        binary_threshold=binary_threshold, hough_param1=hough_param1, hough_param2=hough_param2,
                        min_radius=min_radius, max_radius=max_radius,
                    )

                yield BallDetection(
                    frame_index=frame_index, timestamp=frame_index / fps, position=position, radius=radius,
                )
                frame_index += 1
                if verbose and (frame_index % 30 == 0 or frame_index == total_frames):
                    print(f"\rball detection: frame {frame_index}/{total_frames or '?'}", end="", flush=True)
    finally:
        cap.release()
        if verbose and frame_index:
            print()  # newline after the live-updating progress line


def visualize(
    video_path: str | Path,
    output_path: str | Path,
    model: TrackNet,
    trail_length: int = 15,
    input_size: tuple[int, int] | None = None,
    device: str = "cpu",
    binary_threshold: int = DEFAULT_BINARY_THRESHOLD,
    hough_param1: float = DEFAULT_HOUGH_PARAM1,
    hough_param2: float = DEFAULT_HOUGH_PARAM2,
    min_radius: int = DEFAULT_MIN_RADIUS,
    max_radius: int = DEFAULT_MAX_RADIUS,
) -> int:
    """Write an annotated copy of ``video_path`` with the TrackNet-detected ball + trail overlaid."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    trail: deque = deque(maxlen=trail_length)
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    try:
        for detection in run_tracknet_on_video(
            video_path, model, input_size=input_size, device=device,
            binary_threshold=binary_threshold, hough_param1=hough_param1, hough_param2=hough_param2,
            min_radius=min_radius, max_radius=max_radius,
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
            if frame_count % 10 == 0 or frame_count == total_frames:
                print(f"\rframe {frame_count}/{total_frames or '?'}", end="", flush=True)
    finally:
        cap.release()
        writer.release()
        if frame_count:
            print()  # newline after the live-updating progress line

    return frame_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase E: TrackNet-style ball detector (ported from yastrebksv/TrackNet).")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train on the TrackNet dataset (game*/Clip*/*.jpg + Label.csv).")
    train_p.add_argument("dataset_root", help="Dataset root directory (contains game1/, game2/, ...).")
    train_p.add_argument("--output", required=True, help="Path to save the trained model checkpoint.")
    train_p.add_argument("--games", default=None, help='Comma-separated game numbers, e.g. "1,2,3". Default: all.')
    train_p.add_argument("--max-clips-per-game", type=int, default=None, help="Cap clips per game (quick smoke run).")
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
        help="Resize frames to this width before feeding the network (must be divisible by 8). Smaller = faster, "
        "less VRAM, less precise. Saved into the checkpoint, so visualize/inference automatically match -- no "
        "need to pass this again later.",
    )
    train_p.add_argument(
        "--input-height", type=int, default=DEFAULT_INPUT_SIZE[1],
        help="Resize frames to this height before feeding the network (must be divisible by 8).",
    )
    train_p.add_argument("--gaussian-kernel-size", type=int, default=DEFAULT_GAUSSIAN_KERNEL_SIZE)
    train_p.add_argument("--gaussian-variance", type=float, default=DEFAULT_GAUSSIAN_VARIANCE)
    train_p.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed-precision (fp16) training. On by default on CUDA; harmless to leave on.",
    )

    viz_p = sub.add_parser("visualize", help="Run a trained model over a video, drawing the detected ball trail.")
    viz_p.add_argument("video")
    viz_p.add_argument("output_video")
    viz_p.add_argument("--model", required=True, help="Trained model checkpoint from 'train'.")
    viz_p.add_argument("--device", default="cpu", help='"cpu" or "cuda" -- GPU inference is much faster.')
    viz_p.add_argument(
        "--input-width", type=int, default=None,
        help="Override the resolution baked into the checkpoint at training time. Leave unset to use that.",
    )
    viz_p.add_argument("--input-height", type=int, default=None)
    viz_p.add_argument("--binary-threshold", type=int, default=DEFAULT_BINARY_THRESHOLD)
    viz_p.add_argument("--hough-param1", type=float, default=DEFAULT_HOUGH_PARAM1)
    viz_p.add_argument(
        "--hough-param2", type=float, default=DEFAULT_HOUGH_PARAM2,
        help="Lower = more permissive circle detection (more false positives); higher = stricter.",
    )
    viz_p.add_argument("--min-radius", type=int, default=DEFAULT_MIN_RADIUS)
    viz_p.add_argument("--max-radius", type=int, default=DEFAULT_MAX_RADIUS)

    import_p = sub.add_parser(
        "import-pretrained",
        help="Wrap yastrebksv/TrackNet's own released checkpoint (e.g. their 'Pretrained model' Google Drive "
        "link) for use with this module -- no training required.",
    )
    import_p.add_argument("raw_checkpoint", help="Path to the downloaded raw state_dict checkpoint.")
    import_p.add_argument("--output", required=True, help="Path to save the wrapped checkpoint to.")
    import_p.add_argument(
        "--input-width", type=int, default=PRETRAINED_INPUT_SIZE[0],
        help="Resolution their release was trained/tuned at -- change only if you know it differs.",
    )
    import_p.add_argument("--input-height", type=int, default=PRETRAINED_INPUT_SIZE[1])
    import_p.add_argument("--device", default="cpu")

    args = parser.parse_args(argv)

    if args.command == "train":
        games = [int(g) for g in args.games.split(",")] if args.games else None
        input_size = (args.input_width, args.input_height)
        dataset = build_dataset_from_root(
            args.dataset_root, games=games, max_clips_per_game=args.max_clips_per_game, input_size=input_size,
            gaussian_kernel_size=args.gaussian_kernel_size, gaussian_variance=args.gaussian_variance,
        )
        print(f"Training on {len(dataset)} windows from {args.dataset_root} at {input_size[0]}x{input_size[1]}")
        model = TrackNet(input_size=input_size)
        history = train(
            model, dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device,
            num_workers=args.num_workers, use_amp=not args.no_amp, checkpoint_path=args.output,
            progress_callback=print_training_progress,
        )
        print(f"Saved model to {args.output} (checkpointed every epoch); final loss {history[-1]:.4f}")
        return 0

    if args.command == "visualize":
        model = load_model(args.model, device=args.device)
        input_size = (args.input_width, args.input_height) if args.input_width and args.input_height else None
        count = visualize(
            args.video, args.output_video, model, input_size=input_size, device=args.device,
            binary_threshold=args.binary_threshold, hough_param1=args.hough_param1, hough_param2=args.hough_param2,
            min_radius=args.min_radius, max_radius=args.max_radius,
        )
        print(f"Wrote {count} annotated frames to {args.output_video}")
        return 0

    if args.command == "import-pretrained":
        input_size = (args.input_width, args.input_height)
        import_pretrained_checkpoint(args.raw_checkpoint, args.output, input_size=input_size, device=args.device)
        print(f"Saved {args.output}, ready to use with --tracknet-model {args.output}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
