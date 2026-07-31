"""Phase E: TrackNet-style temporal CNN for ball detection (scaffold).

This is a from-scratch, simplified reimplementation of the TrackNetV2-style
architecture — a VGG-like encoder/decoder that takes several consecutive
frames stacked on the channel axis and outputs one heatmap per input frame.
Using multiple frames at once (rather than one frame in isolation) is what
lets it see through motion blur: a fast-moving ball leaves a trail of
appearance across frames that a single-frame detector can't use.

This is architecture + training-loop scaffolding, not a trained detector.
There are no pretrained weights bundled here (see the project README/plan
discussion for why: third-party pretrained checkpoints for this exact task
are typically distributed as ad-hoc pickled files of uncertain provenance,
which is a real code-execution risk to load blindly). Training this for
real requires a hand-labeled dataset from actual match footage — see
tennis_tracker.labeling for the tool that produces one, and train() below
once you have it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_NUM_FRAMES = 3
DEFAULT_INPUT_SIZE = (512, 288)  # (width, height), matches the published TrackNetV2 setup
DEFAULT_HEATMAP_SIGMA = 5.0


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.bottleneck(x)
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return torch.sigmoid(self.output_conv(x))


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
class LabeledFrame:
    frame_index: int
    x: float
    y: float


class BallHeatmapDataset(Dataset):
    """Builds (stacked_frames, target_heatmaps) training windows from a video + hand labels.

    ``labels`` maps frame_index -> (x, y) in the *original* video's pixel
    coordinates (as produced by tennis_tracker.labeling). Only windows of
    ``num_frames`` consecutive, all-labeled frames are used — TrackNet
    predicts a heatmap for every frame in the window, so every frame in it
    needs a target.
    """

    def __init__(
        self,
        video_path: str | Path,
        labels: dict[int, tuple[float, float]],
        num_frames: int = DEFAULT_NUM_FRAMES,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        heatmap_sigma: float = DEFAULT_HEATMAP_SIGMA,
    ):
        self.num_frames = num_frames
        self.input_size = input_size
        self.heatmap_sigma = heatmap_sigma
        self.labels = labels

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self.frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self.frames.append(frame)
        finally:
            cap.release()

        self.orig_height, self.orig_width = self.frames[0].shape[:2] if self.frames else (0, 0)

        self.window_starts = [
            start
            for start in range(len(self.frames) - num_frames + 1)
            if all((start + i) in labels for i in range(num_frames))
        ]

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.window_starts[idx]
        width, height = self.input_size
        scale_x = width / self.orig_width
        scale_y = height / self.orig_height

        frame_tensors = []
        heatmaps = []
        for i in range(self.num_frames):
            frame_index = start + i
            frame = cv2.resize(self.frames[frame_index], (width, height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frame_tensors.append(frame_rgb.transpose(2, 0, 1))  # HWC -> CHW

            label_x, label_y = self.labels[frame_index]
            heatmaps.append(
                generate_heatmap(label_x * scale_x, label_y * scale_y, height, width, self.heatmap_sigma)
            )

        stacked_frames = np.concatenate(frame_tensors, axis=0)  # (num_frames*3, H, W)
        stacked_heatmaps = np.stack(heatmaps, axis=0)  # (num_frames, H, W)
        return torch.from_numpy(stacked_frames), torch.from_numpy(stacked_heatmaps)


def train(
    model: TrackNet,
    dataset: BallHeatmapDataset,
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Trains ``model`` in place; returns the per-epoch mean loss history."""
    model.to(device)
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    history = []
    for _ in range(epochs):
        epoch_losses = []
        for frames, targets in loader:
            frames, targets = frames.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(frames)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        history.append(float(np.mean(epoch_losses)))

    return history
