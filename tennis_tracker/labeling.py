"""Phase F: hand-labeling tool for producing additional TrackNet training data.

tennis_tracker.tracknet now trains directly from the standard TrackNet
dataset layout (game*/Clip*/*.jpg + Label.csv). This tool is for labeling
*your own* footage beyond that dataset — e.g. fine-tuning on your specific
camera/court/lighting. It dumps a CSV of (frame_index, x, y), which is a
different (simpler) schema than Label.csv's (file name, visibility class,
x, y, trajectory pattern) — to actually train on hand-labeled footage,
extract its frames as individual images (tennis_tracker.ingestion.extract_frames)
and reformat this tool's output into that schema first. There's no way
around actually watching footage and marking the ball by hand — this just
makes that as fast as possible (click, arrow key, click, arrow key) and
checkpoints progress so a labeling session can be resumed later.

The interactive loop needs a real display (cv2.imshow) and isn't something
that can run headlessly in this sandbox — it's meant to be run locally. The
label storage/CSV logic it's built on is plain and fully unit-tested on its
own.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2


class LabelStore:
    """In-memory frame_index -> (x, y) ball label map, with CSV persistence."""

    def __init__(self):
        self._labels: dict[int, tuple[float, float]] = {}

    def set_label(self, frame_index: int, x: float, y: float) -> None:
        self._labels[frame_index] = (float(x), float(y))

    def clear_label(self, frame_index: int) -> None:
        self._labels.pop(frame_index, None)

    def get_label(self, frame_index: int) -> tuple[float, float] | None:
        return self._labels.get(frame_index)

    def as_dict(self) -> dict[int, tuple[float, float]]:
        return dict(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    def save_csv(self, path: str | Path) -> None:
        with Path(path).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "x", "y"])
            for frame_index in sorted(self._labels):
                x, y = self._labels[frame_index]
                writer.writerow([frame_index, x, y])

    @classmethod
    def load_csv(cls, path: str | Path) -> "LabelStore":
        store = cls()
        with Path(path).open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                store.set_label(int(row["frame_index"]), float(row["x"]), float(row["y"]))
        return store


def load_frames(video_path: str | Path) -> list:
    """Preload every frame of ``video_path`` into memory for fast random access while labeling."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def run_labeling_tool(
    video_path: str | Path,
    output_csv: str | Path,
    existing_labels_csv: str | Path | None = None,
    start_frame: int = 0,
) -> LabelStore:
    """Interactive labeling loop. Requires a real display — not runnable headlessly.

    Controls: left-click sets the ball position on the current frame;
    right arrow / 'n' advances a frame; left arrow / 'p' goes back; 'c'
    clears the current frame's label; 's' saves progress to ``output_csv``
    without quitting; 'q' / ESC saves and exits.
    """
    frames = load_frames(video_path)
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")

    label_store = LabelStore.load_csv(existing_labels_csv) if existing_labels_csv else LabelStore()
    current = [max(0, min(start_frame, len(frames) - 1))]  # mutable box for the mouse callback closure

    window_name = "Tennis Ball Labeling  (click=label, arrows=navigate, c=clear, s=save, q=quit)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            label_store.set_label(current[0], x, y)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            frame = frames[current[0]].copy()
            label = label_store.get_label(current[0])
            if label is not None:
                cv2.drawMarker(
                    frame, (int(label[0]), int(label[1])), (0, 0, 255),
                    markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2,
                )
            cv2.putText(
                frame, f"frame {current[0]}/{len(frames) - 1}  labeled: {len(label_store)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                label_store.save_csv(output_csv)
                break
            elif key == ord("s"):
                label_store.save_csv(output_csv)
            elif key == ord("c"):
                label_store.clear_label(current[0])
            elif key in (ord("n"), 83):  # 'n' or right arrow
                current[0] = min(current[0] + 1, len(frames) - 1)
            elif key in (ord("p"), 81):  # 'p' or left arrow
                current[0] = max(current[0] - 1, 0)
    finally:
        cv2.destroyAllWindows()

    return label_store


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Phase F: hand-label ball positions for TrackNet training.")
    parser.add_argument("video", help="Path to the source video file.")
    parser.add_argument("output_csv", help="Path to write labels to (frame_index, x, y).")
    parser.add_argument("--resume-from", default=None, help="Existing labels CSV to resume a session from.")
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args(argv)

    label_store = run_labeling_tool(
        args.video, args.output_csv, existing_labels_csv=args.resume_from, start_frame=args.start_frame
    )
    print(f"Saved {len(label_store)} labels to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
