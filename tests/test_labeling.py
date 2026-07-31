import cv2
import numpy as np
import pytest

from tennis_tracker.labeling import LabelStore, load_frames


def test_label_store_set_get_clear():
    store = LabelStore()
    store.set_label(5, 100.0, 200.0)

    assert store.get_label(5) == (100.0, 200.0)
    assert store.get_label(6) is None
    assert len(store) == 1

    store.clear_label(5)
    assert store.get_label(5) is None
    assert len(store) == 0


def test_label_store_csv_round_trip(tmp_path):
    store = LabelStore()
    store.set_label(0, 10.5, 20.5)
    store.set_label(3, 15.0, 25.0)
    csv_path = tmp_path / "labels.csv"

    store.save_csv(csv_path)
    loaded = LabelStore.load_csv(csv_path)

    assert loaded.as_dict() == store.as_dict()


def test_load_frames_reads_all_frames(tmp_path):
    video_path = tmp_path / "clip.mp4"
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(7):
        out.write(np.full((48, 64, 3), i, dtype=np.uint8))
    out.release()

    frames = load_frames(video_path)

    assert len(frames) == 7
    assert frames[0].shape == (48, 64, 3)


def test_load_frames_raises_on_missing_file(tmp_path):
    with pytest.raises(RuntimeError):
        load_frames(tmp_path / "does_not_exist.mp4")
