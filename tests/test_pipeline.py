import csv
import json

from tennis_tracker.classify import ShotClassification
from tennis_tracker.pipeline import parse_handedness_arg, write_shot_log
from tennis_tracker.trajectory import HitEvent


def test_parse_handedness_arg():
    assert parse_handedness_arg("0:right,1:left") == {0: "right", 1: "left"}


def test_parse_handedness_arg_empty():
    assert parse_handedness_arg("") == {}
    assert parse_handedness_arg(None) == {}


def _sample_classifications():
    return [
        ShotClassification(
            hit=HitEvent(frame_index=10, timestamp=0.333, position=(100.0, 200.0), residual=15.0),
            player_id=0,
            shot_type="forehand",
            confidence=1.5,
        ),
        ShotClassification(
            hit=HitEvent(frame_index=40, timestamp=1.333, position=(300.0, 200.0), residual=20.0),
            player_id=1,
            shot_type="backhand",
            confidence=-0.9,
        ),
    ]


def test_write_shot_log_json(tmp_path):
    out_path = tmp_path / "log.json"
    write_shot_log(_sample_classifications(), out_path)

    data = json.loads(out_path.read_text())

    assert data["summary"] == {"forehand": 1, "backhand": 1, "unclear": 0}
    assert len(data["shots"]) == 2
    assert data["shots"][0]["shot_type"] == "forehand"
    assert data["shots"][0]["frame_index"] == 10


def test_write_shot_log_csv(tmp_path):
    out_path = tmp_path / "log.csv"
    write_shot_log(_sample_classifications(), out_path)

    with out_path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["shot_type"] == "forehand"
    assert rows[1]["player_id"] == "1"
