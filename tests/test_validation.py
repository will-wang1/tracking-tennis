from tennis_tracker.validation import LabeledShot, compute_metrics, match_shots


def test_match_shots_and_compute_metrics():
    ground_truth = [
        LabeledShot(timestamp=1.0, shot_type="forehand"),   # correctly matched
        LabeledShot(timestamp=2.0, shot_type="backhand"),   # matched but misclassified
        LabeledShot(timestamp=5.0, shot_type="forehand"),   # missed entirely
    ]
    predictions = [
        {"timestamp": 1.05, "shot_type": "forehand"},
        {"timestamp": 2.02, "shot_type": "forehand"},  # wrong label
        {"timestamp": 9.0, "shot_type": "backhand"},   # false positive, no nearby ground truth
    ]

    result = match_shots(ground_truth, predictions, tolerance_sec=0.5)

    assert len(result.matches) == 2
    assert len(result.missed) == 1
    assert result.missed[0].timestamp == 5.0
    assert len(result.false_positives) == 1
    assert result.false_positives[0]["timestamp"] == 9.0

    metrics = compute_metrics(result)
    assert metrics["total_ground_truth_shots"] == 3
    assert metrics["matched"] == 2
    assert metrics["correct_classifications"] == 1
    assert metrics["accuracy"] == 1 / 3
    assert metrics["missed_detections"] == 1
    assert metrics["false_positive_detections"] == 1
    assert metrics["confusion_matrix"]["backhand"]["forehand"] == 1


def test_match_shots_prefers_closest_pairing_under_ambiguity():
    ground_truth = [LabeledShot(timestamp=1.0, shot_type="forehand")]
    predictions = [
        {"timestamp": 1.4, "shot_type": "forehand"},
        {"timestamp": 1.05, "shot_type": "forehand"},
    ]

    result = match_shots(ground_truth, predictions, tolerance_sec=0.5)

    assert len(result.matches) == 1
    _, matched_pred = result.matches[0]
    assert matched_pred["timestamp"] == 1.05
    assert len(result.false_positives) == 1


def test_no_ground_truth_shots_gives_zero_accuracy_not_error():
    result = match_shots([], [{"timestamp": 1.0, "shot_type": "forehand"}])

    metrics = compute_metrics(result)

    assert metrics["accuracy"] == 0.0
    assert metrics["false_positive_detections"] == 1
