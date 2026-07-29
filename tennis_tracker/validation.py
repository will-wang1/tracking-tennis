"""Phase 7: validation tooling.

Per the plan, real accuracy can only come from comparing the pipeline's
output against a hand-labeled clip — eyeballing the annotated video isn't
enough to know whether classification is actually working. This module does
the comparison; producing the label set (watching real footage and writing
down each shot's timestamp and type) is the user's part, not something this
pipeline can do for itself.

Ground truth CSV format: columns "timestamp" (seconds) and "shot_type"
(forehand/backhand/other). Predictions are the JSON produced by
tennis_tracker.pipeline (the {"shots": [...]} log).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabeledShot:
    timestamp: float
    shot_type: str


@dataclass
class MatchResult:
    matches: list[tuple[LabeledShot, dict]]
    missed: list[LabeledShot]  # ground-truth shots with no matching prediction
    false_positives: list[dict]  # predictions with no matching ground-truth shot


def load_ground_truth(path: str | Path) -> list[LabeledShot]:
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        return [LabeledShot(timestamp=float(row["timestamp"]), shot_type=row["shot_type"].strip().lower()) for row in reader]


def load_predictions(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        data = json.load(f)
    return data["shots"]


def match_shots(
    ground_truth: list[LabeledShot], predictions: list[dict], tolerance_sec: float = 0.5
) -> MatchResult:
    """Greedily pair each ground-truth shot with its closest-in-time prediction.

    Matching is by timestamp proximity only (not shot type), so a prediction
    that lands at the right time but with the wrong label still counts as a
    "match" (a classification error) rather than a miss + false positive.
    """
    unmatched_gt = list(ground_truth)
    unmatched_pred = list(predictions)

    candidate_pairs = []
    for gt in ground_truth:
        for pred in predictions:
            dt = abs(gt.timestamp - pred["timestamp"])
            if dt <= tolerance_sec:
                candidate_pairs.append((dt, gt, pred))
    candidate_pairs.sort(key=lambda triple: triple[0])

    matches: list[tuple[LabeledShot, dict]] = []
    matched_gt_ids = set()
    matched_pred_ids = set()
    for _, gt, pred in candidate_pairs:
        if id(gt) in matched_gt_ids or id(pred) in matched_pred_ids:
            continue
        matches.append((gt, pred))
        matched_gt_ids.add(id(gt))
        matched_pred_ids.add(id(pred))

    missed = [gt for gt in unmatched_gt if id(gt) not in matched_gt_ids]
    false_positives = [pred for pred in unmatched_pred if id(pred) not in matched_pred_ids]

    return MatchResult(matches=matches, missed=missed, false_positives=false_positives)


def compute_metrics(result: MatchResult) -> dict:
    total_gt = len(result.matches) + len(result.missed)
    correct = sum(1 for gt, pred in result.matches if gt.shot_type == pred["shot_type"])

    confusion: dict[str, dict[str, int]] = {}
    for gt, pred in result.matches:
        confusion.setdefault(gt.shot_type, {}).setdefault(pred["shot_type"], 0)
        confusion[gt.shot_type][pred["shot_type"]] += 1

    return {
        "total_ground_truth_shots": total_gt,
        "matched": len(result.matches),
        "correct_classifications": correct,
        "accuracy": correct / total_gt if total_gt else 0.0,
        "missed_detections": len(result.missed),
        "false_positive_detections": len(result.false_positives),
        "confusion_matrix": confusion,
    }


def _print_report(metrics: dict) -> None:
    print(f"Ground-truth shots:       {metrics['total_ground_truth_shots']}")
    print(f"Matched (by timing):      {metrics['matched']}")
    print(f"Correctly classified:     {metrics['correct_classifications']}")
    print(f"Accuracy:                 {metrics['accuracy']:.1%}")
    print(f"Missed (no detection):    {metrics['missed_detections']}")
    print(f"False positives:          {metrics['false_positive_detections']}")
    print("\nConfusion matrix (rows=ground truth, cols=predicted):")
    for true_type, predicted_counts in metrics["confusion_matrix"].items():
        print(f"  {true_type}: {predicted_counts}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7: score pipeline output against hand-labeled shots.")
    parser.add_argument("ground_truth_csv", help="CSV with columns: timestamp, shot_type.")
    parser.add_argument("predictions_json", help="Shot log JSON produced by tennis_tracker.pipeline.")
    parser.add_argument("--tolerance-sec", type=float, default=0.5)
    args = parser.parse_args(argv)

    ground_truth = load_ground_truth(args.ground_truth_csv)
    predictions = load_predictions(args.predictions_json)
    result = match_shots(ground_truth, predictions, tolerance_sec=args.tolerance_sec)
    metrics = compute_metrics(result)
    _print_report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
