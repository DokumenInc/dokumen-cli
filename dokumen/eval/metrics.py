"""eval metrics: accuracy, precision, recall, f1, calibration error."""

from typing import List, Tuple


def accuracy(correct: int, total: int) -> float:
    """fraction of correct predictions."""
    if total == 0:
        return 0.0
    return correct / total


def precision(true_positives: int, false_positives: int) -> float:
    """tp / (tp + fp) — how many predicted passes are correct."""
    denom = true_positives + false_positives
    if denom == 0:
        return 0.0
    return true_positives / denom


def recall(true_positives: int, false_negatives: int) -> float:
    """tp / (tp + fn) — how many actual passes did we catch."""
    denom = true_positives + false_negatives
    if denom == 0:
        return 0.0
    return true_positives / denom


def f1(prec: float, rec: float) -> float:
    """harmonic mean of precision and recall."""
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def expected_calibration_error(
    predictions: List[Tuple[float, bool]],
    n_bins: int = 10,
) -> float:
    """ECE — measures how well confidence matches actual accuracy.

    args:
        predictions: list of (confidence, was_correct) tuples
        n_bins: number of calibration bins

    returns:
        weighted average of |accuracy - confidence| per bin.
        0.0 = perfectly calibrated, 1.0 = maximally miscalibrated.
    """
    if not predictions:
        return 0.0

    # bin boundaries: [0, 0.1), [0.1, 0.2), ... [0.9, 1.0]
    bins = [[] for _ in range(n_bins)]
    for conf, correct in predictions:
        # clamp to [0, 1]
        conf = max(0.0, min(1.0, conf))
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))

    total = len(predictions)
    ece = 0.0
    for bin_preds in bins:
        if not bin_preds:
            continue
        avg_conf = sum(c for c, _ in bin_preds) / len(bin_preds)
        avg_acc = sum(1 for _, correct in bin_preds if correct) / len(bin_preds)
        ece += (len(bin_preds) / total) * abs(avg_acc - avg_conf)

    return ece
