"""eval metrics: accuracy, precision, recall, and f1."""


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
