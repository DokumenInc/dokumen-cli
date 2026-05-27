"""confidence calibration — post-hoc calibration of judge confidence scores.

currently implements:
  - calibration curve computation (binned accuracy vs confidence)
  - expected calibration error (ECE)
  - reliability diagram data

TODO (shelved — too expensive to run right now):
  - platt scaling (logistic regression fit on held-out eval results)
  - isotonic regression calibration
  - temperature scaling
  - apply_calibration() that transforms raw confidence → calibrated confidence
  these need a decent-sized eval dataset to fit, which means running judges
  on 100+ cases. revisit after #4 eval harness has ground truth datasets.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CalibrationBin:
    """a single bin in the calibration curve."""
    bin_start: float
    bin_end: float
    avg_confidence: float
    avg_accuracy: float
    count: int
    gap: float  # |accuracy - confidence|


@dataclass
class CalibrationReport:
    """full calibration analysis."""
    ece: float
    bins: List[CalibrationBin]
    total_samples: int
    # overconfident = confidence > accuracy on average
    overconfident: bool = False
    # average gap across all bins
    avg_gap: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "ece": self.ece,
            "total_samples": self.total_samples,
            "overconfident": self.overconfident,
            "avg_gap": self.avg_gap,
            "bins": [
                {
                    "bin_start": b.bin_start,
                    "bin_end": b.bin_end,
                    "avg_confidence": b.avg_confidence,
                    "avg_accuracy": b.avg_accuracy,
                    "count": b.count,
                    "gap": b.gap,
                }
                for b in self.bins
            ],
        }


def compute_calibration(
    predictions: List[Tuple[float, bool]],
    n_bins: int = 10,
) -> CalibrationReport:
    """compute calibration curve and ECE from (confidence, was_correct) pairs.

    args:
        predictions: list of (confidence, was_correct) tuples
        n_bins: number of equal-width bins

    returns:
        CalibrationReport with per-bin stats and overall ECE
    """
    if not predictions:
        return CalibrationReport(ece=0.0, bins=[], total_samples=0)

    # bin predictions
    bin_width = 1.0 / n_bins
    bin_data: List[List[Tuple[float, bool]]] = [[] for _ in range(n_bins)]

    for conf, correct in predictions:
        conf = max(0.0, min(1.0, conf))
        idx = min(int(conf * n_bins), n_bins - 1)
        bin_data[idx].append((conf, correct))

    total = len(predictions)
    bins = []
    ece = 0.0
    total_conf = 0.0
    total_acc = 0.0
    weighted_bins = 0

    for i, bin_preds in enumerate(bin_data):
        bin_start = i * bin_width
        bin_end = (i + 1) * bin_width
        count = len(bin_preds)

        if count == 0:
            bins.append(CalibrationBin(
                bin_start=bin_start, bin_end=bin_end,
                avg_confidence=0.0, avg_accuracy=0.0, count=0, gap=0.0,
            ))
            continue

        avg_conf = sum(c for c, _ in bin_preds) / count
        avg_acc = sum(1 for _, correct in bin_preds if correct) / count
        gap = abs(avg_acc - avg_conf)

        ece += (count / total) * gap
        total_conf += avg_conf * count
        total_acc += avg_acc * count
        weighted_bins += count

        bins.append(CalibrationBin(
            bin_start=bin_start, bin_end=bin_end,
            avg_confidence=avg_conf, avg_accuracy=avg_acc,
            count=count, gap=gap,
        ))

    avg_gap = ece  # ece is already the weighted average gap
    overconfident = (total_conf / weighted_bins) > (total_acc / weighted_bins) if weighted_bins > 0 else False

    report = CalibrationReport(
        ece=ece,
        bins=bins,
        total_samples=total,
        overconfident=overconfident,
        avg_gap=avg_gap,
    )

    logger.info(
        "calibration computed",
        extra={"ece": ece, "overconfident": overconfident, "samples": total},
    )

    return report


# ──────────────────────────────────────────────────────────────
# TODO: post-hoc calibration methods (shelved — needs eval data)
# ──────────────────────────────────────────────────────────────

# def fit_platt_scaling(predictions: List[Tuple[float, bool]]) -> Callable[[float], float]:
#     """fit logistic regression to map raw confidence → calibrated confidence.
#     requires scipy or sklearn. needs ~100+ samples for a good fit.
#     returns a function that takes raw confidence and returns calibrated confidence.
#     """
#     pass

# def fit_isotonic(predictions: List[Tuple[float, bool]]) -> Callable[[float], float]:
#     """fit isotonic regression (non-parametric, monotonic).
#     better than platt when the calibration curve is non-sigmoidal.
#     requires sklearn. needs ~200+ samples.
#     """
#     pass

# def fit_temperature_scaling(predictions: List[Tuple[float, bool]]) -> float:
#     """find optimal temperature T such that conf/T is well-calibrated.
#     simplest method — single parameter. good starting point.
#     """
#     pass

# def apply_calibration(
#     raw_confidence: float,
#     calibrator: Callable[[float], float],
# ) -> float:
#     """apply a fitted calibrator to a raw confidence score."""
#     return max(0.0, min(1.0, calibrator(raw_confidence)))
