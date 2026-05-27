#!/usr/bin/env python3
"""standalone tests for confidence calibration.

run: python3 tests/scripts/test_calibration.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dokumen.eval.calibration import compute_calibration, CalibrationReport, CalibrationBin

passed = 0
failed = 0


def test(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}")


def approx(a, b, tol=0.01):
    return abs(a - b) < tol


# ── empty ──

print("── empty input ──")
r = compute_calibration([])
test("empty ece=0", r.ece == 0.0)
test("empty bins=[]", r.bins == [])
test("empty samples=0", r.total_samples == 0)

# ── perfectly calibrated ──

print("\n── perfectly calibrated ──")
# confidence 0.95, all correct → bin 9 has avg_conf≈0.95, avg_acc=1.0
perfect = [(0.95, True)] * 10
r = compute_calibration(perfect)
test("near-perfect ece < 0.1", r.ece < 0.1)
test("10 samples", r.total_samples == 10)

# ── worst case ──

print("\n── worst calibration ──")
# confidence 1.0, all wrong
worst = [(1.0, False)] * 10
r = compute_calibration(worst)
test("worst ece=1.0", approx(r.ece, 1.0))
test("overconfident", r.overconfident is True)

# ── all wrong low confidence ──

print("\n── low confidence all wrong ──")
low_wrong = [(0.05, False)] * 10
r = compute_calibration(low_wrong)
test("low wrong ece near 0", r.ece < 0.1)
test("not overconfident", r.overconfident is False)

# ── mixed realistic ──

print("\n── mixed realistic ──")
mixed = [
    (0.9, True), (0.9, True), (0.9, False),  # bin 8-9: conf=0.9, acc=0.67
    (0.5, True), (0.5, False),                # bin 4-5: conf=0.5, acc=0.5
    (0.1, False), (0.1, False),               # bin 0-1: conf=0.1, acc=0.0
]
r = compute_calibration(mixed)
test("mixed ece > 0", r.ece > 0)
test("mixed ece < 1", r.ece < 1)
test("mixed 7 samples", r.total_samples == 7)

# ── bin structure ──

print("\n── bin structure ──")
r = compute_calibration([(0.5, True)], n_bins=10)
test("10 bins", len(r.bins) == 10)
test("bin 4 has count 1", r.bins[4].count == 1)
test("bin 0 has count 0", r.bins[0].count == 0)
test("bin boundaries", approx(r.bins[0].bin_start, 0.0) and approx(r.bins[0].bin_end, 0.1))

# ── CalibrationBin ──

print("\n── CalibrationBin ──")
b = r.bins[4]
test("avg_confidence", approx(b.avg_confidence, 0.5))
test("avg_accuracy 1.0", approx(b.avg_accuracy, 1.0))
test("gap = |1.0 - 0.5|", approx(b.gap, 0.5))

# ── to_dict ──

print("\n── to_dict ──")
d = r.to_dict()
test("has ece", "ece" in d)
test("has bins", "bins" in d)
test("has overconfident", "overconfident" in d)
test("bins are dicts", isinstance(d["bins"][0], dict))

# ── n_bins=5 ──

print("\n── custom bins ──")
r = compute_calibration([(0.3, True), (0.7, False)], n_bins=5)
test("5 bins", len(r.bins) == 5)

# ── edge: all same confidence ──

print("\n── edge cases ──")
same = [(0.5, True)] * 5 + [(0.5, False)] * 5
r = compute_calibration(same)
test("same conf ece = |0.5 - 0.5| = 0", approx(r.ece, 0.0))

# confidence at boundaries
boundary = [(0.0, False), (1.0, True)]
r = compute_calibration(boundary)
test("boundary ece near 0", r.ece < 0.1)

print(f"\n{'='*50}")
print(f"calibration: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
