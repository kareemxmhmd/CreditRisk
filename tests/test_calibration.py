"""
Unit tests for probability calibration and Expected Calibration Error (ECE).
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.calibration import ProbabilityCalibrator, compute_expected_calibration_error


def test_expected_calibration_error_calculation():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.15, 0.2, 0.25, 0.8, 0.85, 0.9, 0.95])
    
    ece, rel_curve = compute_expected_calibration_error(y_true, y_proba, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert len(rel_curve) == 5
    assert all("empirical_default_rate" in b for b in rel_curve)


def test_probability_calibrator_isotonic():
    np.random.seed(42)
    # Generate synthetic uncalibrated scores
    raw_scores = np.random.uniform(0.1, 0.9, 200)
    y = (raw_scores + np.random.normal(0, 0.1, 200) > 0.5).astype(int)

    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(raw_scores, y)

    cal_probas = calibrator.calibrate_probas(raw_scores)
    assert len(cal_probas) == len(raw_scores)
    assert np.all(cal_probas >= 0.0) and np.all(cal_probas <= 1.0)


def test_probability_calibrator_sigmoid():
    np.random.seed(42)
    raw_scores = np.random.uniform(0.05, 0.95, 100)
    y = (raw_scores > 0.5).astype(int)

    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(raw_scores, y)

    cal_probas = calibrator.calibrate_probas(raw_scores)
    assert len(cal_probas) == len(raw_scores)
    assert np.all(cal_probas >= 0.0) and np.all(cal_probas <= 1.0)
