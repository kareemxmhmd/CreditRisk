"""
Probability calibration and reliability assessment module for CreditRisk.
Implements Isotonic Regression, Platt Scaling (Sigmoid), and Expected Calibration Error (ECE).
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Calculate Expected Calibration Error (ECE) and reliability curve bin metrics.
    ECE = sum_{b=1}^B (n_b / N) * |acc(b) - conf(b)|
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    N = len(y_true)

    if N == 0:
        return 0.0, []

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability_curve = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Determine elements in bin
        if i == n_bins - 1:
            in_bin = (y_proba >= bin_lower) & (y_proba <= bin_upper)
        else:
            in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)

        n_in_bin = int(np.sum(in_bin))
        if n_in_bin > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_proba[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (n_in_bin / N) * bin_error

            reliability_curve.append({
                "bin_index": i + 1,
                "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f})",
                "sample_count": n_in_bin,
                "empirical_default_rate": round(bin_acc, 4),
                "mean_predicted_probability": round(bin_conf, 4),
                "calibration_error": round(bin_error, 4),
            })
        else:
            reliability_curve.append({
                "bin_index": i + 1,
                "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f})",
                "sample_count": 0,
                "empirical_default_rate": 0.0,
                "mean_predicted_probability": round((bin_lower + bin_upper) / 2.0, 4),
                "calibration_error": 0.0,
            })

    return float(round(ece, 5)), reliability_curve


class ProbabilityCalibrator(BaseEstimator, ClassifierMixin):
    """
    Wraps an uncalibrated credit score classifier and fits an Isotonic or Platt calibrator.
    """

    def __init__(self, base_estimator: Any = None, method: str = "isotonic"):
        self.base_estimator = base_estimator
        self.method = method
        self.calibrator = None
        self.is_fitted = False

    def fit(self, X_or_probas: Any, y: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fit calibrator using validation probabilities or feature inputs.
        """
        y = np.asarray(y).astype(int)
        
        if hasattr(X_or_probas, "ndim") and X_or_probas.ndim == 1:
            raw_p = np.asarray(X_or_probas).astype(float)
        elif hasattr(X_or_probas, "shape") and len(X_or_probas.shape) == 2 and X_or_probas.shape[1] == 1:
            raw_p = np.asarray(X_or_probas).flatten().astype(float)
        else:
            # Assume feature matrix and base_estimator exists
            if self.base_estimator is None:
                raise ValueError("base_estimator required when passing feature matrix X to fit()")
            raw_p = self.base_estimator.predict_proba(X_or_probas)[:, 1]

        # Clip for numeric stability
        raw_p = np.clip(raw_p, 1e-7, 1.0 - 1e-7)

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw_p, y)
        elif self.method == "sigmoid":
            # Platt scaling: Logistic Regression on log-odds
            log_odds = np.log(raw_p / (1.0 - raw_p)).reshape(-1, 1)
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(log_odds, y)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}. Choose 'isotonic' or 'sigmoid'.")

        self.is_fitted = True
        return self

    def calibrate_probas(self, raw_probas: np.ndarray) -> np.ndarray:
        """
        Transform raw uncalibrated probabilities to calibrated probabilities.
        """
        if not self.is_fitted:
            return raw_probas

        raw_p = np.asarray(raw_probas).astype(float)
        raw_p = np.clip(raw_p, 1e-7, 1.0 - 1e-7)

        if self.method == "isotonic":
            cal_p = self.calibrator.predict(raw_p)
        else:
            log_odds = np.log(raw_p / (1.0 - raw_p)).reshape(-1, 1)
            cal_p = self.calibrator.predict_proba(log_odds)[:, 1]

        return np.clip(cal_p, 0.0, 1.0)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict calibrated probabilities for input feature matrix.
        """
        if self.base_estimator is None:
            raise ValueError("base_estimator is None; cannot predict from features directly.")
        raw_p = self.base_estimator.predict_proba(X)[:, 1]
        cal_p = self.calibrate_probas(raw_p)
        return np.column_stack([1.0 - cal_p, cal_p])

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Binary classification prediction."""
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)
