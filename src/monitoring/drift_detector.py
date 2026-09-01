"""
Data and model drift detector using Population Stability Index (PSI) and Kolmogorov-Smirnov tests.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Calculate Population Stability Index (PSI) between baseline (expected) and production (actual).
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bucket breakpoints from baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    breakpoints = np.percentile(expected, quantiles)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    # Compute frequency distribution
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convert to percentages
    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    # Normalize to sum to 1
    expected_pct = expected_pct / np.sum(expected_pct)
    actual_pct = actual_pct / np.sum(actual_pct)

    # PSI Formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(max(0.0, psi_value))


class DriftDetector:
    """
    Detects feature distribution shift and prediction drift across production traffic.
    """

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        numeric_features: list[str] | None = None,
        psi_threshold_moderate: float = 0.10,
        psi_threshold_severe: float = 0.25,
    ):
        self.baseline_df = baseline_df
        self.numeric_features = numeric_features or [
            col for col in baseline_df.select_dtypes(include=[np.number]).columns
            if col not in ["SeriousDlqin2yrs", "Unnamed: 0"]
        ]
        self.psi_threshold_moderate = psi_threshold_moderate
        self.psi_threshold_severe = psi_threshold_severe

    def evaluate_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """
        Evaluate drift metrics across all monitored features.
        """
        if current_df.empty:
            return {
                "status": "INSUFFICIENT_DATA",
                "message": "No production data logged yet for drift evaluation.",
                "features_monitored": 0,
                "features_drifted": 0,
                "feature_metrics": [],
            }

        metrics = []
        drifted_count = 0

        for col in self.numeric_features:
            if col not in current_df.columns or col not in self.baseline_df.columns:
                continue

            base_values = self.baseline_df[col].dropna().values
            curr_values = current_df[col].dropna().values

            if len(curr_values) < 5 or len(base_values) < 5:
                continue

            # Compute PSI
            psi_val = calculate_psi(base_values, curr_values)
            
            # Compute KS 2-sample test
            ks_stat, p_val = ks_2samp(base_values, curr_values)

            # Compute Wasserstein distance
            w_dist = float(wasserstein_distance(base_values, curr_values))

            # Status classification
            if psi_val >= self.psi_threshold_severe:
                drift_status = "CRITICAL_DRIFT"
                alert_level = "red"
                drifted_count += 1
            elif psi_val >= self.psi_threshold_moderate:
                drift_status = "MODERATE_DRIFT"
                alert_level = "yellow"
                drifted_count += 1
            else:
                drift_status = "STABLE"
                alert_level = "green"

            metrics.append({
                "feature": col,
                "psi": round(psi_val, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(p_val), 6),
                "wasserstein_distance": round(w_dist, 4),
                "drift_status": drift_status,
                "alert_level": alert_level,
                "baseline_mean": round(float(np.mean(base_values)), 4),
                "current_mean": round(float(np.mean(curr_values)), 4),
                "baseline_std": round(float(np.std(base_values)), 4),
                "current_std": round(float(np.std(curr_values)), 4),
            })

        # Sort with most drifted on top
        metrics.sort(key=lambda x: x["psi"], reverse=True)

        overall_status = "HEALTHY"
        if any(m["drift_status"] == "CRITICAL_DRIFT" for m in metrics):
            overall_status = "ALERT_CRITICAL_DRIFT"
        elif any(m["drift_status"] == "MODERATE_DRIFT" for m in metrics):
            overall_status = "WARNING_MODERATE_DRIFT"

        return {
            "status": overall_status,
            "monitored_samples_count": len(current_df),
            "features_monitored": len(metrics),
            "features_drifted": drifted_count,
            "feature_metrics": metrics,
        }
