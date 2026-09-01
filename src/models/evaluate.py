"""
Model evaluation module computing AUC, KS-statistic, PR-AUC, Brier score, ECE, and financial payoff.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from src.decision_engine.cost_matrix import CostMatrix
from src.models.calibration import compute_expected_calibration_error


def compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov (KS) statistic and the probability threshold where KS is maximized.
    KS = max(TPR - FPR). Standard credit scoring metric.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    sort_idx = np.argsort(-y_proba)
    y_sorted = y_true[sort_idx]
    p_sorted = y_proba[sort_idx]

    n_defaults = np.sum(y_true == 1)
    n_non_defaults = np.sum(y_true == 0)

    if n_defaults == 0 or n_non_defaults == 0:
        return 0.0, 0.5

    cum_defaults = np.cumsum(y_sorted == 1) / n_defaults
    cum_non_defaults = np.cumsum(y_sorted == 0) / n_non_defaults

    ks_curve = np.abs(cum_defaults - cum_non_defaults)
    max_idx = np.argmax(ks_curve)
    ks_stat = float(ks_curve[max_idx])
    ks_threshold = float(p_sorted[max_idx])

    return ks_stat, ks_threshold


def evaluate_model_performance(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.05,
    cost_matrix: CostMatrix = None,
) -> dict[str, Any]:
    """
    Comprehensive credit model evaluation covering discrimination, calibration, and financial return.
    """
    cost_matrix = cost_matrix or CostMatrix()
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # 1. Discrimination & Ranking Metrics
    auc_roc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    ks_stat, ks_thresh = compute_ks_statistic(y_true, y_proba)
    brier = float(brier_score_loss(y_true, y_proba))
    ece, reliability_curve = compute_expected_calibration_error(y_true, y_proba, n_bins=10)

    # 2. Decision classifications at given threshold
    decisions = np.where(y_proba < threshold, "APPROVE", "REJECT")
    y_pred_binary = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    # 3. Cost-Sensitive Financial Payoff
    financial_metrics = cost_matrix.evaluate_financial_impact(y_true, decisions)

    return {
        "model_name": model_name,
        "auc_roc": round(auc_roc, 4),
        "pr_auc": round(pr_auc, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_optimal_threshold": round(ks_thresh, 4),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 5),
        "reliability_curve": reliability_curve,
        "decision_threshold_evaluated": round(threshold, 4),
        "confusion_matrix": {
            "true_non_defaults_approved (TN)": int(tn),
            "true_non_defaults_rejected (FP)": int(fp),
            "defaults_approved (FN)": int(fn),
            "defaults_rejected (TP)": int(tp),
        },
        "financial_impact": financial_metrics,
    }
