"""
Unit tests for Fairness Mitigation post-processing.
"""

import numpy as np
import pandas as pd
import pytest

from src.fairness.mitigation import FairnessMitigator


def test_fairness_mitigator_optimization():
    np.random.seed(42)
    n = 300
    ages = np.random.choice([25, 40, 55, 70], size=n)
    y_true = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    # Give younger cohort artificially higher probabilities to create disparity
    y_proba = np.where(ages == 25, np.random.uniform(0.08, 0.20, n), np.random.uniform(0.01, 0.08, n))

    df = pd.DataFrame({"age": ages})
    mitigator = FairnessMitigator(sensitive_column="age", target_dir=0.80)

    result = mitigator.optimize_mitigated_thresholds(
        df=df,
        y_true=y_true,
        y_proba=y_proba,
        base_approve_thresh=0.05,
        base_reject_thresh=0.15
    )

    assert result["mitigation_applied"] is True
    assert "cohort_thresholds" in result
    assert "Young (<30)" in result["cohort_thresholds"]
    assert "after_mitigation" in result
