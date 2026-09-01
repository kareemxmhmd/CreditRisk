"""
Unit tests for drift detection and fairness auditing.
"""

import numpy as np
import pandas as pd

from src.fairness.audit import FairnessAuditor
from src.monitoring.drift_detector import calculate_psi


def test_calculate_psi():
    # Identical distributions -> PSI near 0
    np.random.seed(42)
    dist1 = np.random.normal(0, 1, 1000)
    dist2 = np.random.normal(0, 1, 1000)
    psi_stable = calculate_psi(dist1, dist2)
    assert psi_stable < 0.10

    # Shifted distribution -> PSI > 0.20
    dist_shifted = np.random.normal(2, 1, 1000)
    psi_shifted = calculate_psi(dist1, dist_shifted)
    assert psi_shifted > 0.20


def test_fairness_auditor():
    auditor = FairnessAuditor(sensitive_column="age")

    df = pd.DataFrame({
        "age": [25, 26, 40, 42, 55, 58, 70, 72] * 10,
    })
    y_true = np.array([0, 1, 0, 0, 0, 0, 0, 0] * 10)
    decisions = np.array(["APPROVE", "REJECT", "APPROVE", "APPROVE", "APPROVE", "APPROVE", "APPROVE", "APPROVE"] * 10)

    audit = auditor.run_fairness_audit(df, y_true, decisions)

    assert "overall_four_fifths_compliant" in audit
    assert "group_metrics" in audit
    assert len(audit["group_metrics"]) == 4

    for g in audit["group_metrics"]:
        assert "disparate_impact_ratio" in g
        assert "four_fifths_compliant" in g
        assert 0.0 <= g["approval_rate"] <= 1.0
