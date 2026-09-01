"""
Unit tests for cost matrix, threshold optimizer, and risk tiering engine.
"""

import numpy as np

from src.decision_engine.cost_matrix import CostMatrix
from src.decision_engine.risk_tiering import RiskDecisionEngine


def test_cost_matrix_payoffs():
    cm = CostMatrix(loan_amount=10000.0, interest_rate=0.15, recovery_rate=0.10)

    # 1. Approved Good Borrower -> +1500
    assert cm.profit_tp == 1500.0

    # 2. Approved Bad Borrower -> -9000 (10000 * 0.90 loss)
    assert cm.loss_fp == -9000.0

    # 3. Rejected Good Borrower -> -1500 (opportunity cost)
    assert cm.cost_fn == -1500.0

    # Evaluate scenario
    y_true = np.array([0, 1, 0, 1])
    decisions = np.array(["APPROVE", "APPROVE", "REJECT", "REJECT"])
    res = cm.evaluate_financial_impact(y_true, decisions)

    # Total profit: 1500 - 9000 - 1500 + 0 = -9000
    assert res["total_net_profit"] == -9000.0
    assert res["approval_rate"] == 0.50


def test_risk_decision_engine():
    engine = RiskDecisionEngine(approve_threshold=0.04, reject_threshold=0.12)

    # Test Approve (Low Risk)
    res_low = engine.evaluate(0.015)
    assert res_low["decision"] == "APPROVE"
    assert res_low["risk_tier"] == "LOW"
    assert res_low["recommended_interest_rate"] < 0.10

    # Test Refer (Moderate/High Risk)
    res_med = engine.evaluate(0.08)
    assert res_med["decision"] == "REFER"
    assert res_med["risk_tier"] == "HIGH"

    # Test Reject (Critical Risk)
    res_high = engine.evaluate(0.25)
    assert res_high["decision"] == "REJECT"
    assert res_high["risk_tier"] == "CRITICAL"
    assert res_high["recommended_interest_rate"] > 0.20
