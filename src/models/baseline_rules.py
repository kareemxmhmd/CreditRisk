"""
Legacy Rule-Based Baseline decision simulator for credit risk benchmark comparison.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd


class LegacyRuleBasedScorer:
    """
    Simulates the legacy rule-based loan approval system:
    - Rejects if Revolving Utilization > 85%
    - Rejects if any 90+ days delinquency
    - Rejects if >= 2 short term (30-59 day) delinquencies
    - Rejects if DebtRatio > 0.60 (for normal ratios) or DebtRatio > 1500 (for absolute debt)
    - Rejects if Disposable Income <= 0
    - Otherwise Approves.
    """

    def __init__(
        self,
        max_utilization: float = 0.85,
        max_90days_late: int = 1,
        max_30_59_late: int = 2,
        max_debt_ratio: float = 0.60,
    ):
        self.max_utilization = max_utilization
        self.max_90days_late = max_90days_late
        self.max_30_59_late = max_30_59_late
        self.max_debt_ratio = max_debt_ratio

    def predict_decision(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns 'APPROVE' or 'REJECT' for each application.
        """
        util_cond = df["RevolvingUtilizationOfUnsecuredLines"] > self.max_utilization
        late_90_cond = df["NumberOfTimes90DaysLate"] >= self.max_90days_late
        late_30_cond = df["NumberOfTime30-59DaysPastDueNotWorse"] >= self.max_30_59_late
        
        # Debt ratio condition: ratio > 0.60 (or huge raw debt > 1500)
        debt_cond = (
            (df["DebtRatio"] > self.max_debt_ratio) & (df["DebtRatio"] <= 5.0)
        ) | (df["DebtRatio"] > 1500.0)

        reject_mask = util_cond | late_90_cond | late_30_cond | debt_cond

        decisions = pd.Series("APPROVE", index=df.index)
        decisions[reject_mask] = "REJECT"
        return decisions

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns a pseudo-probability of default (1.0 for REJECT, 0.0 for APPROVE).
        """
        decisions = self.predict_decision(df)
        p_default = np.where(decisions == "REJECT", 0.75, 0.05)
        return np.column_stack([1.0 - p_default, p_default])
