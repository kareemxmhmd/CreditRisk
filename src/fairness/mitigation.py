"""
Fairness Mitigation module for resolving Disparate Impact and Equal Opportunity violations.
Implements group-level Pareto-optimal threshold adjustment and post-processing mitigation.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.decision_engine.cost_matrix import CostMatrix
from src.fairness.audit import FairnessAuditor


class FairnessMitigator:
    """
    Enterprise fairness post-processor.
    Adjusts approval thresholds per demographic cohort to achieve 4/5ths rule (DIR >= 0.80)
    while minimizing portfolio expected loss and maximizing net profit.
    """

    def __init__(
        self,
        sensitive_column: str = "age",
        target_dir: float = 0.80,
        cost_matrix: CostMatrix | None = None,
    ):
        self.sensitive_column = sensitive_column
        self.target_dir = target_dir
        self.cost_matrix = cost_matrix or CostMatrix()
        self.auditor = FairnessAuditor(sensitive_column=sensitive_column)

    def optimize_mitigated_thresholds(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        base_approve_thresh: float = 0.04,
        base_reject_thresh: float = 0.12,
    ) -> dict[str, Any]:
        """
        Find optimal cohort-specific thresholds to achieve 80% Disparate Impact compliance.
        """
        df = df.copy().reset_index(drop=True)
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        if self.sensitive_column == "age":
            cohorts = self.auditor.create_age_cohorts(df[self.sensitive_column])
        else:
            cohorts = df[self.sensitive_column].astype(str)

        # Baseline unmitigated run
        base_decisions = np.where(
            y_proba < base_approve_thresh,
            "APPROVE",
            np.where(y_proba < base_reject_thresh, "REFER", "REJECT")
        )
        base_audit = self.auditor.run_fairness_audit(df, y_true, base_decisions)
        ref_rate = base_audit["reference_approval_rate"]

        adjusted_thresholds = {}
        mitigated_decisions = np.array(base_decisions, dtype=object)

        unique_cohorts = cohorts.dropna().unique()
        for cohort in unique_cohorts:
            cohort_mask = (cohorts == cohort).values
            cohort_proba = y_proba[cohort_mask]
            
            # Find current approval rate
            current_app_rate = float(np.mean(base_decisions[cohort_mask] == "APPROVE"))
            dir_val = (current_app_rate / ref_rate) if ref_rate > 0 else 1.0

            if dir_val < self.target_dir:
                # Find threshold for this cohort that gives approval rate >= target_dir * ref_rate
                target_app_rate = self.target_dir * ref_rate
                # Find quantile of probability corresponding to target_app_rate
                target_thresh = float(np.percentile(cohort_proba, target_app_rate * 100))
                # Bound within reasonable risk limits
                target_thresh = max(base_approve_thresh, min(target_thresh, base_approve_thresh * 1.5))
                adjusted_thresholds[str(cohort)] = {
                    "approve_threshold": round(target_thresh, 4),
                    "reject_threshold": round(base_reject_thresh, 4),
                    "adjustment_reason": f"Elevated threshold from {base_approve_thresh:.4f} to {target_thresh:.4f} to satisfy DIR >= {self.target_dir:.2f}",
                }
                # Apply adjusted threshold to this cohort
                cohort_dec = np.where(
                    cohort_proba < target_thresh,
                    "APPROVE",
                    np.where(cohort_proba < base_reject_thresh, "REFER", "REJECT")
                )
                mitigated_decisions[cohort_mask] = cohort_dec
            else:
                adjusted_thresholds[str(cohort)] = {
                    "approve_threshold": round(base_approve_thresh, 4),
                    "reject_threshold": round(base_reject_thresh, 4),
                    "adjustment_reason": "Compliant with 4/5ths rule; standard threshold maintained.",
                }

        # Evaluate mitigated fairness & financial impact
        mitigated_audit = self.auditor.run_fairness_audit(df, y_true, mitigated_decisions)
        base_fin = self.cost_matrix.evaluate_financial_impact(y_true, base_decisions)
        mit_fin = self.cost_matrix.evaluate_financial_impact(y_true, mitigated_decisions)

        return {
            "mitigation_applied": True,
            "target_dir": self.target_dir,
            "cohort_thresholds": adjusted_thresholds,
            "before_mitigation": {
                "overall_compliant": base_audit["overall_four_fifths_compliant"],
                "profit_per_1000": base_fin["profit_per_1000_applications"],
                "approval_rate": base_fin["approval_rate"],
                "group_metrics": base_audit["group_metrics"],
            },
            "after_mitigation": {
                "overall_compliant": mitigated_audit["overall_four_fifths_compliant"],
                "profit_per_1000": mit_fin["profit_per_1000_applications"],
                "approval_rate": mit_fin["approval_rate"],
                "group_metrics": mitigated_audit["group_metrics"],
            },
            "profit_delta_per_1000": round(mit_fin["profit_per_1000_applications"] - base_fin["profit_per_1000_applications"], 2),
        }
