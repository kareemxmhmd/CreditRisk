"""
Fair lending and demographic fairness audit module (Disparate Impact, Equal Opportunity).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class FairnessAuditor:
    """
    Audits credit decision models for compliance with Fair Lending regulations (ECOA).
    Evaluates Disparate Impact Ratio (80% / 4-Fifths rule), Demographic Parity, and Equal Opportunity.
    """

    def __init__(self, sensitive_column: str = "age"):
        self.sensitive_column = sensitive_column

    def create_age_cohorts(self, age_series: pd.Series) -> pd.Series:
        """Categorize age into standard demographic audit cohorts."""
        bins = [0, 29, 49, 64, 150]
        labels = ["Young (<30)", "Prime (30-49)", "Mature (50-64)", "Senior (65+)"]
        return pd.cut(age_series, bins=bins, labels=labels, right=True)

    def run_fairness_audit(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        decisions: np.ndarray, # 'APPROVE' / 'REJECT' / 'REFER'
    ) -> Dict[str, Any]:
        """
        Run full fairness analysis across demographic cohorts.
        """
        y_true = np.asarray(y_true)
        # Treat APPROVE as positive outcome
        is_approved = (decisions == "APPROVE")

        # Assign cohorts
        if self.sensitive_column == "age":
            cohorts = self.create_age_cohorts(df[self.sensitive_column])
        else:
            cohorts = df[self.sensitive_column].astype(str)

        audit_df = pd.DataFrame({
            "cohort": cohorts,
            "y_true": y_true,
            "is_approved": is_approved,
        })

        group_stats = []
        unique_cohorts = audit_df["cohort"].dropna().unique()

        # Compute approval rate per group
        for cohort_name in sorted(unique_cohorts, key=lambda x: str(x)):
            sub = audit_df[audit_df["cohort"] == cohort_name]
            n_total = len(sub)
            if n_total == 0:
                continue

            n_approved = int(sub["is_approved"].sum())
            approval_rate = float(n_approved / n_total)
            default_rate = float(sub["y_true"].mean())

            # True Positives: Repayers (0) who got Approved (is_approved == True)
            repayers = sub[sub["y_true"] == 0]
            tpr_repayers = float(repayers["is_approved"].mean()) if len(repayers) > 0 else 0.0

            # Defaulters (1) who got Approved (False Positive in credit sense)
            defaulters = sub[sub["y_true"] == 1]
            fpr_defaulters = float(defaulters["is_approved"].mean()) if len(defaulters) > 0 else 0.0

            group_stats.append({
                "cohort": str(cohort_name),
                "total_applicants": n_total,
                "population_share": float(n_total / len(audit_df)),
                "actual_default_rate": default_rate,
                "approved_count": n_approved,
                "approval_rate": approval_rate,
                "repay_approval_rate_tpr": tpr_repayers,
                "default_approval_rate_fpr": fpr_defaulters,
            })

        # Determine reference group (group with highest approval rate)
        ref_group = max(group_stats, key=lambda x: x["approval_rate"])
        ref_approval_rate = ref_group["approval_rate"]
        ref_tpr = ref_group["repay_approval_rate_tpr"]

        # Compute Disparate Impact Ratio & Equal Opportunity Differences
        overall_compliant = True
        for stat in group_stats:
            di_ratio = (stat["approval_rate"] / ref_approval_rate) if ref_approval_rate > 0 else 1.0
            stat["disparate_impact_ratio"] = round(di_ratio, 4)
            # 80% rule: DIR >= 0.80
            stat["four_fifths_compliant"] = bool(di_ratio >= 0.80)
            if not stat["four_fifths_compliant"]:
                overall_compliant = False

            # Equal opportunity difference
            stat["equal_opportunity_diff"] = round(abs(stat["repay_approval_rate_tpr"] - ref_tpr), 4)

        return {
            "overall_four_fifths_compliant": overall_compliant,
            "reference_group": ref_group["cohort"],
            "reference_approval_rate": round(ref_approval_rate, 4),
            "group_metrics": group_stats,
        }
