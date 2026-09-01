"""
Cost-sensitive threshold optimization and financial impact evaluation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from src.config import (
    DEFAULT_LOAN_AMOUNT,
    DEFAULT_INTEREST_RATE,
    DEFAULT_RECOVERY_RATE,
    DEFAULT_APPROVE_THRESHOLD,
    DEFAULT_REJECT_THRESHOLD,
)


class CostMatrix:
    """
    Computes financial payoff for credit decisions given ground truth outcomes.
    
    Payoff Matrix:
    - Actual 0 (Repays), Decision APPROVE: + (Principal * InterestRate) [Profit]
    - Actual 1 (Defaults), Decision APPROVE: - (Principal * (1 - RecoveryRate)) [Loss Given Default]
    - Actual 0 (Repays), Decision REJECT: - (Principal * InterestRate) [Opportunity Cost / Foregone Interest]
    - Actual 1 (Defaults), Decision REJECT: $0 [Avoided Loss]
    """

    def __init__(
        self,
        loan_amount: float = DEFAULT_LOAN_AMOUNT,
        interest_rate: float = DEFAULT_INTEREST_RATE,
        recovery_rate: float = DEFAULT_RECOVERY_RATE,
    ):
        self.loan_amount = loan_amount
        self.interest_rate = interest_rate
        self.recovery_rate = recovery_rate

    @property
    def profit_tp(self) -> float:
        """Profit from approving a good borrower (repayment)."""
        return self.loan_amount * self.interest_rate

    @property
    def loss_fp(self) -> float:
        """Financial loss from approving a borrower who defaults (Net of recovery)."""
        return -1.0 * self.loan_amount * (1.0 - self.recovery_rate)

    @property
    def cost_fn(self) -> float:
        """Opportunity cost of rejecting a good borrower (lost interest income)."""
        return -1.0 * self.loan_amount * self.interest_rate

    @property
    def gain_tn(self) -> float:
        """Payoff from correctly rejecting a defaulting borrower."""
        return 0.0

    def evaluate_financial_impact(
        self,
        y_true: np.ndarray,
        decisions: np.ndarray, # 'APPROVE', 'REJECT', or 'REFER'
    ) -> Dict[str, Any]:
        """
        Compute total and per-application financial impact for a decision vector.
        Note: For 'REFER', we assume manual review incurs a small review cost ($50)
        and resolves marginal decisions with 80% accuracy.
        """
        y_true = np.asarray(y_true)
        decisions = np.asarray(decisions)
        n = len(y_true)

        # Financial payoff per application
        payoffs = np.zeros(n, dtype=float)

        # Approved cases
        app_mask = (decisions == "APPROVE")
        # Good approved -> + interest
        payoffs[app_mask & (y_true == 0)] = self.profit_tp
        # Bad approved -> - LGD
        payoffs[app_mask & (y_true == 1)] = self.loss_fp

        # Rejected cases
        rej_mask = (decisions == "REJECT")
        # Good rejected -> - opportunity cost
        payoffs[rej_mask & (y_true == 0)] = self.cost_fn
        # Bad rejected -> $0
        payoffs[rej_mask & (y_true == 1)] = self.gain_tn

        # Referred cases (Manual review)
        ref_mask = (decisions == "REFER")
        manual_review_cost = 50.0  # $50 operational review cost
        # Good referred: 80% chance approved -> net profit = 0.8 * profit_tp - cost
        payoffs[ref_mask & (y_true == 0)] = 0.80 * self.profit_tp + 0.20 * self.cost_fn - manual_review_cost
        # Bad referred: 80% chance caught -> net profit = 0.8 * 0 + 0.2 * loss_fp - cost
        payoffs[ref_mask & (y_true == 1)] = 0.20 * self.loss_fp - manual_review_cost

        total_profit = float(np.sum(payoffs))
        profit_per_app = float(total_profit / n) if n > 0 else 0.0
        profit_per_1000 = profit_per_app * 1000.0

        n_approved = int(np.sum(app_mask))
        n_rejected = int(np.sum(rej_mask))
        n_referred = int(np.sum(ref_mask))

        approved_defaults = int(np.sum(app_mask & (y_true == 1)))
        default_rate_in_approved = (approved_defaults / n_approved) if n_approved > 0 else 0.0

        return {
            "total_net_profit": total_profit,
            "profit_per_application": profit_per_app,
            "profit_per_1000_applications": profit_per_1000,
            "total_applications": n,
            "approved_count": n_approved,
            "approval_rate": n_approved / n if n > 0 else 0.0,
            "rejected_count": n_rejected,
            "rejection_rate": n_rejected / n if n > 0 else 0.0,
            "referred_count": n_referred,
            "referral_rate": n_referred / n if n > 0 else 0.0,
            "approved_defaults": approved_defaults,
            "default_rate_in_approved": default_rate_in_approved,
        }


class ThresholdOptimizer:
    """
    Optimizes binary and 3-tier decision thresholds to maximize expected financial payoff.
    """

    def __init__(self, cost_matrix: Optional[CostMatrix] = None):
        self.cost_matrix = cost_matrix or CostMatrix()

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold_steps: int = 200,
    ) -> Dict[str, Any]:
        """
        Scan probability thresholds to find optimal threshold tau* maximizing net profit.
        """
        thresholds = np.linspace(0.01, 0.50, threshold_steps)
        best_threshold = 0.05
        best_profit = -np.inf
        curve = []

        for tau in thresholds:
            decisions = np.where(y_proba < tau, "APPROVE", "REJECT")
            metrics = self.cost_matrix.evaluate_financial_impact(y_true, decisions)
            profit = metrics["total_net_profit"]
            curve.append({
                "threshold": float(tau),
                "total_profit": profit,
                "profit_per_1000": metrics["profit_per_1000_applications"],
                "approval_rate": metrics["approval_rate"],
                "default_rate_in_approved": metrics["default_rate_in_approved"],
            })
            if profit > best_profit:
                best_profit = profit
                best_threshold = float(tau)

        # Theoretical optimal threshold based on cost ratio:
        # Theoretical tau = (Profit_TP - Cost_FN) / ( (Profit_TP - Cost_FN) + (Gain_TN - Loss_FP) )
        # = (1500 - (-1500)) / (3000 + 9000) = 3000 / 12000 = 0.25 (or adjusted for empirical risk)
        
        # Dual thresholds for Approve / Refer / Reject
        approve_threshold = max(0.01, best_threshold * 0.70)
        reject_threshold = min(0.40, best_threshold * 1.50)

        return {
            "optimal_binary_threshold": best_threshold,
            "approve_threshold": float(approve_threshold),
            "reject_threshold": float(reject_threshold),
            "max_profit": best_profit,
            "threshold_curve": curve,
        }
