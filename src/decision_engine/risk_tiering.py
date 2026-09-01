"""
Risk tiering and risk-based interest rate assignment module.
"""

from typing import Dict, Any, Tuple
from src.config import (
    RISK_TIERS,
    DEFAULT_APPROVE_THRESHOLD,
    DEFAULT_REJECT_THRESHOLD,
)


class RiskDecisionEngine:
    """
    Translates model probability of default (PD) into an actionable 3-tier loan decision,
    granular risk tier, and risk-adjusted interest rate.
    """

    def __init__(
        self,
        approve_threshold: float = DEFAULT_APPROVE_THRESHOLD,
        reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
        risk_tiers: Dict[str, Dict[str, Any]] = None,
    ):
        self.approve_threshold = approve_threshold
        self.reject_threshold = reject_threshold
        self.risk_tiers = risk_tiers or RISK_TIERS

    def evaluate(self, pd_value: float) -> Dict[str, Any]:
        """
        Evaluate a single probability of default.
        """
        pd_value = float(max(0.0, min(1.0, pd_value)))

        # 1. Determine 3-tier decision
        if pd_value < self.approve_threshold:
            decision = "APPROVE"
            action_summary = "Approved for automatic origination under standard risk parameters."
        elif pd_value < self.reject_threshold:
            decision = "REFER"
            action_summary = "Referred to senior underwriter for manual documentation and secondary review."
        else:
            decision = "REJECT"
            action_summary = "Application declined due to elevated credit default risk."

        # 2. Determine Risk Tier
        assigned_tier = "CRITICAL"
        tier_info = self.risk_tiers["CRITICAL"]

        if pd_value < self.risk_tiers["LOW"]["max_pd"]:
            assigned_tier = "LOW"
            tier_info = self.risk_tiers["LOW"]
        elif pd_value < self.risk_tiers["MODERATE"]["max_pd"]:
            assigned_tier = "MODERATE"
            tier_info = self.risk_tiers["MODERATE"]
        elif pd_value < self.risk_tiers["HIGH"]["max_pd"]:
            assigned_tier = "HIGH"
            tier_info = self.risk_tiers["HIGH"]
        else:
            assigned_tier = "CRITICAL"
            tier_info = self.risk_tiers["CRITICAL"]

        # 3. Compute Risk-Adjusted Interest Rate
        # Formula: Base rate + Risk premium proportional to PD
        base_rate = tier_info["base_rate"]
        risk_spread = pd_value * 0.20  # +20% spread factor on PD
        recommended_rate = round(base_rate + risk_spread, 4)

        return {
            "decision": decision,
            "action_summary": action_summary,
            "risk_tier": assigned_tier,
            "risk_tier_label": tier_info["label"],
            "risk_tier_color": tier_info["badge_color"],
            "risk_tier_description": tier_info["description"],
            "recommended_interest_rate": recommended_rate,
            "recommended_rate_display": f"{recommended_rate * 100:.2f}% APR",
            "probability_of_default": round(pd_value, 4),
            "approve_threshold": self.approve_threshold,
            "reject_threshold": self.reject_threshold,
        }
