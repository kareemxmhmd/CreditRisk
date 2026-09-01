"""
Explainability engine using SHAP and Adverse Action Reason Code generation.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import shap
from src.config import ADVERSE_ACTION_REASONS


class SHAPExplainerEngine:
    """
    Computes local and global SHAP attributions and maps them to Adverse Action reason codes.
    Uses exact tree-path traversal for high-performance inference (<20ms).
    """

    def __init__(self, model, feature_names: List[str], background_data: Optional[pd.DataFrame] = None):
        self.model = model
        self.feature_names = feature_names
        
        # Use tree-path dependent explainer (fastest and exact for tree ensembles)
        self.explainer = shap.TreeExplainer(model)

    def explain_instance(
        self,
        features_df: pd.DataFrame,
        top_n: int = 3,
        decision: str = "REJECT"
    ) -> Dict[str, Any]:
        """
        Explain a single applicant instance.
        Returns:
        - base_value
        - top_n reason codes
        - full feature contributions dictionary
        """
        # Ensure correct column order
        X = features_df[self.feature_names]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle binary classification output formats
        if isinstance(shap_values, list):
            # Binary classifier: index 1 corresponds to class 1 (Default)
            instance_shap = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Shape (1, n_features, 2)
            instance_shap = shap_values[0, :, 1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
            instance_shap = shap_values[0]
        else:
            instance_shap = np.asarray(shap_values).flatten()

        exp_val = self.explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            base_value = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])
        else:
            base_value = float(exp_val)

        feature_contributions = []
        for feat_name, s_val in zip(self.feature_names, instance_shap):
            raw_val = float(X.iloc[0][feat_name])
            feature_contributions.append({
                "feature": feat_name,
                "value": raw_val,
                "shap_value": float(s_val),
            })

        # For Rejections or Referrals, prioritize features pushing risk UP (highest positive SHAP)
        # For Approvals, show features lowering risk (most negative SHAP)
        if decision in ["REJECT", "REFER"]:
            # Sort descending by shap_value (highest risk drivers first)
            sorted_feats = sorted(feature_contributions, key=lambda x: x["shap_value"], reverse=True)
        else:
            # Sort ascending by shap_value (most favorable factors first)
            sorted_feats = sorted(feature_contributions, key=lambda x: x["shap_value"])

        # Generate top_n reason codes
        reason_codes = []
        for i, item in enumerate(sorted_feats[:top_n], start=1):
            feat = item["feature"]
            s_val = item["shap_value"]
            
            # Look up plain-English Adverse Action explanation
            mapping = ADVERSE_ACTION_REASONS.get(feat, {})
            if s_val > 0:
                reason_text = mapping.get("high", f"Elevated risk associated with {feat}.")
            else:
                reason_text = mapping.get("low", f"Favorable risk factor associated with {feat}.")

            reason_codes.append({
                "rank": i,
                "feature": feat,
                "feature_value": item["value"],
                "shap_impact": round(s_val, 4),
                "reason_code": reason_text,
            })

        return {
            "base_value": round(base_value, 4),
            "reason_codes": reason_codes,
            "plain_reason_texts": [r["reason_code"] for r in reason_codes],
            "feature_contributions": sorted(feature_contributions, key=lambda x: abs(x["shap_value"]), reverse=True),
        }
