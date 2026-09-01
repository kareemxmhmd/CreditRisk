"""
Unit tests for SHAP explainability and Adverse Action reason code mapping.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ALL_FEATURE_COLS
from src.decision_engine.explainability import SHAPExplainerEngine


def test_shap_explainer_generates_adverse_action_reasons():
    # Train dummy model on random features
    np.random.seed(42)
    X_dummy = pd.DataFrame(np.random.randn(50, len(ALL_FEATURE_COLS)), columns=ALL_FEATURE_COLS)
    y_dummy = np.random.randint(0, 2, size=50)

    model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X_dummy, y_dummy)

    explainer_engine = SHAPExplainerEngine(model, feature_names=ALL_FEATURE_COLS, background_data=X_dummy)

    single_row = X_dummy.iloc[[0]]
    explanation = explainer_engine.explain_instance(single_row, top_n=3, decision="REJECT")

    assert "base_value" in explanation
    assert "reason_codes" in explanation
    assert len(explanation["reason_codes"]) == 3
    assert len(explanation["plain_reason_texts"]) == 3

    # Check reason code structure
    top_reason = explanation["reason_codes"][0]
    assert top_reason["rank"] == 1
    assert "feature" in top_reason
    assert "shap_impact" in top_reason
    assert isinstance(top_reason["reason_code"], str)
    assert len(top_reason["reason_code"]) > 5
