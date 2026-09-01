"""
Unit tests for ModelRouter (Champion/Challenger Canary & Shadow routing).
"""

import numpy as np
import pandas as pd
import pytest

from src.serving.router import ModelRouter


class MockModel:
    def __init__(self, fixed_pd: float):
        self.fixed_pd = fixed_pd

    def predict_proba(self, X):
        return np.array([[1.0 - self.fixed_pd, self.fixed_pd]] * len(X))


def test_router_champion_only():
    champ = MockModel(0.03)
    challenger = MockModel(0.06)
    router = ModelRouter(champion_model=champ, challenger_models={"challenger": challenger}, default_mode="champion_only")

    X = pd.DataFrame([{"feat": 1}])
    decision = router.route_and_score(X)
    assert decision.primary_model_name == "creditrisk-lgbm-calibrated"
    assert decision.primary_pd == 0.03
    assert decision.routing_mode == "champion_only"


def test_router_shadow_mode():
    champ = MockModel(0.02)
    challenger = MockModel(0.05)
    router = ModelRouter(champion_model=champ, challenger_models={"challenger": challenger}, default_mode="shadow")

    X = pd.DataFrame([{"feat": 1}])
    decision = router.route_and_score(X)
    assert decision.primary_pd == 0.02
    assert decision.shadow_pd == 0.05
    assert decision.shadow_divergence == 0.03
    assert decision.routing_mode == "shadow"

    metrics = router.get_shadow_metrics()
    assert metrics["shadow_invocations"] == 1
    assert metrics["mean_divergence"] == 0.03


def test_router_model_override():
    champ = MockModel(0.02)
    challenger = MockModel(0.08)
    router = ModelRouter(champion_model=champ, challenger_models={"challenger_xgb": challenger})

    X = pd.DataFrame([{"feat": 1}])
    decision = router.route_and_score(X, model_override="challenger_xgb")
    assert decision.primary_model_name == "challenger_xgb"
    assert decision.primary_pd == 0.08
