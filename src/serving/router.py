"""
Model Router for Champion/Challenger traffic splitting, Canary deployments, and Shadow mode execution.
"""

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CHALLENGER_LR_NAME,
    CHALLENGER_XGB_NAME,
    CHAMPION_MODEL_NAME,
    DEFAULT_CANARY_CHALLENGER_SPLIT,
    DEFAULT_ROUTING_MODE,
)


@dataclass
class RoutingDecision:
    primary_model_name: str
    primary_pd: float
    routing_mode: str
    is_canary: bool = False
    shadow_model_name: str | None = None
    shadow_pd: float | None = None
    shadow_divergence: float | None = None
    shadow_latency_ms: float | None = None


class ModelRouter:
    """
    Enterprise Model Routing Engine.
    Handles:
    - 100% Champion Routing
    - Canary Traffic Splitting (e.g. 90% Champion / 10% Challenger)
    - Zero-Risk Shadow Mode Execution
    - Request header version pinning
    """

    def __init__(
        self,
        champion_model: Any,
        challenger_models: dict[str, Any] | None = None,
        default_mode: str = DEFAULT_ROUTING_MODE,
        canary_split: float = DEFAULT_CANARY_CHALLENGER_SPLIT,
    ):
        self.champion_model = champion_model
        self.challenger_models = challenger_models or {}
        self.default_mode = default_mode
        self.canary_split = canary_split
        self._divergence_history: list[dict[str, Any]] = []

    def route_and_score(
        self,
        features_df: pd.DataFrame,
        mode_override: str | None = None,
        model_override: str | None = None,
    ) -> RoutingDecision:
        """
        Execute routing and predict default probabilities.
        """
        mode = mode_override or self.default_mode

        # Model explicit override
        if model_override and model_override in self.challenger_models:
            challenger = self.challenger_models[model_override]
            pd_val = float(challenger.predict_proba(features_df)[:, 1][0])
            return RoutingDecision(
                primary_model_name=model_override,
                primary_pd=pd_val,
                routing_mode="pinned",
                is_canary=False,
            )

        # 1. Canary Routing Mode
        if mode == "canary" and self.challenger_models:
            # Deterministic/random canary split
            is_canary = random.random() < self.canary_split
            if is_canary:
                challenger_name, challenger = next(iter(self.challenger_models.items()))
                pd_val = float(challenger.predict_proba(features_df)[:, 1][0])
                return RoutingDecision(
                    primary_model_name=challenger_name,
                    primary_pd=pd_val,
                    routing_mode="canary",
                    is_canary=True,
                )
            else:
                pd_val = float(self.champion_model.predict_proba(features_df)[:, 1][0])
                return RoutingDecision(
                    primary_model_name=CHAMPION_MODEL_NAME,
                    primary_pd=pd_val,
                    routing_mode="canary",
                    is_canary=False,
                )

        # 2. Shadow Execution Mode
        elif mode == "shadow" and self.challenger_models:
            champ_pd = float(self.champion_model.predict_proba(features_df)[:, 1][0])

            # Run challenger in shadow mode
            shadow_name, shadow_model = next(iter(self.challenger_models.items()))
            t0 = time.time()
            shadow_pd = float(shadow_model.predict_proba(features_df)[:, 1][0])
            shadow_latency = round((time.time() - t0) * 1000.0, 2)
            divergence = round(abs(champ_pd - shadow_pd), 4)

            # Record shadow telemetry
            self._divergence_history.append({
                "timestamp": time.time(),
                "champion_pd": champ_pd,
                "shadow_pd": shadow_pd,
                "divergence": divergence,
                "shadow_latency_ms": shadow_latency,
            })
            if len(self._divergence_history) > 1000:
                self._divergence_history.pop(0)

            return RoutingDecision(
                primary_model_name=CHAMPION_MODEL_NAME,
                primary_pd=champ_pd,
                routing_mode="shadow",
                is_canary=False,
                shadow_model_name=shadow_name,
                shadow_pd=shadow_pd,
                shadow_divergence=divergence,
                shadow_latency_ms=shadow_latency,
            )

        # 3. Standard Champion Only Mode
        else:
            pd_val = float(self.champion_model.predict_proba(features_df)[:, 1][0])
            return RoutingDecision(
                primary_model_name=CHAMPION_MODEL_NAME,
                primary_pd=pd_val,
                routing_mode="champion_only",
                is_canary=False,
            )

    def get_shadow_metrics(self) -> dict[str, Any]:
        """Return aggregated shadow mode comparison statistics."""
        if not self._divergence_history:
            return {
                "shadow_invocations": 0,
                "mean_divergence": 0.0,
                "max_divergence": 0.0,
                "mean_shadow_latency_ms": 0.0,
                "agreement_rate": 1.0,
            }

        divs = [x["divergence"] for x in self._divergence_history]
        lats = [x["shadow_latency_ms"] for x in self._divergence_history]
        agreements = [1.0 if d < 0.05 else 0.0 for d in divs]

        return {
            "shadow_invocations": len(self._divergence_history),
            "mean_divergence": round(float(np.mean(divs)), 4),
            "max_divergence": round(float(np.max(divs)), 4),
            "mean_shadow_latency_ms": round(float(np.mean(lats)), 2),
            "agreement_rate": round(float(np.mean(agreements)), 4),
        }
