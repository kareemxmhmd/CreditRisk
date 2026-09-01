"""
Reject Inference module for correcting loan approval selection bias.
Implements:
1. Hard Parceling (Pseudo-labeling high-risk rejects as defaults)
2. Soft Fuzzy Augmentation (Probability-weighted sample expansion)
3. Propensity Score Reweighting (Inverse Probability of Acceptance Weighting)
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class RejectInferenceEngine:
    """
    Simulates and executes Reject Inference to reconstruct the through-the-door population.
    """

    def __init__(
        self,
        method: str = "hard_parceling",
        assumed_reject_default_rate: float = 0.20,
    ):
        self.method = method
        self.assumed_reject_default_rate = assumed_reject_default_rate
        self.propensity_model = None

    def fit_propensity(self, X: pd.DataFrame, is_accepted: np.ndarray):
        """Fit propensity model P(Accepted=1 | X)."""
        self.propensity_model = LogisticRegression(max_iter=500, random_state=42)
        self.propensity_model.fit(X, is_accepted)

    def generate_augmented_dataset(
        self,
        X_accepted: pd.DataFrame,
        y_accepted: np.ndarray,
        X_rejected: pd.DataFrame,
        base_scorer: Any,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Augment accepted training data with inferred outcomes for rejected population.
        Returns:
            X_augmented: pd.DataFrame
            y_augmented: np.ndarray
            sample_weights: np.ndarray
        """
        if len(X_rejected) == 0:
            weights = np.ones(len(y_accepted), dtype=float)
            return X_accepted.copy(), np.asarray(y_accepted), weights

        # Predict default probability on rejects using base accepted model
        p_reject_default = base_scorer.predict_proba(X_rejected)[:, 1]

        if self.method == "hard_parceling":
            # Hard parceling: Assign Top (assumed_reject_default_rate) as default (1), rest as 0
            cutoff = np.percentile(p_reject_default, (1.0 - self.assumed_reject_default_rate) * 100)
            inferred_y_reject = (p_reject_default >= cutoff).astype(int)

            X_aug = pd.concat([X_accepted, X_rejected], ignore_index=True)
            y_aug = np.concatenate([y_accepted, inferred_y_reject])
            weights = np.ones(len(y_aug), dtype=float)
            return X_aug, y_aug, weights

        elif self.method == "soft_augmentation":
            # Soft augmentation: Duplicate each rejected row into (y=1 with weight p) and (y=0 with weight 1-p)
            X_rej_dup1 = X_rejected.copy()
            y_rej_dup1 = np.ones(len(X_rejected), dtype=int)
            weights_dup1 = p_reject_default

            X_rej_dup0 = X_rejected.copy()
            y_rej_dup0 = np.zeros(len(X_rejected), dtype=int)
            weights_dup0 = 1.0 - p_reject_default

            X_aug = pd.concat([X_accepted, X_rej_dup1, X_rej_dup0], ignore_index=True)
            y_aug = np.concatenate([y_accepted, y_rej_dup1, y_rej_dup0])
            weights = np.concatenate([np.ones(len(y_accepted)), weights_dup1, weights_dup0])
            return X_aug, y_aug, weights

        elif self.method == "propensity_reweighting":
            # IPW: Weight accepted samples by 1 / P(Accepted | X)
            if self.propensity_model is None:
                # Build synthetic acceptance indicator: higher delinquency -> higher chance of rejection
                is_accepted_sim = np.concatenate([
                    np.ones(len(X_accepted)),
                    np.zeros(len(X_rejected))
                ])
                X_comb = pd.concat([X_accepted, X_rejected], ignore_index=True)
                self.fit_propensity(X_comb, is_accepted_sim)

            prop_accepted = self.propensity_model.predict_proba(X_accepted)[:, 1]
            prop_accepted = np.clip(prop_accepted, 0.05, 0.95)
            weights_accepted = 1.0 / prop_accepted
            # Normalize weights
            weights_accepted = weights_accepted / np.mean(weights_accepted)

            return X_accepted.copy(), np.asarray(y_accepted), weights_accepted

        else:
            raise ValueError(f"Unknown reject inference method: {self.method}")
