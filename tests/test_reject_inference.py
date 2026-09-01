"""
Unit tests for Reject Inference selection-bias correction.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.reject_inference import RejectInferenceEngine


@pytest.fixture
def sample_accepted_rejected_data():
    np.random.seed(42)
    X_acc = pd.DataFrame({
        "feat1": np.random.normal(0, 1, 100),
        "feat2": np.random.normal(0, 1, 100)
    })
    y_acc = np.random.choice([0, 1], size=100, p=[0.9, 0.1])

    X_rej = pd.DataFrame({
        "feat1": np.random.normal(1.5, 1, 50),
        "feat2": np.random.normal(1.5, 1, 50)
    })

    base_model = LogisticRegression()
    base_model.fit(X_acc, y_acc)

    return X_acc, y_acc, X_rej, base_model


def test_hard_parceling(sample_accepted_rejected_data):
    X_acc, y_acc, X_rej, base_model = sample_accepted_rejected_data
    engine = RejectInferenceEngine(method="hard_parceling", assumed_reject_default_rate=0.30)
    
    X_aug, y_aug, weights = engine.generate_augmented_dataset(X_acc, y_acc, X_rej, base_model)
    assert len(X_aug) == len(X_acc) + len(X_rej)
    assert len(y_aug) == len(X_aug)
    assert len(weights) == len(X_aug)


def test_soft_augmentation(sample_accepted_rejected_data):
    X_acc, y_acc, X_rej, base_model = sample_accepted_rejected_data
    engine = RejectInferenceEngine(method="soft_augmentation")

    X_aug, y_aug, weights = engine.generate_augmented_dataset(X_acc, y_acc, X_rej, base_model)
    assert len(X_aug) == len(X_acc) + 2 * len(X_rej)
    assert len(y_aug) == len(X_aug)
    assert len(weights) == len(X_aug)


def test_propensity_reweighting(sample_accepted_rejected_data):
    X_acc, y_acc, X_rej, base_model = sample_accepted_rejected_data
    engine = RejectInferenceEngine(method="propensity_reweighting")

    X_aug, y_aug, weights = engine.generate_augmented_dataset(X_acc, y_acc, X_rej, base_model)
    assert len(X_aug) == len(X_acc)
    assert np.all(weights > 0.0)
