"""
Unit tests for Enterprise Feature Store.
"""

import time
import pandas as pd
import pytest

from src.feature_store.store import EnterpriseFeatureStore, FeatureDefinition


def test_feature_store_registry():
    store = EnterpriseFeatureStore()
    registry = store.get_registry()
    assert len(registry) > 0
    assert any(f["name"] == "DebtRatio" for f in registry)


def test_online_feature_hydration_and_ttl():
    store = EnterpriseFeatureStore(ttl_seconds=1)
    
    # Set features for applicant
    store.set_online_features("APP-TEST-999", {"MonthlyIncome": 8500.0, "DebtRatio": 0.25})
    
    # Retrieve immediately
    feats = store.get_online_features("APP-TEST-999")
    assert feats["MonthlyIncome"] == 8500.0
    assert feats["DebtRatio"] == 0.25

    # Wait for TTL expiration
    time.sleep(1.2)
    expired_feats = store.get_online_features("APP-TEST-999")
    assert expired_feats == {}


def test_offline_batch_feature_extraction():
    store = EnterpriseFeatureStore()
    store.set_online_features("APP-001", {"TotalDelinquencies": 2.0})
    store.set_online_features("APP-002", {"TotalDelinquencies": 0.0})

    entity_df = pd.DataFrame([
        {"application_id": "APP-001", "age": 35},
        {"application_id": "APP-002", "age": 50},
    ])

    extracted = store.get_historical_features(entity_df, feature_names=["TotalDelinquencies", "age"])
    assert len(extracted) == 2
    assert extracted.iloc[0]["TotalDelinquencies"] == 2.0
    assert extracted.iloc[1]["TotalDelinquencies"] == 0.0
