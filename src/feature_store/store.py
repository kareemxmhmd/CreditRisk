"""
Enterprise Feature Store implementation for real-time feature hydration and batch retrieval.
Provides entity-centric feature registry, online low-latency in-memory/caching retrieval,
and point-in-time offline extraction.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURE_COLS,
    FEATURE_STORE_CACHE_TTL_SECONDS,
    RAW_NUMERIC_FEATURES,
)


@dataclass
class FeatureDefinition:
    name: str
    dtype: str
    entity_key: str
    description: str
    category: str = "derived"
    freshness_ttl_seconds: int = FEATURE_STORE_CACHE_TTL_SECONDS


@dataclass
class FeatureView:
    name: str
    entity_key: str
    features: list[FeatureDefinition]
    description: str = ""


class EnterpriseFeatureStore:
    """
    Enterprise-grade Feature Store providing:
    1. Feature Registry & Metadata Discovery.
    2. Online entity feature hydration with in-memory caching and TTL.
    3. Offline batch feature retrieval for training / evaluation datasets.
    """

    def __init__(self, ttl_seconds: int = FEATURE_STORE_CACHE_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self.registry: dict[str, FeatureDefinition] = {}
        self.views: dict[str, FeatureView] = {}
        self._online_store: dict[str, dict[str, Any]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._init_default_registry()

    def _init_default_registry(self):
        """Initialize standard credit risk feature registry."""
        for col in RAW_NUMERIC_FEATURES:
            self.registry[col] = FeatureDefinition(
                name=col,
                dtype="float64",
                entity_key="applicant_id",
                description=f"Raw bureau credit attribute: {col}",
                category="bureau_raw",
            )

        derived_metadata = {
            "TotalDelinquencies": "Sum of 30-59, 60-89, and 90+ day delinquencies",
            "SevereDelinquencyRatio": "Proportion of delinquencies that are 90+ days severe",
            "IncomePerDependent": "Household gross income divided by number of dependents + 1",
            "MonthlyDebtAmount": "Estimated monthly debt obligations in dollars",
            "DisposableIncome": "Monthly income remaining after debt service",
            "CreditLineDensity": "Open lines per year of adult life",
            "RealEstateLoanRatio": "Mortgage lines divided by total credit lines",
            "HighUtilizationFlag": "Indicator whether revolving utilization exceeds 80%",
            "DelinquencyAnomalyFlag": "Flag for anomalous bureau codes (96/98)",
            "MonthlyIncome_is_missing": "Missingness indicator for income",
            "NumberOfDependents_is_missing": "Missingness indicator for dependents",
            "Utilization_x_DebtRatio": "Interaction term of utilization and debt ratio",
        }

        for col, desc in derived_metadata.items():
            self.registry[col] = FeatureDefinition(
                name=col,
                dtype="float64" if not col.endswith("_is_missing") and not col.endswith("Flag") else "int64",
                entity_key="applicant_id",
                description=desc,
                category="engineered",
            )

        self.views["credit_application_view"] = FeatureView(
            name="credit_application_view",
            entity_key="applicant_id",
            features=list(self.registry.values()),
            description="Full feature view for credit risk loan underwriting.",
        )

    def register_feature(self, feature: FeatureDefinition):
        """Register a new feature into the feature store."""
        self.registry[feature.name] = feature

    def get_registry(self) -> list[dict[str, Any]]:
        """Return full registry catalog."""
        return [
            {
                "name": f.name,
                "dtype": f.dtype,
                "entity_key": f.entity_key,
                "description": f.description,
                "category": f.category,
                "freshness_ttl_seconds": f.freshness_ttl_seconds,
            }
            for f in self.registry.values()
        ]

    def set_online_features(self, entity_id: str, features: dict[str, Any]):
        """Store/hydrate online features for an applicant entity."""
        self._online_store[entity_id] = dict(features)
        self._cache_timestamps[entity_id] = time.time()

    def get_online_features(
        self,
        entity_id: str,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve online features for an entity with TTL validation.
        """
        now = time.time()
        if entity_id not in self._online_store:
            return {}

        stored_time = self._cache_timestamps.get(entity_id, 0)
        if (now - stored_time) > self.ttl_seconds:
            del self._online_store[entity_id]
            del self._cache_timestamps[entity_id]
            return {}

        all_feats = self._online_store[entity_id]
        if feature_names is None:
            return dict(all_feats)

        return {k: all_feats.get(k, np.nan) for k in feature_names}

    def seed_from_dataframe(self, df: pd.DataFrame, id_col: str = "application_id", max_rows: int = 500):
        """Seed online feature store with sample entities from a dataframe."""
        for idx, row in df.head(max_rows).iterrows():
            entity_id = str(row[id_col]) if id_col in row else f"APP-{idx+1000:04d}"
            row_dict = row.to_dict()
            self.set_online_features(entity_id, row_dict)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_names: list[str] | None = None,
        entity_key_col: str = "application_id",
    ) -> pd.DataFrame:
        """
        Offline point-in-time feature extraction for training / batch scoring.
        """
        features_to_fetch = feature_names or ALL_FEATURE_COLS
        merged_rows = []

        for _, row in entity_df.iterrows():
            entity_id = str(row[entity_key_col]) if entity_key_col in row else None
            entity_feats = self._online_store.get(entity_id, {}) if entity_id else {}
            row_dict = row.to_dict()
            for feat in features_to_fetch:
                if feat not in row_dict:
                    row_dict[feat] = entity_feats.get(feat, np.nan)
            merged_rows.append(row_dict)

        return pd.DataFrame(merged_rows)

    def get_stats(self) -> dict[str, Any]:
        """Return operational feature store statistics."""
        return {
            "total_registered_features": len(self.registry),
            "total_feature_views": len(self.views),
            "cached_online_entities": len(self._online_store),
            "default_ttl_seconds": self.ttl_seconds,
            "status": "HEALTHY",
        }
