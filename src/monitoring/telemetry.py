"""
Prometheus metrics, OpenTelemetry-compatible counters, and request correlation tracking.
"""

import time
import uuid
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.config import PROMETHEUS_METRICS_NAMESPACE, REQUEST_CORRELATION_HEADER

# Custom registry to avoid default collision
REGISTRY = CollectorRegistry(auto_describe=True)

# 1. Total Requests Counter
METRIC_DECISION_REQUESTS = Counter(
    "scoring_requests_total",
    "Total number of loan underwriting scoring requests processed.",
    ["endpoint", "decision", "risk_tier", "model_version"],
    namespace=PROMETHEUS_METRICS_NAMESPACE,
    registry=REGISTRY,
)

# 2. Latency Histogram
METRIC_SCORING_LATENCY = Histogram(
    "scoring_latency_seconds",
    "End-to-end inference and decision latency in seconds.",
    ["endpoint", "model_version"],
    buckets=(0.005, 0.010, 0.025, 0.050, 0.075, 0.100, 0.250, 0.500, 1.0, 2.5),
    namespace=PROMETHEUS_METRICS_NAMESPACE,
    registry=REGISTRY,
)

# 3. Probability of Default Value Gauge
METRIC_PD_VALUE = Gauge(
    "latest_probability_of_default",
    "Most recent probability of default score output.",
    ["model_version", "risk_tier"],
    namespace=PROMETHEUS_METRICS_NAMESPACE,
    registry=REGISTRY,
)

# 4. Feature Population Stability Index (PSI) Gauge
METRIC_FEATURE_PSI = Gauge(
    "feature_psi",
    "Population Stability Index (PSI) drift metric per feature.",
    ["feature_name", "drift_status"],
    namespace=PROMETHEUS_METRICS_NAMESPACE,
    registry=REGISTRY,
)

# 5. Shadow Model Divergence Gauge
METRIC_SHADOW_DIVERGENCE = Gauge(
    "shadow_model_divergence",
    "Absolute PD score difference between Champion and Shadow Challenger model.",
    ["shadow_model_name"],
    namespace=PROMETHEUS_METRICS_NAMESPACE,
    registry=REGISTRY,
)


class TelemetryService:
    """
    Central telemetry collector for recording Prometheus metrics and request traces.
    """

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate unique UUID for request tracing."""
        return f"CR-{uuid.uuid4().hex[:12].upper()}"

    @staticmethod
    def record_decision(
        endpoint: str,
        decision: str,
        risk_tier: str,
        model_version: str,
        pd_value: float,
        latency_seconds: float,
    ):
        """Record production scoring metrics."""
        METRIC_DECISION_REQUESTS.labels(
            endpoint=endpoint,
            decision=decision,
            risk_tier=risk_tier,
            model_version=model_version,
        ).inc()

        METRIC_SCORING_LATENCY.labels(
            endpoint=endpoint,
            model_version=model_version,
        ).observe(latency_seconds)

        METRIC_PD_VALUE.labels(
            model_version=model_version,
            risk_tier=risk_tier,
        ).set(pd_value)

    @staticmethod
    def record_shadow_divergence(shadow_model_name: str, divergence: float):
        """Record divergence from shadow model execution."""
        METRIC_SHADOW_DIVERGENCE.labels(
            shadow_model_name=shadow_model_name,
        ).set(divergence)

    @staticmethod
    def record_drift_metrics(drift_results: dict[str, Any]):
        """Update Prometheus PSI gauges from drift audit results."""
        feature_metrics = drift_results.get("feature_metrics", [])
        for feat in feature_metrics:
            METRIC_FEATURE_PSI.labels(
                feature_name=feat["feature"],
                drift_status=feat["drift_status"],
            ).set(feat["psi"])

    @staticmethod
    def export_prometheus_metrics() -> tuple[bytes, str]:
        """Export latest formatted Prometheus metrics."""
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
