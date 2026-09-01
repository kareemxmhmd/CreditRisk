"""
Unit tests for Prometheus Telemetry and Correlation Tracking.
"""

import pytest

from src.monitoring.telemetry import TelemetryService


def test_correlation_id_generation():
    cid = TelemetryService.generate_correlation_id()
    assert cid.startswith("CR-")
    assert len(cid) > 5


def test_record_decision_and_export_prometheus():
    TelemetryService.record_decision(
        endpoint="/api/v1/predict",
        decision="APPROVE",
        risk_tier="LOW",
        model_version="creditrisk-lgbm-calibrated",
        pd_value=0.015,
        latency_seconds=0.012
    )

    body, content_type = TelemetryService.export_prometheus_metrics()
    assert len(body) > 0
    assert "creditrisk_scoring_requests_total" in body.decode("utf-8")
    assert "creditrisk_scoring_latency_seconds" in body.decode("utf-8")
