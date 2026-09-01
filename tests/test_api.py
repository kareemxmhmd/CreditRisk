"""
Integration and functional tests for FastAPI decisioning service endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.routes import scoring_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_models_loaded():
    """Ensure models are loaded before running API tests."""
    scoring_service.load()


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "model_version" in data


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert "decision_thresholds" in data
    assert "test_auc" in data


def test_predict_single_application():
    payload = {
        "application_id": "TEST-APP-001",
        "RevolvingUtilizationOfUnsecuredLines": 0.15,
        "age": 42,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.28,
        "MonthlyIncome": 7500.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["application_id"] == "TEST-APP-001"
    assert data["decision"] in ["APPROVE", "REFER", "REJECT"]
    assert 0.0 <= data["probability_of_default"] <= 1.0
    assert "risk_tier" in data
    assert "recommended_interest_rate" in data
    assert len(data["reason_codes"]) > 0
    assert data["latency_ms"] >= 0.0


def test_explain_endpoint():
    payload = {
        "application_id": "TEST-APP-002",
        "RevolvingUtilizationOfUnsecuredLines": 0.95,
        "age": 28,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.70,
        "MonthlyIncome": 2500.0,
        "NumberOfOpenCreditLinesAndLoans": 4,
        "NumberOfTimes90DaysLate": 1,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 1,
        "NumberOfDependents": 2.0
    }
    response = client.post("/api/v1/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "detailed_reason_codes" in data
    assert len(data["detailed_reason_codes"]) == 3
    assert "all_feature_contributions" in data
    assert "base_value" in data


def test_batch_predict_endpoint():
    payload = {
        "applications": [
            {
                "application_id": "BATCH-1",
                "RevolvingUtilizationOfUnsecuredLines": 0.1,
                "age": 50,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.2,
                "MonthlyIncome": 10000.0,
                "NumberOfOpenCreditLinesAndLoans": 10,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 2,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 0.0
            },
            {
                "application_id": "BATCH-2",
                "RevolvingUtilizationOfUnsecuredLines": 1.2,
                "age": 25,
                "NumberOfTime30-59DaysPastDueNotWorse": 3,
                "DebtRatio": 0.9,
                "MonthlyIncome": 2000.0,
                "NumberOfOpenCreditLinesAndLoans": 3,
                "NumberOfTimes90DaysLate": 2,
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 2,
                "NumberOfDependents": 3.0
            }
        ]
    }
    response = client.post("/api/v1/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["decisions"]) == 2


def test_audit_logs_and_drift_endpoints():
    res_logs = client.get("/api/v1/audit-logs?limit=10")
    assert res_logs.status_code == 200
    assert isinstance(res_logs.json(), list)

    res_drift = client.get("/api/v1/drift")
    assert res_drift.status_code == 200
    assert "status" in res_drift.json()
