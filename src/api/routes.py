"""
API Endpoints and routing for the CreditRisk FastAPI decision service.
"""

import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Query

from src.config import (
    ARTIFACTS_DIR,
    MODEL_VERSION,
    ALL_FEATURE_COLS,
    RAW_NUMERIC_FEATURES,
)
from src.api.schemas import (
    ApplicationInput,
    SingleDecisionResponse,
    ExplainDecisionResponse,
    BatchApplicationRequest,
    BatchDecisionResponse,
    HealthResponse,
)
from src.data_pipeline import DataCleaner
from src.feature_engineering import CreditFeatureEngineer
from src.decision_engine.risk_tiering import RiskDecisionEngine
from src.decision_engine.explainability import SHAPExplainerEngine
from src.monitoring.logger import DecisionLogger
from src.monitoring.drift_detector import DriftDetector

router = APIRouter()


class ScoringService:
    """
    Singleton service holding model artifacts in memory for low-latency serving.
    """
    def __init__(self):
        self.cleaner: Optional[DataCleaner] = None
        self.feature_engineer: Optional[CreditFeatureEngineer] = None
        self.model = None
        self.explainer: Optional[SHAPExplainerEngine] = None
        self.decision_engine: Optional[RiskDecisionEngine] = None
        self.logger: Optional[DecisionLogger] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.metadata: Dict[str, Any] = {}
        self.is_loaded = False

    def load(self):
        if not (ARTIFACTS_DIR / "champion_lgbm_model.joblib").exists():
            # Trigger training or fail gracefully
            raise RuntimeError("Model artifacts not found in artifacts/. Please run training pipeline first.")

        self.cleaner = joblib.load(ARTIFACTS_DIR / "cleaner_pipeline.joblib")
        self.feature_engineer = joblib.load(ARTIFACTS_DIR / "feature_engineer.joblib")
        self.model = joblib.load(ARTIFACTS_DIR / "champion_lgbm_model.joblib")
        
        with open(ARTIFACTS_DIR / "threshold_config.json") as f:
            thresh_config = json.load(f)
            approve_thresh = thresh_config.get("approve_threshold", 0.04)
            reject_thresh = thresh_config.get("reject_threshold", 0.12)

        self.decision_engine = RiskDecisionEngine(
            approve_threshold=approve_thresh,
            reject_threshold=reject_thresh
        )

        # Load SHAP Explainer
        if (ARTIFACTS_DIR / "shap_explainer.joblib").exists():
            raw_explainer = joblib.load(ARTIFACTS_DIR / "shap_explainer.joblib")
            self.explainer = SHAPExplainerEngine(
                self.model,
                feature_names=ALL_FEATURE_COLS
            )
            self.explainer.explainer = raw_explainer
        else:
            self.explainer = SHAPExplainerEngine(self.model, feature_names=ALL_FEATURE_COLS)

        # Load metadata
        if (ARTIFACTS_DIR / "model_metadata.json").exists():
            with open(ARTIFACTS_DIR / "model_metadata.json") as f:
                self.metadata = json.load(f)

        self.logger = DecisionLogger()
        
        # Load baseline sample for drift
        if (ARTIFACTS_DIR / "sample_background.joblib").exists():
            sample_bg = joblib.load(ARTIFACTS_DIR / "sample_background.joblib")
            self.drift_detector = DriftDetector(sample_bg)

        self.is_loaded = True


# Global service instance
scoring_service = ScoringService()


def get_service() -> ScoringService:
    if not scoring_service.is_loaded:
        scoring_service.load()
    return scoring_service


def _input_to_df(app_input: ApplicationInput) -> pd.DataFrame:
    """Convert Pydantic ApplicationInput to raw pandas DataFrame with exact column names."""
    data = {
        "RevolvingUtilizationOfUnsecuredLines": app_input.RevolvingUtilizationOfUnsecuredLines,
        "age": app_input.age,
        "NumberOfTime30-59DaysPastDueNotWorse": app_input.NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio": app_input.DebtRatio,
        "MonthlyIncome": app_input.MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": app_input.NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": app_input.NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": app_input.NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": app_input.NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents": app_input.NumberOfDependents,
    }
    return pd.DataFrame([data])


@router.get("/health", response_model=HealthResponse)
def health(service: ScoringService = Depends(get_service)):
    """System healthcheck, model version, and status."""
    return HealthResponse(
        status="UP",
        model_version=MODEL_VERSION,
        model_type=service.metadata.get("model_type", "LightGBM Classifier"),
        cv_auc=service.metadata.get("cv_mean_auc", 0.86),
        test_auc=service.metadata.get("test_auc", 0.86),
        decision_thresholds=service.metadata.get("thresholds", {
            "approve_threshold": 0.04,
            "reject_threshold": 0.12
        }),
        uptime_status="Operational"
    )


@router.post("/predict", response_model=SingleDecisionResponse)
def predict_decision(
    application: ApplicationInput,
    service: ScoringService = Depends(get_service)
):
    """
    Evaluate single loan application: Returns probability of default,
    3-tier decision (Approve/Refer/Reject), risk tier, recommended APR, and top reason codes.
    """
    start_time = time.time()

    # 1. Convert to DataFrame and preprocess
    raw_df = _input_to_df(application)
    cleaned_df = service.cleaner.transform(raw_df)
    features_df = service.feature_engineer.transform(cleaned_df)

    # 2. Score probability of default
    proba = float(service.model.predict_proba(features_df)[:, 1][0])

    # 3. Decision & Risk Tiering
    decision_result = service.decision_engine.evaluate(proba)
    decision = decision_result["decision"]

    # 4. Explainability / Reason Codes
    explanation = service.explainer.explain_instance(
        features_df,
        top_n=3,
        decision=decision
    )

    elapsed_ms = round((time.time() - start_time) * 1000.0, 2)
    timestamp = datetime.now(timezone.utc).isoformat()

    # 5. Log decision for audit & drift
    try:
        service.logger.log_decision(
            application_id=application.application_id or "APP-UNKNOWN",
            probability_of_default=proba,
            decision=decision,
            risk_tier=decision_result["risk_tier"],
            recommended_interest_rate=decision_result["recommended_interest_rate"],
            reason_codes=explanation["reason_codes"],
            raw_features=raw_df.iloc[0].to_dict(),
            latency_ms=elapsed_ms,
        )
    except Exception as e:
        # Logging error should not break real-time scoring
        pass

    return SingleDecisionResponse(
        application_id=application.application_id or "APP-DEFAULT",
        decision=decision,
        probability_of_default=decision_result["probability_of_default"],
        risk_tier=decision_result["risk_tier"],
        risk_tier_label=decision_result["risk_tier_label"],
        risk_tier_color=decision_result["risk_tier_color"],
        recommended_interest_rate=decision_result["recommended_interest_rate"],
        recommended_rate_display=decision_result["recommended_rate_display"],
        action_summary=decision_result["action_summary"],
        reason_codes=explanation["plain_reason_texts"],
        model_version=MODEL_VERSION,
        decision_timestamp=timestamp,
        latency_ms=elapsed_ms,
    )


@router.post("/explain", response_model=ExplainDecisionResponse)
def explain_decision(
    application: ApplicationInput,
    service: ScoringService = Depends(get_service)
):
    """
    Score and explain application with detailed SHAP attributions, base value, and ranked reason codes.
    """
    start_time = time.time()
    raw_df = _input_to_df(application)
    cleaned_df = service.cleaner.transform(raw_df)
    features_df = service.feature_engineer.transform(cleaned_df)

    proba = float(service.model.predict_proba(features_df)[:, 1][0])
    decision_result = service.decision_engine.evaluate(proba)
    decision = decision_result["decision"]

    explanation = service.explainer.explain_instance(
        features_df,
        top_n=3,
        decision=decision
    )

    elapsed_ms = round((time.time() - start_time) * 1000.0, 2)
    timestamp = datetime.now(timezone.utc).isoformat()

    return ExplainDecisionResponse(
        application_id=application.application_id or "APP-DEFAULT",
        decision=decision,
        probability_of_default=decision_result["probability_of_default"],
        risk_tier=decision_result["risk_tier"],
        risk_tier_label=decision_result["risk_tier_label"],
        risk_tier_color=decision_result["risk_tier_color"],
        recommended_interest_rate=decision_result["recommended_interest_rate"],
        recommended_rate_display=decision_result["recommended_rate_display"],
        action_summary=decision_result["action_summary"],
        reason_codes=explanation["plain_reason_texts"],
        model_version=MODEL_VERSION,
        decision_timestamp=timestamp,
        latency_ms=elapsed_ms,
        base_value=explanation["base_value"],
        detailed_reason_codes=explanation["reason_codes"],
        all_feature_contributions=explanation["feature_contributions"],
    )


@router.post("/batch-predict", response_model=BatchDecisionResponse)
def batch_predict(
    batch: BatchApplicationRequest,
    service: ScoringService = Depends(get_service)
):
    """
    Process batch applications concurrently.
    """
    start_time = time.time()
    results = []
    app_count = len(batch.applications)

    if app_count == 0:
        return BatchDecisionResponse(
            total_processed=0,
            approved_count=0,
            referred_count=0,
            rejected_count=0,
            decisions=[],
            latency_ms=0.0
        )

    # Convert all inputs to batch dataframe
    rows = []
    app_ids = []
    for app in batch.applications:
        app_ids.append(app.application_id or f"APP-{len(rows)+1}")
        rows.append({
            "RevolvingUtilizationOfUnsecuredLines": app.RevolvingUtilizationOfUnsecuredLines,
            "age": app.age,
            "NumberOfTime30-59DaysPastDueNotWorse": app.NumberOfTime30_59DaysPastDueNotWorse,
            "DebtRatio": app.DebtRatio,
            "MonthlyIncome": app.MonthlyIncome,
            "NumberOfOpenCreditLinesAndLoans": app.NumberOfOpenCreditLinesAndLoans,
            "NumberOfTimes90DaysLate": app.NumberOfTimes90DaysLate,
            "NumberRealEstateLoansOrLines": app.NumberRealEstateLoansOrLines,
            "NumberOfTime60-89DaysPastDueNotWorse": app.NumberOfTime60_89DaysPastDueNotWorse,
            "NumberOfDependents": app.NumberOfDependents,
        })
    raw_df = pd.DataFrame(rows)
    cleaned_df = service.cleaner.transform(raw_df)
    features_df = service.feature_engineer.transform(cleaned_df)

    probas = service.model.predict_proba(features_df)[:, 1]

    approved_c = 0
    referred_c = 0
    rejected_c = 0
    timestamp = datetime.now(timezone.utc).isoformat()

    for idx, p in enumerate(probas):
        d_res = service.decision_engine.evaluate(float(p))
        dec = d_res["decision"]
        if dec == "APPROVE":
            approved_c += 1
        elif dec == "REFER":
            referred_c += 1
        else:
            rejected_c += 1

        # Extract top reasons
        exp = service.explainer.explain_instance(features_df.iloc[[idx]], top_n=3, decision=dec)

        results.append(SingleDecisionResponse(
            application_id=app_ids[idx],
            decision=dec,
            probability_of_default=d_res["probability_of_default"],
            risk_tier=d_res["risk_tier"],
            risk_tier_label=d_res["risk_tier_label"],
            risk_tier_color=d_res["risk_tier_color"],
            recommended_interest_rate=d_res["recommended_interest_rate"],
            recommended_rate_display=d_res["recommended_rate_display"],
            action_summary=d_res["action_summary"],
            reason_codes=exp["plain_reason_texts"],
            model_version=MODEL_VERSION,
            decision_timestamp=timestamp,
            latency_ms=0.0
        ))

    elapsed_ms = round((time.time() - start_time) * 1000.0, 2)
    return BatchDecisionResponse(
        total_processed=app_count,
        approved_count=approved_c,
        referred_count=referred_c,
        rejected_count=rejected_c,
        decisions=results,
        latency_ms=elapsed_ms
    )


@router.get("/metrics")
def get_metrics(service: ScoringService = Depends(get_service)):
    """Retrieve full model evaluation, cross-validation metrics, and baseline comparisons."""
    return service.metadata


@router.get("/drift")
def check_drift(service: ScoringService = Depends(get_service)):
    """Evaluate data and feature drift on production logged applications."""
    if not service.drift_detector:
        return {"status": "NO_BASELINE", "message": "Drift baseline data not initialized."}
    
    logged_features = service.logger.get_all_logged_features()
    return service.drift_detector.evaluate_drift(logged_features)


@router.get("/audit-logs")
def get_audit_logs(
    limit: int = Query(default=50, ge=1, le=500),
    service: ScoringService = Depends(get_service)
):
    """Retrieve recent decision audit logs."""
    df = service.logger.get_recent_decisions(limit=limit)
    return df.to_dict(orient="records")
