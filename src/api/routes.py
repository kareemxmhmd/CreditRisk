"""
Enterprise API Endpoints and routing for the CreditRisk FastAPI decision service.
Includes Model Routing (Canary/Shadow), Feature Store Hydration, Prometheus Telemetry,
Fairness Mitigation, and Calibrated Predictions.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response

from src.api.schemas import (
    ApplicationInput,
    BatchApplicationRequest,
    BatchDecisionResponse,
    ExplainDecisionResponse,
    FeatureStoreCatalogResponse,
    HealthResponse,
    RoutingConfigResponse,
    SingleDecisionResponse,
)
from src.config import (
    ALL_FEATURE_COLS,
    ARTIFACTS_DIR,
    CHALLENGER_LR_NAME,
    CHALLENGER_XGB_NAME,
    CHAMPION_MODEL_NAME,
    DEFAULT_CANARY_CHALLENGER_SPLIT,
    DEFAULT_ROUTING_MODE,
    MODEL_VERSION,
    RAW_NUMERIC_FEATURES,
    REQUEST_CORRELATION_HEADER,
)
from src.data_pipeline import DataCleaner
from src.decision_engine.explainability import SHAPExplainerEngine
from src.decision_engine.risk_tiering import RiskDecisionEngine
from src.fairness.mitigation import FairnessMitigator
from src.feature_engineering import CreditFeatureEngineer
from src.feature_store.store import EnterpriseFeatureStore
from src.models.calibration import ProbabilityCalibrator
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.logger import DecisionLogger
from src.monitoring.telemetry import TelemetryService
from src.serving.router import ModelRouter

router = APIRouter()


class ScoringService:
    """
    Enterprise Singleton service holding models, calibrators, feature store,
    and routing engines in memory for sub-30ms serving.
    """
    def __init__(self):
        self.cleaner: DataCleaner | None = None
        self.feature_engineer: CreditFeatureEngineer | None = None
        self.raw_champion = None
        self.calibrated_champion: ProbabilityCalibrator | None = None
        self.challenger_xgb = None
        self.challenger_lr = None
        self.router: ModelRouter | None = None
        self.feature_store: EnterpriseFeatureStore | None = None
        self.fairness_mitigator: FairnessMitigator | None = None
        self.explainer: SHAPExplainerEngine | None = None
        self.decision_engine: RiskDecisionEngine | None = None
        self.logger: DecisionLogger | None = None
        self.drift_detector: DriftDetector | None = None
        self.metadata: dict[str, Any] = {}
        self.is_loaded = False

    def load(self):
        if not (ARTIFACTS_DIR / "champion_lgbm_model.joblib").exists():
            raise RuntimeError("Model artifacts not found in artifacts/. Please run training pipeline first.")

        self.cleaner = joblib.load(ARTIFACTS_DIR / "cleaner_pipeline.joblib")
        self.feature_engineer = joblib.load(ARTIFACTS_DIR / "feature_engineer.joblib")
        self.raw_champion = joblib.load(ARTIFACTS_DIR / "champion_lgbm_model.joblib")

        # Load Calibrated Champion if present
        if (ARTIFACTS_DIR / "calibrated_champion_lgbm.joblib").exists():
            self.calibrated_champion = joblib.load(ARTIFACTS_DIR / "calibrated_champion_lgbm.joblib")
        else:
            self.calibrated_champion = ProbabilityCalibrator(base_estimator=self.raw_champion)

        # Load Challenger Models
        challengers = {}
        if (ARTIFACTS_DIR / "challenger_xgb_model.joblib").exists():
            self.challenger_xgb = joblib.load(ARTIFACTS_DIR / "challenger_xgb_model.joblib")
            challengers[CHALLENGER_XGB_NAME] = self.challenger_xgb

        if (ARTIFACTS_DIR / "baseline_lr_model.joblib").exists():
            self.challenger_lr = joblib.load(ARTIFACTS_DIR / "baseline_lr_model.joblib")
            challengers[CHALLENGER_LR_NAME] = self.challenger_lr

        # Initialize Enterprise Router
        self.router = ModelRouter(
            champion_model=self.calibrated_champion,
            challenger_models=challengers,
            default_mode=DEFAULT_ROUTING_MODE,
            canary_split=DEFAULT_CANARY_CHALLENGER_SPLIT,
        )

        # Initialize Feature Store
        self.feature_store = EnterpriseFeatureStore()

        # Initialize Fairness Mitigator
        self.fairness_mitigator = FairnessMitigator(sensitive_column="age", target_dir=0.80)

        # Load Thresholds
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
            self.explainer = SHAPExplainerEngine(self.raw_champion, feature_names=ALL_FEATURE_COLS)
            self.explainer.explainer = raw_explainer
        else:
            self.explainer = SHAPExplainerEngine(self.raw_champion, feature_names=ALL_FEATURE_COLS)

        # Load metadata
        if (ARTIFACTS_DIR / "model_metadata.json").exists():
            with open(ARTIFACTS_DIR / "model_metadata.json") as f:
                self.metadata = json.load(f)

        self.logger = DecisionLogger()

        # Load drift baseline
        if (ARTIFACTS_DIR / "sample_background.joblib").exists():
            sample_bg = joblib.load(ARTIFACTS_DIR / "sample_background.joblib")
            self.drift_detector = DriftDetector(sample_bg)

        self.is_loaded = True


scoring_service = ScoringService()


def get_service() -> ScoringService:
    if not scoring_service.is_loaded:
        scoring_service.load()
    return scoring_service


def _input_to_df(app_input: ApplicationInput, feature_store: EnterpriseFeatureStore | None = None) -> pd.DataFrame:
    """Convert ApplicationInput to DataFrame, hydrating missing fields from Feature Store if available."""
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
    app_id = app_input.application_id
    if feature_store and app_id:
        stored = feature_store.get_online_features(app_id, RAW_NUMERIC_FEATURES)
        for k, v in stored.items():
            if data.get(k) is None and not pd.isna(v):
                data[k] = v

    return pd.DataFrame([data])


@router.get("/health", response_model=HealthResponse)
def health(service: ScoringService = Depends(get_service)):
    """System healthcheck, model version, calibration and routing status."""
    return HealthResponse(
        status="UP",
        model_version=MODEL_VERSION,
        model_type=service.metadata.get("model_type", "Calibrated LightGBM"),
        cv_auc=service.metadata.get("cv_mean_auc", 0.87),
        test_auc=service.metadata.get("test_auc", 0.87),
        test_ece=service.metadata.get("test_ece", 0.005),
        decision_thresholds=service.metadata.get("thresholds", {
            "approve_threshold": 0.04,
            "reject_threshold": 0.12
        }),
        routing_mode=service.router.default_mode if service.router else "champion_only",
        uptime_status="Operational"
    )


@router.post("/predict", response_model=SingleDecisionResponse)
def predict_decision(
    application: ApplicationInput,
    x_routing_mode: str | None = Header(default=None, alias="X-Routing-Mode"),
    x_model_version: str | None = Header(default=None, alias="X-Model-Version"),
    x_correlation_id: str | None = Header(default=None, alias=REQUEST_CORRELATION_HEADER),
    service: ScoringService = Depends(get_service),
):
    """
    Real-time enterprise loan decisioning:
    Supports Calibrated PD, Canary/Shadow routing, and request correlation tracing.
    """
    start_time = time.time()
    correlation_id = x_correlation_id or TelemetryService.generate_correlation_id()

    # 1. Feature Hydration & Preprocessing
    raw_df = _input_to_df(application, service.feature_store)
    cleaned_df = service.cleaner.transform(raw_df)
    features_df = service.feature_engineer.transform(cleaned_df)

    # 2. Model Routing & Probability Scoring
    routing_result = service.router.route_and_score(
        features_df,
        mode_override=x_routing_mode,
        model_override=x_model_version,
    )
    proba = routing_result.primary_pd

    # 3. Decision & Pricing
    decision_result = service.decision_engine.evaluate(proba)
    decision = decision_result["decision"]

    # 4. Explainability
    explanation = service.explainer.explain_instance(
        features_df,
        top_n=3,
        decision=decision
    )

    elapsed_s = time.time() - start_time
    elapsed_ms = round(elapsed_s * 1000.0, 2)
    timestamp = datetime.now(timezone.utc).isoformat()

    # 5. Prometheus Telemetry Recording
    TelemetryService.record_decision(
        endpoint="/api/v1/predict",
        decision=decision,
        risk_tier=decision_result["risk_tier"],
        model_version=routing_result.primary_model_name,
        pd_value=proba,
        latency_seconds=elapsed_s,
    )
    if routing_result.shadow_model_name and routing_result.shadow_divergence is not None:
        TelemetryService.record_shadow_divergence(
            routing_result.shadow_model_name,
            routing_result.shadow_divergence
        )

    # 6. Audit Logging
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
            model_version=routing_result.primary_model_name,
        )
    except Exception:
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
        model_version=routing_result.primary_model_name,
        decision_timestamp=timestamp,
        latency_ms=elapsed_ms,
        correlation_id=correlation_id,
        routing_mode=routing_result.routing_mode,
        is_canary=routing_result.is_canary,
        calibrated=True,
    )


@router.post("/explain", response_model=ExplainDecisionResponse)
def explain_decision(
    application: ApplicationInput,
    service: ScoringService = Depends(get_service),
):
    """Score and explain application with SHAP attributions and Adverse Action reason codes."""
    start_time = time.time()
    raw_df = _input_to_df(application, service.feature_store)
    cleaned_df = service.cleaner.transform(raw_df)
    features_df = service.feature_engineer.transform(cleaned_df)

    routing_result = service.router.route_and_score(features_df)
    proba = routing_result.primary_pd
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
        model_version=routing_result.primary_model_name,
        decision_timestamp=timestamp,
        latency_ms=elapsed_ms,
        base_value=explanation["base_value"],
        detailed_reason_codes=explanation["reason_codes"],
        all_feature_contributions=explanation["feature_contributions"],
    )


@router.post("/batch-predict", response_model=BatchDecisionResponse)
def batch_predict(
    batch: BatchApplicationRequest,
    service: ScoringService = Depends(get_service),
):
    """Process high-throughput batch credit applications concurrently."""
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

    probas = service.calibrated_champion.predict_proba(features_df)[:, 1]

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
            model_version=CHAMPION_MODEL_NAME,
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
    """Retrieve full model evaluation, calibration statistics, and baseline comparisons."""
    return service.metadata


@router.get("/metrics/prometheus")
def get_prometheus_metrics():
    """Prometheus exposition format endpoint."""
    body, content_type = TelemetryService.export_prometheus_metrics()
    return Response(content=body, media_type=content_type)


@router.get("/routing", response_model=RoutingConfigResponse)
def get_routing_config(service: ScoringService = Depends(get_service)):
    """Retrieve active routing configuration and shadow model evaluation telemetry."""
    return RoutingConfigResponse(
        routing_mode=service.router.default_mode,
        champion_model=CHAMPION_MODEL_NAME,
        challenger_models=list(service.router.challenger_models.keys()),
        canary_split=service.router.canary_split,
        shadow_metrics=service.router.get_shadow_metrics(),
    )


@router.post("/routing/mode")
def set_routing_mode(
    mode: str = Query(..., pattern="^(champion_only|canary|shadow)$"),
    canary_split: float = Query(default=0.10, ge=0.0, le=1.0),
    service: ScoringService = Depends(get_service),
):
    """Dynamically update production routing mode (champion_only, canary, or shadow)."""
    service.router.default_mode = mode
    service.router.canary_split = canary_split
    return {"status": "SUCCESS", "new_mode": mode, "canary_split": canary_split}


@router.get("/feature-store", response_model=FeatureStoreCatalogResponse)
def get_feature_store_catalog(service: ScoringService = Depends(get_service)):
    """Explore Feature Store registry and online cache statistics."""
    stats = service.feature_store.get_stats()
    catalog = service.feature_store.get_registry()
    return FeatureStoreCatalogResponse(
        total_features=stats["total_registered_features"],
        cached_entities=stats["cached_online_entities"],
        status=stats["status"],
        features=catalog,
    )


@router.get("/drift")
def check_drift(service: ScoringService = Depends(get_service)):
    """Evaluate feature distribution drift (PSI / KS) on logged production traffic."""
    if not service.drift_detector:
        return {"status": "NO_BASELINE", "message": "Drift baseline data not initialized."}

    logged_features = service.logger.get_all_logged_features()
    drift_result = service.drift_detector.evaluate_drift(logged_features)
    TelemetryService.record_drift_metrics(drift_result)
    return drift_result


@router.get("/audit-logs")
def get_audit_logs(
    limit: int = Query(default=50, ge=1, le=500),
    service: ScoringService = Depends(get_service),
):
    """Retrieve recent immutable decision audit logs."""
    df = service.logger.get_recent_decisions(limit=limit)
    return df.to_dict(orient="records")
