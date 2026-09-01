"""
Pydantic schemas for CreditRisk FastAPI decision service.
"""

from pydantic import BaseModel, Field


class ApplicationInput(BaseModel):
    """Input payload for a single loan credit application."""
    application_id: str | None = Field(default="APP-DEFAULT-001", description="Unique identifier for applicant")
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ...,
        ge=0.0,
        description="Total balance on credit cards and personal lines of credit divided by the sum of credit limits"
    )
    age: int = Field(..., ge=18, le=120, description="Age of borrower in years")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ...,
        ge=0,
        alias="NumberOfTime30-59DaysPastDueNotWorse",
        description="Number of times borrower has been 30-59 days past due but no worse in the last 2 years"
    )
    DebtRatio: float = Field(
        ...,
        ge=0.0,
        description="Monthly debt payments, alimony, living costs divided by monthly gross income (or monthly debt amount if income is unverified)"
    )
    MonthlyIncome: float | None = Field(
        default=None,
        ge=0.0,
        description="Monthly gross income in USD (optional, median imputed if omitted)"
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ...,
        ge=0,
        description="Number of open loans (installment, auto, mortgage) and lines of credit (credit cards)"
    )
    NumberOfTimes90DaysLate: int = Field(
        ...,
        ge=0,
        description="Number of times borrower has been 90 days or more past due"
    )
    NumberRealEstateLoansOrLines: int = Field(
        ...,
        ge=0,
        description="Number of mortgage and real estate loans including home equity lines of credit"
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ...,
        ge=0,
        alias="NumberOfTime60-89DaysPastDueNotWorse",
        description="Number of times borrower has been 60-89 days past due but no worse in the last 2 years"
    )
    NumberOfDependents: float | None = Field(
        default=0.0,
        ge=0.0,
        description="Number of dependents in family excluding applicant (children, spouse etc.)"
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "application_id": "APP-2026-8941",
                "RevolvingUtilizationOfUnsecuredLines": 0.28,
                "age": 38,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.32,
                "MonthlyIncome": 6800.0,
                "NumberOfOpenCreditLinesAndLoans": 9,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2
            }
        }
    }


class ReasonCode(BaseModel):
    rank: int
    feature: str
    feature_value: float
    shap_impact: float
    reason_code: str


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class SingleDecisionResponse(BaseModel):
    application_id: str
    decision: str  # APPROVE, REFER, REJECT
    probability_of_default: float
    risk_tier: str
    risk_tier_label: str
    risk_tier_color: str
    recommended_interest_rate: float
    recommended_rate_display: str
    action_summary: str
    reason_codes: list[str]
    model_version: str
    decision_timestamp: str
    latency_ms: float


class ExplainDecisionResponse(SingleDecisionResponse):
    base_value: float
    detailed_reason_codes: list[ReasonCode]
    all_feature_contributions: list[FeatureContribution]


class BatchApplicationRequest(BaseModel):
    applications: list[ApplicationInput]


class BatchDecisionResponse(BaseModel):
    total_processed: int
    approved_count: int
    referred_count: int
    rejected_count: int
    decisions: list[SingleDecisionResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_type: str
    cv_auc: float
    test_auc: float
    decision_thresholds: dict[str, float]
    uptime_status: str
