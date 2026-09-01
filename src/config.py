"""
Configuration and constants for the CreditRisk decisioning engine.
"""

from pathlib import Path
from typing import Any

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_TRAIN_DATA_PATH = DATA_DIR / "cs-training.csv"
RAW_TEST_DATA_PATH = DATA_DIR / "cs-test.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data_processed"

# Artifacts & Reports
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Database path for decision audit logging
DB_PATH = BASE_DIR / "audit_decisions.db"

# Model Version
MODEL_VERSION = "creditrisk-lgbm-v1.0"

# Target variable
TARGET_COL = "SeriousDlqin2yrs"
ID_COL = "Unnamed: 0"

# Raw feature names
RAW_NUMERIC_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Engineered feature names
ENGINEERED_FEATURES = [
    "TotalDelinquencies",
    "SevereDelinquencyRatio",
    "IncomePerDependent",
    "MonthlyDebtAmount",
    "DisposableIncome",
    "CreditLineDensity",
    "RealEstateLoanRatio",
    "HighUtilizationFlag",
    "DelinquencyAnomalyFlag",
    "MonthlyIncome_is_missing",
    "NumberOfDependents_is_missing",
    "Utilization_x_DebtRatio",
]

ALL_FEATURE_COLS = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES

# Default Cost-Sensitive Matrix Parameters
# Loan Assumptions:
DEFAULT_LOAN_AMOUNT = 10000.0       # Average loan principal ($)
DEFAULT_INTEREST_RATE = 0.15        # Expected return on repaid loan (15% = +$1,500)
DEFAULT_RECOVERY_RATE = 0.10        # Fraction of default loan recovered (10% -> Loss Given Default = 90% = -$9,000)

# Decision Threshold Defaults (Can be tuned/overridden)
DEFAULT_APPROVE_THRESHOLD = 0.040   # PD < 4.0% -> Approve
DEFAULT_REJECT_THRESHOLD = 0.120    # PD >= 12.0% -> Reject (Between 4% and 12% -> Refer)

# Risk Tiers Configuration
RISK_TIERS: dict[str, dict[str, Any]] = {
    "LOW": {
        "max_pd": 0.025,
        "label": "Low Risk (Prime)",
        "base_rate": 0.075,  # 7.5% APR
        "badge_color": "green",
        "description": "High creditworthiness, strong debt coverage, no recent delinquency."
    },
    "MODERATE": {
        "max_pd": 0.060,
        "label": "Moderate Risk (Near-Prime)",
        "base_rate": 0.115,  # 11.5% APR
        "badge_color": "blue",
        "description": "Satisfactory credit profile, moderate utilization or debt ratio."
    },
    "HIGH": {
        "max_pd": 0.150,
        "label": "High Risk (Subprime)",
        "base_rate": 0.165,  # 16.5% APR
        "badge_color": "orange",
        "description": "Elevated probability of default, multiple risk indicators present."
    },
    "CRITICAL": {
        "max_pd": 1.000,
        "label": "Critical Risk (Deep Subprime)",
        "base_rate": 0.245,  # 24.5% APR or Decline
        "badge_color": "red",
        "description": "Severe risk of default, significant past delinquency or unmanageable debt."
    }
}

# Adverse Action Reason Code Mapping (for FCRA / ECOA compliance)
# Maps features whose high SHAP values push probability of default upwards to human-readable explanations.
ADVERSE_ACTION_REASONS: dict[str, dict[str, str]] = {
    "RevolvingUtilizationOfUnsecuredLines": {
        "high": "High revolving credit card balance relative to total credit limits (high utilization).",
        "low": "Favorable low credit utilization across revolving lines."
    },
    "NumberOfTimes90DaysLate": {
        "high": "Presence of severe past-due delinquencies (90+ days late) in credit bureau records.",
        "low": "Zero severe past-due delinquencies (90+ days late)."
    },
    "DebtRatio": {
        "high": "High total monthly debt payments and fixed obligations relative to monthly gross income.",
        "low": "Healthy debt-to-income ratio."
    },
    "NumberOfTime30-59DaysPastDueNotWorse": {
        "high": "Recent short-term payment delinquency (30-59 days past due) observed.",
        "low": "Clean short-term payment history."
    },
    "NumberOfTime60-89DaysPastDueNotWorse": {
        "high": "Moderate payment delinquency (60-89 days past due) in recent history.",
        "low": "Clean payment history on moderate-term delinquency."
    },
    "TotalDelinquencies": {
        "high": "Cumulative history of multiple payment delinquencies across all accounts.",
        "low": "Impeccable overall delinquency record."
    },
    "SevereDelinquencyRatio": {
        "high": "High proportion of severe 90+ day delinquent events relative to total late payments.",
        "low": "Low severity delinquency history."
    },
    "MonthlyIncome": {
        "high": "High verified gross monthly income supporting loan repayment.",
        "low": "Insufficient monthly gross income relative to the requested obligation."
    },
    "IncomePerDependent": {
        "high": "Adequate disposable household income buffer per dependent.",
        "low": "Low net household income per dependent family member."
    },
    "DisposableIncome": {
        "high": "Strong monthly disposable cash flow after existing debt service.",
        "low": "Constrained monthly disposable income after servicing current debt obligations."
    },
    "MonthlyDebtAmount": {
        "high": "Substantial existing monthly debt burden requiring significant cash outflow.",
        "low": "Low existing monthly debt obligations."
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "high": "Large number of concurrent open credit lines indicating high potential leverage.",
        "low": "Limited number of established open trade lines."
    },
    "NumberRealEstateLoansOrLines": {
        "high": "Multiple outstanding real estate mortgages or home equity lines of credit.",
        "low": "Manageable or zero real estate mortgage obligations."
    },
    "HighUtilizationFlag": {
        "high": "Revolving credit line utilization exceeds safe 80% threshold.",
        "low": "Revolving credit lines well within standard operating limits."
    },
    "CreditLineDensity": {
        "high": "High concentration of active credit lines relative to adult credit history tenure.",
        "low": "Balanced credit line acquisition pace over adult years."
    },
    "Utilization_x_DebtRatio": {
        "high": "Compound high credit card utilization combined with heavy monthly debt burden.",
        "low": "Healthy combination of low utilization and balanced debt burden."
    }
}

# Enterprise Serving & Champion-Challenger Configuration
CHAMPION_MODEL_NAME = "creditrisk-lgbm-calibrated"
CHALLENGER_XGB_NAME = "creditrisk-xgb-challenger"
CHALLENGER_LR_NAME = "creditrisk-lr-challenger"
DEFAULT_ROUTING_MODE = "champion_only"  # "champion_only", "canary", "shadow"
DEFAULT_CANARY_CHALLENGER_SPLIT = 0.10  # 10% to Challenger in Canary mode

# Enterprise Feature Store Configuration
FEATURE_STORE_CACHE_TTL_SECONDS = 3600
FEATURE_STORE_MAX_CACHE_SIZE = 10000

# Probability Calibration Settings
CALIBRATION_METHOD = "isotonic"  # "isotonic" or "sigmoid"

# Reject Inference Settings
REJECT_INFERENCE_METHODS = ["hard_parceling", "soft_augmentation", "propensity_reweighting"]
DEFAULT_REJECT_INFERENCE_METHOD = "hard_parceling"
REJECT_INFERENCE_PARCELING_DEF_RATE = 0.20  # Assumed default rate among rejects in hard parceling

# Telemetry & Observability
PROMETHEUS_METRICS_NAMESPACE = "creditrisk"
REQUEST_CORRELATION_HEADER = "X-Correlation-ID"
