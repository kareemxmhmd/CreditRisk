"""
Feature engineering pipeline for CreditRisk scoring engine.
Transforms cleaned raw inputs into domain-rich risk signals.
"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import ALL_FEATURE_COLS


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer for credit risk data.
    """

    def __init__(self, feature_cols: list[str] | None = None):
        self.feature_cols = feature_cols or ALL_FEATURE_COLS

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Total Delinquencies count across all 3 buckets
        df["TotalDelinquencies"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"].fillna(0)
            + df["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0)
            + df["NumberOfTimes90DaysLate"].fillna(0)
        )

        # 2. Severe Delinquency Ratio (proportion of late events that are >= 90 days)
        df["SevereDelinquencyRatio"] = df["NumberOfTimes90DaysLate"].fillna(0) / (df["TotalDelinquencies"] + 1.0)

        # 3. Income per dependent (living buffer)
        df["IncomePerDependent"] = df["MonthlyIncome"].fillna(5400.0) / (df["NumberOfDependents"].fillna(0) + 1.0)

        # 4. Estimated Monthly Debt Amount
        # In Give Me Some Credit, if DebtRatio > 5 or income was missing, DebtRatio represents raw dollar debt.
        # Otherwise, DebtRatio * MonthlyIncome gives actual monthly debt obligation.
        income = df["MonthlyIncome"].fillna(5400.0)
        debt_ratio = df["DebtRatio"].fillna(0.35)
        is_missing_inc = (df["MonthlyIncome_is_missing"] == 1) if "MonthlyIncome_is_missing" in df.columns else False
        
        is_raw_dollar_debt = (debt_ratio > 5.0) | is_missing_inc
        df["MonthlyDebtAmount"] = np.where(
            is_raw_dollar_debt,
            debt_ratio,
            debt_ratio * income
        )
        # Clip debt to sensible upper bound
        df["MonthlyDebtAmount"] = np.clip(df["MonthlyDebtAmount"], 0.0, 100000.0)

        # 5. Disposable Income after debt obligations
        df["DisposableIncome"] = income - df["MonthlyDebtAmount"]

        # 6. Credit Line Density (lines per year of adult life)
        adult_years = np.maximum(df["age"].fillna(50) - 17.0, 1.0)
        df["CreditLineDensity"] = df["NumberOfOpenCreditLinesAndLoans"].fillna(8) / adult_years

        # 7. Real Estate Loan Ratio (proportion of open lines that are real estate mortgages)
        df["RealEstateLoanRatio"] = (
            df["NumberRealEstateLoansOrLines"].fillna(1)
            / (df["NumberOfOpenCreditLinesAndLoans"].fillna(8) + 1.0)
        )

        # 8. High Utilization Flag (> 80% utilization is standard critical threshold)
        df["HighUtilizationFlag"] = (
            df["RevolvingUtilizationOfUnsecuredLines"].fillna(0.15) > 0.80
        ).astype(int)

        # 9. Compound Interaction: Utilization * DebtRatio (clipped)
        clipped_dr = np.clip(debt_ratio, 0.0, 5.0)
        df["Utilization_x_DebtRatio"] = df["RevolvingUtilizationOfUnsecuredLines"].fillna(0.15) * clipped_dr

        # Ensure missing flags exist
        if "MonthlyIncome_is_missing" not in df.columns:
            df["MonthlyIncome_is_missing"] = 0
        if "NumberOfDependents_is_missing" not in df.columns:
            df["NumberOfDependents_is_missing"] = 0
        if "DelinquencyAnomalyFlag" not in df.columns:
            df["DelinquencyAnomalyFlag"] = 0

        # Return dataframe with strictly the expected features
        return df[self.feature_cols]
