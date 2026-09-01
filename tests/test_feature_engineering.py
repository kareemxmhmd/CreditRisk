"""
Unit tests for credit feature engineering.
"""

import pandas as pd

from src.config import ALL_FEATURE_COLS
from src.feature_engineering import CreditFeatureEngineer


def test_feature_engineering_transforms():
    sample_df = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": [0.3, 0.9],
        "age": [37, 57],
        "NumberOfTime30-59DaysPastDueNotWorse": [1, 2],
        "DebtRatio": [0.4, 0.8],
        "MonthlyIncome": [5000.0, 3000.0],
        "NumberOfOpenCreditLinesAndLoans": [6, 12],
        "NumberOfTimes90DaysLate": [0, 1],
        "NumberRealEstateLoansOrLines": [1, 2],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 1],
        "NumberOfDependents": [1.0, 2.0],
        "MonthlyIncome_is_missing": [0, 0],
        "NumberOfDependents_is_missing": [0, 0],
        "DelinquencyAnomalyFlag": [0, 0],
    })

    engineer = CreditFeatureEngineer()
    feat_df = engineer.transform(sample_df)

    # Check that all expected features are present
    for col in ALL_FEATURE_COLS:
        assert col in feat_df.columns

    # Row 0: Total Delinquencies = 1 + 0 + 0 = 1
    assert feat_df.loc[0, "TotalDelinquencies"] == 1
    # Row 1: Total Delinquencies = 2 + 1 + 1 = 4
    assert feat_df.loc[1, "TotalDelinquencies"] == 4

    # Income per dependent: Row 0 = 5000 / (1 + 1) = 2500
    assert feat_df.loc[0, "IncomePerDependent"] == 2500.0

    # High utilization flag
    assert feat_df.loc[0, "HighUtilizationFlag"] == 0
    assert feat_df.loc[1, "HighUtilizationFlag"] == 1

    # Disposable income
    assert feat_df.loc[0, "DisposableIncome"] > 0
