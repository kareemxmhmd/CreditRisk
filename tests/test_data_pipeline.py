"""
Unit tests for data pipeline and data cleaning.
"""

import pandas as pd
import numpy as np
from src.data_pipeline import DataCleaner, split_data


def test_data_cleaner_imputation_and_anomalies():
    # Sample raw dataframe with missing values and anomaly codes (96/98)
    sample_df = pd.DataFrame({
        "Unnamed: 0": [1, 2, 3],
        "SeriousDlqin2yrs": [0, 1, 0],
        "RevolvingUtilizationOfUnsecuredLines": [0.2, 55000.0, np.nan],
        "age": [0, 45, 120],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 98, 1],
        "DebtRatio": [0.35, 15000.0, 0.4],
        "MonthlyIncome": [5000.0, np.nan, 8000.0],
        "NumberOfOpenCreditLinesAndLoans": [5, 10, 8],
        "NumberOfTimes90DaysLate": [0, 96, 0],
        "NumberRealEstateLoansOrLines": [1, 0, 2],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 98, 0],
        "NumberOfDependents": [np.nan, 2.0, 0.0]
    })

    cleaner = DataCleaner()
    cleaned = cleaner.fit_transform(sample_df)

    # Check unnamed column dropped
    assert "Unnamed: 0" not in cleaned.columns

    # Check missing flags added
    assert "MonthlyIncome_is_missing" in cleaned.columns
    assert "NumberOfDependents_is_missing" in cleaned.columns
    assert cleaned.loc[1, "MonthlyIncome_is_missing"] == 1
    assert cleaned.loc[0, "NumberOfDependents_is_missing"] == 1

    # Check age 0 was corrected
    assert cleaned.loc[0, "age"] >= 18

    # Check anomaly codes 96/98 were caught
    assert cleaned.loc[1, "DelinquencyAnomalyFlag"] == 1
    assert cleaned.loc[1, "NumberOfTime30-59DaysPastDueNotWorse"] == 0

    # Check outlier utilization clipped
    assert cleaned.loc[1, "RevolvingUtilizationOfUnsecuredLines"] <= 15.0


def test_split_data_proportions():
    sample_df = pd.DataFrame({
        "SeriousDlqin2yrs": [0] * 90 + [1] * 10,
        "age": np.random.randint(20, 70, size=100),
    })

    train_df, val_df, test_df = split_data(sample_df, test_size=0.15, val_size=0.15, random_state=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(sample_df)
    assert 65 <= len(train_df) <= 75
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert train_df["SeriousDlqin2yrs"].sum() > 0
