"""
Data ingestion, cleaning, and preprocessing pipeline for CreditRisk.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ID_COL,
    RAW_TRAIN_DATA_PATH,
    TARGET_COL,
)


class DataCleaner:
    """
    Robust cleaning logic for raw credit risk application datasets.
    Handles anomalous codes (96/98), outlier clipping, and missing value indicators.
    """

    def __init__(self, median_values: dict[str, float] | None = None):
        self.median_values = median_values or {}

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Compute training set statistics for reproducible imputation."""
        # Calculate medians for imputation
        clean_df = df.copy()
        
        # Compute median income on non-missing
        self.median_values["MonthlyIncome"] = float(clean_df["MonthlyIncome"].median(skipna=True))
        self.median_values["NumberOfDependents"] = float(clean_df["NumberOfDependents"].median(skipna=True))
        self.median_values["DebtRatio"] = float(clean_df["DebtRatio"].median(skipna=True))
        self.median_values["RevolvingUtilizationOfUnsecuredLines"] = float(
            clean_df["RevolvingUtilizationOfUnsecuredLines"].median(skipna=True)
        )
        self.median_values["age"] = float(clean_df["age"].median(skipna=True))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning transformations deterministically."""
        data = df.copy()

        # Drop unnamed index column if present
        if ID_COL in data.columns:
            data = data.drop(columns=[ID_COL])

        # Track missingness flags
        if "MonthlyIncome" in data.columns:
            data["MonthlyIncome_is_missing"] = data["MonthlyIncome"].isnull().astype(int)
        else:
            data["MonthlyIncome_is_missing"] = 0

        if "NumberOfDependents" in data.columns:
            data["NumberOfDependents_is_missing"] = data["NumberOfDependents"].isnull().astype(int)
        else:
            data["NumberOfDependents_is_missing"] = 0

        # Handle Age anomalies (e.g. age < 18 or age == 0 -> replace with median age)
        if "age" in data.columns:
            median_age = self.median_values.get("age", 52.0)
            data["age"] = data["age"].apply(lambda x: median_age if (pd.isna(x) or x < 18 or x > 105) else x)

        # Handle delinquency anomaly codes: 96 and 98 in Give Me Some Credit represent error/missing codes
        delinq_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate"
        ]
        
        # Check for delinquency anomalies
        anomaly_mask = pd.Series(False, index=data.index)
        for col in delinq_cols:
            if col in data.columns:
                anomaly_mask = anomaly_mask | (data[col] >= 96)
        data["DelinquencyAnomalyFlag"] = anomaly_mask.astype(int)

        # Clip / replace anomalous 96/98 with 0 (or maximum realistic past due count)
        for col in delinq_cols:
            if col in data.columns:
                data[col] = data[col].apply(lambda x: 0 if (pd.isna(x) or x >= 96) else min(float(x), 20.0))

        # Impute missing values with learned medians
        if "MonthlyIncome" in data.columns:
            med_income = self.median_values.get("MonthlyIncome", 5400.0)
            data["MonthlyIncome"] = data["MonthlyIncome"].fillna(med_income)
            # Clip negative income
            data["MonthlyIncome"] = data["MonthlyIncome"].clip(lower=0.0, upper=500000.0)

        if "NumberOfDependents" in data.columns:
            med_dep = self.median_values.get("NumberOfDependents", 0.0)
            data["NumberOfDependents"] = data["NumberOfDependents"].fillna(med_dep)
            data["NumberOfDependents"] = data["NumberOfDependents"].clip(lower=0.0, upper=20.0)

        # Handle extreme outliers in RevolvingUtilization (clip to 15.0 for stability, ratio > 1 already means > 100% maxed)
        if "RevolvingUtilizationOfUnsecuredLines" in data.columns:
            data["RevolvingUtilizationOfUnsecuredLines"] = data["RevolvingUtilizationOfUnsecuredLines"].fillna(
                self.median_values.get("RevolvingUtilizationOfUnsecuredLines", 0.15)
            ).clip(lower=0.0, upper=15.0)

        # Handle DebtRatio: when DebtRatio is astronomical (> 10000), it often represents raw debt dollar amount in rows missing income
        if "DebtRatio" in data.columns:
            # We preserve high debt signal but cap astronomical values at 10,000 for numeric stability
            data["DebtRatio"] = data["DebtRatio"].fillna(
                self.median_values.get("DebtRatio", 0.36)
            ).clip(lower=0.0, upper=10000.0)

        if "NumberOfOpenCreditLinesAndLoans" in data.columns:
            data["NumberOfOpenCreditLinesAndLoans"] = data["NumberOfOpenCreditLinesAndLoans"].clip(lower=0.0, upper=60.0)

        if "NumberRealEstateLoansOrLines" in data.columns:
            data["NumberRealEstateLoansOrLines"] = data["NumberRealEstateLoansOrLines"].clip(lower=0.0, upper=30.0)

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit cleaner and transform dataset."""
        return self.fit(df).transform(df)


def load_raw_train_data(filepath: str | None = None) -> pd.DataFrame:
    """Load raw training CSV dataset."""
    path = filepath or RAW_TRAIN_DATA_PATH
    df = pd.read_csv(path)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into stratified Train, Validation, and Test sets.
    """
    stratify_col = df[TARGET_COL] if TARGET_COL in df.columns else None

    # First split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state
    )

    # Then split validation from train
    adjusted_val_size = val_size / (1.0 - test_size)
    stratify_train_val = train_val_df[TARGET_COL] if TARGET_COL in train_val_df.columns else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=stratify_train_val,
        random_state=random_state
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
