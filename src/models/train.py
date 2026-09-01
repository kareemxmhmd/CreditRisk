"""
Model training and artifact generation pipeline for CreditRisk decisioning engine.
Trains Baseline Rule Scorer, Logistic Regression, XGBoost, and Champion LightGBM with:
1. Stratified 5-Fold Cross-Validation with Out-of-Fold (OOF) probability generation.
2. Isotonic/Platt Probability Calibration for accurate Empirical Default Rate alignment.
3. Leak-free OOF Cost-Sensitive Threshold Optimization.
4. Selection-bias correction via Reject Inference Augmentation.
5. TreeExplainer and Fairness Auditing.
"""

import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ALL_FEATURE_COLS,
    ARTIFACTS_DIR,
    CHAMPION_MODEL_NAME,
    MODEL_VERSION,
    RAW_TRAIN_DATA_PATH,
    REPORTS_DIR,
    TARGET_COL,
)
from src.data_pipeline import DataCleaner, load_raw_train_data, split_data
from src.decision_engine.cost_matrix import CostMatrix, ThresholdOptimizer
from src.fairness.audit import FairnessAuditor
from src.feature_engineering import CreditFeatureEngineer
from src.models.baseline_rules import LegacyRuleBasedScorer
from src.models.calibration import ProbabilityCalibrator, compute_expected_calibration_error
from src.models.evaluate import evaluate_model_performance
from src.models.reject_inference import RejectInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate_all():
    """
    Execute end-to-end training, hyperparameter fitting, OOF probability calibration,
    threshold tuning, reject inference, SHAP explainer initialization, and artifact generation.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Loading raw training dataset from %s", RAW_TRAIN_DATA_PATH)
    raw_df = load_raw_train_data()
    logger.info("Loaded raw training data with shape: %s", raw_df.shape)

    # Step 2: Split data before fitting transformations to prevent data leakage
    logger.info("Step 2: Splitting data into Stratified Train (70%), Val (15%), and Test (15%)")
    train_raw, val_raw, test_raw = split_data(raw_df, test_size=0.15, val_size=0.15, random_state=42)

    # Step 3: Fit DataCleaner on training set only
    logger.info("Step 3: Fitting DataCleaner on training partition...")
    cleaner = DataCleaner()
    cleaner.fit(train_raw)
    joblib.dump(cleaner, ARTIFACTS_DIR / "cleaner_pipeline.joblib")

    train_cleaned = cleaner.transform(train_raw)
    val_cleaned = cleaner.transform(val_raw)
    test_cleaned = cleaner.transform(test_raw)

    # Step 4: Feature Engineering
    logger.info("Step 4: Applying domain Feature Engineering...")
    feature_engineer = CreditFeatureEngineer()
    feature_engineer.fit(train_cleaned)
    joblib.dump(feature_engineer, ARTIFACTS_DIR / "feature_engineer.joblib")

    X_train = feature_engineer.transform(train_cleaned)
    y_train = train_raw[TARGET_COL].values

    X_val = feature_engineer.transform(val_cleaned)
    y_val = val_raw[TARGET_COL].values

    X_test = feature_engineer.transform(test_cleaned)
    y_test = test_raw[TARGET_COL].values

    logger.info("Engineered feature set size: %d columns", X_train.shape[1])
    logger.info("Features: %s", list(X_train.columns))

    cost_matrix = CostMatrix()

    # Step 5: Train Baseline Models
    logger.info("Step 5.1: Evaluating Legacy Rule-Based Baseline...")
    legacy_scorer = LegacyRuleBasedScorer()
    legacy_test_proba = legacy_scorer.predict_proba(test_raw)[:, 1]
    legacy_eval = evaluate_model_performance(
        "Legacy Baseline", y_test, legacy_test_proba, threshold=0.50, cost_matrix=cost_matrix
    )

    logger.info("Step 5.2: Training & Evaluating Logistic Regression Baseline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    lr_preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, ALL_FEATURE_COLS)],
        remainder='drop'
    )
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', lr_preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    joblib.dump(lr_pipeline, ARTIFACTS_DIR / "baseline_lr_model.joblib")

    lr_test_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_eval = evaluate_model_performance(
        "Baseline Logistic Regression", y_test, lr_test_proba, threshold=0.05, cost_matrix=cost_matrix
    )
    logger.info("Logistic Regression Test AUC: %.4f, KS: %.4f, ECE: %.4f",
                lr_eval["auc_roc"], lr_eval["ks_statistic"], lr_eval["expected_calibration_error"])

    # Step 5.3: Train XGBoost Challenger Model
    logger.info("Step 5.3: Training XGBoost Challenger Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.5,
        random_state=42,
        eval_metric="auc",
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(xgb_model, ARTIFACTS_DIR / "challenger_xgb_model.joblib")

    xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_eval = evaluate_model_performance(
        "XGBoost Classifier", y_test, xgb_test_proba, threshold=0.05, cost_matrix=cost_matrix
    )
    logger.info("XGBoost Test AUC: %.4f, KS: %.4f, ECE: %.4f",
                xgb_eval["auc_roc"], xgb_eval["ks_statistic"], xgb_eval["expected_calibration_error"])

    # Step 6: Train Champion LightGBM with 5-Fold Stratified CV and OOF Predictions
    logger.info("Step 6: Training Champion LightGBM with 5-Fold Stratified CV & OOF Generation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    oof_raw_probas = np.zeros(len(X_train), dtype=float)

    lgb_params = {
        "n_estimators": 350,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "random_state": 42,
        "n_jobs": -1,
        "importance_type": "gain",
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train[train_idx]
        X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train[val_idx]

        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            X_fold_train,
            y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        fold_proba = clf.predict_proba(X_fold_val)[:, 1]
        oof_raw_probas[val_idx] = fold_proba
        fold_eval = evaluate_model_performance(f"LGBM Fold {fold}", y_fold_val, fold_proba)
        cv_scores.append(fold_eval["auc_roc"])
        logger.info("Fold %d AUC: %.4f | ECE: %.4f", fold, fold_eval["auc_roc"], fold_eval["expected_calibration_error"])

    mean_cv_auc = float(np.mean(cv_scores))
    logger.info("Mean 5-Fold CV AUC for LightGBM: %.4f (std: %.4f)", mean_cv_auc, float(np.std(cv_scores)))

    # Train final Champion model on all training data
    champion_lgbm = lgb.LGBMClassifier(**lgb_params)
    champion_lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )
    joblib.dump(champion_lgbm, ARTIFACTS_DIR / "champion_lgbm_model.joblib")

    # Step 7: Fit Probability Calibrator on Out-of-Fold (OOF) predictions
    logger.info("Step 7: Fitting Isotonic Probability Calibrator on OOF predictions...")
    calibrator = ProbabilityCalibrator(base_estimator=champion_lgbm, method="isotonic")
    calibrator.fit(oof_raw_probas, y_train)
    joblib.dump(calibrator, ARTIFACTS_DIR / "calibrated_champion_lgbm.joblib")

    oof_calibrated_probas = calibrator.calibrate_probas(oof_raw_probas)
    raw_ece, _ = compute_expected_calibration_error(y_train, oof_raw_probas)
    cal_ece, cal_curve = compute_expected_calibration_error(y_train, oof_calibrated_probas)
    logger.info("Raw OOF ECE: %.5f -> Calibrated OOF ECE: %.5f (Calibration error reduced by %.1f%%)",
                raw_ece, cal_ece, (1.0 - cal_ece / max(raw_ece, 1e-6)) * 100)

    # Step 8: Leak-Free Threshold Optimization on OOF Calibrated Probabilities
    logger.info("Step 8: Optimizing Cost-Sensitive Thresholds on OOF Calibrated Probabilities...")
    optimizer = ThresholdOptimizer(cost_matrix=cost_matrix)
    opt_result = optimizer.find_optimal_threshold(y_train, oof_calibrated_probas)

    opt_binary_thresh = opt_result["optimal_binary_threshold"]
    approve_thresh = opt_result["approve_threshold"]
    reject_thresh = opt_result["reject_threshold"]

    logger.info("Optimal Binary Cutoff: %.4f", opt_binary_thresh)
    logger.info("Tuned 3-Tier Thresholds: Approve < %.4f, Refer [%.4f, %.4f), Reject >= %.4f",
                approve_thresh, approve_thresh, reject_thresh, reject_thresh)

    threshold_config = {
        "optimal_binary_threshold": opt_binary_thresh,
        "approve_threshold": approve_thresh,
        "reject_threshold": reject_thresh,
        "loan_amount": cost_matrix.loan_amount,
        "interest_rate": cost_matrix.interest_rate,
        "recovery_rate": cost_matrix.recovery_rate,
    }
    with open(ARTIFACTS_DIR / "threshold_config.json", "w") as f:
        json.dump(threshold_config, f, indent=2)

    # Step 9: Selection Bias Correction via Reject Inference
    logger.info("Step 9: Evaluating Reject Inference (Parceling & Selection Bias Correction)...")
    # Simulate unobserved rejected applicants (top risk quintile from val set treated as unaccepted candidates)
    val_raw_p = champion_lgbm.predict_proba(X_val)[:, 1]
    rej_candidate_mask = val_raw_p >= 0.15
    X_sim_rejected = X_val[rej_candidate_mask]
    
    rej_engine = RejectInferenceEngine(method="hard_parceling", assumed_reject_default_rate=0.25)
    X_aug, y_aug, weights_aug = rej_engine.generate_augmented_dataset(
        X_train, y_train, X_sim_rejected, base_scorer=champion_lgbm
    )
    rej_lgbm = lgb.LGBMClassifier(**lgb_params)
    rej_lgbm.fit(X_aug, y_aug, sample_weight=weights_aug)
    joblib.dump(rej_lgbm, ARTIFACTS_DIR / "reject_inference_lgbm.joblib")
    logger.info("Reject Inference model trained on augmented dataset (%d total records).", len(X_aug))

    # Step 10: Final Test Set Evaluation of Calibrated Champion Model
    logger.info("Step 10: Evaluating Final Calibrated Champion on Holdout Test Set...")
    test_raw_proba = champion_lgbm.predict_proba(X_test)[:, 1]
    test_calibrated_proba = calibrator.calibrate_probas(test_raw_proba)

    champion_eval = evaluate_model_performance(
        "Champion LightGBM (Calibrated)",
        y_test,
        test_calibrated_proba,
        threshold=approve_thresh,
        cost_matrix=cost_matrix
    )
    logger.info("Champion LightGBM Test AUC: %.4f | KS: %.4f | ECE: %.5f",
                champion_eval["auc_roc"], champion_eval["ks_statistic"], champion_eval["expected_calibration_error"])
    logger.info("Champion LightGBM Test Expected Profit per 1000 apps: $%.2f (vs Legacy $%.2f)",
                champion_eval["financial_impact"]["profit_per_1000_applications"],
                legacy_eval["financial_impact"]["profit_per_1000_applications"])

    # Step 11: Initialize and Cache SHAP TreeExplainer
    logger.info("Step 11: Initializing and caching SHAP TreeExplainer...")
    sample_background = X_train.sample(min(200, len(X_train)), random_state=42)
    joblib.dump(sample_background, ARTIFACTS_DIR / "sample_background.joblib")

    shap_explainer = shap.TreeExplainer(champion_lgbm)
    joblib.dump(shap_explainer, ARTIFACTS_DIR / "shap_explainer.joblib")

    # Step 12: Demographic Fairness & Disparate Impact Audit
    logger.info("Step 12: Executing Demographic Fairness & Disparate Impact Audit...")
    fairness_auditor = FairnessAuditor(sensitive_column="age")
    test_decisions = np.where(
        test_calibrated_proba < approve_thresh,
        "APPROVE",
        np.where(test_calibrated_proba < reject_thresh, "REFER", "REJECT")
    )
    fairness_results = fairness_auditor.run_fairness_audit(test_raw, y_test, test_decisions)
    logger.info("Fairness Audit Complete. 4/5ths Rule Compliant: %s", fairness_results["overall_four_fifths_compliant"])

    # Step 13: Save Comprehensive Model Metadata
    model_metadata = {
        "model_version": MODEL_VERSION,
        "champion_name": CHAMPION_MODEL_NAME,
        "model_type": "Calibrated LightGBM (Isotonic)",
        "n_features": len(ALL_FEATURE_COLS),
        "feature_names": ALL_FEATURE_COLS,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "cv_mean_auc": round(mean_cv_auc, 4),
        "cv_std_auc": round(float(np.std(cv_scores)), 4),
        "test_auc": champion_eval["auc_roc"],
        "test_pr_auc": champion_eval["pr_auc"],
        "test_ks_statistic": champion_eval["ks_statistic"],
        "test_brier_score": champion_eval["brier_score"],
        "test_ece": champion_eval["expected_calibration_error"],
        "raw_oof_ece": round(raw_ece, 5),
        "calibrated_oof_ece": round(cal_ece, 5),
        "reliability_curve": cal_curve,
        "thresholds": threshold_config,
        "comparison": {
            "legacy_baseline": {
                "auc_roc": legacy_eval["auc_roc"],
                "profit_per_1000": legacy_eval["financial_impact"]["profit_per_1000_applications"],
                "approval_rate": legacy_eval["financial_impact"]["approval_rate"],
                "ece": legacy_eval["expected_calibration_error"],
            },
            "logistic_regression": {
                "auc_roc": lr_eval["auc_roc"],
                "profit_per_1000": lr_eval["financial_impact"]["profit_per_1000_applications"],
                "approval_rate": lr_eval["financial_impact"]["approval_rate"],
                "ece": lr_eval["expected_calibration_error"],
            },
            "xgboost": {
                "auc_roc": xgb_eval["auc_roc"],
                "profit_per_1000": xgb_eval["financial_impact"]["profit_per_1000_applications"],
                "approval_rate": xgb_eval["financial_impact"]["approval_rate"],
                "ece": xgb_eval["expected_calibration_error"],
            },
            "champion_lgbm": {
                "auc_roc": champion_eval["auc_roc"],
                "profit_per_1000": champion_eval["financial_impact"]["profit_per_1000_applications"],
                "approval_rate": champion_eval["financial_impact"]["approval_rate"],
                "ece": champion_eval["expected_calibration_error"],
            }
        },
        "fairness_audit": fairness_results,
    }

    with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)

    logger.info("All enterprise model artifacts and metadata saved successfully to %s", ARTIFACTS_DIR)
    return model_metadata
