# Model Card: CreditRisk — AI-Powered Loan Decisioning Engine
**Model ID:** creditrisk-lgbm-v1.0  
**Model Type:** LightGBM Gradient Boosted Decision Trees  
**Intended Domain:** Retail & Consumer Loan Credit Risk Underwriting  
**Owner:** Senior Data Scientist (Credit Risk & Algorithmic Governance)  

---

## 1. Intended Use
- **Primary Use Case:** Real-time credit risk evaluation, probability of default (PD) estimation, 3-tier loan decisioning (Approve, Refer to Manual Review, Reject), risk tiering, and risk-adjusted APR suggestion.
- **Explainability:** Automatic generation of top 3 adverse action reason codes for FCRA / ECOA notice compliance.
- **Out of Scope:** Commercial lending, mortgage underwriting without human-in-the-loop verification, real-time fraud detection.

---

## 2. Quantitative Performance Metrics
- **Stratified 5-Fold Cross-Validation AUC:** 0.8634 ± 0.0058
- **Holdout Test Set AUC-ROC:** 0.8725
- **Precision-Recall AUC (PR-AUC):** 0.4172
- **Kolmogorov-Smirnov (KS) Separation:** 0.5894
- **Brier Score Loss (Calibration):** 0.0483
- **Inference Latency (p95):** < 30ms

---

## 3. Decision Policy & Thresholds
- **Optimal Probability Threshold (\tau^*):** 0.2144
- **Automated Approval Cutoff (\tau_approve):** < 0.1501 (PD < 15.01%)
- **Manual Referral Zone (\tau_refer):** [0.1501, 0.3216)
- **Adverse Rejection Cutoff (\tau_reject):** \ge 0.3216 (PD \ge 32.16%)

---

## 4. Explainability & Governance
- Explanations computed via **SHAP TreeExplainer** with fast exact tree path traversal.
- Raw SHAP attributions mapped to Adverse Action reason codes.
- Continuous monitoring with Population Stability Index (PSI) and Kolmogorov-Smirnov distribution tests.
- Every scored decision logged to SQLite audit log with input snapshots.
