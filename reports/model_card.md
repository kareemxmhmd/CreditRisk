# Enterprise Model Card: CreditRisk Decisioning Engine
**Model ID:** creditrisk-lgbm-v1.0  
**Champion Model:** creditrisk-lgbm-calibrated  
**Model Type:** Calibrated LightGBM (Isotonic Scaling)  
**Intended Domain:** Retail & Consumer Loan Credit Risk Underwriting  
**Owner:** Senior Data Scientist & Principal ML Engineer  

---

## 1. Intended Use
- **Primary Use Case:** Real-time credit risk evaluation, calibrated probability of default (PD) estimation, 3-tier loan decisioning (Approve, Refer to Manual Review, Reject), risk tiering, and risk-adjusted APR pricing.
- **Explainability:** Automatic generation of top 3 adverse action reason codes for FCRA / ECOA notice compliance.
- **Serving Modes:** Supports 100% Champion, Canary Traffic Splitting (e.g., 90/10), and Shadow Mode execution.

---

## 2. Quantitative Performance Metrics
- **Stratified 5-Fold Cross-Validation AUC:** 0.8632 ± 0.0056
- **Holdout Test Set AUC-ROC:** 0.8723
- **Precision-Recall AUC (PR-AUC):** 0.4057
- **Kolmogorov-Smirnov (KS) Separation:** 0.5894
- **Expected Calibration Error (ECE):** 0.00440
- **Brier Score Loss:** 0.0484
- **Inference Latency (p95):** < 25ms

---

## 3. Decision Policy & Thresholds
- **Optimal Probability Threshold (\tau^*):** 0.2513
- **Automated Approval Cutoff (\tau_approve):** < 0.1759 (PD < 17.59%)
- **Manual Referral Zone (\tau_refer):** [0.1759, 0.3770)
- **Adverse Rejection Cutoff (\tau_reject):** \ge 0.3770 (PD \ge 37.70%)

---

## 4. Explainability & Governance
- Explanations computed via **SHAP TreeExplainer** with fast exact tree path traversal.
- Raw SHAP attributions mapped to Adverse Action reason codes.
- Continuous monitoring with Population Stability Index (PSI) and Prometheus telemetry.
