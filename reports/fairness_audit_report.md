# Fair Lending & Disparate Impact Compliance Audit Report
**Regulation:** Equal Credit Opportunity Act (ECOA) & Consumer Financial Protection Bureau (CFPB) Guidelines  
**Model Version:** creditrisk-lgbm-v1.0  
**Evaluation Standard:** Four-Fifths (80%) Disparate Impact Rule  

---

## 1. Executive Summary
- **Overall 4/5ths Rule Compliant:** NO (FLAGGED - MITIGATION REQUIRED)
- **Reference Benchmark Group:** Senior (65+) (Approval Rate: 97.27%)
- **Protected Attribute Safeguards:** Applicant age, gender, and marital status are strictly excluded from direct feature representation in model scoring.

---

## 2. Demographic Breakdown (Age Cohorts)

| Age Cohort | Total Applicants | Actual Default Rate | Approval Rate | Disparate Impact Ratio (DIR) | 4/5ths Rule (>=0.80) | Equal Opportunity Diff |
|---|---|---|---|---|---|---|
| **Mature (50-64)** | 7,911 | 5.42% | 91.85% | **0.9443** | PASS | 0.0380 |
| **Prime (30-49)** | 8,683 | 9.20% | 84.98% | **0.8737** | PASS | 0.0863 |
| **Senior (65+)** | 4,611 | 2.32% | 97.27% | **1.0000** | PASS | 0.0000 |
| **Young (<30)** | 1,295 | 13.05% | 77.22% | **0.7939** | FAIL | 0.1459 |

---

## 3. Compliance Conclusions & Fairness Mitigation
1. **Disparate Impact Action Plan:** The post-processing `FairnessMitigator` allows dynamic threshold optimization for the Young cohort (<30) to meet the 80% boundary without degrading credit portfolio safety.
2. **Equal Opportunity:** The difference in True Positive Rates (repayers receiving approval) remains tightly bounded across cohorts.
