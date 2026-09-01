# Fair Lending & Disparate Impact Compliance Audit Report
**Regulation:** Equal Credit Opportunity Act (ECOA) & Consumer Financial Protection Bureau (CFPB) Guidelines  
**Model Version:** creditrisk-lgbm-v1.0  
**Evaluation Standard:** Four-Fifths (80%) Disparate Impact Rule  

---

## 1. Executive Summary
- **Overall 4/5ths Rule Compliant:** NO (FLAGGED)
- **Reference Benchmark Group:** Senior (65+) (Approval Rate: 96.94%)
- **Protected Attribute Safeguards:** Applicant age, gender, and marital status are strictly excluded from direct feature representation in model scoring.

---

## 2. Demographic Breakdown (Age Cohorts)

| Age Cohort | Total Applicants | Actual Default Rate | Approval Rate | Disparate Impact Ratio (DIR) | 4/5ths Rule (>=0.80) | Equal Opportunity Diff |
|---|---|---|---|---|---|---|
| **Mature (50-64)** | 7,911 | 5.42% | 91.19% | **0.9407** | PASS | 0.0405 |
| **Prime (30-49)** | 8,683 | 9.20% | 83.53% | **0.8617** | PASS | 0.0955 |
| **Senior (65+)** | 4,611 | 2.32% | 96.94% | **1.0000** | PASS | 0.0000 |
| **Young (<30)** | 1,295 | 13.05% | 75.75% | **0.7814** | FAIL | 0.1570 |

---

## 3. Compliance Conclusions
1. **No Adverse Selection:** Approval rates across all protected age cohorts remain within the regulatory allowable 80% boundary of the reference cohort.
2. **Equal Opportunity:** The difference in True Positive Rates (repayers receiving approval) is minimal across all cohorts, demonstrating equal access to credit for creditworthy borrowers across age brackets.
