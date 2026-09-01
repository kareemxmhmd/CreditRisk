# CreditRisk Engine Audit Report

## 1. What Was Audited & Verified
- **End-to-End Pipeline**: Verified that `python run_all.py` executes successfully, generating all expected artifacts (model files, config, thresholds, SHAP explainer) and reports without errors.
- **FastAPI Endpoints**: Confirmed that the API at `src/api/` exposes all documented endpoints (`/health`, `/predict`, `/explain`, `/batch-predict`, `/metrics`, `/drift`, `/audit-logs`) using proper Pydantic request/response schemas.
- **Streamlit UI**: Verified that the Streamlit application starts successfully and renders all 5 views.
- **Automated Tests**: Confirmed that running `pytest tests/ -v` successfully passes all 14 tests, covering endpoints, feature engineering, the decision engine, pipeline data cleaning, explainability, and fairness checks.
- **Code Quality**: Verified the presence of type hints, docstrings, lack of hardcoded secrets, and proper configuration abstraction via `src/config.py`.

## 2. What Was Broken & Fixed
1. **Inaccurate README Metrics**: The README presented aspirational or outdated metrics. It claimed an expected net profit of +$1,250,000/1K apps, an AUC of 0.864, and a KS of 0.582. The actual run yields +$932,266/1K apps, AUC 0.872, and KS 0.589. The legacy baseline profit was corrected from +$820k to +$330k.
2. **Inaccurate README Fairness Results**: The README falsely claimed the 4/5ths Rule Fairness Audit was a `PASS` across the board. The actual audit (`reports/fairness_audit_report.md`) shows that the `Young (<30)` cohort has a Disparate Impact Ratio of `0.781` which is a `FAIL` (flagged). The README was updated to reflect this reality.
3. **Hardcoded Prints**: Discovered `print()` statements in `src/api/app.py`. These were replaced with standard Python `logging`.
4. **Missing GitHub Actions**: Added `.github/workflows/ci.yml` to run `pytest` and `ruff` on pushes and pull requests.
5. **Missing CONTRIBUTING.md**: Created `CONTRIBUTING.md` to establish developer guidelines.

## 3. What Still Needs Manual Attention
- **Full UI/API Decoupling**: The Streamlit UI currently imports the backend model logic directly from `src.api.routes` (e.g., `get_service()`) to perform in-memory inference, rather than routing all application traffic through the FastAPI endpoints via HTTP. While I added an active HTTP healthcheck to `src/ui/app.py` (which successfully verifies that the UI and API containers can talk to each other over the Docker Compose network), a full architectural decouple—refactoring all 5 Streamlit views to use `requests.post("http://api:8000/api/v1/predict")`—should be scheduled for a future sprint to achieve true microservice isolation without risking breaking the current interactive "What-If" simulators.
