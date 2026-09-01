import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.ui.loan_officer_page import render_loan_officer_page
from src.ui.risk_manager_page import render_risk_manager_page
from src.ui.fairness_page import render_fairness_page
from src.ui.batch_page import render_batch_page
from src.ui.monitoring_page import render_monitoring_page
from src.config import MODEL_VERSION

# Set page configuration
st.set_page_config(
    page_title="CreditRisk Decisioning Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished fintech design
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## **CreditRisk AI**")
        st.markdown("*Loan Decisioning Engine*")
        st.caption(f"Model Version: `{MODEL_VERSION}`")
        st.markdown("---")

        page = st.radio(
            "Select Workspace View:",
            [
                "Loan Officer Workspace",
                "Risk Manager Portfolio",
                "Fair Lending & Bias Audit",
                "Batch Application Evaluator",
                "Data Drift & Decision Audit"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown("### System Status")
        st.success("API: Online (Port 8000)")
        st.success("Engine: Champion LightGBM")
        st.info("Explainability: SHAP Active")
        st.caption("CreditRisk Decisioning Engine. All rights reserved.")

    # Page Routing
    if page == "Loan Officer Workspace":
        render_loan_officer_page()
    elif page == "Risk Manager Portfolio":
        render_risk_manager_page()
    elif page == "Fair Lending & Bias Audit":
        render_fairness_page()
    elif page == "Batch Application Evaluator":
        render_batch_page()
    elif page == "Data Drift & Decision Audit":
        render_monitoring_page()


if __name__ == "__main__":
    main()
