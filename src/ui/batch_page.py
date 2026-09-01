"""
Streamlit Page: Batch Application Evaluator & Report Generator
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.api.routes import get_service
from src.config import RAW_TEST_DATA_PATH


def render_batch_page():
    st.markdown("## Batch Application Evaluator")
    st.markdown(
        "Upload a batch CSV file of loan applications to score thousands of records simultaneously. "
        "Download enriched decision reports with automated risk tiers, recommended APRs, and Adverse Action reason codes."
    )

    service = get_service()

    # Template generator
    sample_df = pd.DataFrame([
        {
            "application_id": "APP-2026-0001",
            "RevolvingUtilizationOfUnsecuredLines": 0.25,
            "age": 42,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.30,
            "MonthlyIncome": 7500.0,
            "NumberOfOpenCreditLinesAndLoans": 9,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 2,
        },
        {
            "application_id": "APP-2026-0002",
            "RevolvingUtilizationOfUnsecuredLines": 0.92,
            "age": 31,
            "NumberOfTime30-59DaysPastDueNotWorse": 2,
            "DebtRatio": 0.75,
            "MonthlyIncome": 3200.0,
            "NumberOfOpenCreditLinesAndLoans": 4,
            "NumberOfTimes90DaysLate": 1,
            "NumberRealEstateLoansOrLines": 0,
            "NumberOfTime60-89DaysPastDueNotWorse": 1,
            "NumberOfDependents": 1,
        }
    ])
    csv_template = sample_df.to_csv(index=False)

    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        st.download_button(
            label="Download CSV Template",
            data=csv_template,
            file_name="creditrisk_batch_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Upload options
    upload_tab1, upload_tab2 = st.tabs(["Upload Custom CSV", "Load Kaggle Test Sample"])

    input_df = None

    with upload_tab1:
        uploaded_file = st.file_uploader("Upload CSV containing applicant records:", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(input_df):,} applicant records.")

    with upload_tab2:
        if st.button("Load 500 Samples from Test Dataset", key="btn_load_sample"):
            if RAW_TEST_DATA_PATH.exists():
                raw_sample = pd.read_csv(RAW_TEST_DATA_PATH, nrows=500)
                if "Unnamed: 0" in raw_sample.columns:
                    raw_sample["application_id"] = "APP-TEST-" + raw_sample["Unnamed: 0"].astype(str)
                input_df = raw_sample
                st.session_state["loaded_sample_df"] = input_df
                st.success(f"Loaded {len(input_df)} records from test dataset.")

        if "loaded_sample_df" in st.session_state and input_df is None:
            input_df = st.session_state["loaded_sample_df"]

    if input_df is not None:
        st.markdown("### Raw Input Data Preview")
        st.dataframe(input_df.head(5), use_container_width=True)

        if st.button("Process Batch Scoring", type="primary", use_container_width=True):
            with st.spinner(f"Scoring {len(input_df):,} applications and computing explainability..."):
                clean_df = service.cleaner.transform(input_df)
                feat_df = service.feature_engineer.transform(clean_df)

                # Batch model inference
                probas = service.model.predict_proba(feat_df)[:, 1]

                results = []
                for idx, p in enumerate(probas):
                    d_eval = service.decision_engine.evaluate(float(p))
                    dec = d_eval["decision"]
                    
                    # Generate top reason codes
                    exp = service.explainer.explain_instance(feat_df.iloc[[idx]], top_n=2, decision=dec)
                    reasons_str = "; ".join(exp["plain_reason_texts"])

                    app_id = input_df.iloc[idx].get("application_id", f"APP-{idx+1}")

                    results.append({
                        "Application ID": app_id,
                        "Decision": dec,
                        "Probability of Default": round(float(p), 4),
                        "Risk Tier": d_eval["risk_tier_label"],
                        "Recommended Rate": d_eval["recommended_rate_display"],
                        "Primary Reason Codes": reasons_str,
                    })

                scored_df = pd.DataFrame(results)
                st.session_state["batch_scored_df"] = scored_df

        # Display Batch Results
        if "batch_scored_df" in st.session_state:
            scored_df = st.session_state["batch_scored_df"]
            st.markdown("### Batch Scoring Results")

            # Metrics
            b1, b2, b3, b4 = st.columns(4)
            n_tot = len(scored_df)
            n_app = (scored_df["Decision"] == "APPROVE").sum()
            n_ref = (scored_df["Decision"] == "REFER").sum()
            n_rej = (scored_df["Decision"] == "REJECT").sum()

            with b1:
                st.metric("Total Applications", f"{n_tot:,}")
            with b2:
                st.metric("Approved", f"{n_app:,} ({n_app/n_tot*100:.1f}%)")
            with b3:
                st.metric("Referred (Review)", f"{n_ref:,} ({n_ref/n_tot*100:.1f}%)")
            with b4:
                st.metric("Rejected", f"{n_rej:,} ({n_rej/n_tot*100:.1f}%)")

            # Chart breakdown
            fig_pie = px.pie(
                scored_df,
                names="Decision",
                title="Batch Decision Distribution",
                color="Decision",
                color_discrete_map={"APPROVE": "#28a745", "REFER": "#ffc107", "REJECT": "#dc3545"}
            )
            fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

            # Filterable Table
            filter_dec = st.multiselect("Filter Table by Decision:", ["APPROVE", "REFER", "REJECT"], default=["APPROVE", "REFER", "REJECT"])
            filtered_df = scored_df[scored_df["Decision"].isin(filter_dec)]
            st.dataframe(filtered_df, use_container_width=True)

            # Download Enriched Decision Report
            csv_output = scored_df.to_csv(index=False)
            st.download_button(
                label="Download Full Scored Decisions Report (CSV)",
                data=csv_output,
                file_name="creditrisk_scored_decisions_report.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
