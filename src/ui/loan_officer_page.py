"""
Streamlit Page: Loan Officer Workspace (Single Application Evaluation & Adverse Action Reasons)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.api.routes import get_service, ApplicationInput


# Sample borrower archetypes for quick demonstration
PRESET_PROFILES = {
    "Prime Borrower (Low Risk)": {
        "RevolvingUtilizationOfUnsecuredLines": 0.12,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.22,
        "MonthlyIncome": 9500.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0,
    },
    "Near-Prime / Borderline (Referral)": {
        "RevolvingUtilizationOfUnsecuredLines": 0.65,
        "age": 34,
        "NumberOfTime30-59DaysPastDueNotWorse": 1,
        "DebtRatio": 0.48,
        "MonthlyIncome": 4200.0,
        "NumberOfOpenCreditLinesAndLoans": 10,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2.0,
    },
    "Subprime / High Delinquency (Rejection)": {
        "RevolvingUtilizationOfUnsecuredLines": 0.95,
        "age": 29,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.85,
        "MonthlyIncome": 2800.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 2,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 1,
        "NumberOfDependents": 3.0,
    },
    "Young Professional (Clean History)": {
        "RevolvingUtilizationOfUnsecuredLines": 0.25,
        "age": 26,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.28,
        "MonthlyIncome": 6200.0,
        "NumberOfOpenCreditLinesAndLoans": 4,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0.0,
    },
    "Custom Input": None
}


def render_loan_officer_page():
    st.markdown("## Loan Officer Workspace")
    st.markdown(
        "Evaluate individual loan applications in real time. Receive an automated credit decision, "
        "risk tier, suggested APR, and **regulatory Adverse Action reason codes** derived from SHAP."
    )

    service = get_service()

    # Preset selection
    st.markdown("### 1. Select Application Profile or Enter Custom Data")
    selected_preset = st.selectbox("Borrower Preset Profile:", list(PRESET_PROFILES.keys()))

    preset_data = PRESET_PROFILES[selected_preset]

    col1, col2 = st.columns(2)
    with col1:
        app_id = st.text_input("Application Reference ID:", value="APP-2026-0901")
        utilization = st.number_input(
            "Revolving Credit Utilization (Balance / Limit):",
            min_value=0.0,
            max_value=15.0,
            value=float(preset_data["RevolvingUtilizationOfUnsecuredLines"]) if preset_data else 0.30,
            step=0.05,
            help="e.g. 0.30 represents 30% utilization of total available credit card lines."
        )
        age = st.number_input(
            "Applicant Age (Years):",
            min_value=18,
            max_value=110,
            value=int(preset_data["age"]) if preset_data else 35,
            step=1
        )
        monthly_income = st.number_input(
            "Gross Monthly Income ($):",
            min_value=0.0,
            max_value=500000.0,
            value=float(preset_data["MonthlyIncome"]) if preset_data else 6000.0,
            step=250.0
        )
        debt_ratio = st.number_input(
            "Debt-to-Income Ratio (DTI):",
            min_value=0.0,
            max_value=50.0,
            value=float(preset_data["DebtRatio"]) if preset_data else 0.35,
            step=0.05,
            help="Monthly debt payments divided by gross monthly income."
        )
        dependents = st.number_input(
            "Number of Dependents:",
            min_value=0,
            max_value=20,
            value=int(preset_data["NumberOfDependents"]) if preset_data else 1,
            step=1
        )

    with col2:
        open_lines = st.number_input(
            "Number of Open Credit Lines & Loans:",
            min_value=0,
            max_value=60,
            value=int(preset_data["NumberOfOpenCreditLinesAndLoans"]) if preset_data else 8,
            step=1
        )
        re_lines = st.number_input(
            "Number of Real Estate Mortgages / Lines:",
            min_value=0,
            max_value=30,
            value=int(preset_data["NumberRealEstateLoansOrLines"]) if preset_data else 1,
            step=1
        )
        late_30_59 = st.number_input(
            "30-59 Days Past Due Count (Last 2 Yrs):",
            min_value=0,
            max_value=20,
            value=int(preset_data["NumberOfTime30-59DaysPastDueNotWorse"]) if preset_data else 0,
            step=1
        )
        late_60_89 = st.number_input(
            "60-89 Days Past Due Count (Last 2 Yrs):",
            min_value=0,
            max_value=20,
            value=int(preset_data["NumberOfTime60-89DaysPastDueNotWorse"]) if preset_data else 0,
            step=1
        )
        late_90 = st.number_input(
            "90+ Days Past Due Count (Severe Delinquency):",
            min_value=0,
            max_value=20,
            value=int(preset_data["NumberOfTimes90DaysLate"]) if preset_data else 0,
            step=1
        )

    st.markdown("---")

    # Evaluate Button
    if st.button("Evaluate Credit Decision & Explain", type="primary", use_container_width=True):
        app_input = ApplicationInput(
            application_id=app_id,
            RevolvingUtilizationOfUnsecuredLines=utilization,
            age=age,
            NumberOfTime30_59DaysPastDueNotWorse=late_30_59,
            DebtRatio=debt_ratio,
            MonthlyIncome=monthly_income,
            NumberOfOpenCreditLinesAndLoans=open_lines,
            NumberOfTimes90DaysLate=late_90,
            NumberRealEstateLoansOrLines=re_lines,
            NumberOfTime60_89DaysPastDueNotWorse=late_60_89,
            NumberOfDependents=float(dependents)
        )

        with st.spinner("Executing Risk Scoring & SHAP Explainer..."):
            # Clean and feature engineer
            raw_dict = {
                "RevolvingUtilizationOfUnsecuredLines": utilization,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
                "DebtRatio": debt_ratio,
                "MonthlyIncome": monthly_income,
                "NumberOfOpenCreditLinesAndLoans": open_lines,
                "NumberOfTimes90DaysLate": late_90,
                "NumberRealEstateLoansOrLines": re_lines,
                "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
                "NumberOfDependents": float(dependents),
            }
            raw_df = pd.DataFrame([raw_dict])
            clean_df = service.cleaner.transform(raw_df)
            feat_df = service.feature_engineer.transform(clean_df)

            # Score model
            pd_val = float(service.model.predict_proba(feat_df)[:, 1][0])
            eval_res = service.decision_engine.evaluate(pd_val)
            decision = eval_res["decision"]

            # Explain model
            explanation = service.explainer.explain_instance(feat_df, top_n=3, decision=decision)

            # Log decision
            service.logger.log_decision(
                application_id=app_id,
                probability_of_default=pd_val,
                decision=decision,
                risk_tier=eval_res["risk_tier"],
                recommended_interest_rate=eval_res["recommended_interest_rate"],
                reason_codes=explanation["reason_codes"],
                raw_features=raw_dict,
                latency_ms=12.5
            )

        # Store in session state for interactive what-if
        st.session_state["current_eval"] = {
            "eval_res": eval_res,
            "explanation": explanation,
            "raw_dict": raw_dict,
            "feat_df": feat_df,
            "pd_val": pd_val,
        }

    # Display results if available
    if "current_eval" in st.session_state:
        eval_data = st.session_state["current_eval"]
        res = eval_data["eval_res"]
        exp = eval_data["explanation"]
        pd_score = eval_data["pd_val"]

        st.markdown("### 2. Decision Summary Card")
        
        # Decision Badge Styling
        dec_color = {"APPROVE": "#28a745", "REFER": "#ffc107", "REJECT": "#dc3545"}[res["decision"]]
        badge_text_color = "#000" if res["decision"] == "REFER" else "#fff"

        st.markdown(
            f"""
            <div style="background-color: {dec_color}; color: {badge_text_color}; padding: 18px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800;">DECISION: {res['decision']}</h1>
                <p style="margin: 5px 0 0 0; font-size: 1.1rem;">{res['action_summary']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # KPI metric cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.metric("Probability of Default (PD)", f"{pd_score * 100:.2f}%")
        with kpi_col2:
            st.metric("Assigned Risk Tier", res["risk_tier_label"])
        with kpi_col3:
            st.metric("Recommended APR", res["recommended_rate_display"])
        with kpi_col4:
            st.metric("Approval Cutoff Threshold", f"{res['approve_threshold'] * 100:.2f}%")

        # Gauge Chart for Probability of Default
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pd_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Default Risk Score Meter (%)", 'font': {'size': 18}},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 50], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': dec_color},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, res['approve_threshold'] * 100], 'color': "rgba(40, 167, 69, 0.25)"},
                    {'range': [res['approve_threshold'] * 100, res['reject_threshold'] * 100], 'color': "rgba(255, 193, 7, 0.25)"},
                    {'range': [res['reject_threshold'] * 100, 50], 'color': "rgba(220, 53, 69, 0.25)"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': pd_score * 100
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Adverse Action Reason Codes Section (FR3, FR4)
        st.markdown("### 3. Adverse Action & Credit Reason Codes (FCRA / ECOA Compliance)")
        st.markdown(
            "Under the Equal Credit Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA), "
            "adverse decisions must provide the principal reasons that contributed to the scoring."
        )

        for item in exp["reason_codes"]:
            st.info(f"**Reason #{item['rank']}:** {item['reason_code']} *(Feature: `{item['feature']}` = {item['feature_value']}, SHAP Impact: `{item['shap_impact']:+.4f}`)*")

        # Feature Attribution Waterfall / Bar Chart
        st.markdown("### 4. Local SHAP Attribution Breakdown")
        top_contribs = exp["feature_contributions"][:10]
        feats = [c["feature"] for c in top_contribs][::-1]
        shap_vals = [c["shap_value"] for c in top_contribs][::-1]
        colors = ["#dc3545" if v > 0 else "#28a745" for v in shap_vals]

        fig_shap = go.Figure(go.Bar(
            x=shap_vals,
            y=feats,
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.3f}" for v in shap_vals],
            textposition='auto',
        ))
        fig_shap.update_layout(
            title="Top 10 Feature Contributions to Default Probability (SHAP)",
            xaxis_title="SHAP Impact (Red: Increases Default Risk, Green: Lowers Default Risk)",
            yaxis_title="Feature Name",
            height=380,
            margin=dict(l=10, r=10, t=40, b=30)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # What-If Interactive Scenario Simulator
        st.markdown("### 5. What-If Scenario Simulator")
        st.markdown(
            "Simulate how borrower profile modifications (e.g. paying down credit card balances or increasing verified income) "
            "would alter the credit decision in real time."
        )

        wcol1, wcol2, wcol3 = st.columns(3)
        with wcol1:
            sim_util = st.slider("Simulated Utilization:", 0.0, 1.5, float(eval_data["raw_dict"]["RevolvingUtilizationOfUnsecuredLines"]), 0.05)
        with wcol2:
            sim_income = st.slider("Simulated Monthly Income ($):", 1000.0, 25000.0, float(eval_data["raw_dict"]["MonthlyIncome"]), 500.0)
        with wcol3:
            sim_late30 = st.slider("Simulated 30-59 Days Late:", 0, 5, int(eval_data["raw_dict"]["NumberOfTime30-59DaysPastDueNotWorse"]), 1)

        sim_raw = eval_data["raw_dict"].copy()
        sim_raw["RevolvingUtilizationOfUnsecuredLines"] = sim_util
        sim_raw["MonthlyIncome"] = sim_income
        sim_raw["NumberOfTime30-59DaysPastDueNotWorse"] = sim_late30

        sim_clean = service.cleaner.transform(pd.DataFrame([sim_raw]))
        sim_feat = service.feature_engineer.transform(sim_clean)
        sim_pd = float(service.model.predict_proba(sim_feat)[:, 1][0])
        sim_eval = service.decision_engine.evaluate(sim_pd)

        delta_pd = sim_pd - pd_score
        st.markdown(
            f"""
            **Simulated Outcome:** Decision: **`{sim_eval['decision']}`** | New PD: **`{sim_pd * 100:.2f}%`** (Change: `{delta_pd * 100:+.2f}%`) | New Rate: **`{sim_eval['recommended_rate_display']}`**
            """
        )
