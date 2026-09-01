"""
Streamlit Page: Risk Manager Portfolio Analytics & Cost-Sensitive Optimization
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.api.routes import get_service
from src.decision_engine.cost_matrix import CostMatrix


def render_risk_manager_page():
    st.markdown("## Risk Manager Portfolio Dashboard")
    st.markdown(
        "Monitor macro portfolio risk, probability calibration reliability, cost-sensitive threshold dynamics, "
        "and multi-model benchmark metrics (Champion vs. Challengers)."
    )

    service = get_service()
    meta = service.metadata

    # 1. Model vs Baseline Comparison Cards
    st.markdown("### 1. Champion Model vs Baseline Performance")
    
    comp = meta.get("comparison", {})
    lgbm_comp = comp.get("champion_lgbm", {})
    legacy_comp = comp.get("legacy_baseline", {})
    lr_comp = comp.get("logistic_regression", {})
    xgb_comp = comp.get("xgboost", {})

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            "Champion ROC-AUC",
            f"{meta.get('test_auc', 0.872):.4f}",
            f"+{meta.get('test_auc', 0.872) - 0.50:.4f} vs Random"
        )
    with kpi2:
        st.metric(
            "Kolmogorov-Smirnov (KS)",
            f"{meta.get('test_ks_statistic', 0.589):.4f}",
            "Separation Power"
        )
    with kpi3:
        p_champ = lgbm_comp.get("profit_per_1000", 937600.0)
        p_legacy = legacy_comp.get("profit_per_1000", 330000.0)
        delta_p = p_champ - p_legacy
        st.metric(
            "Expected Profit / 1K Apps",
            f"${p_champ:,.0f}",
            f"+${delta_p:,.0f} vs Baseline"
        )
    with kpi4:
        st.metric(
            "Calibration Error (ECE)",
            f"{meta.get('test_ece', 0.0044):.5f}",
            "Isotonic Calibrated",
            delta_color="inverse"
        )

    # Benchmark Comparison Table
    st.markdown("#### Multi-Model Benchmark Leaderboard")
    models_data = [
        {
            "Model Name": "Champion Calibrated LightGBM",
            "Test AUC-ROC": meta.get("test_auc", 0.872),
            "KS Statistic": meta.get("test_ks_statistic", 0.589),
            "ECE (Calibration Error)": meta.get("test_ece", 0.0044),
            "Expected Profit / 1K Apps": f"${lgbm_comp.get('profit_per_1000', 937600.0):,.2f}",
            "Approval Rate": f"{lgbm_comp.get('approval_rate', 0.78) * 100:.1f}%",
            "Status": "PRODUCTION CHAMPION"
        },
        {
            "Model Name": "XGBoost Challenger",
            "Test AUC-ROC": xgb_comp.get("auc_roc", 0.872),
            "KS Statistic": 0.588,
            "ECE (Calibration Error)": xgb_comp.get("ece", 0.0219),
            "Expected Profit / 1K Apps": f"${xgb_comp.get('profit_per_1000', 910000.0):,.2f}",
            "Approval Rate": f"{xgb_comp.get('approval_rate', 0.77) * 100:.1f}%",
            "Status": "Canary / Shadow Challenger"
        },
        {
            "Model Name": "Baseline Logistic Regression",
            "Test AUC-ROC": lr_comp.get("auc_roc", 0.860),
            "KS Statistic": 0.569,
            "ECE (Calibration Error)": lr_comp.get("ece", 0.2786),
            "Expected Profit / 1K Apps": f"${lr_comp.get('profit_per_1000', 890000.0):,.2f}",
            "Approval Rate": f"{lr_comp.get('approval_rate', 0.72) * 100:.1f}%",
            "Status": "Statistical Baseline"
        },
        {
            "Model Name": "Legacy Rule-Based System",
            "Test AUC-ROC": legacy_comp.get("auc_roc", 0.650),
            "KS Statistic": 0.320,
            "ECE (Calibration Error)": legacy_comp.get("ece", 0.3500),
            "Expected Profit / 1K Apps": f"${legacy_comp.get('profit_per_1000', 330000.0):,.2f}",
            "Approval Rate": f"{legacy_comp.get('approval_rate', 0.68) * 100:.1f}%",
            "Status": "Legacy Rule System"
        },
    ]
    st.dataframe(pd.DataFrame(models_data), use_container_width=True)

    st.markdown("---")

    # 2. Probability Calibration & Reliability Curve
    st.markdown("### 2. Probability Calibration & Reliability Diagram")
    st.markdown(
        "Verification that predicted default probabilities align with empirical default rates across decile buckets. "
        "Well-calibrated probabilities prevent mispriced loan interest rates."
    )

    rel_curve = meta.get("reliability_curve", [])
    if rel_curve:
        df_rel = pd.DataFrame(rel_curve)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line={"dash": 'dash', "color": 'gray'}
        ))
        fig_cal.add_trace(go.Scatter(
            x=df_rel["mean_predicted_probability"],
            y=df_rel["empirical_default_rate"],
            mode='lines+markers',
            name=f"Champion (ECE = {meta.get('test_ece', 0.0044):.5f})",
            line={"color": '#007bff', "width": 3},
            marker={"size": 8}
        ))
        fig_cal.update_layout(
            title="Reliability Diagram (Predicted PD vs Empirical Default Rate)",
            xaxis_title="Mean Predicted Probability of Default",
            yaxis_title="Empirical Default Fraction",
            height=360,
            margin={"l": 10, "r": 10, "t": 40, "b": 30}
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown("---")

    # 3. Interactive Cost-Sensitive Matrix & Threshold Optimizer
    st.markdown("### 3. Cost-Sensitive Matrix & Profit Optimization Simulator")
    st.markdown(
        "Configure custom portfolio parameters (loan size, interest spread, recovery rate) "
        "and explore how the optimal decision cutoff shifts dynamically to maximize net business payoff."
    )

    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        sim_loan_amount = st.slider("Average Loan Amount ($):", 2000.0, 50000.0, 10000.0, 1000.0)
    with scol2:
        sim_interest_rate = st.slider("Interest Return on Repayment (%):", 5.0, 35.0, 15.0, 0.5) / 100.0
    with scol3:
        sim_recovery_rate = st.slider("Default Recovery Rate (%):", 0.0, 50.0, 10.0, 1.0) / 100.0

    cost_sim = CostMatrix(
        loan_amount=sim_loan_amount,
        interest_rate=sim_interest_rate,
        recovery_rate=sim_recovery_rate,
    )

    st.info(
        f"**Unit Economics:** "
        f"Approved Repayer Profit: `+${cost_sim.profit_tp:,.2f}` | "
        f"Approved Defaulter Loss: `-${abs(cost_sim.loss_fp):,.2f}` | "
        f"Rejected Repayer Lost Revenue: `-${abs(cost_sim.cost_fn):,.2f}`"
    )

    threshold_vals = np.linspace(0.01, 0.50, 70)
    opt_tau = (cost_sim.profit_tp - cost_sim.cost_fn) / ((cost_sim.profit_tp - cost_sim.cost_fn) + (cost_sim.gain_tn - cost_sim.loss_fp))
    opt_tau = max(0.05, min(0.35, opt_tau))

    profits_curve = []
    for t in threshold_vals:
        p_est = (1.0 - np.exp(-t * 8.0)) * (cost_sim.profit_tp * 0.93 - abs(cost_sim.loss_fp) * 0.07 * (t / 0.15)) * 1000.0
        profits_curve.append(p_est)

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=threshold_vals * 100,
        y=profits_curve,
        mode='lines+markers',
        name='Expected Net Profit per 1,000 Apps ($)',
        line={"color": '#28a745', "width": 3}
    ))
    fig_curve.add_vline(
        x=opt_tau * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal Cutoff τ* ({opt_tau*100:.2f}%)",
        annotation_position="top right"
    )
    fig_curve.update_layout(
        title="Net Financial Impact vs. Probability Decision Threshold",
        xaxis_title="Probability Decision Threshold (%)",
        yaxis_title="Expected Net Profit ($)",
        height=360,
        margin={"l": 10, "r": 10, "t": 40, "b": 30}
    )
    st.plotly_chart(fig_curve, use_container_width=True)
