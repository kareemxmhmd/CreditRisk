"""
Streamlit Page: Data Drift Monitoring & Live Decision Audit Logger
"""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.api.routes import get_service


def render_monitoring_page():
    st.markdown("## Data Drift & Real-Time Decision Audit Logger")
    st.markdown(
        "Continuous post-deployment monitoring for feature distribution shifts (Population Stability Index - PSI) "
        "and complete audit trail of every credit decision logged to SQLite."
    )

    service = get_service()

    mon_tab1, mon_tab2 = st.tabs(["Feature Drift Monitoring", "Decision Audit Trail"])

    with mon_tab1:
        st.markdown("### 1. Population Stability Index (PSI) & Statistical Drift Tests")
        st.markdown(
            "Detect covariate shifts between the training baseline population and live production applicant streams. "
            "**PSI < 0.10:** Stable | **0.10 <= PSI < 0.25:** Moderate Shift | **PSI >= 0.25:** Significant Drift (Action Required)."
        )

        logged_features_df = service.logger.get_all_logged_features()

        if logged_features_df.empty:
            st.info("No production decisions logged yet. Evaluating baseline sample stability benchmark.")
            # Use sample background if available
            sample_data = service.drift_detector.baseline_df.sample(min(150, len(service.drift_detector.baseline_df)), random_state=99)
            drift_report = service.drift_detector.evaluate_drift(sample_data)
        else:
            drift_report = service.drift_detector.evaluate_drift(logged_features_df)

        status = drift_report.get("status", "HEALTHY")
        if status == "HEALTHY":
            st.success("**Monitoring Status: HEALTHY** — No critical feature distribution drift detected.")
        elif "WARNING" in status:
            st.warning("**Monitoring Status: WARNING** — Moderate drift observed on one or more monitored features.")
        else:
            st.error("**Monitoring Status: CRITICAL ALERT** — Severe distribution shift detected. Retraining recommended.")

        # KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Features Monitored", drift_report.get("features_monitored", 0))
        with m2:
            st.metric("Features Drifted", drift_report.get("features_drifted", 0))
        with m3:
            st.metric("Monitored Sample Size", drift_report.get("monitored_samples_count", 0))
        with m4:
            st.metric("Drift Standard", "PSI + KS 2-Sample")

        # Drift table
        feat_metrics = drift_report.get("feature_metrics", [])
        if feat_metrics:
            drift_rows = []
            for fm in feat_metrics:
                status_label = {
                    "STABLE": "Stable",
                    "MODERATE_DRIFT": "Moderate Shift",
                    "CRITICAL_DRIFT": "Critical Drift"
                }.get(fm["drift_status"], fm["drift_status"])

                drift_rows.append({
                    "Feature Name": fm["feature"],
                    "PSI Score": f"{fm['psi']:.4f}",
                    "KS Test Stat": f"{fm['ks_statistic']:.4f}",
                    "KS p-value": f"{fm['ks_p_value']:.4e}",
                    "Wasserstein Dist": f"{fm['wasserstein_distance']:.4f}",
                    "Baseline Mean": f"{fm['baseline_mean']:.3f}",
                    "Current Mean": f"{fm['current_mean']:.3f}",
                    "Status": status_label,
                })
            st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)

            # Interactive Distribution Comparison Overlay
            st.markdown("### 2. Feature Distribution Overlay (Baseline vs. Production)")
            selected_feat = st.selectbox(
                "Select Feature to Inspect Distribution:",
                [fm["feature"] for fm in feat_metrics]
            )

            if selected_feat and service.drift_detector:
                base_vals = service.drift_detector.baseline_df[selected_feat].dropna().values
                curr_vals = logged_features_df[selected_feat].dropna().values if not logged_features_df.empty else base_vals * 1.05

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=base_vals,
                    name="Training Baseline",
                    opacity=0.6,
                    marker_color="#007bff",
                    nbinsx=30,
                    histnorm='probability'
                ))
                fig_dist.add_trace(go.Histogram(
                    x=curr_vals,
                    name="Production Stream",
                    opacity=0.6,
                    marker_color="#28a745",
                    nbinsx=30,
                    histnorm='probability'
                ))
                fig_dist.update_layout(
                    title=f"Distribution Comparison for '{selected_feat}'",
                    xaxis_title=selected_feat,
                    yaxis_title="Probability Density",
                    barmode='overlay',
                    height=340,
                    margin={"l": 10, "r": 10, "t": 40, "b": 30}
                )
                st.plotly_chart(fig_dist, use_container_width=True)

    with mon_tab2:
        st.markdown("### 3. Immutable Decision Audit Trail (SQLite)")
        st.markdown(
            "Every loan application scored via API or UI is persisted with model version, "
            "timestamp, input snapshots, and explainability reason codes for compliance and regulatory inspection."
        )

        recent_logs = service.logger.get_recent_decisions(limit=100)

        if recent_logs.empty:
            st.info("No decision records currently in audit database.")
        else:
            st.markdown(f"**Total Records in Audit Log:** `{len(recent_logs)}`")
            
            # Formatted table
            display_logs = recent_logs[[
                "id", "application_id", "timestamp", "decision",
                "probability_of_default", "risk_tier", "recommended_interest_rate", "latency_ms"
            ]].copy()
            display_logs["probability_of_default"] = display_logs["probability_of_default"].apply(lambda p: f"{p*100:.2f}%")
            display_logs["recommended_interest_rate"] = display_logs["recommended_interest_rate"].apply(lambda r: f"{r*100:.2f}% APR")
            
            st.dataframe(display_logs, use_container_width=True)

            # Record inspector
            st.markdown("#### Inspect Specific Application Record")
            selected_app_id = st.selectbox("Select Application ID to Inspect:", recent_logs["application_id"].tolist())
            if selected_app_id:
                row = recent_logs[recent_logs["application_id"] == selected_app_id].iloc[0]
                
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.markdown("**Application Snapshot:**")
                    st.json(json.loads(row["raw_features_json"]))
                with col_i2:
                    st.markdown("**Reason Codes & Attributions:**")
                    st.json(json.loads(row["reason_codes_json"]))
